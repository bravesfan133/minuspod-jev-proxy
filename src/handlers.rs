use axum::{
    Json,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::Instant;
use tokio::sync::Semaphore;

use crate::classify::{RequestKind, classify};
use crate::config::Config;
use crate::health::BackendHealth;
use crate::jev::{JevError, Question, decide, questions_map};
use crate::openai::{ChatRequest, TokenUsage, chat_response};
use crate::transcript::{
    Ad, ClassifiedSpan, Line, Segment, collect_sponsor_ads, detection_state, end_text_for,
    parse_transcript, round3, shrink_bounds, split_into_pieces, to_segments,
};

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<Config>,
    pub client: reqwest::Client,
    pub sem: Arc<Semaphore>,
    pub health: Arc<BackendHealth>,
}

impl AppState {
    /// Pause and unpause are recorded in memory. Integration tests use this
    /// so they never talk to a Docker daemon.
    pub fn for_tests(cfg: Arc<Config>, client: reqwest::Client, sem: Arc<Semaphore>) -> Self {
        let health = BackendHealth::recording(cfg.clone());
        Self { cfg, client, sem, health }
    }
}

static REVIEW_BOUNDS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"Original boundaries:\s*([\d.]+)\s*s\s*-\s*([\d.]+)\s*s").expect("bounds regex")
});

/// Jev choice options for segment classification.
///
/// These descriptions are the model's definition of each label, not hints.
/// A decision model scores the state against every option's text in parallel
/// and splits probability across all of them, so anything not named here has
/// nothing to match against and lands in whatever label is closest by
/// accident. Write them for a schema, not for a reader: each one says what
/// belongs in that label, what does not, and how it differs from its
/// neighbours.
///
/// `self_promo` in particular covers the show's own app, community, and
/// notification plugs, not just Patreon and merch. A narrow description here
/// caused a real miss: Jev read "subscribe, join the community, get push
/// notifications for live streams" as `content` at 0.97, because an app
/// subscribe was not in the list.
fn detection_criteria() -> HashMap<String, String> {
    [
        (
            "content",
            "The episode talking: news, interview, opinion, story, analysis, or \
             banter. Also a brand discussed as a subject, with no ask to subscribe, \
             download, visit, or buy.",
        ),
        (
            "host_read_sponsor",
            "The host reads a paid sponsor's pitch: the sponsor is named as the \
             thing being sold, and the listener is asked to go somewhere, buy \
             something, or enter a promo code.",
        ),
        (
            "inserted_ad",
            "A produced or dynamically inserted commercial: someone other than the \
             host selling, reading a script, or a break where the topic abruptly \
             changes to an advertiser.",
        ),
        (
            "self_promo",
            "The show or network asking the listener for their own stuff: its app, \
             Patreon or membership, merch, newsletter or mailing list, community \
             or group chat, notification bell, live stream, event, or back \
             catalogue. Also a link to the show's own site, show notes, or feed. \
             Trigger words: subscribe, join the community, sign up, get the app, \
             tap the bell, support us, member.",
        ),
        (
            "cross_promo",
            "A pitch for someone else's show, podcast, newsletter, or video, rather \
             than a paid sponsor and not this show's own material.",
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// Labels that count as removable.
///
/// `self_promo` is included deliberately: the show's own Patreon, merch, and
/// community plugs are cuts here, matching how MinusPod treats that category
/// and how this proxy behaved before the detection rewrite. `content` and
/// `cross_promo` are not cuts: ordinary editorial talk stays, and a pitch for
/// a different show is left alone.
fn is_paid_sponsor(choice: &str) -> bool {
    matches!(choice, "paid_ad" | "host_read_sponsor" | "inserted_ad" | "self_promo")
}

fn err(status: StatusCode, msg: String) -> Response {
    let body = serde_json::json!({"error": {"message": msg, "code": status.as_u16()}});
    (status, Json(body)).into_response()
}

/// Map Jev failures. Rate-limited on all backends => HTTP 429 so MinusPod
/// defers + retries the episode instead of dropping windows as failed.
fn jev_err(e: JevError) -> Response {
    match e {
        JevError::RateLimited { retry_after_secs, message } => {
            let mut headers = HeaderMap::new();
            if let Some(s) = retry_after_secs {
                if let Ok(v) = HeaderValue::from_str(&s.min(300).to_string()) {
                    headers.insert(axum::http::header::RETRY_AFTER, v);
                }
            }
            let body =
                serde_json::json!({"error": {"message": message, "code": 429}});
            (StatusCode::TOO_MANY_REQUESTS, headers, Json(body)).into_response()
        }
        JevError::Failed { message } => err(StatusCode::BAD_GATEWAY, message),
        // No Retry-After. A 429 here makes MinusPod defer for at most 300s
        // and try again, which is the loop this pause exists to stop.
        // Dead is internal: decide() turns it into failover or AllBackendsDead
        // before a handler sees it. If one leaks, it is still not retryable.
        JevError::Dead { message, .. } | JevError::AllBackendsDead { message } => {
            err(StatusCode::SERVICE_UNAVAILABLE, message)
        }
    }
}

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let t0 = Instant::now();
    if req.stream.unwrap_or(false) {
        return err(StatusCode::BAD_REQUEST, "streaming is not supported".into());
    }
    let model = if req.model.is_empty() { "jev".to_string() } else { req.model.clone() };
    match classify(&req) {
        RequestKind::Detection { verification } => {
            handle_detection(&state, &req, &model, verification, t0).await
        }
        RequestKind::Review => handle_review(&state, &req, &model, t0).await,
        RequestKind::CategoryRepair => handle_repair(&state, &req, &model, t0).await,
        RequestKind::TrimRecovery => {
            Json(chat_response(
                &model,
                r#"{"ad_start": null, "ad_end": null}"#.into(),
                TokenUsage::default(),
            ))
            .into_response()
        }
        RequestKind::Probe => {
            Json(chat_response(&model, r#"{"ok": true}"#.into(), TokenUsage::default()))
                .into_response()
        }
        RequestKind::Chapters => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "chapter generation is not served by the Jev proxy; MinusPod falls back to generic chapters".into(),
        ),
        RequestKind::Unknown => err(
            StatusCode::BAD_REQUEST,
            "unrecognized request type; refusing to guess".into(),
        ),
    }
}

#[allow(clippy::too_many_lines)]
async fn handle_detection(
    state: &AppState,
    req: &ChatRequest,
    model: &str,
    verification: bool,
    t0: Instant,
) -> Response {
    let cfg = &state.cfg;
    let lines = parse_transcript(&req.user_text());
    if lines.is_empty() {
        return err(StatusCode::UNPROCESSABLE_ENTITY, "no timestamped transcript lines found".into());
    }
    let span_secs = lines.last().map(|l| l.end).unwrap_or(0.0) - lines.first().map(|l| l.start).unwrap_or(0.0);
    let segments = to_segments(
        &lines,
        cfg.segment_target_secs,
        cfg.segment_max_secs,
        cfg.segment_gap_secs,
    );

    let verify_note = if verification {
        " This audio was already edited. The words [transition tone] are an edit marker, not an ad. A leftover promo code, URL, or partial sponsor sentence is still an ad."
    } else {
        ""
    };

    let mut spans: Vec<ClassifiedSpan> = Vec::new();
    let mut jev_ms_total: u64 = 0;
    let mut usage = TokenUsage::default();
    let mut backend_used = "primary";
    let criteria = detection_criteria();

    let mut offset = 0usize;
    for chunk in segments.chunks(cfg.max_segments_per_call.max(1)) {
        // State is this batch only. Questions name `segments[i].text`.
        let state_value = detection_state(&segments, offset, chunk);
        let mut qs = Vec::new();
        for j in 0..chunk.len() {
            // No path references and no repeated target text. The state already
            // carries the segment under `segments[j].text`, and a decision
            // model scores the state against each option rather than
            // dereferencing a pointer inside the instruction. Naming the path
            // adds characters and buys nothing.
            qs.push((
                format!("seg{j}"),
                Question::choice(
                    format!("What kind of speech is segments[{j}].text?"),
                    criteria.clone(),
                ),
            ));
            qs.push((
                format!("ad{j}"),
                Question::noul_with(
                    // Direct and positive. The old phrasing leaned on
                    // "judge this segment itself" and "are only the adjacent
                    // speech", which is scaffolding that helps a chat model and
                    // only dilutes schema scoring.
                    format!(
                        "Should segments[{j}].text be cut from the episode as \
                         advertising or self-promotion?{verify_note}"
                    ),
                    "A sponsor pitch, an inserted commercial, or the show plugging \
                     its own app, membership, community, or event",
                    "Episode talk that should stay: discussion, interview, opinion, \
                     or a brand mentioned without an ask",
                ),
            ));
        }
        let outcome = match decide(
            &state.client,
            cfg,
            &state.health,
            &state.sem,
            t0,
            &state_value,
            &questions_map(qs),
        )
        .await
        {
            Ok(o) => o,
            Err(e) => return jev_err(e),
        };
        jev_ms_total += outcome.latency_ms;
        usage.add(outcome.usage);
        backend_used = outcome.backend;
        for (j, seg) in chunk.iter().enumerate() {
            let noul = outcome
                .answers
                .get(&format!("ad{j}"))
                .and_then(|a| a.noul)
                .unwrap_or(0.0);
            let choice = outcome
                .answers
                .get(&format!("seg{j}"))
                .and_then(|a| a.choice.clone())
                .unwrap_or_else(|| "content".to_string());
            // Both signals have to agree. Noul alone was cutting lines the
            // label called the show. The label alone was dropping reads
            // whose yes-probability was already high.
            spans.push(ClassifiedSpan {
                segment: seg.clone(),
                noul,
                sponsor: is_paid_sponsor(&choice),
                choice,
            });
        }
        offset += chunk.len();
    }

    let ads = collect_sponsor_ads(
        &lines,
        &spans,
        cfg.recall_threshold,
        cfg.attach_threshold,
        cfg.attach_gap_secs,
    );

    // Precision pass. Detection spans flush around the target, so a cut can
    // still start or end mid-sentence. Re-judge just the head and tail of
    // each cut at finer granularity and pull the bounds inward.
    let trim = match edge_trim(state, &lines, ads, verification, t0).await {
        Ok(t) => t,
        Err(e) => return jev_err(e),
    };
    usage.add(trim.usage);

    let content = serde_json::to_string(&trim.ads).unwrap_or_else(|_| "[]".to_string());
    tracing::info!(
        kind = "detection",
        verification,
        backend = backend_used,
        span_secs = round3(span_secs),
        segments = spans.len(),
        candidates = trim.candidates,
        ads = trim.ads.len(),
        edge_pieces = trim.pieces,
        trimmed_secs = round3(trim.trimmed_secs),
        jev_ms = jev_ms_total,
        total_ms = t0.elapsed().as_millis() as u64,
        "detection request served"
    );
    Json(chat_response(model, content, usage)).into_response()
}

/// Outcome of the edge-trim pass, plus counters for the log line.
#[derive(Debug, Default)]
struct TrimOutcome {
    ads: Vec<Ad>,
    /// Candidate cuts that came out of the detection pass, before trimming.
    candidates: usize,
    /// Pieces re-judged by the classifier.
    pieces: usize,
    /// Total seconds removed from cut edges.
    trimmed_secs: f64,
    usage: TokenUsage,
}

/// One piece of a cut edge that needs a decision.
struct TrimPiece {
    /// Index of the ad this piece belongs to.
    ad: usize,
    /// Which end of that ad: 0 = head, 1 = tail.
    end_idx: usize,
    /// Position of this piece within its own end, so flags can be reassembled
    /// in the same order they were requested.
    slot: usize,
    start: f64,
    end: f64,
}

/// Re-judge the first and last seconds of each cut, then pull the bounds in.
///
/// Detection spans are ~4s, so a cut tends to start and end mid-sentence: the
/// read's opening clause is left in, and a little show talk is taken out with
/// it. This pass re-asks about just those ends at finer granularity.
///
/// Shrink-only. A piece the classifier calls content is evidence the cut was
/// too wide, so bounds only ever move inward. This can therefore improve a
/// cut but can never invent one, and can never grow a cut into show talk.
/// Neither end moves unless at least one of its own pieces was kept, so a
/// uniformly-rejected end is left alone rather than collapsing the cut.
///
/// A failure here must not cost the caller confirmed cuts: on error the
/// untrimmed cuts are returned. The exception is every backend being dead:
/// publishing the cuts would look like success and keep MinusPod running.
async fn edge_trim(
    state: &AppState,
    lines: &[Line],
    ads: Vec<Ad>,
    verification: bool,
    started: Instant,
) -> Result<TrimOutcome, JevError> {
    let cfg = &state.cfg;
    let mut out = TrimOutcome {
        candidates: ads.len(),
        ..Default::default()
    };
    if ads.is_empty() {
        return Ok(out);
    }

    let verify_note = if verification {
        " This audio was already edited. [transition tone] is an edit marker, not an ad."
    } else {
        ""
    };
    let criteria = detection_criteria();

    // Re-judge a few pieces' worth of each end.
    let depth = (cfg.edge_piece_secs * 3.0).max(cfg.edge_piece_secs);

    // Per ad, per end: the pieces we asked about and their keep flags. The
    // pieces are retained so bounds can be reassembled from the exact
    // intervals the classifier saw, rather than by re-splitting and hoping
    // the second split matches the first. Flags start false so a piece whose
    // answer never arrives cannot be mistaken for "kept".
    let mut head: Vec<(Vec<Segment>, Vec<bool>)> = Vec::with_capacity(ads.len());
    let mut tail: Vec<(Vec<Segment>, Vec<bool>)> = Vec::with_capacity(ads.len());
    for ad in &ads {
        let head_end = (ad.start + depth).min(ad.end);
        let tail_start = (ad.end - depth).max(ad.start);
        let h = split_into_pieces(lines, ad.start, head_end, cfg.edge_piece_secs);
        let t = split_into_pieces(lines, tail_start, ad.end, cfg.edge_piece_secs);
        head.push((h.clone(), vec![false; h.len()]));
        tail.push((t.clone(), vec![false; t.len()]));
    }

    // Build the work list, grouped by context so each Decisions call carries
    // one state. Context differs between ads and between the two ends of a
    // long cut, so group on the exact window rather than assuming.
    struct Group {
        context: String,
        pieces: Vec<TrimPiece>,
    }
    let mut groups: Vec<Group> = Vec::new();

    for (i, ad) in ads.iter().enumerate() {
        let head_end = (ad.start + depth).min(ad.end);
        let tail_start = (ad.end - depth).max(ad.start);
        for (end, (lo, hi)) in [(0usize, (ad.start, head_end)), (1usize, (tail_start, ad.end))] {
            if hi - lo <= 0.0 {
                continue;
            }
            let pieces = split_into_pieces(lines, lo, hi, cfg.edge_piece_secs);
            if pieces.is_empty() {
                continue;
            }
            let context = lines
                .iter()
                .filter(|l| {
                    l.end > lo - cfg.edge_context_secs && l.start < hi + cfg.edge_context_secs
                })
                .map(|l| format!("[{:.1}s - {:.1}s] {}", l.start, l.end, l.text))
                .collect::<Vec<_>>()
                .join("\n");
            groups.push(Group {
                context,
                pieces: pieces
                    .iter()
                    .enumerate()
                    .map(|(slot, p)| TrimPiece {
                        ad: i,
                        end_idx: end,
                        slot,
                        start: p.start,
                        end: p.end,
                    })
                    .collect(),
            });
        }
    }
    out.pieces = groups.iter().map(|g| g.pieces.len()).sum();

    for group in &groups {
        for batch in group.pieces.chunks(cfg.max_segments_per_call.max(1)) {
let mut qs = Vec::new();
            for (j, piece) in batch.iter().enumerate() {
                // The state for this group is the surrounding transcript and
                // the piece text is already inside it, so the instruction names
                // the position without repeating the words. The old phrasing
                // embedded the whole piece here as well, which duplicated it.
                let label =
                    format!("[{:.1}s - {:.1}s]", piece.start, piece.end);
                qs.push((
                    format!("p{j}"),
                    Question::noul_with(
                        format!(
                            "Should the text at {label} be cut from the episode as \
                             advertising or self-promotion?{verify_note}"
                        ),
                        "A sponsor pitch, an inserted commercial, or the show plugging \
                         its own app, membership, community, or event",
                        "Episode talk that should stay: discussion, interview, opinion, \
                         or a brand mentioned without an ask",
                    ),
                ));
                qs.push((
                    format!("c{j}"),
                    Question::choice(
                        format!("What kind of speech is the text at {label}?"),
                        criteria.clone(),
                    ),
                ));
            }

            let outcome = match decide(
                &state.client,
                cfg,
                &state.health,
                &state.sem,
                started,
                &Value::String(group.context.clone()),
                &questions_map(qs),
            )
            .await
            {
                Ok(o) => o,
                Err(e @ JevError::AllBackendsDead { .. }) => return Err(e),
                Err(e) => {
                    tracing::warn!("edge trim decide failed, keeping untrimmed cuts: {e}");
                    out.ads = ads;
                    return Ok(out);
                }
            };
            out.usage.add(outcome.usage);

            for (j, piece) in batch.iter().enumerate() {
                let noul = outcome
                    .answers
                    .get(&format!("p{j}"))
                    .and_then(|a| a.noul)
                    .unwrap_or(0.0);
                let choice = outcome
                    .answers
                    .get(&format!("c{j}"))
                    .and_then(|a| a.choice.clone())
                    .unwrap_or_default();
                // Both signals must agree, same rule as the detection gate.
                let keep = noul >= cfg.edge_threshold && is_paid_sponsor(&choice);
                let ends: &mut [(Vec<Segment>, Vec<bool>)] = if piece.end_idx == 0 {
                    &mut head
                } else {
                    &mut tail
                };
                if let Some((_, flags)) = ends.get_mut(piece.ad) {
                    if let Some(slot) = flags.get_mut(piece.slot) {
                        *slot = keep;
                    }
                }
            }
        }
    }

    for (i, ad) in ads.iter().enumerate() {
        let (head_pieces, head_flags) = &head[i];
        let (tail_pieces, tail_flags) = &tail[i];

        let (new_start, new_end) =
            shrink_bounds(ad, head_pieces, head_flags, tail_pieces, tail_flags);

        if new_start > ad.start || new_end < ad.end {
            out.trimmed_secs += (new_start - ad.start).max(0.0) + (ad.end - new_end).max(0.0);
            let mut trimmed = ad.clone();
            trimmed.start = new_start;
            trimmed.end = new_end;
            // end_text is a required field and must describe the span actually
            // cut, not the pre-trim bounds.
            if let Some(t) = end_text_for(lines, new_start, new_end) {
                trimmed.end_text = t;
            }
            out.ads.push(trimmed);
        } else {
            out.ads.push(ad.clone());
        }
    }

    Ok(out)
}

async fn handle_review(
    state: &AppState,
    req: &ChatRequest,
    model: &str,
    t0: Instant,
) -> Response {
    let cfg = &state.cfg;
    let user = req.user_text();
    let caps = match REVIEW_BOUNDS_RE.captures(&user) {
        Some(c) => c,
        None => {
            // Cannot confirm without bounds; fail loudly so MinusPod keeps
            // the candidate via its native failure path.
            return err(StatusCode::UNPROCESSABLE_ENTITY, "cannot parse original boundaries".into());
        }
    };
    let orig_start: f64 = caps.get(1).and_then(|m| m.as_str().parse().ok()).unwrap_or(0.0);
    let orig_end: f64 = caps.get(2).and_then(|m| m.as_str().parse().ok()).unwrap_or(0.0);

    let qs = questions_map(vec![(
        "is_ad".to_string(),
        Question::noul_with(
            format!("The transcript below marks a candidate ad span [{orig_start:.1}s - {orig_end:.1}s]. Is that span primarily a paid sponsor read or inserted commercial that should be removed?"),
            "The span is a paid ad and should be cut",
            "The span should stay: show talk, a sign-off, or the show promoting itself",
        ),
    )]);
    let outcome = match decide(
        &state.client,
        cfg,
        &state.health,
        &state.sem,
        t0,
        &Value::String(user),
        &qs,
    )
    .await
    {
        Ok(o) => o,
        Err(e) => return jev_err(e),
    };
    let noul = outcome.answers.get("is_ad").and_then(|a| a.noul).unwrap_or(0.0);
    // Empty array = reject; element with is_ad = confirm original bounds.
    // Jev never adjusts bounds (it cannot emit timestamps).
    let content = if noul >= cfg.review_threshold {
        serde_json::json!([{
            "is_ad": true,
            "start": orig_start,
            "end": orig_end,
            "confidence": round3(noul),
            "reason": format!("jev review p={noul:.2}, bounds unchanged"),
        }])
        .to_string()
    } else {
        "[]".to_string()
    };
    tracing::info!(
        kind = "review",
        backend = outcome.backend,
        noul = round3(noul),
        jev_ms = outcome.latency_ms,
        total_ms = t0.elapsed().as_millis() as u64,
        "review request served"
    );
    Json(chat_response(model, content, outcome.usage)).into_response()
}

#[derive(Debug, serde::Deserialize)]
struct RepairItem {
    index: i64,
    #[serde(default)]
    start: Option<f64>,
    #[serde(default)]
    end: Option<f64>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    end_text: Option<String>,
}

fn repair_categories() -> HashMap<String, String> {
    [
        ("sponsor", "A paid host read, produced ad spot, dynamically inserted ad, or platform-inserted ad."),
        ("cross_promo", "A produced segment promoting a different show, inserted by platform or network."),
        ("self_promo", "The show promotes its own other content: another show, Patreon, merch, mailing list."),
        ("interaction", "Asks listeners to subscribe, rate, review, or follow the show."),
        ("intro", "Opening theme music and/or host introduction."),
        ("outro", "Closing credits, sign-off, or theme music."),
        ("recap", "A 'coming up' preview, headline bumper, or 'listen next' segment."),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// Extract the `[{index,...}]` array after "Segments needing a category:".
fn extract_repair_items(user: &str) -> Option<Vec<RepairItem>> {
    let marker = "Segments needing a category:";
    let pos = user.find(marker)?;
    let rest = &user[pos + marker.len()..];
    let start = rest.find('[')?;
    let bytes = rest.as_bytes();
    let mut depth = 0;
    let mut in_str = false;
    let mut esc = false;
    let mut end = None;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
        } else if b == b'"' {
            in_str = true;
        } else if b == b'[' {
            depth += 1;
        } else if b == b']' {
            depth -= 1;
            if depth == 0 {
                end = Some(i);
                break;
            }
        }
    }
    let json = &rest[start..=end?];
    serde_json::from_str(json).ok()
}

async fn handle_repair(
    state: &AppState,
    req: &ChatRequest,
    model: &str,
    t0: Instant,
) -> Response {
    let cfg = &state.cfg;
    let user = req.user_text();
    let items = match extract_repair_items(&user) {
        Some(v) if !v.is_empty() => v,
        _ => {
            // Native path defaults unrepaired ads to sponsor.
            return err(StatusCode::UNPROCESSABLE_ENTITY, "cannot parse repair segments".into());
        }
    };
    let excerpt: String = user.lines().take(40).collect::<Vec<_>>().join("\n");
    let mut qs = Vec::new();
    for item in &items {
        let desc = format!(
            "Segment index {} [{:?}s - {:?}s]: {} Last words: {}",
            item.index,
            item.start.unwrap_or(-1.0),
            item.end.unwrap_or(-1.0),
            item.reason.clone().unwrap_or_default(),
            item.end_text.clone().unwrap_or_default()
        );
        qs.push((
            format!("cat{}", item.index),
            Question::choice(
                format!("Assign exactly one category to this already-identified ad segment. Do NOT detect new segments. {desc}\nTranscript excerpt:\n{excerpt}"),
                repair_categories(),
            ),
        ));
    }
    let outcome = match decide(
        &state.client,
        cfg,
        &state.health,
        &state.sem,
        t0,
        &Value::String(excerpt),
        &questions_map(qs),
    )
    .await
    {
        Ok(o) => o,
        Err(e) => return jev_err(e),
    };
    let mut out = Vec::new();
    for item in &items {
        let cat = outcome
            .answers
            .get(&format!("cat{}", item.index))
            .and_then(|a| a.choice.clone())
            .unwrap_or_else(|| "sponsor".to_string());
        out.push(serde_json::json!({"index": item.index, "category": cat}));
    }
    tracing::info!(
        kind = "repair",
        backend = outcome.backend,
        items = items.len(),
        jev_ms = outcome.latency_ms,
        total_ms = t0.elapsed().as_millis() as u64,
        "repair request served"
    );
    Json(chat_response(model, Value::Array(out).to_string(), outcome.usage)).into_response()
}

pub async fn models() -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": "jev-ad-detection",
            "object": "model",
            "created": 0,
            "owned_by": "minuspod-jev-proxy",
        }],
    }))
}

pub async fn status(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.health.status().await)
}

pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "primary": {"base_url": state.cfg.primary.base_url, "model": state.cfg.primary.model},
        "secondary": {"base_url": state.cfg.secondary.base_url, "model": state.cfg.secondary.model},
        // Reported so it is visible whether the third slot is configured.
        // base_url is empty when the slot is unused.
        "tertiary": {"base_url": state.cfg.tertiary.base_url, "model": state.cfg.tertiary.model},
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::last_words;

    #[test]
    fn repair_items_extracted() {
        let user = "Transcript excerpt:\nxxx\n\nSegments needing a category:\n[{\"index\": 2, \"start\": 10.0, \"end\": 20.0, \"reason\": \"r\", \"end_text\": \"buy now [bracket]\"}]";
        let items = extract_repair_items(user).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].index, 2);
    }

    #[test]
    fn review_bounds_parsed() {
        let caps = REVIEW_BOUNDS_RE
            .captures("Original boundaries: 10.50s-50.25s")
            .unwrap();
        assert_eq!(&caps[1], "10.50");
        assert_eq!(&caps[2], "50.25");
    }

    #[test]
    fn choice_mapping() {
        // Paid sponsors and the show's own plugs are all removable.
        assert!(is_paid_sponsor("paid_ad"));
        assert!(is_paid_sponsor("host_read_sponsor"));
        assert!(is_paid_sponsor("inserted_ad"));
        assert!(is_paid_sponsor("self_promo"));
        // Editorial talk and someone else's show are left alone.
        assert!(!is_paid_sponsor("cross_promo"));
        assert!(!is_paid_sponsor("content"));
        assert!(!is_paid_sponsor("uncertain"));
    }

    #[test]
    fn last_words_import_used() {
        assert_eq!(last_words("a b c", 2), "b c");
    }
}

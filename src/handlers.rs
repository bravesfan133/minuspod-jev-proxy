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
use crate::jev::{JevError, Question, decide, questions_map};
use crate::openai::{ChatRequest, chat_response};
use crate::transcript::{
    Ad, Segment, context_for, end_text_for, merge_trimmed_run, parse_transcript, round3,
    runs_of, split_span, to_segments,
};

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<Config>,
    pub client: reqwest::Client,
    pub sem: Arc<Semaphore>,
}

static REVIEW_BOUNDS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"Original boundaries:\s*([\d.]+)\s*s\s*-\s*([\d.]+)\s*s").expect("bounds regex")
});

/// Jev choice options for segment classification.
fn detection_criteria() -> HashMap<String, String> {
    [
        ("content", "Normal editorial show content: discussion, interviews, news, jokes, stories, or incidental brand mentions without any sales pitch."),
        ("paid_ad", "A paid sponsor read or produced commercial for an external product/service."),
        ("host_read_sponsor", "The host personally endorses a sponsor with a call to action (URL, promo code, 'go to')."),
        ("inserted_ad", "A dynamically/platform-inserted commercial break, jarring topic shift to an advertiser."),
        ("self_promo", "The show promotes its own other content: Patreon, merch, mailing list, live shows."),
        ("cross_promo", "Promotion of a different show or network content."),
        ("uncertain", "Cannot tell from this segment alone whether it is an ad or content."),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// Map a Jev choice to a MinusPod category. None = not an ad.
fn map_choice(choice: &str) -> Option<&'static str> {
    match choice {
        "paid_ad" | "host_read_sponsor" | "inserted_ad" => Some("sponsor"),
        "self_promo" => Some("self_promo"),
        "cross_promo" => Some("cross_promo"),
        _ => None,
    }
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
            Json(chat_response(&model, r#"{"ad_start": null, "ad_end": null}"#.into()))
                .into_response()
        }
        RequestKind::Probe => {
            Json(chat_response(&model, r#"{"ok": true}"#.into())).into_response()
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
    let segments = to_segments(&lines, cfg.segment_target_secs);
    let state_text = lines
        .iter()
        .map(|l| format!("[{:.1}s - {:.1}s] {}", l.start, l.end, l.text))
        .collect::<Vec<_>>()
        .join("\n");
    let state_value = Value::String(state_text);

    let verify_note = if verification {
        " This audio already had ads removed; [transition tone] markers are edit points, NOT ads. Orphaned URLs, promo codes, or partial sponsor reads ARE missed-ad fragments."
    } else {
        ""
    };

    let mut decisions: Vec<(Segment, Option<(f64, String, String)>)> = Vec::new();
    let mut jev_ms_total: u64 = 0;
    let mut backend_used = "primary";
    let criteria = detection_criteria();

    for chunk in segments.chunks(cfg.max_segments_per_call.max(1)) {
        let mut qs = Vec::new();
        for (j, seg) in chunk.iter().enumerate() {
            let label = format!("[{:.1}s - {:.1}s]: {}", seg.start, seg.end, seg.text);
            qs.push((
                format!("seg{j}"),
                Question::choice(
                    format!("Classify this target segment in its transcript context. Target segment {label}"),
                    criteria.clone(),
                ),
            ));
            qs.push((
                format!("ad{j}"),
                Question::noul_with(
                    format!("Is this target segment removable advertising/promotional content rather than normal editorial show content? A host discussing a company as normal content is NOT an ad. Paid sponsor reads, inserted commercials, and explicit promotional calls to action ARE ads.{verify_note} Target segment {label}"),
                    "The segment is primarily an ad/promo that should be removed",
                    "The segment is normal editorial content that must be kept",
                ),
            ));
        }
        let outcome =
            match decide(&state.client, cfg, &state.sem, &state_value, &questions_map(qs)).await
            {
                Ok(o) => o,
                Err(e) => return jev_err(e),
            };
        jev_ms_total += outcome.latency_ms;
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
                .unwrap_or_else(|| "uncertain".to_string());
            match map_choice(&choice) {
                Some(cat) if noul >= cfg.ad_threshold => {
                    let reason = format!("jev:{choice} p={noul:.2}");
                    decisions.push((seg.clone(), Some((noul, cat.to_string(), reason))));
                }
                _ => decisions.push((seg.clone(), None)),
            }
        }
    }

    // L2 edge pass: subdivide the first/last block of each ad run into
    // ~5s pieces and re-ask Jev per piece with tight local context.
    // Shrink-only: edges move inward, never outward.
    let runs = runs_of(&decisions);
    let mut ads: Vec<Ad> = Vec::new();
    let mut l2_pieces = 0usize;
    let mut l2_trimmed_secs = 0.0f64;
    for run in &runs {
        let first = &run[0].0;
        let last = &run[run.len() - 1].0;
        let lead = split_span(&lines, first.start, first.end, cfg.edge_piece_secs);
        let tail = if run.len() > 1 {
            split_span(&lines, last.start, last.end, cfg.edge_piece_secs)
        } else {
            Vec::new()
        };
        let state_value =
            Value::String(context_for(&lines, first.start, last.end, cfg.edge_context_secs));
        // (question key, piece index within combined edge list)
        let mut keys: Vec<(String, usize)> = Vec::new();
        let mut pieces: Vec<Segment> = Vec::new();
        for p in &lead {
            keys.push((format!("e{}", pieces.len()), pieces.len()));
            pieces.push(p.clone());
        }
        let tail_off = pieces.len();
        for p in &tail {
            keys.push((format!("e{}", pieces.len()), pieces.len()));
            pieces.push(p.clone());
        }
        let mut ad_flags = vec![false; pieces.len()];
        for group in keys.chunks(16) {
            let mut qs = Vec::new();
            for (qkey, pi) in group {
                let p = &pieces[*pi];
                qs.push((
                    format!("{qkey}p"),
                    Question::noul_with(
                        format!("This short piece [{:.1}s - {:.1}s]: \"{}\" — is it advertising content that should be removed? Judge ONLY this piece. An incidental brand mention without a sales pitch is content, NOT an ad.{verify_note}", p.start, p.end, p.text),
                        "The piece is ad content to remove",
                        "The piece is normal content to keep",
                    ),
                ));
                qs.push((
                    format!("{qkey}c"),
                    Question::choice(
                        format!("Classify this short piece [{:.1}s - {:.1}s]: \"{}\"", p.start, p.end, p.text),
                        criteria.clone(),
                    ),
                ));
            }
            let outcome =
                match decide(&state.client, cfg, &state.sem, &state_value, &questions_map(qs))
                    .await
                {
                    Ok(o) => o,
                    Err(e) => return jev_err(e),
                };
            jev_ms_total += outcome.latency_ms;
            backend_used = outcome.backend;
            for (qkey, pi) in group {
                let noul = outcome
                    .answers
                    .get(&format!("{qkey}p"))
                    .and_then(|a| a.noul)
                    .unwrap_or(0.0);
                let choice = outcome
                    .answers
                    .get(&format!("{qkey}c"))
                    .and_then(|a| a.choice.clone())
                    .unwrap_or_else(|| "uncertain".to_string());
                // Trim only on agreement: low probability AND non-ad choice.
                // Either signal alone is too noisy on 5s pieces.
                ad_flags[*pi] = noul >= cfg.edge_threshold && map_choice(&choice).is_some();
            }
        }
        l2_pieces += pieces.len();
        // Trim leading content pieces (lead) and trailing content pieces
        // (tail). Single-segment runs share one piece list for both ends.
        let lead_keep = &ad_flags[0..lead.len()];
        let tail_keep: &[bool] = if run.len() > 1 {
            &ad_flags[tail_off..]
        } else {
            &ad_flags[..]
        };
        let mut new_start = first.start;
        let mut new_end = last.end;
        match lead_keep.iter().position(|&k| k) {
            Some(k) => {
                if k > 0 {
                    l2_trimmed_secs += lead[k].start - first.start;
                    new_start = lead[k].start;
                }
            }
            None => {
                // Whole lead block is content: start at next block if any.
                if run.len() > 1 {
                    l2_trimmed_secs += run[1].0.start - first.start;
                    new_start = run[1].0.start;
                } else {
                    continue; // entire run is content: drop it.
                }
            }
        }
        match tail_keep.iter().rposition(|&k| k) {
            Some(k) => {
                let tail_pieces = if run.len() > 1 { &tail } else { &lead };
                if k + 1 < tail_pieces.len() {
                    l2_trimmed_secs += last.end - tail_pieces[k].end;
                    new_end = tail_pieces[k].end;
                }
            }
            None => {
                if run.len() > 1 {
                    l2_trimmed_secs += last.end - run[run.len() - 2].0.end;
                    new_end = run[run.len() - 2].0.end;
                } else {
                    continue;
                }
            }
        }
        if new_end > new_start {
            ads.push(merge_trimmed_run(run, new_start, new_end));
        }
    }
    // end_text must reflect the trimmed span, not the raw edge block.
    for ad in &mut ads {
        if let Some(t) = end_text_for(&lines, ad.start, ad.end) {
            ad.end_text = t;
        }
    }
    let content = serde_json::to_string(&ads).unwrap_or_else(|_| "[]".to_string());
    tracing::info!(
        kind = "detection",
        verification,
        backend = backend_used,
        span_secs = round3(span_secs),
        segments = decisions.len(),
        ads = ads.len(),
        l2_pieces = l2_pieces,
        l2_trimmed_secs = round3(l2_trimmed_secs),
        jev_ms = jev_ms_total,
        total_ms = t0.elapsed().as_millis() as u64,
        "detection request served"
    );
    Json(chat_response(model, content)).into_response()
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
            format!("The transcript below marks a candidate ad span [{orig_start:.1}s - {orig_end:.1}s]. Is that span primarily paid/promotional advertising content that should be removed?"),
            "The span is an ad and should be cut",
            "The span is normal content and must be kept",
        ),
    )]);
    let outcome = match decide(&state.client, cfg, &state.sem, &Value::String(user), &qs).await
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
    Json(chat_response(model, content)).into_response()
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
        &state.sem,
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
    Json(chat_response(model, Value::Array(out).to_string())).into_response()
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

pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "primary": {"base_url": state.cfg.primary.base_url, "model": state.cfg.primary.model},
        "secondary": {"base_url": state.cfg.secondary.base_url, "model": state.cfg.secondary.model},
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
        assert_eq!(map_choice("paid_ad"), Some("sponsor"));
        assert_eq!(map_choice("host_read_sponsor"), Some("sponsor"));
        assert_eq!(map_choice("inserted_ad"), Some("sponsor"));
        assert_eq!(map_choice("self_promo"), Some("self_promo"));
        assert_eq!(map_choice("cross_promo"), Some("cross_promo"));
        assert_eq!(map_choice("content"), None);
        assert_eq!(map_choice("uncertain"), None);
    }

    #[test]
    fn last_words_import_used() {
        assert_eq!(last_words("a b c", 2), "b c");
    }
}

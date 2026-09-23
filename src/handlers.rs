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
    ClassifiedSpan, collect_sponsor_ads, detection_state, parse_transcript, round3,
    to_segments,
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
        ("content", "Editorial speech: news, interview, opinion, story, or a brand mentioned with no ask to buy, visit, or use a code."),
        ("paid_ad", "A paid sponsor read or produced commercial that asks the listener to buy, visit, or use a code."),
        ("host_read_sponsor", "The host reads a sponsor pitch with a call to action, URL, or promo code."),
        ("inserted_ad", "A commercial break that is not the host's editorial topic."),
        ("self_promo", "The show promotes its own Patreon, merch, mailing list, or live event."),
        ("cross_promo", "A pitch for a different show or network."),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// Paid sponsor reads are the only cuts. The show promoting itself,
/// a sign-off, or ordinary talk is not a sponsor.
fn is_paid_sponsor(choice: &str) -> bool {
    matches!(choice, "paid_ad" | "host_read_sponsor" | "inserted_ad")
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
    let mut backend_used = "primary";
    let criteria = detection_criteria();

    let mut offset = 0usize;
    for chunk in segments.chunks(cfg.max_segments_per_call.max(1)) {
        // State is this batch only. Questions name `segments[i].text`.
        let state_value = detection_state(&segments, offset, chunk);
        let mut qs = Vec::new();
        for j in 0..chunk.len() {
            let path = format!("segments[{j}]");
            qs.push((
                format!("seg{j}"),
                Question::choice(
                    format!(
                        "Which label best describes `{path}.text`? `{path}.before` and `{path}.after` are only the adjacent speech."
                    ),
                    criteria.clone(),
                ),
            ));
            qs.push((
                format!("ad{j}"),
                Question::noul_with(
                    format!(
                        "Is `{path}.text` a paid advertisement that should be cut? `{path}.before` and `{path}.after` are only the adjacent speech, to show whether this stretch sits inside a pitch. Judge `{path}.text` itself.{verify_note}"
                    ),
                    "A paid sponsor read, host-read ad, or inserted commercial with a call to action, URL, or promo code",
                    "Show talk that should stay: news, interview, opinion, a sign-off, the show promoting itself, or a brand mentioned with no call to action",
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
        cfg.ad_threshold,
        cfg.attach_threshold,
        cfg.attach_gap_secs,
    );
    let content = serde_json::to_string(&ads).unwrap_or_else(|_| "[]".to_string());
    tracing::info!(
        kind = "detection",
        verification,
        backend = backend_used,
        span_secs = round3(span_secs),
        segments = spans.len(),
        ads = ads.len(),
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
            format!("The transcript below marks a candidate ad span [{orig_start:.1}s - {orig_end:.1}s]. Is that span primarily a paid sponsor read or inserted commercial that should be removed?"),
            "The span is a paid ad and should be cut",
            "The span should stay: show talk, a sign-off, or the show promoting itself",
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
        assert!(is_paid_sponsor("paid_ad"));
        assert!(is_paid_sponsor("host_read_sponsor"));
        assert!(is_paid_sponsor("inserted_ad"));
        assert!(!is_paid_sponsor("self_promo"));
        assert!(!is_paid_sponsor("cross_promo"));
        assert!(!is_paid_sponsor("content"));
        assert!(!is_paid_sponsor("uncertain"));
    }

    #[test]
    fn last_words_import_used() {
        assert_eq!(last_words("a b c", 2), "b c");
    }
}

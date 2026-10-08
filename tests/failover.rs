//! Failover across all three backend slots.
//!
//! The third slot exists so a new paid provider can be added without
//! displacing the keyless free tier that already sits in secondary. These
//! tests pin the two properties that make that safe: an unconfigured slot is
//! skipped, and a configured one is actually reached when the slots before it
//! fail.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
};
use serde_json::{Value, json};

use minuspod_jev_proxy::config::{Config, JevBackendCfg};
use minuspod_jev_proxy::handlers::AppState;
use minuspod_jev_proxy::openai::ChatRequest;

/// A transcript with one sponsor read, enough to make one Decisions call.
fn transcript_with_ad() -> String {
    let mut lines = Vec::new();
    for i in 0..10 {
        let s = i as f64 * 5.0;
        lines.push(format!(
            "[{s:.1}s - {:.1}s] So that is the thing about how the team ships.",
            s + 4.0
        ));
    }
    lines.push("[50.0s - 54.0s] This segment is brought to you by Acorns.".to_string());
    lines.push("[54.0s - 60.0s] Acorns is a money app, go to acorns.com/lenny, code ACORNS10.".to_string());
    for i in 0..10 {
        let s = 60.0 + i as f64 * 5.0;
        lines.push(format!("[{s:.1}s - {:.1}s] Anyway, back to the show.", s + 4.0));
    }
    lines.join("\n")
}

fn detection_request(transcript: &str) -> ChatRequest {
    ChatRequest {
        model: "jev-ad-detection".to_string(),
        messages: vec![
            minuspod_jev_proxy::openai::ChatMessage {
                role: "system".to_string(),
                content: json!("Identify all advertisement segments."),
            },
            minuspod_jev_proxy::openai::ChatMessage {
                role: "user".to_string(),
                content: json!(format!("Podcast: Test\nTranscript:\n{transcript}")),
            },
        ],
        temperature: None,
        stream: None,
        max_tokens: Some(2000),
        max_completion_tokens: None,
        response_format: Some(json!({
            "type": "json_schema",
            "json_schema": {"name": "ad_detection", "schema": {}},
        })),
    }
}

#[derive(Clone)]
struct Target {
    base_url: String,
    hits: Arc<AtomicUsize>,
}

/// A stub backend. Answers every question as "this is an ad", so any call
/// that lands here produces a cut and proves the slot was reached.
async fn answering_stub(State(hits): State<Arc<AtomicUsize>>, Json(body): Json<Value>) -> Json<Value> {
    hits.fetch_add(1, Ordering::SeqCst);
    let questions = body
        .get("questions")
        .and_then(|q| q.as_object())
        .cloned()
        .unwrap_or_default();
    let mut answers = serde_json::Map::new();
    for (key, q) in &questions {
        match q.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "noul" => {
                answers.insert(key.clone(), json!({"type": "noul", "noul": 0.95}));
            }
            "choice" => {
                answers.insert(
                    key.clone(),
                    json!({
                        "type": "choice",
                        "choice": "host_read_sponsor",
                        "probabilities": {"host_read_sponsor": 0.9, "content": 0.02},
                        "confidence": 0.9,
                    }),
                );
            }
            _ => {}
        }
    }
    Json(json!({
        "model": "stub",
        "answers": answers,
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }))
}

/// Spawn a stub and return its base URL plus a call counter.
async fn spawn_stub() -> Target {
    let hits = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route("/v1/systemone", post(answering_stub))
        .with_state(hits.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Target {
        base_url: format!("http://{addr}/v1/systemone"),
        hits,
    }
}

/// A backend that cannot connect. Used to exhaust the earlier slots.
fn dead_backend() -> JevBackendCfg {
    JevBackendCfg {
        // Port 1 refuses connections immediately, so failover is fast.
        base_url: "http://127.0.0.1:1/none".to_string(),
        model: "stub".to_string(),
        api_key: None,
    }
}

fn empty_backend() -> JevBackendCfg {
    JevBackendCfg {
        base_url: String::new(),
        model: String::new(),
        api_key: None,
    }
}

fn config(primary: JevBackendCfg, secondary: JevBackendCfg, tertiary: JevBackendCfg) -> Config {
    Config {
        port: 0,
        primary,
        secondary,
        tertiary,
        timeout_secs: 1,
        recall_threshold: 0.35,
        edge_threshold: 0.50,
        edge_piece_secs: 2.0,
        edge_context_secs: 45.0,
        attach_threshold: 0.40,
        attach_gap_secs: 8.0,
        review_threshold: 0.5,
        segment_target_secs: 4.0,
        segment_max_secs: 8.0,
        segment_gap_secs: 1.25,
        max_segments_per_call: 16,
        max_concurrent: 4,
        billing_dead_secs: 1800,
        client_timeout_secs: 600,
        hold_margin_secs: 30,
        minuspod_container: "minuspod".into(),
        docker_socket: String::new(),
    }
}

fn state_for(cfg: Config) -> AppState {
    let slots = cfg.max_concurrent;
    AppState::for_tests(
        Arc::new(cfg),
        reqwest::Client::new(),
        Arc::new(tokio::sync::Semaphore::new(slots)),
    )
}

async fn run(cfg: Config, transcript: &str) -> (axum::http::StatusCode, String) {
    let state = state_for(cfg);
    let resp = minuspod_jev_proxy::handlers::completions(
        axum::extract::State(state),
        Json(detection_request(transcript)),
    )
    .await;
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

#[tokio::test]
async fn tertiary_is_reached_when_primary_and_secondary_fail() {
    let t = spawn_stub().await;
    let cfg = config(dead_backend(), dead_backend(), JevBackendCfg {
        base_url: t.base_url.clone(),
        model: "stub".to_string(),
        api_key: None,
    });

    let (status, body) = run(cfg, &transcript_with_ad()).await;

    assert_eq!(status, 200, "third slot should have answered: {body}");
    assert!(
        t.hits.load(Ordering::SeqCst) > 0,
        "tertiary backend was never called",
    );
    assert!(body.contains("sponsor"), "expected a cut from the third slot: {body}");
}

#[tokio::test]
async fn unconfigured_tertiary_is_skipped_not_attempted() {
    // The regression this guards: an unset slot must not turn a working
    // two-backend setup into a failure.
    let p = spawn_stub().await;
    let cfg = config(
        JevBackendCfg { base_url: p.base_url.clone(), model: "stub".into(), api_key: None },
        empty_backend(),
        empty_backend(),
    );

    let (status, body) = run(cfg, &transcript_with_ad()).await;
    assert_eq!(status, 200, "primary should have answered: {body}");
    assert!(p.hits.load(Ordering::SeqCst) > 0);
}

#[tokio::test]
async fn secondary_is_reached_when_primary_fails() {
    // Adding the third slot must not change the existing two-backend order.
    let p = spawn_stub().await;
    let cfg = config(
        dead_backend(),
        JevBackendCfg { base_url: p.base_url.clone(), model: "stub".into(), api_key: None },
        empty_backend(),
    );

    let (status, body) = run(cfg, &transcript_with_ad()).await;
    assert_eq!(status, 200, "secondary should have answered: {body}");
    assert!(p.hits.load(Ordering::SeqCst) > 0);
}

#[tokio::test]
async fn all_slots_failing_is_a_bad_gateway_not_a_false_zero() {
    // If every backend is dead the proxy must fail loudly. A 200 with an empty
    // ad array would read as "no ads here" and keep every ad in the episode.
    let cfg = config(dead_backend(), dead_backend(), dead_backend());
    let (status, _body) = run(cfg, &transcript_with_ad()).await;
    assert_eq!(status, 502, "all backends down must surface as 502");
}

#[tokio::test]
async fn health_reports_all_three_slots() {
    let t = spawn_stub().await;
    let cfg = config(dead_backend(), dead_backend(), JevBackendCfg {
        base_url: t.base_url.clone(),
        model: "clef".into(),
        api_key: None,
    });
    let state = state_for(cfg);

    let resp = minuspod_jev_proxy::handlers::health(axum::extract::State(state))
        .await
        .into_response();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 16).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();

    // Every slot must be visible, and none of them may leak a key.
    for slot in ["primary", "secondary", "tertiary"] {
        assert!(v.get(slot).is_some(), "health is missing {slot}");
    }
    assert_eq!(v["tertiary"]["model"], "clef");
    let text = bytes_text(&v);
    assert!(
        !text.contains("api_key") && !text.contains("bearer"),
        "health must not expose credentials",
    );
}

fn bytes_text(v: &Value) -> String {
    v.to_string()
}

#[derive(Clone)]
struct Script {
    hits: Arc<AtomicUsize>,
    /// How many calls return the error before a normal answer.
    fail_for: usize,
    status: u16,
    body: Value,
    retry_after: Option<String>,
}

async fn scripted(State(script): State<Script>, Json(body): Json<Value>) -> impl IntoResponse {
    let n = script.hits.fetch_add(1, Ordering::SeqCst);
    if n < script.fail_for {
        let mut headers = HeaderMap::new();
        if let Some(ra) = &script.retry_after {
            headers.insert("retry-after", ra.parse().unwrap());
        }
        let status = StatusCode::from_u16(script.status).unwrap();
        return (status, headers, Json(script.body)).into_response();
    }
    let questions = body.get("questions").and_then(|q| q.as_object()).cloned().unwrap_or_default();
    let mut answers = serde_json::Map::new();
    for (key, q) in &questions {
        match q.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "noul" => {
                answers.insert(key.clone(), json!({"type": "noul", "noul": 0.95}));
            }
            "choice" => {
                answers.insert(key.clone(), json!({"type": "choice", "choice": "host_read_sponsor"}));
            }
            _ => {}
        }
    }
    (
        StatusCode::OK,
        HeaderMap::new(),
        Json(json!({
            "model": "stub",
            "answers": answers,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        })),
    )
        .into_response()
}

async fn spawn_script(script: Script) -> String {
    let app = Router::new()
        .route("/v1/systemone", post(scripted))
        .with_state(script);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    format!("http://{addr}/v1/systemone")
}

fn url_backend(url: String) -> JevBackendCfg {
    JevBackendCfg { base_url: url, model: "stub".into(), api_key: None }
}

async fn exchange(state: &AppState, transcript: &str) -> (StatusCode, HeaderMap, String) {
    let resp = minuspod_jev_proxy::handlers::completions(
        axum::extract::State(state.clone()),
        Json(detection_request(transcript)),
    )
    .await;
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, headers, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
async fn billing_failure_is_skipped_on_the_next_request() {
    let primary_hits = Arc::new(AtomicUsize::new(0));
    let primary = spawn_script(Script {
        hits: primary_hits.clone(),
        fail_for: usize::MAX,
        status: 402,
        body: json!({"error": "billing_error", "message": "no available TypeSafe API credits"}),
        retry_after: None,
    })
    .await;
    let secondary = spawn_stub().await;
    let mut cfg = config(
        url_backend(primary),
        url_backend(secondary.base_url.clone()),
        empty_backend(),
    );
    cfg.billing_dead_secs = 3600;
    cfg.client_timeout_secs = 30;
    let state = state_for(cfg);

    let (status, _, body) = exchange(&state, &transcript_with_ad()).await;
    assert_eq!(status, 200, "secondary should answer after primary billing failure: {body}");
    assert_eq!(primary_hits.load(Ordering::SeqCst), 1, "primary was retried");

    let primary_before = primary_hits.load(Ordering::SeqCst);
    let secondary_before = secondary.hits.load(Ordering::SeqCst);
    let (status, _, body) = exchange(&state, &transcript_with_ad()).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        primary_hits.load(Ordering::SeqCst),
        primary_before,
        "a dead primary must be skipped",
    );
    assert!(secondary.hits.load(Ordering::SeqCst) > secondary_before);
}

#[tokio::test]
async fn all_dead_is_503_without_retry_after_and_pauses() {
    let hits = Arc::new(AtomicUsize::new(0));
    let url = spawn_script(Script {
        hits: hits.clone(),
        fail_for: usize::MAX,
        status: 402,
        body: json!({"error": "billing_error", "message": "no available credits"}),
        retry_after: Some("300".into()),
    })
    .await;
    let mut cfg = config(url_backend(url), empty_backend(), empty_backend());
    // 30 min is longer than the client budget, so the request must not be held.
    cfg.billing_dead_secs = 1800;
    cfg.client_timeout_secs = 30;
    cfg.hold_margin_secs = 30;
    let state = state_for(cfg);

    let (status, headers, body) = exchange(&state, &transcript_with_ad()).await;
    assert_eq!(status, 503, "{body}");
    assert!(headers.get("retry-after").is_none(), "Retry-After must not be forwarded: {headers:?}");
    assert!(!body.to_lowercase().contains("rate limit"));
    assert_eq!(hits.load(Ordering::SeqCst), 1, "billing is not retried");

    let log = state.health.pause_log();
    {
        let g = log.lock().unwrap();
        assert!(g.events.iter().any(|e| e.action == "pause"), "{:?}", g.events);
        assert!(
            !g.events.iter().any(|e| e.action == "unpause"),
            "must not unpause inside the timeout"
        );
        assert!(g.paused);
    }

    let (status, headers, _) = exchange(&state, &transcript_with_ad()).await;
    assert_eq!(status, 503);
    assert!(headers.get("retry-after").is_none());
    assert_eq!(hits.load(Ordering::SeqCst), 1, "a second request must not call upstream");
}

#[tokio::test]
async fn daily_quota_stays_dead_until_utc_midnight() {
    let hits = Arc::new(AtomicUsize::new(0));
    let url = spawn_script(Script {
        hits: hits.clone(),
        fail_for: usize::MAX,
        status: 429,
        body: json!({
            "code": 4006,
            "message": "you have used up your daily free allocation of 10,000 neurons"
        }),
        retry_after: Some("300".into()),
    })
    .await;
    let mut cfg = config(empty_backend(), url_backend(url), empty_backend());
    cfg.client_timeout_secs = 30;
    cfg.hold_margin_secs = 30;
    let state = state_for(cfg);

    let (status, headers, body) = exchange(&state, &transcript_with_ad()).await;
    assert_eq!(status, 503, "{body}");
    assert!(headers.get("retry-after").is_none(), "Zen/Clef Retry-After must not be forwarded");

    let snap = state.health.status().await;
    let until = snap["backends"]["secondary"]["dead_until"].as_str().unwrap_or("");
    assert!(until.ends_with("T00:00:00Z"), "quota should last until midnight, got {until}");
    assert_eq!(snap["backends"]["secondary"]["kind"], "quota");
    assert_eq!(snap["backends"]["secondary"]["state"], "dead");
    assert_eq!(snap["minuspod_paused"], true);
    assert!(snap["next_unpause"].as_str().unwrap_or("").ends_with("T00:00:00Z"));

    let _ = exchange(&state, &transcript_with_ad()).await;
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn short_outage_holds_then_retries_the_same_request() {
    let hits = Arc::new(AtomicUsize::new(0));
    let url = spawn_script(Script {
        hits: hits.clone(),
        fail_for: 1,
        status: 402,
        body: json!({"error": "billing_error", "message": "no available credits"}),
        retry_after: None,
    })
    .await;
    let mut cfg = config(url_backend(url), empty_backend(), empty_backend());
    cfg.billing_dead_secs = 1;
    cfg.client_timeout_secs = 600;
    cfg.hold_margin_secs = 30;
    // One span, so the hold is the whole request.
    cfg.segment_target_secs = 600.0;
    cfg.segment_max_secs = 600.0;
    let state = state_for(cfg);

    let (status, _, body) = exchange(&state, &transcript_with_ad()).await;
    assert_eq!(status, 200, "held request should succeed after the dead window: {body}");
    assert!(hits.load(Ordering::SeqCst) >= 2, "the retry after unpause is the probe");
    let log = state.health.pause_log();
    let g = log.lock().unwrap();
    assert!(g.events.iter().any(|e| e.action == "pause"));
    assert!(g.events.iter().any(|e| e.action == "unpause"));
    assert!(!g.paused);
}
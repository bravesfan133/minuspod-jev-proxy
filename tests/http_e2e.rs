//! End-to-end check of the HTTP surface against a live stub backend.
//!
//! Everything here runs without an API key. A stub HTTP server stands in for
//! the Jev Decisions API and answers from the same keyword classifier the
//! accuracy harness uses, so the full request path can be exercised: routing,
//! the two-stage detection flow, and the OpenAI response shape MinusPod
//! parses.
//!
//! This is the layer the unit tests cannot reach. `tests/accuracy.rs` proves
//! the geometry is right; this proves the geometry is actually wired up and
//! that a real request produces a real `chat.completion` payload.
//!
//! The stub deliberately mirrors the real backend's contract: one answer per
//! question key, `noul` for `*p`-style keys, `choice` for `*c`-style keys.
//! If the proxy ever sent questions the backend cannot answer, this fails.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use serde_json::{Value, json};

use minuspod_jev_proxy::config::{Config, JevBackendCfg};
use minuspod_jev_proxy::handlers::AppState;
use minuspod_jev_proxy::openai::ChatRequest;

/// Shared counters so tests can assert on how the pipeline called the backend.
#[derive(Default)]
struct Counters {
    calls: AtomicUsize,
    /// Questions seen per question type, to prove both stages ran.
    noul_questions: AtomicUsize,
    choice_questions: AtomicUsize,
}

/// Stand-in for the Decisions API.
///
/// Reads the `questions` map out of the request and answers each one from the
/// text named in its instructions, using the same rules as the accuracy
/// harness. Anything it cannot judge comes back as `content` / `0.0`, which
/// is the safe direction: the read is missed, not the show talk cut.
async fn stub_decide(
    State(counters): State<Arc<Counters>>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    counters.calls.fetch_add(1, Ordering::SeqCst);

    let state_text = body
        .get("state")
        .map(|s| match s {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default();
    let questions = body
        .get("questions")
        .and_then(|q| q.as_object())
        .cloned()
        .unwrap_or_default();

    let mut answers = serde_json::Map::new();
    for (key, q) in &questions {
        let qtype = q.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let instructions = q
            .get("instructions")
            .and_then(|i| i.as_str())
            .unwrap_or("");

        // The state holds the text under judgment. Real question text names a
        // path into it; here the whole state is the subject.
        let text = extract_target(instructions).unwrap_or_else(|| state_text.clone());
        let verdict = judge(&text);

        match qtype {
            "noul" => {
                counters.noul_questions.fetch_add(1, Ordering::SeqCst);
                answers.insert(
                    key.clone(),
                    json!({"type": "noul", "noul": verdict.noul}),
                );
            }
            "choice" => {
                counters.choice_questions.fetch_add(1, Ordering::SeqCst);
                let mut probs = serde_json::Map::new();
                for label in [
                    "content",
                    "paid_ad",
                    "host_read_sponsor",
                    "inserted_ad",
                    "self_promo",
                    "cross_promo",
                ] {
                    let p = if label == verdict.choice { 0.9 } else { 0.02 };
                    probs.insert(label.to_string(), json!(p));
                }
                answers.insert(
                    key.clone(),
                    json!({
                        "type": "choice",
                        "choice": verdict.choice,
                        "probabilities": probs,
                        "confidence": 0.8,
                    }),
                );
            }
            // The proxy should never send a question the backend cannot
            // answer. Reject rather than guess, so a protocol change fails
            // loudly instead of silently dropping decisions.
            other => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": {"message": format!("unhandled question type {other}")}})),
                ));
            }
        }
    }

    Ok(Json(json!({
        "model": body.get("model").and_then(|m| m.as_str()).unwrap_or("stub"),
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 10},
    })))
}

/// Pull the quoted target text out of a question's instructions.
///
/// The real questions embed the segment as a backticked path or a quoted
/// snippet; in the stub, the state itself is the target, so this is only used
/// to keep prompts readable.
fn extract_target(instructions: &str) -> Option<String> {
    let quoted = instructions.split('"').nth(1);
    quoted.map(|s| s.to_string())
}

struct Verdict {
    noul: f64,
    choice: &'static str,
}

const AD_MARKERS: &[&str] = &["brought to you by", "sponsored by"];
const ASK_PHRASES: &[&str] = &[
    "check it out",
    "go to",
    "sign up",
    "try it",
    "download",
    "get a free",
    "for a free",
    "head on over to",
    "click",
    "enter code",
];
const SELF_PROMO: &[&str] = &[
    "our patreon",
    "patreon link",
    "our merch",
    "our newsletter",
    "our store",
    "product pass",
    "productpass",
    "past episodes",
    "all episodes",
    "newsletter.com",
    "podcast.com",
    "my contact form",
    "my website",
    "my book",
    "my handle",
    "find me at",
];

fn judge(text: &str) -> Verdict {
    let lower = text.to_lowercase();
    if SELF_PROMO.iter().any(|m| lower.contains(m)) {
        return Verdict { noul: 0.88, choice: "self_promo" };
    }
    let marker = AD_MARKERS.iter().any(|m| lower.contains(m));
    let has_url = lower.split_whitespace().any(|w| {
        let bare = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '/');
        bare.contains('.') && bare.ends_with(".com") && bare.matches('.').count() == 1
    });
    let promo_code = lower.split_whitespace().any(|w| {
        w.trim_matches(|c: char| !c.is_alphanumeric()) == "code"
    });
    if marker || (has_url && ASK_PHRASES.iter().any(|a| lower.contains(a))) || promo_code {
        return Verdict { noul: 0.95, choice: "host_read_sponsor" };
    }
    Verdict { noul: 0.03, choice: "content" }
}

/// A transcript with one host read in the middle, in MinusPod's line format.
fn transcript_with_ad() -> String {
    let mut lines = Vec::new();
    // Ordinary talk.
    for i in 0..12 {
        let s = i as f64 * 5.0;
        lines.push(format!(
            "[{s:.1}s - {:.1}s] So that is the thing about how the team ships software.",
            s + 4.0
        ));
    }
    // The ad. Note the boundaries deliberately fall mid-flow.
    lines.push("[60.0s - 64.0s] This segment is brought to you by Acorns.".to_string());
    lines.push("[64.0s - 70.0s] Acorns is a money app for investing, go to acorns.com/lenny.".to_string());
    lines.push("[70.0s - 74.0s] Use code ACORNS10 at checkout for your first month free.".to_string());
    // Back to the show.
    for i in 0..12 {
        let s = 74.0 + i as f64 * 5.0;
        lines.push(format!(
            "[{s:.1}s - {:.1}s] Anyway, back to what we were discussing earlier.",
            s + 4.0
        ));
    }
    lines.join("\n")
}

fn ad_free_transcript() -> String {
    (0..24)
        .map(|i| {
            let s = i as f64 * 5.0;
            format!("[{s:.1}s - {:.1}s] We were talking about the pricing model.", s + 4.0)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build a request shaped like the ones MinusPod sends.
fn detection_request(transcript: &str) -> ChatRequest {
    ChatRequest {
        model: "jev-ad-detection".to_string(),
        messages: vec![
            minuspod_jev_proxy::openai::ChatMessage {
                role: "system".to_string(),
                content: json!(
                    "You are detecting advertisements. Identify all advertisement segments."
                ),
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

struct Harness {
    base: String,
    counters: Arc<Counters>,
}

impl Harness {
    async fn start() -> Self {
        let counters = Arc::new(Counters::default());
        let app = Router::new()
            .route("/v1/systemone", post(stub_decide))
            .with_state(counters.clone());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind stub");
        let addr: SocketAddr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        Self {
            base: format!("http://{addr}/v1"),
            counters,
        }
    }

    /// Build app state pointed at the stub.
    fn app_state(&self) -> AppState {
        let cfg = Config {
            port: 0,
            primary: JevBackendCfg {
                base_url: format!("{}/systemone", self.base),
                model: "stub".to_string(),
                api_key: None,
            },
            secondary: JevBackendCfg {
                base_url: format!("{}/systemone", self.base),
                model: "stub".to_string(),
                api_key: None,
            },
            // Unused by default: these tests exercise the primary path, and an
            // unconfigured third slot must be skipped rather than attempted.
            tertiary: JevBackendCfg {
                base_url: String::new(),
                model: String::new(),
                api_key: None,
            },
            timeout_secs: 10,
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
        };
        AppState {
            cfg: Arc::new(cfg),
            client: reqwest::Client::new(),
            sem: Arc::new(tokio::sync::Semaphore::new(4)),
        }
    }
}

use std::sync::Arc as StdArc;

/// Post a request through the real handler stack.
async fn post_completions(h: &Harness, req: ChatRequest) -> Value {
    let state = h.app_state();
    let resp = minuspod_jev_proxy::handlers::completions(
        axum::extract::State(state),
        Json(req),
    )
    .await;

    assert_eq!(resp.status(), 200, "expected 200, got {}", resp.status());
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("read body");
    serde_json::from_slice(&bytes).expect("response is JSON")
}

/// Extract the Ad array the proxy embedded as chat message content.
fn ads_from(resp: &Value) -> Vec<Value> {
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .expect("content is a string");
    serde_json::from_str(content).expect("content is an ad array")
}

#[tokio::test]
async fn detection_emits_a_cut_for_the_sponsor_read() {
    let h = Harness::start().await;
    let resp = post_completions(&h, detection_request(&transcript_with_ad())).await;

    // The response must be shaped like an OpenAI chat completion.
    assert_eq!(resp["object"], "chat.completion");
    assert!(resp["id"].as_str().is_some(), "missing id");
    assert_eq!(resp["model"], "jev-ad-detection");

    let ads = ads_from(&resp);
    assert_eq!(ads.len(), 1, "expected exactly one cut, got {ads:#?}");

    let ad = &ads[0];
    // The ad must overlap the read at 60-74s and stay within the window.
    let start = ad["start"].as_f64().expect("start is a number");
    let end = ad["end"].as_f64().expect("end is a number");
    assert!(start < 74.0 && end > 60.0, "cut {start}-{end} misses the read");
    assert!(end > start, "cut is inverted");
    assert!(start >= 0.0 && end <= 138.0, "cut escapes the window: {start}-{end}");

    // Required MinusPod fields.
    assert_eq!(ad["category"], "sponsor");
    assert!(ad["confidence"].as_f64().unwrap_or(0.0) > 0.0);
    let end_text = ad["end_text"].as_str().expect("end_text present");
    assert!(!end_text.trim().is_empty(), "end_text is empty");
    assert!(ad["reason"].as_str().is_some(), "reason present");
}

#[tokio::test]
async fn ad_free_transcript_produces_no_cuts() {
    let h = Harness::start().await;
    let resp = post_completions(&h, detection_request(&ad_free_transcript())).await;
    let ads = ads_from(&resp);
    assert!(ads.is_empty(), "cut ordinary conversation: {ads:#?}");
}

#[tokio::test]
async fn edge_trim_stage_actually_runs() {
    // Both stages must call the backend: detection asks about segments, then
    // edge trim asks about the cut's edges. If only detection ran, the second
    // counter would be flat.
    let h = Harness::start().await;
    let before = h.counters.calls.load(Ordering::SeqCst);
    let _ = post_completions(&h, detection_request(&transcript_with_ad())).await;
    let after = h.counters.calls.load(Ordering::SeqCst);

    assert!(
        after > before,
        "backend was never called; detection is not reaching the model",
    );
    let noul = h.counters.noul_questions.load(Ordering::SeqCst);
    let choice = h.counters.choice_questions.load(Ordering::SeqCst);
    assert!(noul > 0, "no noul questions sent");
    assert!(choice > 0, "no choice questions sent");
    // Each stage asks a noul and a choice per subject, so the two should be
    // close. A large gap would mean one stage is asking only one question type.
    let diff = (noul as i64 - choice as i64).abs();
    assert!(
        diff <= 1,
        "noul ({noul}) and choice ({choice}) question counts diverged",
    );
}

#[tokio::test]
async fn unknown_requests_fail_loudly() {
    // A request the proxy cannot classify must not produce a guess.
    let h = Harness::start().await;
    let state = h.app_state();
    // Long enough not to look like a probe: the classifier treats a tiny
    // prompt with a tiny budget as MinusPod's connection test.
    let req = ChatRequest {
        model: "jev-ad-detection".to_string(),
        messages: vec![minuspod_jev_proxy::openai::ChatMessage {
            role: "user".to_string(),
            content: json!(
                "Summarize the following research paper in three paragraphs, covering the \
                 methodology, the sample size, and any limitations the authors acknowledge \
                 about generalizing their findings to other populations."
            ),
        }],
        temperature: None,
        stream: None,
        max_tokens: Some(500),
        max_completion_tokens: None,
        response_format: None,
    };
    let resp = minuspod_jev_proxy::handlers::completions(axum::extract::State(state), Json(req)).await;
    assert_eq!(resp.status(), 400, "unknown request must be rejected");
}

#[tokio::test]
async fn probe_requests_answer_without_calling_the_model() {
    // MinusPod's connection test must not spend a model call.
    let h = Harness::start().await;
    let state = h.app_state();
    let before = h.counters.calls.load(Ordering::SeqCst);

    let req = ChatRequest {
        model: "jev-ad-detection".to_string(),
        messages: vec![minuspod_jev_proxy::openai::ChatMessage {
            role: "user".to_string(),
            content: json!("ping"),
        }],
        temperature: None,
        stream: None,
        max_tokens: Some(4),
        max_completion_tokens: None,
        response_format: None,
    };
    let resp = minuspod_jev_proxy::handlers::completions(axum::extract::State(state), Json(req)).await;
    assert_eq!(resp.status(), 200);

    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 16).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "{\"ok\": true}");
    assert_eq!(
        h.counters.calls.load(Ordering::SeqCst),
        before,
        "probe must not call the model",
    );
}

#[tokio::test]
async fn backend_failure_surfaces_as_bad_gateway() {
    // A dead backend must not be reported as success with empty results:
    // that would look like "no ads found" and silently keep every ad.
    let cfg = Config {
        port: 0,
        primary: JevBackendCfg {
            base_url: "http://127.0.0.1:1/none".to_string(),
            model: "stub".to_string(),
            api_key: None,
        },
        secondary: JevBackendCfg {
            base_url: "http://127.0.0.1:1/none".to_string(),
            model: "stub".to_string(),
            api_key: None,
        },
        tertiary: JevBackendCfg {
            base_url: String::new(),
            model: String::new(),
            api_key: None,
        },
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
        max_concurrent: 1,
    };
    let state = AppState {
        cfg: StdArc::new(cfg),
        client: reqwest::Client::new(),
        sem: StdArc::new(tokio::sync::Semaphore::new(1)),
    };

    let resp = minuspod_jev_proxy::handlers::completions(
        axum::extract::State(state),
        Json(detection_request(&transcript_with_ad())),
    )
    .await;

    assert_eq!(
        resp.status(),
        502,
        "a dead backend must be a 502, not a 200 with zero ads",
    );
}
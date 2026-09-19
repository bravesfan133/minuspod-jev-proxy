use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

use crate::config::{Config, JevBackendCfg};

/// Jev question types. See https://docs.typesafe.ai/api
#[derive(Debug, Clone, Serialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub qtype: String,
    pub instructions: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Value>,
}

impl Question {
    pub fn noul_with(instructions: String, yes: &str, no: &str) -> Self {
        Self {
            qtype: "noul".into(),
            instructions,
            criteria: Some(serde_json::json!({"true": yes, "false": no})),
        }
    }

    pub fn choice(instructions: String, criteria: HashMap<String, String>) -> Self {
        Self {
            qtype: "choice".into(),
            instructions,
            criteria: Some(Value::Object(
                criteria.into_iter().map(|(k, v)| (k, Value::String(v))).collect(),
            )),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecideResponse {
    #[allow(dead_code)]
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub answers: HashMap<String, Answer>,
    #[allow(dead_code)]
    #[serde(default)]
    pub usage: Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Answer {
    #[allow(dead_code)]
    #[serde(rename = "type", default)]
    pub atype: String,
    #[serde(default)]
    pub noul: Option<f64>,
    #[serde(default)]
    pub choice: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub probabilities: Option<HashMap<String, f64>>,
    #[allow(dead_code)]
    #[serde(default)]
    pub confidence: Option<f64>,
}

#[derive(Debug)]
pub struct DecideOutcome {
    pub answers: HashMap<String, Answer>,
    /// "primary" or "secondary".
    pub backend: &'static str,
    pub latency_ms: u64,
}

/// RateLimited must reach MinusPod as HTTP 429 (it defers + retries the
/// episode) — never as 502 (windows get dropped as failed).
#[derive(Debug)]
pub enum JevError {
    RateLimited { retry_after_secs: Option<u64>, message: String },
    Failed { message: String },
}

impl std::fmt::Display for JevError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JevError::RateLimited { message, .. } => write!(f, "{message}"),
            JevError::Failed { message } => write!(f, "{message}"),
        }
    }
}

fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}

/// Backoff for attempt n (0-based), honoring server Retry-After, capped.
fn backoff_secs(attempt: u32, retry_after: Option<u64>) -> u64 {
    match retry_after {
        Some(s) => s.min(30),
        None => (1u64 << attempt.min(4)).min(8),
    }
}

/// Tiny std-only jitter so parallel retries don't march in lockstep.
fn jitter_ms() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0)
        .hash(&mut h);
    h.finish() % 500
}

fn classify_status(
    status: reqwest::StatusCode,
    retry_after: Option<u64>,
    body_head: String,
) -> JevError {
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status.as_u16() == 529
        || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
    {
        JevError::RateLimited {
            retry_after_secs: retry_after,
            message: format!("HTTP {status}: {body_head}"),
        }
    } else {
        JevError::Failed {
            message: format!("HTTP {status}: {body_head}"),
        }
    }
}

/// POST a Decisions request to one backend. No API keys are logged.
async fn decide_once(
    client: &Client,
    backend: &JevBackendCfg,
    state: &Value,
    questions: &Map<String, Value>,
    timeout: Duration,
) -> Result<DecideResponse, JevError> {
    let body = serde_json::json!({
        "model": backend.model,
        "state": state,
        "questions": questions,
    });
    let mut req = client.post(&backend.base_url).timeout(timeout).json(&body);
    if let Some(key) = backend.api_key.as_ref() {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.map_err(|e| JevError::Failed {
        message: format!("request failed: {e:?}"),
    })?;
    let status = resp.status();
    if !status.is_success() {
        let retry_after = parse_retry_after(resp.headers());
        let head: String = resp
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(300)
            .collect();
        return Err(classify_status(status, retry_after, head));
    }
    resp.json::<DecideResponse>().await.map_err(|e| JevError::Failed {
        message: format!("malformed Jev response: {e}"),
    })
}

/// Try primary, fall back to secondary. Each backend gets bounded retries
/// with backoff on rate limiting (per TypeSafe guidance). Calls are gated
/// by a semaphore so MinusPod's parallel windows don't burst the quota.
/// Returns Err only if BOTH backends fail; RateLimited if the failure was
/// quota/overload on both (MinusPod must defer, not drop).
pub async fn decide(
    client: &Client,
    cfg: &Config,
    sem: &Arc<Semaphore>,
    state: &Value,
    questions: &Map<String, Value>,
) -> Result<DecideOutcome, JevError> {
    const MAX_ATTEMPTS: u32 = 3;
    let timeout = Duration::from_secs(cfg.timeout_secs.max(5));
    let backends = [(&cfg.primary, "primary"), (&cfg.secondary, "secondary")];
    let mut last_err: Option<JevError> = None;
    let mut saw_rate_limited: Option<u64> = None;

    for (backend, name) in backends {
        for attempt in 0..MAX_ATTEMPTS {
            let result = {
                let _permit = sem.acquire().await.map_err(|_| JevError::Failed {
                    message: "semaphore closed".to_string(),
                })?;
                let t0 = Instant::now();
                let r = decide_once(client, backend, state, questions, timeout).await;
                r.map(|resp| (resp, t0.elapsed().as_millis() as u64))
            };
            match result {
                Ok((resp, latency_ms)) => {
                    return Ok(DecideOutcome {
                        answers: resp.answers,
                        backend: name,
                        latency_ms,
                    })
                }
                Err(JevError::RateLimited { retry_after_secs, message }) => {
                    tracing::warn!(
                        "jev {name} backend rate-limited (attempt {}/{MAX_ATTEMPTS}), backing off",
                        attempt + 1,
                    );
                    saw_rate_limited = Some(
                        saw_rate_limited
                            .unwrap_or(u64::MAX)
                            .min(retry_after_secs.unwrap_or(u64::MAX)),
                    );
                    last_err = Some(JevError::RateLimited {
                        retry_after_secs,
                        message,
                    });
                    if attempt + 1 < MAX_ATTEMPTS {
                        let sleep =
                            backoff_secs(attempt, retry_after_secs) * 1000 + jitter_ms();
                        tokio::time::sleep(Duration::from_millis(sleep)).await;
                        continue;
                    }
                    break;
                }
                Err(e) => {
                    tracing::warn!("jev {name} backend failed: {e}");
                    last_err = Some(e);
                    break;
                }
            }
        }
    }

    match (last_err, saw_rate_limited) {
        (Some(JevError::RateLimited { message, .. }), _) => Err(JevError::RateLimited {
            retry_after_secs: saw_rate_limited.filter(|&s| s != u64::MAX),
            message,
        }),
        (Some(e), _) => Err(e),
        (None, _) => Err(JevError::Failed {
            message: "no Jev backends configured".to_string(),
        }),
    }
}

pub fn questions_map(qs: Vec<(String, Question)>) -> Map<String, Value> {
    let mut m = Map::new();
    for (k, q) in qs {
        m.insert(k, serde_json::to_value(q).unwrap_or(Value::Null));
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_shapes_serialize() {
        let q = Question::noul_with("Is it an ad?".into(), "ad", "content");
        let v = serde_json::to_value(&q).unwrap();
        assert_eq!(v["type"], "noul");
        let mut c = HashMap::new();
        c.insert("a".to_string(), "A".to_string());
        let q2 = Question::choice("Pick".into(), c);
        let v2 = serde_json::to_value(&q2).unwrap();
        assert_eq!(v2["criteria"]["a"], "A");
    }

    #[test]
    fn answer_parses() {
        let r: DecideResponse = serde_json::from_value(serde_json::json!({
            "model": "jev-1.13-free",
            "answers": {
                "q": {"type": "noul", "noul": 0.92},
                "c": {"type": "choice", "choice": "x",
                      "probabilities": {"x": 0.8, "y": 0.2}, "confidence": 0.75}
            },
            "usage": {}
        }))
        .unwrap();
        assert_eq!(r.answers["q"].noul, Some(0.92));
        assert_eq!(r.answers["c"].choice.as_deref(), Some("x"));
    }

    #[test]
    fn status_classification() {
        let ra = Some(7);
        assert!(matches!(
            classify_status(reqwest::StatusCode::TOO_MANY_REQUESTS, ra, "x".into()),
            JevError::RateLimited { retry_after_secs: Some(7), .. }
        ));
        assert!(matches!(
            classify_status(
                reqwest::StatusCode::from_u16(529).unwrap(),
                None,
                "x".into()
            ),
            JevError::RateLimited { .. }
        ));
        assert!(matches!(
            classify_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR, None, "x".into()),
            JevError::Failed { .. }
        ));
        assert!(matches!(
            classify_status(reqwest::StatusCode::BAD_REQUEST, None, "x".into()),
            JevError::Failed { .. }
        ));
    }

    #[test]
    fn backoff_shape() {
        assert_eq!(backoff_secs(0, None), 1);
        assert_eq!(backoff_secs(1, None), 2);
        assert_eq!(backoff_secs(2, None), 4);
        assert_eq!(backoff_secs(9, None), 8);
        assert_eq!(backoff_secs(0, Some(120)), 30);
        assert_eq!(backoff_secs(0, Some(3)), 3);
    }

    #[test]
    fn retry_after_parses() {
        let mut h = reqwest::header::HeaderMap::new();
        h.insert(reqwest::header::RETRY_AFTER, "12".parse().unwrap());
        assert_eq!(parse_retry_after(&h), Some(12));
        let empty = reqwest::header::HeaderMap::new();
        assert_eq!(parse_retry_after(&empty), None);
    }
}

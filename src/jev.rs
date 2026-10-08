use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{OwnedMutexGuard, Semaphore};

use crate::config::{Config, JevBackendCfg};
use crate::health::{BackendHealth, DeadKind};
use crate::openai::TokenUsage;

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
    /// "primary", "secondary", or "tertiary".
    pub backend: &'static str,
    pub latency_ms: u64,
    pub usage: TokenUsage,
}

/// RateLimited must reach MinusPod as HTTP 429 (it defers + retries the
/// episode) — never as 502 (windows get dropped as failed).
///
/// Dead is one backend's billing or daily quota. AllBackendsDead means every
/// configured backend is in that state: MinusPod has been paused (when Docker
/// is reachable) and the HTTP answer is 503 with no Retry-After.
#[derive(Debug)]
pub enum JevError {
    RateLimited { retry_after_secs: Option<u64>, message: String },
    Dead { kind: DeadKind, message: String },
    AllBackendsDead { message: String },
    Failed { message: String },
}

impl std::fmt::Display for JevError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JevError::RateLimited { message, .. }
            | JevError::Dead { message, .. }
            | JevError::AllBackendsDead { message }
            | JevError::Failed { message } => write!(f, "{message}"),
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
    // Billing and daily quota are checked before the 429/503 bucket. A Clef
    // 4006 or a Zen FreeUsageLimitError is not a short rate limit, and must
    // not carry Retry-After back to MinusPod.
    if is_billing(status, &body_head) {
        return JevError::Dead {
            kind: DeadKind::Billing,
            message: format!("HTTP {status}: {body_head}"),
        };
    }
    if is_daily_quota(&body_head) {
        return JevError::Dead {
            kind: DeadKind::Quota,
            message: format!("HTTP {status}: {body_head}"),
        };
    }
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

/// HTTP 402, or a body that says the account has no credits left.
pub fn is_billing(status: reqwest::StatusCode, body: &str) -> bool {
    if status == reqwest::StatusCode::PAYMENT_REQUIRED {
        return true;
    }
    let l = body.to_ascii_lowercase();
    l.contains("billing_error")
        || l.contains("insufficient_quota")
        || l.contains("payment required")
        || (l.contains("credit")
            && (l.contains("no available") || l.contains("insufficient") || l.contains("exhausted")))
}

/// Clef's daily neuron cap (code 4006) and Zen's daily free-tier cap.
/// A plain 429 without these markers stays a short rate limit.
pub fn is_daily_quota(body: &str) -> bool {
    let l = body.to_ascii_lowercase();
    if l.contains("freeusagelimiterror")
        || l.contains("free usage limit")
        || l.contains("daily free allocation")
        || l.contains("used up your daily")
    {
        return true;
    }
    l.contains("4006") && (l.contains("neuron") || l.contains("daily") || l.contains("allocation"))
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

fn configured(cfg: &Config) -> Vec<(&'static str, &JevBackendCfg)> {
    [
        ("primary", &cfg.primary),
        ("secondary", &cfg.secondary),
        ("tertiary", &cfg.tertiary),
    ]
    .into_iter()
    .filter(|(_, b)| !b.base_url.trim().is_empty())
    .collect()
}

/// Try primary, then secondary, then tertiary.
///
/// A billing or daily-quota answer marks that backend dead and is not retried.
/// Transient 429/529/503 still get bounded backoff. When every configured
/// backend is dead, MinusPod is paused. The in-flight request is held and
/// retried only when the wait fits inside MinusPod's client timeout; otherwise
/// it returns [`JevError::AllBackendsDead`] (HTTP 503, no Retry-After).
/// Unconfigured slots (empty base_url) are skipped entirely.
pub async fn decide(
    client: &Client,
    cfg: &Config,
    health: &BackendHealth,
    sem: &Arc<Semaphore>,
    started: Instant,
    state: &Value,
    questions: &Map<String, Value>,
) -> Result<DecideOutcome, JevError> {
    let backends = configured(cfg);
    let names: Vec<&str> = backends.iter().map(|(n, _)| *n).collect();
    if names.is_empty() {
        return Err(JevError::Failed {
            message: "no Jev backends configured".to_string(),
        });
    }

    let mut guard: Option<OwnedMutexGuard<()>> = None;
    let mut held_once = false;

    loop {
        if health.all_dead(&names).await {
            if guard.is_none() {
                guard = Some(health.lock_serial().await);
                continue;
            }
            let until = health.earliest().await;
            let wait = until
                .and_then(|t| t.duration_since(SystemTime::now()).ok())
                .unwrap_or_default();
            let budget = health.budget(started);
            let why = health.summary().await;
            if !held_once && !health.probe_spent().await && wait <= budget {
                held_once = true;
                health.set_probe_spent(true).await;
                health
                    .pause(&format!(
                        "holding the in-flight request until a backend should be available; {why}"
                    ))
                    .await;
                if let Some(t) = until {
                    health.wait_until(t).await;
                }
                health
                    .unpause(
                        "dead window ended; retrying the held request as the probe",
                        false,
                    )
                    .await;
                continue;
            }
            health.set_probe_spent(true).await;
            health
                .pause(&format!(
                    "wait {wait:?} exceeds the remaining client budget {budget:?}; {why}"
                ))
                .await;
            return Err(JevError::AllBackendsDead {
                message: health.exhausted_message().await,
            });
        }

        if health.revival().await && guard.is_none() {
            guard = Some(health.lock_serial().await);
            continue;
        }
        if guard.is_some() && !health.revival().await && !health.all_dead(&names).await {
            guard = None;
        }

        let mut last_rate: Option<JevError> = None;
        let mut saw_retry_after: Option<u64> = None;
        let mut last_failed: Option<JevError> = None;

        for (name, backend) in &backends {
            if health.is_dead(name).await {
                continue;
            }
            match try_backend(client, cfg, sem, backend, name, state, questions).await {
                Ok(outcome) => {
                    health.note_success().await;
                    return Ok(outcome);
                }
                Err(JevError::Dead { kind, message }) => {
                    tracing::warn!("jev {name} backend dead ({kind:?}): {message}");
                    health
                        .mark_dead(name, kind, &message, cfg.billing_dead_secs)
                        .await;
                }
                Err(JevError::RateLimited {
                    retry_after_secs,
                    message,
                }) => {
                    saw_retry_after = Some(
                        saw_retry_after
                            .unwrap_or(u64::MAX)
                            .min(retry_after_secs.unwrap_or(u64::MAX)),
                    );
                    last_rate = Some(JevError::RateLimited {
                        retry_after_secs,
                        message,
                    });
                }
                Err(e) => {
                    tracing::warn!("jev {name} backend failed: {e}");
                    last_failed = Some(e);
                }
            }
        }

        if health.all_dead(&names).await {
            continue;
        }

        if let Some(JevError::RateLimited { message, .. }) = last_rate {
            return Err(JevError::RateLimited {
                retry_after_secs: saw_retry_after.filter(|&s| s != u64::MAX),
                message,
            });
        }
        return Err(last_failed.unwrap_or(JevError::Failed {
            message: "no Jev backends configured".to_string(),
        }));
    }
}

/// Bounded retries for a short rate limit. Billing and daily quota return
/// immediately so the caller can skip this backend.
async fn try_backend(
    client: &Client,
    cfg: &Config,
    sem: &Arc<Semaphore>,
    backend: &JevBackendCfg,
    name: &'static str,
    state: &Value,
    questions: &Map<String, Value>,
) -> Result<DecideOutcome, JevError> {
    const MAX_ATTEMPTS: u32 = 3;
    let timeout = Duration::from_secs(cfg.timeout_secs.max(5));
    let mut last_rate: Option<JevError> = None;
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
                    usage: TokenUsage::from_upstream(&resp.usage),
                })
            }
            Err(e @ JevError::Dead { .. }) => return Err(e),
            Err(JevError::RateLimited {
                retry_after_secs,
                message,
            }) => {
                tracing::warn!(
                    "jev {name} backend rate-limited (attempt {}/{MAX_ATTEMPTS}), backing off",
                    attempt + 1,
                );
                last_rate = Some(JevError::RateLimited {
                    retry_after_secs,
                    message,
                });
                if attempt + 1 < MAX_ATTEMPTS {
                    let sleep = backoff_secs(attempt, retry_after_secs) * 1000 + jitter_ms();
                    tokio::time::sleep(Duration::from_millis(sleep)).await;
                    continue;
                }
                return Err(last_rate.expect("rate limit recorded"));
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_rate.unwrap_or(JevError::Failed {
        message: format!("jev {name} produced no result"),
    }))
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
        assert!(matches!(
            classify_status(
                reqwest::StatusCode::PAYMENT_REQUIRED,
                None,
                r#"{"error":"billing_error","message":"no available TypeSafe API credits"}"#.into(),
            ),
            JevError::Dead { kind: DeadKind::Billing, .. }
        ));
        assert!(matches!(
            classify_status(
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                Some(300),
                r#"{"code":4006,"message":"you have used up your daily free allocation of 10,000 neurons"}"#.into(),
            ),
            JevError::Dead { kind: DeadKind::Quota, .. }
        ));
        assert!(matches!(
            classify_status(
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                Some(300),
                r#"{"type":"FreeUsageLimitError","message":"daily limit"}"#.into(),
            ),
            JevError::Dead { kind: DeadKind::Quota, .. }
        ));
        // A short 429 is still a rate limit, and its Retry-After is kept
        // here. It is dropped only when every backend is dead.
        assert!(matches!(
            classify_status(
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                Some(3),
                "slow down".into(),
            ),
            JevError::RateLimited { retry_after_secs: Some(3), .. }
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

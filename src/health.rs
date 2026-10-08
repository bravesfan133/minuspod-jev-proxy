//! Which backends are unusable, and when MinusPod may run again.
//!
//! A billing failure (HTTP 402, no credits) parks that backend for
//! `JEV_BILLING_DEAD_SECS`. A daily quota failure (Clef 4006, Zen
//! `FreeUsageLimitError`) parks it until the next 00:00 UTC. Nothing here
//! polls upstream: the next real request after unpause is the probe.

use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Notify, OwnedMutexGuard};

use crate::config::{Config, JevBackendCfg};
use crate::docker::PauseCtl;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadKind {
    Billing,
    Quota,
}

impl DeadKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DeadKind::Billing => "billing",
            DeadKind::Quota => "quota",
        }
    }
}

#[derive(Debug, Clone)]
struct DeadMark {
    until: SystemTime,
    kind: DeadKind,
    message: String,
}

#[derive(Debug, Clone)]
pub struct PauseEvent {
    pub action: String,
    pub reason: String,
    pub at: String,
}

#[derive(Debug, Default)]
pub struct PauseLog {
    pub events: Vec<PauseEvent>,
    pub paused: bool,
    pub last_error: Option<String>,
}

struct Inner {
    dead: HashMap<String, DeadMark>,
    next_unpause: Option<SystemTime>,
    /// Set when a scheduled unpause happens, so the next request is the only
    /// one that talks to upstream.
    revival: bool,
    /// A probe already ran for this outage. Further requests must not call
    /// upstream or start another hold.
    probe_spent: bool,
}

pub struct BackendHealth {
    cfg: Arc<Config>,
    ctl: PauseCtl,
    log: Arc<Mutex<PauseLog>>,
    inner: tokio::sync::Mutex<Inner>,
    notify: Notify,
    /// Held across a hold or a revival probe so parallel windows do not each
    /// burn a request.
    gate: Arc<tokio::sync::Mutex<()>>,
}

impl BackendHealth {
    pub fn recording(cfg: Arc<Config>) -> Arc<Self> {
        Arc::new(Self::new(cfg, PauseCtl::Record))
    }

    pub fn with_docker(cfg: Arc<Config>) -> Arc<Self> {
        let ctl = PauseCtl::docker(&cfg.docker_socket, &cfg.minuspod_container);
        Arc::new(Self::new(cfg, ctl))
    }

    fn new(cfg: Arc<Config>, ctl: PauseCtl) -> Self {
        Self {
            cfg,
            ctl,
            log: Arc::new(Mutex::new(PauseLog::default())),
            inner: tokio::sync::Mutex::new(Inner {
                dead: HashMap::new(),
                next_unpause: None,
                revival: false,
                probe_spent: false,
            }),
            notify: Notify::new(),
            gate: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    pub fn pause_log(&self) -> Arc<Mutex<PauseLog>> {
        self.log.clone()
    }

    pub async fn lock_serial(&self) -> OwnedMutexGuard<()> {
        self.gate.clone().lock_owned().await
    }

    pub async fn is_dead(&self, name: &str) -> bool {
        let g = self.inner.lock().await;
        match g.dead.get(name) {
            Some(m) => m.until > SystemTime::now(),
            None => false,
        }
    }

    pub async fn all_dead(&self, names: &[&str]) -> bool {
        if names.is_empty() {
            return false;
        }
        let g = self.inner.lock().await;
        let now = SystemTime::now();
        names.iter().all(|n| g.dead.get(*n).is_some_and(|m| m.until > now))
    }

    pub async fn revival(&self) -> bool {
        self.inner.lock().await.revival
    }

    pub async fn probe_spent(&self) -> bool {
        self.inner.lock().await.probe_spent
    }

    pub async fn set_probe_spent(&self, spent: bool) {
        self.inner.lock().await.probe_spent = spent;
    }

    pub async fn mark_dead(&self, name: &str, kind: DeadKind, message: &str, billing_secs: u64) {
        let until = dead_until(kind, SystemTime::now(), billing_secs);
        let mut g = self.inner.lock().await;
        g.dead.insert(
            name.to_string(),
            DeadMark {
                until,
                kind,
                message: message.chars().take(300).collect(),
            },
        );
    }

    pub async fn note_success(&self) {
        let mut g = self.inner.lock().await;
        g.revival = false;
        g.probe_spent = false;
    }

    pub fn budget(&self, started: Instant) -> Duration {
        Duration::from_secs(self.cfg.client_timeout_secs)
            .saturating_sub(started.elapsed())
            .saturating_sub(Duration::from_secs(self.cfg.hold_margin_secs))
    }

    pub async fn earliest(&self) -> Option<SystemTime> {
        let g = self.inner.lock().await;
        let now = SystemTime::now();
        g.dead
            .values()
            .filter(|m| m.until > now)
            .map(|m| m.until)
            .min()
    }

    pub async fn summary(&self) -> String {
        let g = self.inner.lock().await;
        let now = SystemTime::now();
        let mut parts: Vec<String> = g
            .dead
            .iter()
            .filter(|(_, m)| m.until > now)
            .map(|(name, m)| format!("{name} {} until {}", m.kind.as_str(), rfc3339(m.until)))
            .collect();
        parts.sort();
        if parts.is_empty() {
            "no backend is inside a dead window".to_string()
        } else {
            parts.join("; ")
        }
    }

    pub async fn exhausted_message(&self) -> String {
        let when = self
            .earliest()
            .await
            .map(rfc3339)
            .unwrap_or_else(|| "unknown".into());
        format!(
            "all Jev backends are out of credit or daily quota; MinusPod paused until {when}"
        )
    }

    /// Freeze MinusPod and arm the scheduler for the earliest dead window.
    pub async fn pause(&self, reason: &str) {
        if let Some(until) = self.earliest().await {
            self.inner.lock().await.next_unpause = Some(until);
        }
        self.notify.notify_waiters();
        let container = &self.cfg.minuspod_container;
        tracing::warn!(container = %container, reason, "pausing MinusPod");
        let result = self.ctl.pause().await;
        self.record("pause", reason, result, true).await;
    }

    /// `fresh_probe` clears the spent flag so the next real request may call
    /// upstream once. The in-request hold passes false: it is about to probe
    /// itself and must not open a second wave.
    pub async fn unpause(&self, reason: &str, fresh_probe: bool) {
        let container = &self.cfg.minuspod_container;
        tracing::warn!(container = %container, reason, fresh_probe, "unpausing MinusPod");
        let result = self.ctl.unpause().await;
        {
            let mut g = self.inner.lock().await;
            // A failed probe may already have armed a later deadline. Do not
            // erase that one.
            if g.next_unpause.is_some_and(|t| t <= SystemTime::now()) {
                g.next_unpause = None;
            }
            if fresh_probe {
                g.probe_spent = false;
                g.revival = true;
            }
        }
        self.notify.notify_waiters();
        self.record("unpause", reason, result, false).await;
    }

    pub async fn unpause_if_due(&self) {
        let due = self
            .inner
            .lock()
            .await
            .next_unpause
            .is_some_and(|t| t <= SystemTime::now());
        if due {
            self.unpause(
                "scheduled: a backend dead window ended; the next request is the probe",
                true,
            )
            .await;
        }
    }

    pub async fn wait_until(&self, until: SystemTime) {
        if let Ok(delay) = until.duration_since(SystemTime::now()) {
            tokio::time::sleep(delay).await;
        }
    }

    async fn record(&self, action: &str, reason: &str, result: Result<(), String>, paused: bool) {
        let mut log = self.log.lock().unwrap_or_else(|e| e.into_inner());
        log.events.push(PauseEvent {
            action: action.to_string(),
            reason: reason.to_string(),
            at: rfc3339(SystemTime::now()),
        });
        match result {
            Ok(()) => {
                log.paused = paused;
                log.last_error = None;
            }
            Err(e) => {
                tracing::error!(action, error = %e, "docker {action} of MinusPod failed");
                log.last_error = Some(e);
            }
        }
    }

    pub async fn next_unpause(&self) -> Option<SystemTime> {
        self.inner.lock().await.next_unpause
    }

    pub async fn status(&self) -> Value {
        let paused = match self.ctl.inspect_paused().await {
            Ok(v) => v,
            Err(_) => self.log.lock().unwrap_or_else(|e| e.into_inner()).paused,
        };
        let g = self.inner.lock().await;
        let now = SystemTime::now();
        let log = self.log.lock().unwrap_or_else(|e| e.into_inner());
        let last = |action: &str| -> Value {
            log.events
                .iter()
                .rev()
                .find(|e| e.action == action)
                .map(|e| json!({"at": e.at, "reason": e.reason}))
                .unwrap_or(Value::Null)
        };
        json!({
            "backends": {
                "primary": backend_status(&self.cfg.primary, g.dead.get("primary"), now),
                "secondary": backend_status(&self.cfg.secondary, g.dead.get("secondary"), now),
                "tertiary": backend_status(&self.cfg.tertiary, g.dead.get("tertiary"), now),
            },
            "minuspod_container": self.cfg.minuspod_container,
            "minuspod_paused": paused,
            "next_unpause": g.next_unpause.map(rfc3339),
            "revival": g.revival,
            "probe_spent": g.probe_spent,
            "last_pause": last("pause"),
            "last_unpause": last("unpause"),
            "docker_error": log.last_error,
        })
    }
}

fn backend_status(cfg: &JevBackendCfg, mark: Option<&DeadMark>, now: SystemTime) -> Value {
    if cfg.base_url.trim().is_empty() {
        return json!({"configured": false, "state": "unconfigured"});
    }
    match mark {
        Some(m) if m.until > now => json!({
            "configured": true,
            "state": "dead",
            "kind": m.kind.as_str(),
            "dead_until": rfc3339(m.until),
            "model": cfg.model,
            "base_url": cfg.base_url,
            "detail": m.message,
        }),
        Some(m) => json!({
            "configured": true,
            "state": "live",
            "model": cfg.model,
            "base_url": cfg.base_url,
            "last_kind": m.kind.as_str(),
            "last_dead_until": rfc3339(m.until),
        }),
        None => json!({
            "configured": true,
            "state": "live",
            "model": cfg.model,
            "base_url": cfg.base_url,
        }),
    }
}

/// Sleep until the next armed unpause, then thaw MinusPod. Does not call
/// upstream. Started once from `main`.
pub async fn run_scheduler(health: Arc<BackendHealth>) {
    loop {
        let notified = health.notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if let Some(when) = health.next_unpause().await {
            let delay = when
                .duration_since(SystemTime::now())
                .unwrap_or_default();
            tokio::select! {
                _ = tokio::time::sleep(delay) => health.unpause_if_due().await,
                _ = notified => {}
            }
        } else {
            notified.await;
        }
    }
}

pub fn dead_until(kind: DeadKind, now: SystemTime, billing_secs: u64) -> SystemTime {
    match kind {
        DeadKind::Billing => now + Duration::from_secs(billing_secs.max(1)),
        DeadKind::Quota => next_utc_midnight(now),
    }
}

pub fn next_utc_midnight(now: SystemTime) -> SystemTime {
    let secs = now
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    UNIX_EPOCH + Duration::from_secs((secs / 86_400 + 1) * 86_400)
}

pub fn rfc3339(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    let hh = tod / 3600;
    let mm = (tod % 3600) / 60;
    let ss = tod % 60;
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Days since 1970-01-01 to a civil date. Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Arc<Config> {
        Arc::new(Config {
            port: 0,
            primary: crate::config::JevBackendCfg {
                base_url: "http://primary".into(),
                model: "p".into(),
                api_key: None,
            },
            secondary: crate::config::JevBackendCfg {
                base_url: "http://secondary".into(),
                model: "s".into(),
                api_key: None,
            },
            tertiary: crate::config::JevBackendCfg {
                base_url: String::new(),
                model: String::new(),
                api_key: None,
            },
            timeout_secs: 5,
            recall_threshold: 0.35,
            edge_threshold: 0.5,
            edge_piece_secs: 2.0,
            edge_context_secs: 45.0,
            attach_threshold: 0.4,
            attach_gap_secs: 8.0,
            review_threshold: 0.5,
            segment_target_secs: 4.0,
            segment_max_secs: 8.0,
            segment_gap_secs: 1.25,
            max_segments_per_call: 16,
            max_concurrent: 1,
            billing_dead_secs: 1800,
            client_timeout_secs: 600,
            hold_margin_secs: 30,
            minuspod_container: "minuspod".into(),
            docker_socket: String::new(),
        })
    }

    #[test]
    fn midnight_and_billing_deadlines() {
        let t0 = UNIX_EPOCH + Duration::from_secs(10);
        assert_eq!(next_utc_midnight(t0), UNIX_EPOCH + Duration::from_secs(86_400));
        assert_eq!(
            next_utc_midnight(UNIX_EPOCH + Duration::from_secs(86_400)),
            UNIX_EPOCH + Duration::from_secs(172_800)
        );
        assert_eq!(rfc3339(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(UNIX_EPOCH + Duration::from_secs(86_400)), "1970-01-02T00:00:00Z");
        // 2023-11-14 22:13:20 UTC.
        assert_eq!(rfc3339(UNIX_EPOCH + Duration::from_secs(1_700_000_000)), "2023-11-14T22:13:20Z");
        let billing = dead_until(DeadKind::Billing, t0, 1800);
        assert_eq!(billing, t0 + Duration::from_secs(1800));
        let quota = dead_until(DeadKind::Quota, t0, 1800);
        assert_eq!(quota, UNIX_EPOCH + Duration::from_secs(86_400));
    }

    #[tokio::test]
    async fn scheduler_unpauses_at_the_billing_deadline() {
        let health = BackendHealth::recording(cfg());
        health
            .mark_dead("primary", DeadKind::Billing, "HTTP 402: no credits", 1)
            .await;
        health.pause("all backends dead").await;
        let child = Arc::clone(&health);
        tokio::spawn(async move { run_scheduler(child).await });
        tokio::time::sleep(Duration::from_millis(1500)).await;
        let log = health.pause_log();
        let (saw_unpause, paused) = {
            let g = log.lock().unwrap();
            (
                g.events.iter().any(|e| e.action == "unpause"),
                g.paused,
            )
        };
        assert!(saw_unpause, "scheduler should unpause");
        assert!(!paused);
        assert!(health.revival().await);
        assert!(!health.probe_spent().await);
    }
}

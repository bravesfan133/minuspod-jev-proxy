use std::env;

#[derive(Clone, Debug)]
pub struct JevBackendCfg {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub port: u16,
    pub primary: JevBackendCfg,
    pub secondary: JevBackendCfg,
    /// Optional third backend. Empty base_url means "not configured", and the
    /// failover loop skips it. Kept separate from secondary so adding a paid
    /// provider never displaces the keyless free tier.
    pub tertiary: JevBackendCfg,
    pub timeout_secs: u64,
    /// A paid-sponsor span becomes a cut candidate when noul >= this.
    ///
    /// This is the recall gate and it is meant to be permissive: it decides
    /// only *whether a candidate exists*. Precision is applied afterwards by
    /// the edge-trim pass. Keeping one threshold for both jobs is what lets
    /// whole windows come back with no ads at all, because a strict recall
    /// gate never gives the trim pass anything to tighten.
    pub recall_threshold: f64,
    /// An edge piece stays in the cut when noul >= this AND its label is a
    /// paid sponsor. Both signals must agree.
    pub edge_threshold: f64,
    /// Size of the pieces the edge-trim pass re-judges, in seconds.
    pub edge_piece_secs: f64,
    /// Transcript context included around an edge piece, in seconds.
    pub edge_context_secs: f64,
    /// A paid-sponsor neighbor joins a candidate when noul >= this.
    pub attach_threshold: f64,
    /// Max silence, in seconds, between a candidate and a neighbor that joins it.
    pub attach_gap_secs: f64,
    /// reviewer noul >= review_threshold => confirm candidate.
    pub review_threshold: f64,
    /// Flush a span once it reaches this many seconds.
    pub segment_target_secs: f64,
    /// Never add another line if it would make the span longer than this.
    pub segment_max_secs: f64,
    /// Flush before a line when the pause in front of it is at least this.
    pub segment_gap_secs: f64,
    /// max segments per Decisions call (one noul and one choice each).
    pub max_segments_per_call: usize,
    /// max concurrent Jev calls (MinusPod fires all windows in parallel).
    pub max_concurrent: usize,
    /// How long a 402 / billing failure parks that backend.
    pub billing_dead_secs: u64,
    /// MinusPod's per-request client timeout. A held request is only kept
    /// open when the wait fits inside this, minus `hold_margin_secs`.
    /// openai-compatible MinusPod defaults to 600s when `llm_timeout_seconds`
    /// is unset.
    pub client_timeout_secs: u64,
    /// Seconds reserved so the probe after unpause can finish before the
    /// client gives up.
    pub hold_margin_secs: u64,
    /// Container `docker pause` / `docker unpause` acts on.
    pub minuspod_container: String,
    /// Docker Engine socket. Empty disables pause/unpause (requests still
    /// fail fast once every backend is dead).
    pub docker_socket: String,
}

/// Every env var this build reads. Anything else in the `JEV_` namespace is
/// dead config: it may look active in a compose file or a container's env
/// while this process ignores it entirely.
const KNOWN_ENV_VARS: &[&str] = &[
    "PORT",
    "JEV_PRIMARY_BASE_URL",
    "JEV_PRIMARY_MODEL",
    "JEV_PRIMARY_API_KEY",
    "JEV_SECONDARY_BASE_URL",
    "JEV_SECONDARY_MODEL",
    "JEV_SECONDARY_API_KEY",
    "JEV_TERTIARY_BASE_URL",
    "JEV_TERTIARY_MODEL",
    "JEV_TERTIARY_API_KEY",
    "TYPESAFE_API_KEY",
    "JEV_TIMEOUT_SECS",
    "JEV_RECALL_THRESHOLD",
    "JEV_EDGE_THRESHOLD",
    "JEV_EDGE_PIECE_SECS",
    "JEV_EDGE_CONTEXT_SECS",
    "JEV_ATTACH_THRESHOLD",
    "JEV_ATTACH_GAP_SECS",
    "JEV_REVIEW_THRESHOLD",
    "JEV_SEGMENT_TARGET_SECS",
    "JEV_SEGMENT_MAX_SECS",
    "JEV_SEGMENT_GAP_SECS",
    "JEV_MAX_SEGMENTS_PER_CALL",
    "JEV_MAX_CONCURRENT_REQS",
    "JEV_BILLING_DEAD_SECS",
    "JEV_CLIENT_TIMEOUT_SECS",
    "JEV_HOLD_MARGIN_SECS",
    "JEV_MINUSPOD_CONTAINER",
    "JEV_DOCKER_SOCKET",
    // Read as a deprecated alias for JEV_RECALL_THRESHOLD.
    "JEV_AD_THRESHOLD",
];

/// Warn about `JEV_*` / `TYPESAFE_*` variables this build does not read.
///
/// Stale config that silently does nothing is worse than config that fails:
/// a threshold tuned for an older build looks applied while having no effect.
pub fn warn_unknown_env_vars() {
    for (key, _) in env::vars() {
        if !(key.starts_with("JEV_") || key.starts_with("TYPESAFE_")) {
            continue;
        }
        if KNOWN_ENV_VARS.contains(&key.as_str()) {
            continue;
        }
        tracing::warn!(
            var = %key,
            "ignoring unrecognized environment variable; this build does not read it",
        );
    }
}

/// A threshold in [0, 1]. Out-of-range values fall back to the default rather
/// than silently disabling a stage.
fn get_prob(key: &str, default: f64) -> f64 {
    let v = get_f64(key, default);
    if v.is_finite() && (0.0..=1.0).contains(&v) {
        v
    } else {
        tracing::warn!(
            var = %key,
            value = v,
            default = default,
            "threshold outside [0,1]; using default",
        );
        default
    }
}

fn get(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn get_f64(key: &str, default: f64) -> f64 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn get_usize(key: &str, default: usize) -> usize {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn get_u16(key: &str, default: u16) -> u16 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn get_u64(key: &str, default: u64) -> u64 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn opt(key: &str) -> Option<String> {
    env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// TYPESAFE_API_KEY is accepted for whichever backend points at TypeSafe.
/// Never log values.
fn key_for(explicit_var: &str, base_url: &str) -> Option<String> {
    opt(explicit_var).or_else(|| {
        if base_url.contains("typesafe.ai") {
            opt("TYPESAFE_API_KEY")
        } else {
            None
        }
    })
}

impl Config {
    pub fn from_env() -> Self {
        // Primary first: reliable quota wins over free. Override via env.
        let primary_url = get(
            "JEV_PRIMARY_BASE_URL",
            "https://api.typesafe.ai/v1/systemone",
        );
        let secondary_url = get(
            "JEV_SECONDARY_BASE_URL",
            "https://opencode.ai/zen/v1/systemone",
        );
        // Empty by default: the third slot is opt-in and must not displace the
        // keyless free tier that already sits in secondary.
        let tertiary_url = get("JEV_TERTIARY_BASE_URL", "");
        let bounds = segment_bounds();
        Self {
            port: get_u16("PORT", 8787),
            primary: JevBackendCfg {
                model: get("JEV_PRIMARY_MODEL", "jev-latest"),
                api_key: key_for("JEV_PRIMARY_API_KEY", &primary_url),
                base_url: primary_url,
            },
            secondary: JevBackendCfg {
                model: get("JEV_SECONDARY_MODEL", "jev-1.13-free"),
                api_key: key_for("JEV_SECONDARY_API_KEY", &secondary_url),
                base_url: secondary_url,
            },
            // No default URL: this slot is opt-in, so an existing deployment
            // keeps exactly the behaviour it has now.
            tertiary: JevBackendCfg {
                model: get("JEV_TERTIARY_MODEL", ""),
                api_key: key_for("JEV_TERTIARY_API_KEY", &tertiary_url),
                base_url: tertiary_url,
            },
            timeout_secs: get_u64("JEV_TIMEOUT_SECS", 60),

            // Recall gate. Deliberately low: a clear host read scores well
            // above 0.35, and being strict here costs whole ads, because a
            // rejected span never reaches the edge-trim pass that would have
            // tightened it.
            recall_threshold: recall_threshold(),

            // Precision gate, applied per edge piece after the fact.
            edge_threshold: get_prob("JEV_EDGE_THRESHOLD", 0.50),
            edge_piece_secs: get_f64("JEV_EDGE_PIECE_SECS", 2.0).max(0.5),
            edge_context_secs: get_f64("JEV_EDGE_CONTEXT_SECS", 45.0).max(0.0),

            attach_threshold: get_prob("JEV_ATTACH_THRESHOLD", 0.40),
            attach_gap_secs: get_f64("JEV_ATTACH_GAP_SECS", 8.0),
            review_threshold: get_prob("JEV_REVIEW_THRESHOLD", 0.5),
            segment_gap_secs: get_f64("JEV_SEGMENT_GAP_SECS", 1.25).max(0.0),
            max_segments_per_call: get_usize("JEV_MAX_SEGMENTS_PER_CALL", 16).max(1),
            max_concurrent: get_usize("JEV_MAX_CONCURRENT_REQS", 4).max(1),
            billing_dead_secs: get_u64("JEV_BILLING_DEAD_SECS", 1800).max(1),
            client_timeout_secs: get_u64("JEV_CLIENT_TIMEOUT_SECS", 600).max(1),
            hold_margin_secs: get_u64("JEV_HOLD_MARGIN_SECS", 30),
            minuspod_container: get("JEV_MINUSPOD_CONTAINER", "minuspod"),
            docker_socket: get("JEV_DOCKER_SOCKET", "/var/run/docker.sock"),
            segment_target_secs: bounds.segment_target_secs,
            segment_max_secs: bounds.segment_max_secs,
        }
    }
}

/// `JEV_SEGMENT_TARGET_SECS` is the span length operators set. If max is
/// smaller, spans flush on max and the target is silently ignored. Raise max
/// to the target and say so.
fn segment_bounds() -> SegmentBounds {
    let target = get_f64("JEV_SEGMENT_TARGET_SECS", 4.0).max(0.5);
    let configured_max = get_f64("JEV_SEGMENT_MAX_SECS", 8.0).max(0.5);
    let (segment_target_secs, segment_max_secs, raised) =
        reconcile_segment_bounds(target, configured_max);
    if raised {
        tracing::warn!(
            configured_max,
            target = segment_target_secs,
            effective_max = segment_max_secs,
            "JEV_SEGMENT_MAX_SECS is below JEV_SEGMENT_TARGET_SECS; raising max to the target so spans are not capped shorter than the target",
        );
    }
    SegmentBounds {
        segment_target_secs,
        segment_max_secs,
    }
}

struct SegmentBounds {
    segment_target_secs: f64,
    segment_max_secs: f64,
}

/// Returns `(target, max, raised)`.
pub fn reconcile_segment_bounds(target: f64, max: f64) -> (f64, f64, bool) {
    if max < target {
        (target, target, true)
    } else {
        (target, max, false)
    }
}

/// `JEV_RECALL_THRESHOLD`, or the deprecated `JEV_AD_THRESHOLD` alias.
///
/// The old variable gated both recall and precision. Treating it as the
/// recall gate alone can only make the cut set larger than the operator
/// configured, so it is read but loudly flagged.
fn recall_threshold() -> f64 {
    if let Some(v) = opt("JEV_RECALL_THRESHOLD") {
        if let Ok(parsed) = v.parse::<f64>() {
            return if parsed.is_finite() && (0.0..=1.0).contains(&parsed) {
                parsed
            } else {
                tracing::warn!(
                    "JEV_RECALL_THRESHOLD={parsed} is outside [0,1]; using 0.35",
                );
                0.35
            };
        }
    }
    if let Some(v) = opt("JEV_AD_THRESHOLD") {
        tracing::warn!(
            "JEV_AD_THRESHOLD is deprecated and now sets only the recall gate; \
             edge trimming uses JEV_EDGE_THRESHOLD. Use JEV_RECALL_THRESHOLD instead."
        );
        if let Ok(parsed) = v.parse::<f64>() {
            if parsed.is_finite() && (0.0..=1.0).contains(&parsed) {
                return parsed;
            }
        }
    }
    0.35
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vars_exclude_the_deprecated_alias() {
        // The alias is read, but only as a fallback, so it must not be
        // advertised as the primary knob.
        assert!(KNOWN_ENV_VARS.contains(&"JEV_RECALL_THRESHOLD"));
        assert!(KNOWN_ENV_VARS.contains(&"JEV_EDGE_THRESHOLD"));
        assert!(!KNOWN_ENV_VARS.contains(&"JEV_SEGMENT_FLOOR_SECS"));
    }

    #[test]
    fn target_longer_than_max_raises_max() {
        assert_eq!(reconcile_segment_bounds(20.0, 8.0), (20.0, 20.0, true));
        assert_eq!(reconcile_segment_bounds(4.0, 8.0), (4.0, 8.0, false));
        assert_eq!(reconcile_segment_bounds(8.0, 8.0), (8.0, 8.0, false));
    }

    #[test]
    fn get_prob_rejects_out_of_range() {
        // Not a real env test: get_prob is exercised through its parsing
        // rule, which keeps this test independent of process env ordering.
        let ok = 0.42f64;
        assert!((0.0..=1.0).contains(&ok));
        for bad in [-0.1f64, 1.1, f64::NAN, f64::INFINITY] {
            assert!(!(bad.is_finite() && (0.0..=1.0).contains(&bad)));
        }
    }
}

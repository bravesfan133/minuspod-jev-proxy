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
    pub timeout_secs: u64,
    /// Cut when noul >= ad_threshold. Noul is the yes probability.
    /// The choice labels a cut; it does not veto one.
    pub ad_threshold: f64,
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
            timeout_secs: get_u64("JEV_TIMEOUT_SECS", 60),
            // 0.5 is "ad is at least as likely as content". Clear reads
            // score well above this once the state is the span itself.
            ad_threshold: get_f64("JEV_AD_THRESHOLD", 0.5),
            review_threshold: get_f64("JEV_REVIEW_THRESHOLD", 0.5),
            segment_target_secs: get_f64("JEV_SEGMENT_TARGET_SECS", 4.0),
            segment_max_secs: get_f64("JEV_SEGMENT_MAX_SECS", 8.0),
            segment_gap_secs: get_f64("JEV_SEGMENT_GAP_SECS", 1.25),
            max_segments_per_call: get_usize("JEV_MAX_SEGMENTS_PER_CALL", 16),
            max_concurrent: get_usize("JEV_MAX_CONCURRENT_REQS", 4).max(1),
        }
    }
}

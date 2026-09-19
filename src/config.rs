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
    /// noul >= ad_threshold AND ad-ish choice => ad candidate.
    pub ad_threshold: f64,
    /// reviewer noul >= review_threshold => confirm candidate.
    pub review_threshold: f64,
    /// target seconds per classification segment.
    pub segment_target_secs: f64,
    /// max segments per Decisions call (2 questions each, limit is 32).
    pub max_segments_per_call: usize,
    /// max concurrent Jev calls (MinusPod fires all windows in parallel).
    pub max_concurrent: usize,
    /// L2 edge pass: sub-piece seconds for trimming run boundaries.
    pub edge_piece_secs: f64,
    /// L2 edge pass: noul >= edge_threshold keeps a piece as ad.
    pub edge_threshold: f64,
    /// L2 edge pass: context pad seconds around an ad run.
    pub edge_context_secs: f64,
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
            // L1 is the recall gate: deliberately permissive. L2 edge
            // agreement is the precision gate. Keep L1 low, tune L2.
            ad_threshold: get_f64("JEV_AD_THRESHOLD", 0.6),
            review_threshold: get_f64("JEV_REVIEW_THRESHOLD", 0.5),
            segment_target_secs: get_f64("JEV_SEGMENT_TARGET_SECS", 30.0),
            max_segments_per_call: get_usize("JEV_MAX_SEGMENTS_PER_CALL", 16),
            max_concurrent: get_usize("JEV_MAX_CONCURRENT_REQS", 4).max(1),
            edge_piece_secs: get_f64("JEV_EDGE_PIECE_SECS", 5.0),
            edge_threshold: get_f64("JEV_EDGE_THRESHOLD", 0.5),
            edge_context_secs: get_f64("JEV_EDGE_CONTEXT_SECS", 90.0),
        }
    }
}

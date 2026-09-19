use regex::Regex;
use std::sync::LazyLock;

use crate::openai::ChatRequest;

/// MinusPod request types, identified without depending on brittle
/// exact prompt text: response_format schema name first, then stable
/// marker phrases, then structural heuristics.
#[derive(Debug, PartialEq, Eq)]
pub enum RequestKind {
    /// Primary ad detection (pass 1). verification=false.
    /// Verification re-scan (pass 2, post-cut audio). verification=true.
    Detection { verification: bool },
    /// Single-candidate review pass.
    Review,
    /// Contradiction trim recovery (no transcript).
    TrimRecovery,
    /// Category assignment for already-found segments.
    CategoryRepair,
    /// Chapter generation (generative; Jev cannot do this).
    Chapters,
    /// Connection/probe calls (tiny prompts).
    Probe,
    Unknown,
}

static TS_LINE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[\s*[\d.]+\s*s\s*-\s*[\d.]+\s*s\s*\]").expect("ts regex")
});

fn count_ts_lines(text: &str) -> usize {
    TS_LINE_RE.find_iter(text).count()
}

pub fn classify(req: &ChatRequest) -> RequestKind {
    let system = req.system_text();
    let user = req.user_text();
    let sys_lc = system.to_lowercase();

    // 1. Authoritative: response_format schema name.
    match req.schema_name().as_deref() {
        Some("trim_recovery") => return RequestKind::TrimRecovery,
        Some("segment_categories") => return RequestKind::CategoryRepair,
        Some("ad_review") => return RequestKind::Review,
        Some("ad_detection") => {
            return RequestKind::Detection {
                verification: is_verification_system(&system),
            }
        }
        _ => {}
    }

    // 2. System marker phrases (stable across MinusPod versions).
    if sys_lc.contains("assigning a category") {
        return RequestKind::CategoryRepair;
    }
    if sys_lc.contains("unchanged boundaries while its reasoning") {
        return RequestKind::TrimRecovery;
    }
    if sys_lc.contains("already been detected") || sys_lc.contains("second look") {
        return RequestKind::Review;
    }
    if sys_lc.contains("identify all advertisement") {
        return RequestKind::Detection {
            verification: is_verification_system(&system),
        };
    }

    // 3. User-prompt structure.
    if user.contains(">>> CANDIDATE AD") {
        return RequestKind::Review;
    }
    if user.contains("Segments needing a category") {
        return RequestKind::CategoryRepair;
    }
    if user.contains("Original candidate span:") && user.contains("Reviewer reasoning:") {
        return RequestKind::TrimRecovery;
    }
    if is_chapters(&system, &user) {
        return RequestKind::Chapters;
    }

    // 4. Heuristic catch-all: dense [Ns - Ns] transcript lines in an
    // ad-related prompt => detection (robust to prompt rewording).
    let ts_lines = count_ts_lines(&user);
    if ts_lines >= 3 && (sys_lc.contains("advertisement") || sys_lc.contains("ad segments")) {
        return RequestKind::Detection {
            verification: is_verification_system(&system),
        };
    }

    // 5. Tiny prompts (probes, verify-connection, "hi") => neutral answer.
    // Real MinusPod work prompts are kilobytes (transcripts); probes are a
    // few dozen chars. Anything in between that reaches here is Unknown.
    let budget = req.token_budget().unwrap_or(u32::MAX);
    if budget <= 16 || system.len() + user.len() < 100 {
        return RequestKind::Probe;
    }

    RequestKind::Unknown
}

fn is_verification_system(system: &str) -> bool {
    let s = system.to_lowercase();
    s.contains("already had advertisements removed") || s.contains("transition tone")
}

fn is_chapters(system: &str, user: &str) -> bool {
    let u = user.to_lowercase();
    (system.trim().is_empty() && (u.contains("topic") || u.contains("chapter")))
        || u.contains("major topic changes")
        || u.contains("chapter titles")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req(system: &str, user: &str, schema: Option<&str>, budget: Option<u32>) -> ChatRequest {
        let rf = schema.map(|name| {
            json!({"type": "json_schema", "json_schema": {"name": name, "schema": {}}})
        });
        ChatRequest {
            model: "m".to_string(),
            messages: vec![
                crate::openai::ChatMessage {
                    role: "system".to_string(),
                    content: json!(system),
                },
                crate::openai::ChatMessage {
                    role: "user".to_string(),
                    content: json!(user),
                },
            ],
            temperature: None,
            max_tokens: budget,
            max_completion_tokens: None,
            response_format: rf,
            stream: None,
        }
    }

    fn detection_user() -> String {
        let mut s = "Podcast: X\nEpisode: Y\nTranscript:\n".to_string();
        for i in 0..5 {
            s.push_str(&format!(
                "[{:.1}s - {:.1}s] some transcript line number {}\n",
                i as f64 * 10.0,
                i as f64 * 10.0 + 9.5,
                i
            ));
        }
        s.push_str("=== WINDOW 1/3: 0.0-10.0 minutes ===\n");
        s
    }

    #[test]
    fn detection_by_schema() {
        let r = req("sys", &detection_user(), Some("ad_detection"), None);
        assert_eq!(
            classify(&r),
            RequestKind::Detection { verification: false }
        );
    }

    #[test]
    fn detection_by_markers_without_schema() {
        let r = req(
            "Analyze this podcast transcript and identify ALL advertisement segments.\nOUTPUT FORMAT: ...",
            &detection_user(),
            None,
            None,
        );
        assert_eq!(
            classify(&r),
            RequestKind::Detection { verification: false }
        );
    }

    #[test]
    fn verification_detected() {
        let r = req(
            "You are reviewing a podcast episode that has ALREADY had advertisements removed with transition tone markers.",
            &detection_user(),
            Some("ad_detection"),
            None,
        );
        assert_eq!(
            classify(&r),
            RequestKind::Detection { verification: true }
        );
    }

    #[test]
    fn reviewer_markers() {
        let r = req(
            "You are reviewing a candidate advertisement that has already been detected.",
            "Original boundaries: 10.0s-50.0s\n>>> CANDIDATE AD START [10.0s] >>>\n",
            Some("ad_review"),
            None,
        );
        assert_eq!(classify(&r), RequestKind::Review);
    }

    #[test]
    fn trim_recovery_markers() {
        let r = req(
            "A podcast ad reviewer returned a candidate ad span with unchanged boundaries while its reasoning says part of the span is not ad content.",
            "Original candidate span: 10.0s-50.0s\nReviewer reasoning:\n...",
            Some("trim_recovery"),
            Some(300),
        );
        assert_eq!(classify(&r), RequestKind::TrimRecovery);
    }

    #[test]
    fn category_repair_markers() {
        let r = req(
            "You are assigning a category to podcast segments already identified in a prior pass.",
            "Segments needing a category:\n[{\"index\": 0}]",
            Some("segment_categories"),
            None,
        );
        assert_eq!(classify(&r), RequestKind::CategoryRepair);
    }

    #[test]
    fn chapters_detected() {
        let r = req(
            "",
            "Analyze this podcast transcript segment and identify 3 major topic changes.\nTranscript:\n[02:30] hello",
            None,
            None,
        );
        assert_eq!(classify(&r), RequestKind::Chapters);
    }

    #[test]
    fn probe_detected() {
        let r = req("", "hi", None, Some(1));
        assert_eq!(classify(&r), RequestKind::Probe);
    }

    #[test]
    fn unknown_for_random() {
        let r = req(
            "You are a helpful assistant.",
            "Tell me a moderately long story about the sea and sailing ships and storms. \
             It should have several paragraphs describing the voyage, the crew, the weather, \
             the strange island they found, and the long journey back home across the ocean.",
            None,
            None,
        );
        assert_eq!(classify(&r), RequestKind::Unknown);
    }
}

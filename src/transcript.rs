use regex::Regex;
use serde::Serialize;
use std::sync::LazyLock;

/// MinusPod detection-window transcript line:
/// `[45.0s - 48.0s] text` (lenient about spacing).
static LINE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\[\s*([\d.]+)\s*s\s*-\s*([\d.]+)\s*s\s*\]\s?(.*)$").expect("line regex")
});

#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// MinusPod ad object. `end_text` (last 3-5 words) is REQUIRED.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Ad {
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
    pub category: String,
    pub reason: String,
    pub end_text: String,
}

pub fn parse_transcript(user_text: &str) -> Vec<Line> {
    let mut out = Vec::new();
    for raw in user_text.lines() {
        let line = raw.trim();
        let Some(caps) = LINE_RE.captures(line) else {
            continue;
        };
        let (Some(s), Some(e)) = (
            caps.get(1).and_then(|m| m.as_str().parse::<f64>().ok()),
            caps.get(2).and_then(|m| m.as_str().parse::<f64>().ok()),
        ) else {
            continue;
        };
        if !s.is_finite() || !e.is_finite() || e < s {
            continue;
        }
        out.push(Line {
            start: s,
            end: e,
            text: caps.get(3).map(|m| m.as_str().trim().to_string()).unwrap_or_default(),
        });
    }
    out
}

/// Group lines into segments of roughly `target_secs` (never reprocess
/// audio; timestamps are preserved verbatim from the transcript).
pub fn to_segments(lines: &[Line], target_secs: f64) -> Vec<Segment> {
    let mut segs = Vec::new();
    let mut cur: Vec<&Line> = Vec::new();
    for line in lines {
        cur.push(line);
        let span = cur.last().map(|l| l.end).unwrap_or(0.0)
            - cur.first().map(|l| l.start).unwrap_or(0.0);
        if span >= target_secs {
            segs.push(flush(&mut cur));
        }
    }
    if !cur.is_empty() {
        segs.push(flush(&mut cur));
    }
    segs
}

fn flush(cur: &mut Vec<&Line>) -> Segment {
    let start = cur.first().map(|l| l.start).unwrap_or(0.0);
    let end = cur.last().map(|l| l.end).unwrap_or(0.0);
    let text = cur
        .iter()
        .map(|l| l.text.clone())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    cur.clear();
    Segment { start, end, text }
}

/// Lines within [start, end] regrouped into ~piece_secs segments.
/// Used to subdivide edge blocks for fine boundary decisions.
pub fn split_span(lines: &[Line], start: f64, end: f64, piece_secs: f64) -> Vec<Segment> {
    let owned: Vec<Line> = lines
        .iter()
        .filter(|l| l.end > start && l.start < end)
        .cloned()
        .collect();
    if owned.is_empty() {
        return Vec::new();
    }
    to_segments(&owned, piece_secs.max(1.0))
}

/// Canonical transcript text for lines within [start-pad, end+pad].
/// Tight per-question context: only what the decision needs.
pub fn context_for(lines: &[Line], start: f64, end: f64, pad_secs: f64) -> String {
    lines
        .iter()
        .filter(|l| l.end > start - pad_secs && l.start < end + pad_secs)
        .map(|l| format!("[{:.1}s - {:.1}s] {}", l.start, l.end, l.text))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Last `n` words of text (for the REQUIRED `end_text` field).
pub fn last_words(text: &str, n: usize) -> String {
    text.split_whitespace()
        .rev()
        .take(n)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split decisions into consecutive ad-only runs.
pub fn runs_of(
    decisions: &[(Segment, Option<(f64, String, String)>)],
) -> Vec<Vec<(Segment, f64, String, String)>> {
    let mut runs = Vec::new();
    let mut cur = Vec::new();
    for (seg, opt) in decisions {
        match opt {
            Some((conf, cat, reason)) => {
                cur.push((seg.clone(), *conf, cat.clone(), reason.clone()));
            }
            None => {
                if !cur.is_empty() {
                    runs.push(std::mem::take(&mut cur));
                }
            }
        }
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    runs
}

/// Merge a run of ad flags with explicit bounds (from edge trimming).
/// Bounds must stay within [first.start, last.end]: shrink-only.
pub fn merge_trimmed_run(
    flags: &[(Segment, f64, String, String)],
    start: f64,
    end: f64,
) -> Ad {
    let first = &flags[0].0;
    let last = &flags[flags.len() - 1].0;
    let min_conf = flags.iter().map(|f| f.1).fold(1.0_f64, f64::min);
    Ad {
        start: start.max(first.start).min(end),
        end: end.min(last.end).max(start),
        confidence: round3(min_conf),
        category: flags[0].2.clone(),
        reason: flags[0].3.clone(),
        end_text: last_words(&last.text, 5),
    }
}

/// Last words of transcript lines fully inside [start, end].
/// Used for the REQUIRED `end_text` field so it reflects the actual span.
pub fn end_text_for(lines: &[Line], start: f64, end: f64) -> Option<String> {
    let text = lines
        .iter()
        .filter(|l| l.start >= start - 0.01 && l.end <= end + 0.01)
        .map(|l| l.text.clone())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let w = last_words(&text, 5);
    if w.is_empty() {
        None
    } else {
        Some(w)
    }
}

pub fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "Podcast: X\nEpisode: Y\nTranscript:\n\
        [45.0s - 48.0s] That's a great point. Let's take a quick break.\n\
        [48.5s - 52.0s] This episode is brought to you by Athletic Greens.\n\
        not a transcript line\n\
        [bad - data] nope\n\
        [60.0s - 61.0s] Welcome back to the show.\n";

    #[test]
    fn parses_lines_and_skips_junk() {
        let lines = parse_transcript(SAMPLE);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].start, 45.0);
        assert_eq!(lines[1].text, "This episode is brought to you by Athletic Greens.");
    }

    #[test]
    fn segments_cover_full_span_without_gaps() {
        let lines = parse_transcript(SAMPLE);
        let segs = to_segments(&lines, 5.0);
        assert!(segs.len() >= 2);
        assert_eq!(segs[0].start, 45.0);
        assert_eq!(segs.last().unwrap().end, 61.0);
    }

    #[test]
    fn merge_run_uses_existing_timestamps() {
        let segs = vec![
            Segment { start: 20.0, end: 30.0, text: "a b c d e f".into() },
            Segment { start: 30.0, end: 40.0, text: "g h i j k l".into() },
            Segment { start: 40.0, end: 50.0, text: "m n o p q r".into() },
            Segment { start: 50.0, end: 60.0, text: "content here stays".into() },
            Segment { start: 60.0, end: 70.0, text: "s t u v w x".into() },
        ];
        let decisions: Vec<(Segment, Option<(f64, String, String)>)> = segs
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                if i == 3 {
                    (s, None)
                } else {
                    (s, Some((0.9, "sponsor".into(), "r".into())))
                }
            })
            .collect();
        let runs = runs_of(&decisions);
        assert_eq!(runs.len(), 2);
        let ads: Vec<Ad> = runs
            .iter()
            .map(|r| merge_trimmed_run(r, r[0].0.start, r[r.len() - 1].0.end))
            .collect();
        assert_eq!(ads.len(), 2);
        assert_eq!(ads[0].start, 20.0);
        assert_eq!(ads[0].end, 50.0);
        assert_eq!(ads[0].end_text, "n o p q r");
        assert_eq!(ads[1].start, 60.0);
        assert_eq!(ads[1].end, 70.0);
    }

    #[test]
    fn last_words_handles_short_text() {
        assert_eq!(last_words("a b", 5), "a b");
        assert_eq!(last_words("a b c d e f", 3), "d e f");
    }

    #[test]
    fn split_span_subdivides_and_covers() {
        let lines = vec![
            Line { start: 0.0, end: 5.0, text: "a".into() },
            Line { start: 5.0, end: 10.0, text: "b".into() },
            Line { start: 10.0, end: 15.0, text: "c".into() },
            Line { start: 15.0, end: 20.0, text: "d".into() },
        ];
        let pieces = split_span(&lines, 0.0, 20.0, 5.0);
        assert_eq!(pieces.len(), 4);
        assert_eq!(pieces[0].start, 0.0);
        assert_eq!(pieces[3].end, 20.0);
        // Outside range => empty.
        assert!(split_span(&lines, 100.0, 200.0, 5.0).is_empty());
    }

    #[test]
    fn context_for_pads_both_sides() {
        let lines = vec![
            Line { start: 0.0, end: 10.0, text: "far before".into() },
            Line { start: 100.0, end: 110.0, text: "near before".into() },
            Line { start: 120.0, end: 130.0, text: "target here".into() },
            Line { start: 140.0, end: 150.0, text: "near after".into() },
            Line { start: 500.0, end: 510.0, text: "far after".into() },
        ];
        let ctx = context_for(&lines, 120.0, 130.0, 60.0);
        assert!(ctx.contains("target here"));
        assert!(ctx.contains("near before"));
        assert!(ctx.contains("near after"));
        assert!(!ctx.contains("far before"));
        assert!(!ctx.contains("far after"));
    }

    #[test]
    fn end_text_uses_in_bounds_lines() {
        let lines = vec![
            Line { start: 0.0, end: 5.0, text: "before content here".into() },
            Line { start: 5.0, end: 10.0, text: "buy now at shop com".into() },
            Line { start: 10.0, end: 15.0, text: "after content here".into() },
        ];
        assert_eq!(
            end_text_for(&lines, 5.0, 10.0).as_deref(),
            Some("buy now at shop com")
        );
        assert!(end_text_for(&lines, 100.0, 200.0).is_none());
    }

    #[test]
    fn trimmed_run_clamps_shrink_only() {
        let flags = vec![
            (
                Segment { start: 20.0, end: 40.0, text: "a b c".into() },
                0.9,
                "sponsor".into(),
                "r".into(),
            ),
            (
                Segment { start: 40.0, end: 60.0, text: "d e f".into() },
                0.8,
                "sponsor".into(),
                "r".into(),
            ),
        ];
        let ad = merge_trimmed_run(&flags, 25.0, 55.0);
        assert_eq!(ad.start, 25.0);
        assert_eq!(ad.end, 55.0);
        assert_eq!(ad.confidence, 0.8);
        // Out-of-range bounds clamp back inside.
        let ad2 = merge_trimmed_run(&flags, 0.0, 999.0);
        assert_eq!(ad2.start, 20.0);
        assert_eq!(ad2.end, 60.0);
    }
}

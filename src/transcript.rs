use regex::Regex;
use serde::Serialize;
use serde_json::{Value, json};
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

/// Group lines into short spans the model can judge as one message.
///
/// Flush when the span reaches `target_secs`, when the next line would
/// push it past `max_secs`, or when the pause before the next line is at
/// least `gap_secs`. Pauses and timestamps stay in code. A single long
/// transcript line is kept whole — there is no finer timestamp to cut on.
pub fn to_segments(lines: &[Line], target_secs: f64, max_secs: f64, gap_secs: f64) -> Vec<Segment> {
    let target = target_secs.max(0.5);
    let max = max_secs.max(0.5);
    let gap_limit = gap_secs.max(0.0);
    let mut segs = Vec::new();
    let mut cur: Vec<&Line> = Vec::new();
    for line in lines {
        if let Some(prev) = cur.last() {
            let gap = line.start - prev.end;
            let span_if_added = line.end - cur.first().map(|l| l.start).unwrap_or(line.start);
            if gap >= gap_limit || span_if_added > max {
                segs.push(flush(&mut cur));
            }
        }
        cur.push(line);
        let span = cur.last().map(|l| l.end).unwrap_or(0.0)
            - cur.first().map(|l| l.start).unwrap_or(0.0);
        if span >= target {
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

/// State for one Decisions call: only this batch, as named fields.
///
/// `before` / `after` are the adjacent spans (local pitch context).
/// The rest of the episode is not included. Questions point at
/// `segments[i].text` with a backticked path.
pub fn detection_state(all: &[Segment], offset: usize, chunk: &[Segment]) -> Value {
    let segments: Vec<Value> = chunk
        .iter()
        .enumerate()
        .map(|(j, seg)| {
            json!({
                "text": seg.text,
                "before": adjacent_text(all, offset + j, -1),
                "after": adjacent_text(all, offset + j, 1),
            })
        })
        .collect();
    json!({ "segments": segments })
}

fn adjacent_text(all: &[Segment], index: usize, dir: isize) -> String {
    let Some(n) = index.checked_add_signed(dir) else {
        return String::new();
    };
    let Some(seg) = all.get(n) else {
        return String::new();
    };
    const MAX_CHARS: usize = 500;
    if seg.text.chars().count() <= MAX_CHARS {
        seg.text.clone()
    } else {
        seg.text.chars().take(MAX_CHARS).collect()
    }
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

/// Turn consecutive ad decisions into MinusPod spans.
///
/// Bounds are the first and last flagged segment's own timestamps.
/// A content segment between two ads stays in the episode: it breaks
/// the run. Silence inside a run is included because the span runs
/// from the first start to the last end. Confidence is the mean noul
/// so one borderline edge line does not sink a clear read. Category
/// and reason come from the strongest line in the run.
pub fn collect_ads(
    lines: &[Line],
    decisions: &[(Segment, Option<(f64, String, String)>)],
) -> Vec<Ad> {
    let mut ads = Vec::new();
    for run in runs_of(decisions) {
        if run.is_empty() {
            continue;
        }
        let start = run[0].0.start;
        let end = run[run.len() - 1].0.end;
        if end <= start {
            continue;
        }
        let mean = run.iter().map(|f| f.1).sum::<f64>() / run.len() as f64;
        let best = run.iter().max_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let (category, reason) = best
            .map(|f| (f.2.clone(), f.3.clone()))
            .unwrap_or_else(|| ("sponsor".to_string(), "jev".to_string()));
        let end_text = end_text_for(lines, start, end)
            .unwrap_or_else(|| last_words(&run[run.len() - 1].0.text, 5));
        ads.push(Ad {
            start,
            end,
            confidence: round3(mean),
            category,
            reason,
            end_text,
        });
    }
    ads
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
        let segs = to_segments(&lines, 5.0, 30.0, 100.0);
        assert!(segs.len() >= 2);
        assert_eq!(segs[0].start, 45.0);
        assert_eq!(segs.last().unwrap().end, 61.0);
    }

    #[test]
    fn segments_split_on_pause_and_cap_length() {
        let lines = vec![
            Line { start: 0.0, end: 3.0, text: "a".into() },
            Line { start: 3.2, end: 6.0, text: "b".into() },
            Line { start: 6.1, end: 9.0, text: "c".into() },
            Line { start: 20.0, end: 23.0, text: "after gap".into() },
        ];
        let segs = to_segments(&lines, 4.0, 8.0, 1.25);
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0].start, 0.0);
        assert_eq!(segs[0].end, 6.0);
        assert_eq!(segs[0].text, "a b");
        assert_eq!(segs[1].text, "c");
        assert_eq!(segs[1].start, 6.1);
        assert_eq!(segs[2].text, "after gap");
        assert_eq!(segs[2].start, 20.0);

        let capped = vec![
            Line { start: 0.0, end: 3.0, text: "a".into() },
            Line { start: 3.0, end: 6.0, text: "b".into() },
            Line { start: 6.0, end: 9.0, text: "c".into() },
        ];
        // Target is 10s, but adding "c" would pass the 7s cap, so the
        // span flushes before it reaches the target.
        let segs = to_segments(&capped, 10.0, 7.0, 5.0);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].end, 6.0);
        assert_eq!(segs[1].text, "c");
    }

    #[test]
    fn detection_state_keeps_only_local_neighbors() {
        let segs = vec![
            Segment { start: 0.0, end: 4.0, text: "far content".into() },
            Segment { start: 4.0, end: 8.0, text: "before words".into() },
            Segment { start: 8.0, end: 12.0, text: "use code SAVE".into() },
            Segment { start: 12.0, end: 16.0, text: "after words".into() },
            Segment { start: 16.0, end: 20.0, text: "much later".into() },
        ];
        let state = detection_state(&segs, 2, &segs[2..3]);
        let text = state.to_string();
        assert!(text.contains("use code SAVE"));
        assert!(text.contains("before words"));
        assert!(text.contains("after words"));
        assert!(!text.contains("far content"));
        assert!(!text.contains("much later"));
        assert!(state["segments"][0]["text"].as_str().unwrap().contains("SAVE"));
    }

    #[test]
    fn collect_ads_uses_segment_timestamps_and_keeps_content() {
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
        let lines = vec![
            Line { start: 20.0, end: 30.0, text: "a b c d e f".into() },
            Line { start: 30.0, end: 40.0, text: "g h i j k l".into() },
            Line { start: 40.0, end: 50.0, text: "m n o p q r".into() },
            Line { start: 50.0, end: 60.0, text: "content here stays".into() },
            Line { start: 60.0, end: 70.0, text: "s t u v w x".into() },
        ];
        let ads = collect_ads(&lines, &decisions);
        assert_eq!(ads.len(), 2);
        assert_eq!(ads[0].start, 20.0);
        assert_eq!(ads[0].end, 50.0);
        assert_eq!(ads[0].end_text, "n o p q r");
        assert_eq!(ads[0].confidence, 0.9);
        assert_eq!(ads[1].start, 60.0);
        assert_eq!(ads[1].end, 70.0);
    }

    #[test]
    fn last_words_handles_short_text() {
        assert_eq!(last_words("a b", 5), "a b");
        assert_eq!(last_words("a b c d e f", 3), "d e f");
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

}

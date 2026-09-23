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

/// One classified stretch. `sponsor` is true only for a paid read
/// (host-read, produced spot, or inserted commercial). Show talk,
/// sign-offs, and the show promoting itself are not sponsors.
#[derive(Debug, Clone)]
pub struct ClassifiedSpan {
    pub segment: Segment,
    pub noul: f64,
    pub sponsor: bool,
    pub choice: String,
}

/// Build MinusPod cuts from paid-sponsor spans only.
///
/// A span is a core when it is a sponsor and noul >= `cut_threshold`.
/// A neighboring sponsor span with noul >= `attach_threshold` joins that
/// core when the gap is at most `attach_gap_secs`, so one read becomes
/// one cut instead of a few short scraps. A non-sponsor span never joins,
/// so show talk between reads stays. Confidence is the strongest noul in
/// the cut, which is the probability MinusPod compares to its slider.
pub fn collect_sponsor_ads(
    lines: &[Line],
    spans: &[ClassifiedSpan],
    cut_threshold: f64,
    attach_threshold: f64,
    attach_gap_secs: f64,
) -> Vec<Ad> {
    let mut ads = Vec::new();
    let mut i = 0;
    while i < spans.len() {
        if !is_core(&spans[i], cut_threshold) {
            i += 1;
            continue;
        }
        let mut lo = i;
        let mut hi = i;
        while lo > 0 && can_attach(&spans[lo - 1], &spans[lo], attach_threshold, attach_gap_secs)
        {
            lo -= 1;
        }
        while hi + 1 < spans.len()
            && can_attach(&spans[hi + 1], &spans[hi], attach_threshold, attach_gap_secs)
        {
            hi += 1;
        }
        ads.push(ad_from_spans(lines, &spans[lo..=hi]));
        i = hi + 1;
    }
    ads
}

fn is_core(span: &ClassifiedSpan, cut_threshold: f64) -> bool {
    span.sponsor && span.noul >= cut_threshold
}

fn can_attach(
    candidate: &ClassifiedSpan,
    neighbor: &ClassifiedSpan,
    attach_threshold: f64,
    attach_gap_secs: f64,
) -> bool {
    if !candidate.sponsor || candidate.noul < attach_threshold {
        return false;
    }
    let gap = if candidate.segment.start >= neighbor.segment.start {
        candidate.segment.start - neighbor.segment.end
    } else {
        neighbor.segment.start - candidate.segment.end
    };
    (-0.05..=attach_gap_secs).contains(&gap)
}

fn ad_from_spans(lines: &[Line], spans: &[ClassifiedSpan]) -> Ad {
    let start = spans[0].segment.start;
    let end = spans[spans.len() - 1].segment.end;
    let best = spans.iter().max_by(|a, b| {
        a.noul
            .partial_cmp(&b.noul)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (choice, noul) = best
        .map(|s| (s.choice.as_str(), s.noul))
        .unwrap_or(("sponsor", 0.0));
    let end_text = end_text_for(lines, start, end).unwrap_or_else(|| {
        last_words(&spans[spans.len() - 1].segment.text, 5)
    });
    Ad {
        start,
        end,
        confidence: round3(noul),
        category: "sponsor".to_string(),
        reason: format!("jev:{choice} noul={noul:.2}"),
        end_text,
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

    fn span(start: f64, end: f64, text: &str, noul: f64, sponsor: bool, choice: &str) -> ClassifiedSpan {
        ClassifiedSpan {
            segment: Segment { start, end, text: text.into() },
            noul,
            sponsor,
            choice: choice.into(),
        }
    }

    #[test]
    fn collect_sponsor_ads_keeps_show_and_uses_strongest_score() {
        let spans = vec![
            span(20.0, 30.0, "a b c d e f", 0.55, true, "host_read_sponsor"),
            span(30.0, 40.0, "g h i j k l", 0.96, true, "host_read_sponsor"),
            span(40.0, 50.0, "m n o p q r", 0.42, true, "paid_ad"),
            span(50.0, 60.0, "content here stays put", 0.91, false, "content"),
            span(60.0, 70.0, "comment of the day today", 0.93, false, "self_promo"),
            span(80.0, 90.0, "s t u v w x", 0.88, true, "inserted_ad"),
        ];
        let lines: Vec<Line> = spans
            .iter()
            .map(|s| Line {
                start: s.segment.start,
                end: s.segment.end,
                text: s.segment.text.clone(),
            })
            .collect();
        let ads = collect_sponsor_ads(&lines, &spans, 0.5, 0.4, 8.0);
        assert_eq!(ads.len(), 2);
        assert_eq!(ads[0].start, 20.0);
        assert_eq!(ads[0].end, 50.0);
        assert_eq!(ads[0].confidence, 0.96);
        assert_eq!(ads[0].category, "sponsor");
        assert!(ads[0].reason.contains("host_read_sponsor"));
        assert_eq!(ads[0].end_text, "n o p q r");
        assert_eq!(ads[1].start, 80.0);
        assert_eq!(ads[1].end, 90.0);
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

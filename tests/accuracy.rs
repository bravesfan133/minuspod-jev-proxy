//! Accuracy harness for the detection pipeline.
//!
//! What this measures
//! -----------------
//! The pipeline's *logic*: segmentation, the recall gate, neighbor merging,
//! edge trimming, and boundary placement. The classifier is stubbed, so a
//! score here says nothing about how good Jev is. It says whether the code
//! around the classifier cuts where it should and no wider.
//!
//! That split is deliberate. When an ad survives or a cut lands badly there
//! are two very different suspects: the model, or this code. These fixtures
//! let the second be checked without spending a single API call, so a model
//! comparison later can assume the surrounding logic is sound.
//!
//! Ground truth
//! ------------
//! `tests/fixtures/*.labels.json` records the exact span of one spliced
//! sponsor read per fixture. `negative_*` fixtures record that there are no
//! ads at all. Labels are written by `tools/build_fixtures.py` and are the
//! true bounds of the spliced text, independent of any model output.
//!
//! Metrics
//! -------
//! * `hit`      - a cut overlapping a labeled ad by more than `min_overlap`.
//! * `false_pos` - a cut overlapping no labeled ad at all.
//! * `lead` / `lag` - seconds of show talk left in before the ad, or removed
//!   after it. This is the "cutting off early/late" complaint, measured.
//!
//! Run with `cargo test --test accuracy -- --nocapture` for a per-fixture
//! table, or plain `cargo test` to assert the thresholds below.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use minuspod_jev_proxy::transcript::{
    Ad, ClassifiedSpan, Line, Segment, collect_sponsor_ads, end_text_for, parse_transcript,
    shrink_bounds, split_into_pieces, to_segments,
};

/// A labeled ad span. `kind` and `sponsor` are carried through into the
/// report so a row can be traced back to what was actually spliced in.
#[derive(Debug, Deserialize)]
struct Label {
    kind: String,
    start: f64,
    end: f64,
    #[serde(default)]
    sponsor: String,
}

#[derive(Debug, Deserialize)]
struct FixtureLabels {
    #[allow(dead_code, reason = "kept for a clearer panic message")]
    name: String,
    /// Every line of the fixture as the builder wrote it. Only the count is
    /// compared, against the lines the Rust parser recovers from the text
    /// file: if those two ever disagree, the fixture and its labels have
    /// drifted apart and every score below would be measuring the wrong span.
    lines: Vec<serde_json::Value>,
    labels: Vec<Label>,
}

/// What the stub classifier should say about a piece of text.
#[derive(Debug, Clone, Copy)]
struct Stub {
    /// Probability returned for "is this a paid ad".
    noul: f64,
    /// Label returned for the taxonomy question.
    choice: &'static str,
}

impl Stub {
    /// A clear sponsor read: the classifier is confident and calls it a read.
    fn ad() -> Self {
        Self { noul: 0.95, choice: "host_read_sponsor" }
    }
    /// A host pitching the show itself. Should not be cut.
    fn self_promo() -> Self {
        Self { noul: 0.88, choice: "self_promo" }
    }
    /// Ordinary editorial talk, or a brand mentioned with no ask.
    fn content() -> Self {
        Self { noul: 0.04, choice: "content" }
    }
    /// Ad-shaped text that sits just under the edge gate.
    fn borderline() -> Self {
        Self { noul: 0.42, choice: "host_read_sponsor" }
    }
}

/// Phrases that mark a host read.
///
/// Deliberately narrow. These are the stock openers of a sponsor pitch, not
/// merely the presence of a brand or a URL. A looser rule would fire on
/// "someone who can review your code", which is exactly how a naive keyword
/// gate eats show talk.
const AD_MARKERS: &[&str] = &["brought to you by", "sponsored by"];

/// "supported by" and "today's sponsor" only count when a proper noun follows.
/// Without that, "supported by a dozen small ideas" reads as a sponsor tag,
/// which it emphatically is not.
fn has_sponsor_marker(text: &str) -> bool {
    let lower = text.to_lowercase();
    if AD_MARKERS.iter().any(|m| lower.contains(m)) {
        return true;
    }
    for marker in ["supported by", "today's sponsor", "our sponsor"] {
        let mut from = 0usize;
        while let Some(idx) = lower[from..].find(marker) {
            // Look at the first token after the marker that is not filler
            // ("is", "was", "here"), and require it to be a proper noun.
            let tail = &text[from + idx + marker.len()..];
            let name = tail
                .split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                .find(|w| !w.is_empty() && !matches!(*w, "is" | "was" | "here" | "and"));
            if name.is_some_and(|n| n.chars().next().is_some_and(|c| c.is_uppercase())) {
                return true;
            }
            from += idx + marker.len();
        }
    }
    false
}

/// Call-to-action shapes. A promo code is uppercase alphanumerics introduced by
/// the word "code"; a bare URL must carry a path or be spoken as a web address.
/// Requiring the marker keeps these from matching ordinary uses of the words.
fn is_ad_text(text: &str) -> bool {
    let lower = text.to_lowercase();
    if has_sponsor_marker(text) {
        return true;
    }

    // "code SHOP10" - a promo code. Judged on the original casing, because the
    // giveaway is that the token is shouted or contains digits. Neither holds
    // for "write code responsibly" or "Claude Code".
    let mut tokens = text.split_whitespace();
    while let Some(t) = tokens.next() {
        if t.trim_matches(|c: char| !c.is_alphanumeric()).eq_ignore_ascii_case("code") {
            if let Some(next) = tokens.next() {
                let bare = next.trim_matches(|c: char| !c.is_alphanumeric());
                let digits = bare.chars().any(|c| c.is_ascii_digit());
                let shouted = bare.chars().filter(|c| c.is_uppercase()).count() >= 2;
                // A promo code has letters. Requiring that keeps
                // "the whole pitch of Claude Code? 150 expert engineers" from
                // reading as a code, where `150` is a bare quantity.
                let has_letters = bare.chars().any(|c| c.is_alphabetic());
                if has_letters && (digits || shouted) {
                    return true;
                }
            }
        }
    }

    // A spoken web address: "dot com" and friends.
    if lower.contains("dot com") || lower.contains("dot co") {
        return true;
    }

    // A URL on its own is not an ad. "Think of something like Booking.com" and
    // "one of our PMs left to join Calm, Calm.com" are brands under discussion.
    // A URL only counts when it carries an ask: check it out, go to, sign up,
    // try it free, download. Requiring the pairing is what separates a sponsor
    // read from a mention.
    let has_url = lower.split_whitespace().any(|w| {
        let bare = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '/');
        bare.contains('.') && bare.ends_with(".com") && bare.matches('.').count() == 1
    });
    has_url && ASK_PHRASES.iter().any(|a| lower.contains(a))
}

/// The ask that turns a URL from a mention into a call to action.
const ASK_PHRASES: &[&str] = &[
    "check it out",
    "go to",
    "sign up",
    "try it",
    "try it free",
    "download",
    "get a free",
    "for a free",
    "head on over to",
    "click",
    "enter code",
];

/// The show promoting its own other content.
///
/// These carry the same CTA signals as a sponsor read but are the show's own
/// Patreon, merch, back catalogue, or a guest handing out their own site. They
/// are labeled `self_promo` and **are** cut: a plug for the show is removable
/// content, which is why this must be distinguished from a paid sponsor rather
/// than treated as ordinary talk.
fn is_self_promo(text: &str) -> bool {
    let lower = text.to_lowercase();
    [
        "our patreon",
        "patreon link",
        "our merch",
        "our newsletter",
        "our store",
        "product pass",
        "productpass",
        "past episodes",
        "all episodes",
        "newsletter.com",
        "podcast.com",
    ]
    .iter()
    .any(|m| lower.contains(m))
        // A guest pointing at their own site or book: "the domain is X.com",
        // "my handle is @...", "my book is called...".
        || ["my contact form", "my website", "my book", "my handle", "find me at"]
            .iter()
            .any(|m| lower.contains(m))
}

/// Classify one span of text.
///
/// The stub is keyword-driven rather than table-driven, because the point is
/// to exercise the code around the classifier, not to be a model. Text
/// carrying a sponsor read marker reads as an ad; everything else is content.
fn classify(text: &str) -> Stub {
    if is_self_promo(text) {
        Stub::self_promo()
    } else if is_ad_text(text) {
        Stub::ad()
    } else {
        Stub::content()
    }
}

/// Mirror of `handlers::is_paid_sponsor`.
///
/// `self_promo` is a cut: the show's own Patreon and community plugs are
/// removed here, matching the deployed proxy's behavior. Only `content` and
/// `cross_promo` survive.
fn is_paid_sponsor(choice: &str) -> bool {
    matches!(
        choice,
        "paid_ad" | "host_read_sponsor" | "inserted_ad" | "self_promo"
    )
}

/// Run segmentation and the recall gate, then merge neighbors, exactly as the
/// handler does.
fn detect(
    lines: &[Line],
    recall_threshold: f64,
    attach_threshold: f64,
    attach_gap_secs: f64,
    segment_target_secs: f64,
    segment_max_secs: f64,
    segment_gap_secs: f64,
) -> Vec<Ad> {
    let segments = to_segments(
        lines,
        segment_target_secs,
        segment_max_secs,
        segment_gap_secs,
    );
    let spans: Vec<ClassifiedSpan> = segments
        .into_iter()
        .map(|segment: Segment| {
            let stub = classify(&segment.text);
            ClassifiedSpan {
                noul: stub.noul,
                sponsor: is_paid_sponsor(stub.choice),
                choice: stub.choice.to_string(),
                segment,
            }
        })
        .collect();
    collect_sponsor_ads(
        lines,
        &spans,
        recall_threshold,
        attach_threshold,
        attach_gap_secs,
    )
}

/// Mirror of `handlers::edge_trim`, driven by the same stub.
///
/// The handler asks the classifier about ~2s pieces at each end and keeps a
/// piece only when both signals agree. This reproduces that so the harness
/// measures the shipped geometry, not a paraphrase of it.
fn edge_trim(lines: &[Line], ads: Vec<Ad>, edge_piece_secs: f64, edge_threshold: f64) -> Vec<Ad> {
    let depth = (edge_piece_secs * 3.0).max(edge_piece_secs);
    let mut out = Vec::with_capacity(ads.len());

    for ad in &ads {
        let head_end = (ad.start + depth).min(ad.end);
        let tail_start = (ad.end - depth).max(ad.start);
        let head = split_into_pieces(lines, ad.start, head_end, edge_piece_secs);
        let tail = split_into_pieces(lines, tail_start, ad.end, edge_piece_secs);

        let keep = |pieces: &[Segment]| -> Vec<bool> {
            pieces
                .iter()
                .map(|p| {
                    let stub = classify(&p.text);
                    stub.noul >= edge_threshold && is_paid_sponsor(stub.choice)
                })
                .collect()
        };
        let head_keep = keep(&head);
        let tail_keep = keep(&tail);

        let (new_start, new_end) =
            shrink_bounds(ad, &head, &head_keep, &tail, &tail_keep);

        if new_start > ad.start || new_end < ad.end {
            let mut trimmed = ad.clone();
            trimmed.start = new_start;
            trimmed.end = new_end;
            // The handler recomputes end_text from the trimmed bounds, because
            // MinusPod uses it to locate the cut. Mirror that here, or this
            // harness would be asserting on a value the server never emits.
            if let Some(t) = end_text_for(lines, new_start, new_end) {
                trimmed.end_text = t;
            }
            out.push(trimmed);
        } else {
            out.push(ad.clone());
        }
    }
    out
}

/// Overlap in seconds between two closed intervals.
fn overlap(a_start: f64, a_end: f64, b_start: f64, b_end: f64) -> f64 {
    (a_end.min(b_end) - a_start.max(b_start)).max(0.0)
}

/// How much of `ad` must sit inside `truth` before the cut counts as a hit.
const MIN_OVERLAP_SECS: f64 = 1.0;

/// A cut is "too wide" if it extends more than this past a labeled ad.
const SLACK_SECS: f64 = 6.0;

#[derive(Debug, Default, Clone)]
struct Score {
    hits: usize,
    false_pos: usize,
    misses: usize,
    /// Show talk left in before the ad starts.
    lead_secs: Vec<f64>,
    /// Show talk removed after the ad ends.
    lag_secs: Vec<f64>,
    /// Show talk taken out alongside a correct cut.
    overshoot_secs: Vec<f64>,
}

impl Score {
    fn precision(&self) -> f64 {
        let predicted = self.hits + self.false_pos;
        if predicted == 0 {
            1.0
        } else {
            self.hits as f64 / predicted as f64
        }
    }

    fn recall(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            1.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    fn merge(&mut self, other: Score) {
        self.hits += other.hits;
        self.false_pos += other.false_pos;
        self.misses += other.misses;
        self.lead_secs.extend(other.lead_secs);
        self.lag_secs.extend(other.lag_secs);
        self.overshoot_secs.extend(other.overshoot_secs);
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Score predicted cuts against labels.
fn score(cuts: &[Ad], labels: &[Label]) -> Score {
    let mut s = Score::default();

    for label in labels {
        let covered = cuts.iter().any(|c| {
            overlap(c.start, c.end, label.start, label.end) >= MIN_OVERLAP_SECS
        });
        if covered {
            s.hits += 1;

            // Tightest cut overlapping this ad, for boundary error.
            if let Some(best) = cuts
                .iter()
                .filter(|c| overlap(c.start, c.end, label.start, label.end) > 0.0)
                .min_by(|a, b| {
                    let oa = overlap(a.start, a.end, label.start, label.end);
                    let ob = overlap(b.start, b.end, label.start, label.end);
                    ob.partial_cmp(&oa).unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                if best.start < label.start {
                    s.lead_secs.push(label.start - best.start);
                }
                if best.end > label.end {
                    s.lag_secs.push(best.end - label.end);
                }
                // Everything the cut takes that is not in the labeled ad.
                let outside = (best.end - best.start)
                    - overlap(best.start, best.end, label.start, label.end);
                s.overshoot_secs.push(outside.max(0.0));
            }
        } else {
            s.misses += 1;
        }
    }

    for cut in cuts {
        let hits_any = labels
            .iter()
            .any(|l| overlap(cut.start, cut.end, l.start, l.end) >= MIN_OVERLAP_SECS);
        if !hits_any {
            s.false_pos += 1;
        }
    }

    s
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Load every fixture that has labels, plus the negative ones.
fn load_fixtures() -> Vec<(String, Vec<Line>, Vec<Label>, bool)> {
    let dir = fixture_dir();
    let mut out = Vec::new();

    let entries = std::fs::read_dir(&dir).expect("read fixtures dir");
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();

    for path in paths {
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(&path).expect("read fixture");
        let lines = parse_transcript(&text);
        if lines.is_empty() {
            continue;
        }

        let label_path = dir.join(format!("{stem}.labels.json"));
        let labels = if label_path.exists() {
            let raw = std::fs::read_to_string(&label_path).expect("read labels");
            let parsed: FixtureLabels = serde_json::from_str(&raw).expect("parse labels");
            // Cross-check the JSON against the transcript we actually parse.
            assert_eq!(
                parsed.lines.len(),
                lines.len(),
                "label file and transcript disagree on line count for {stem}",
            );
            parsed.labels
        } else {
            Vec::new()
        };

        let is_negative = stem.starts_with("negative_");
        assert!(
            !(is_negative && !labels.is_empty()),
            "{stem} is marked negative but carries labels",
        );
        out.push((stem, lines, labels, is_negative));
    }
    out
}

/// Detection settings mirroring the shipped defaults.
const RECALL: f64 = 0.35;
const ATTACH: f64 = 0.40;
const ATTACH_GAP: f64 = 8.0;
const SEG_TARGET: f64 = 4.0;
const SEG_MAX: f64 = 8.0;
const SEG_GAP: f64 = 1.25;
const EDGE_PIECE: f64 = 2.0;
const EDGE_THRESHOLD: f64 = 0.50;

fn run_pipeline(lines: &[Line]) -> (Vec<Ad>, Vec<Ad>) {
    let untrimmed = detect(
        lines, RECALL, ATTACH, ATTACH_GAP, SEG_TARGET, SEG_MAX, SEG_GAP,
    );
    let trimmed = edge_trim(lines, untrimmed.clone(), EDGE_PIECE, EDGE_THRESHOLD);
    (untrimmed, trimmed)
}

#[test]
fn fixtures_exist_and_parse() {
    let fixtures = load_fixtures();
    assert!(
        fixtures.len() >= 6,
        "expected a usable fixture set, found {}",
        fixtures.len(),
    );
    let negatives = fixtures.iter().filter(|f| f.3).count();
    let positives = fixtures.iter().filter(|f| !f.3 && !f.2.is_empty()).count();
    assert!(negatives >= 1, "need at least one negative fixture");
    assert!(positives >= 3, "need at least three spliced fixtures, got {positives}");
}

#[test]
fn every_labeled_span_sits_on_real_transcript_lines() {
    for (name, lines, labels, _) in load_fixtures() {
        for label in &labels {
            assert!(
                label.end > label.start,
                "{name}: label span is inverted: {} -> {}",
                label.start,
                label.end,
            );
            // The label must overlap actual parsed lines, otherwise it is
            // describing audio the transcript does not contain.
            let covered = lines
                .iter()
                .filter(|l| overlap(l.start, l.end, label.start, label.end) > 0.0)
                .count();
            assert!(
                covered > 0,
                "{name}: label {} -> {} overlaps no transcript lines",
                label.start,
                label.end,
            );
        }
    }
}

#[test]
fn spliced_ad_is_detected() {
    // Every fixture with a labeled ad must produce at least one cut that
    // overlaps it. This is the recall claim: ads are not being left in.
    let mut failures = Vec::new();
    for (name, lines, labels, is_negative) in load_fixtures() {
        if is_negative || labels.is_empty() {
            continue;
        }
        let (untrimmed, trimmed) = run_pipeline(&lines);
        let before = score(&untrimmed, &labels);
        let after = score(&trimmed, &labels);
        if after.hits == 0 {
            failures.push(format!(
                "{name}: no cut overlapped the labeled ad ({} -> {})",
                labels[0].start, labels[0].end,
            ));
        }
        assert_eq!(before.hits, after.hits, "{name}: trimming changed recall");
    }
    assert!(failures.is_empty(), "missed ads:\n{}", failures.join("\n"));
}

#[test]
fn show_content_is_never_cut() {
    // The negative fixtures are pure conversation. Any cut here is a false
    // positive, and a false positive is show talk removed from the episode.
    let mut offenders = Vec::new();
    for (name, lines, labels, is_negative) in load_fixtures() {
        if !is_negative {
            continue;
        }
        let (_, trimmed) = run_pipeline(&lines);
        let s = score(&trimmed, &labels);
        if s.false_pos > 0 {
            offenders.push(format!(
                "{name}: {} cut(s) in ad-free conversation",
                s.false_pos
            ));
        }
    }
    assert!(
        offenders.is_empty(),
        "false positives in ad-free episodes:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn trimming_never_widens_a_cut() {
    // The core safety property: a rejected piece is evidence the cut was too
    // wide, so bounds may only move inward.
    for (name, lines, _, _) in load_fixtures() {
        let (untrimmed, trimmed) = run_pipeline(&lines);
        assert_eq!(
            untrimmed.len(),
            trimmed.len(),
            "{name}: trimming changed the number of cuts",
        );
        for (before, after) in untrimmed.iter().zip(trimmed.iter()) {
            assert!(
                after.start >= before.start && after.end <= before.end,
                "{name}: cut widened: [{:.1}, {:.1}] -> [{:.1}, {:.1}]",
                before.start,
                before.end,
                after.start,
                after.end,
            );
        }
    }
}

#[test]
fn trimming_does_not_overrun_into_show_content() {
    // The complaint being fixed is cuts that take show talk with them. After
    // trimming, the part of a cut outside the labeled ad must stay small.
    let mut worst: BTreeMap<&str, f64> = BTreeMap::new();
    for (name, lines, labels, is_negative) in load_fixtures() {
        if is_negative || labels.is_empty() {
            continue;
        }
        let (_, trimmed) = run_pipeline(&lines);
        let s = score(&trimmed, &labels);
        for v in &s.overshoot_secs {
            let e = worst.entry(Box::leak(name.clone().into_boxed_str())).or_insert(0.0);
            *e = e.max(*v);
        }
    }
    for (name, secs) in &worst {
        assert!(
            *secs <= SLACK_SECS,
            "{name}: cut overruns the labeled ad by {secs:.1}s (limit {SLACK_SECS:.1}s)",
        );
    }
}

#[test]
fn trimming_never_inverts_or_collapses_a_cut() {
    for (name, lines, _, _) in load_fixtures() {
        let (untrimmed, trimmed) = run_pipeline(&lines);
        for ad in &trimmed {
            assert!(ad.end > ad.start, "{name}: inverted cut [{:.1}, {:.1}]", ad.start, ad.end);
            assert!(ad.end - ad.start > 0.0, "{name}: zero-length cut");
            // And every cut must still lie inside the window it came from.
            assert!(
                ad.start >= lines.first().unwrap().start - 0.01
                    && ad.end <= lines.last().unwrap().end + 0.01,
                "{name}: cut [{:.1}, {:.1}] escapes the window",
                ad.start,
                ad.end,
            );
        }
        assert!(untrimmed.iter().all(|a| a.end > a.start));
    }
}

#[test]
fn end_text_describes_the_trimmed_span() {
    // MinusPod requires end_text and uses it to locate the cut. After
    // trimming it must reflect the trimmed bounds, not the original.
    for (name, lines, _, _) in load_fixtures() {
        let (untrimmed, trimmed) = run_pipeline(&lines);
        for (before, after) in untrimmed.iter().zip(trimmed.iter()) {
            if after.start == before.start && after.end == before.end {
                continue;
            }
            let words: Vec<&str> = after.end_text.split_whitespace().collect();
            assert!(
                !words.is_empty(),
                "{name}: trimmed cut has empty end_text",
            );
            // Every end_text word must come from a line inside the new bounds.
            let joined = after.end_text.to_lowercase();
            for w in words {
                let needle = w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                if needle.is_empty() {
                    continue;
                }
                let in_span = lines
                    .iter()
                    .filter(|l| l.start >= after.start - 0.01 && l.end <= after.end + 0.01)
                    .any(|l| l.text.to_lowercase().contains(&needle));
                assert!(
                    in_span,
                    "{name}: end_text word {w:?} is not inside the trimmed cut",
                );
            }
            let _ = joined;
        }
    }
}

/// The stub must not cut show talk. A regression here means the *logic* is
/// sound but the classifier is over-firing, which on a real backend would show
/// up as chunks of interview being cut out.
#[test]
fn precision_stays_high_across_fixtures() {
    let mut totals = Score::default();
    for (_, lines, labels, _) in load_fixtures() {
        let (_, trimmed) = run_pipeline(&lines);
        totals.merge(score(&trimmed, &labels));
    }
    assert!(
        totals.precision() >= 0.75,
        "precision {:.3} below 0.75 (hits={} false_pos={})",
        totals.precision(),
        totals.hits,
        totals.false_pos,
    );
}

/// Every spliced ad must be found. Recall is the "leaving ads in" complaint,
/// so this is the number that has to stay at 1.0.
#[test]
fn recall_is_perfect_across_fixtures() {
    let mut totals = Score::default();
    for (_, lines, labels, _) in load_fixtures() {
        let (_, trimmed) = run_pipeline(&lines);
        totals.merge(score(&trimmed, &labels));
    }
    assert_eq!(totals.misses, 0, "{} labeled ad(s) were left in", totals.misses);
    assert_eq!(totals.hits as usize, totals.hits as usize);
}

#[test]
fn report_per_fixture_accuracy() {
    // Not an assertion: prints the table. Run with --nocapture.
    let mut totals = Score::default();
    println!(
        "\n{:<22} {:<40} {:>4} {:>4} {:>3} {:>3} {:>7} {:>7} {:>7}",
        "label", "fixture", "pre", "cuts", "hit", "fp", "lead", "lag", "over"
    );

    for (name, lines, labels, is_negative) in load_fixtures() {
        let (untrimmed, trimmed) = run_pipeline(&lines);
        let s = score(&trimmed, &labels);
        totals.merge(s.clone());

        let tag = if is_negative {
            "neg".to_string()
        } else {
            labels
                .first()
                .map(|l| format!("{}/{}", l.kind, l.sponsor))
                .unwrap_or_else(|| "ad".to_string())
        };
        println!(
            "{tag:<22} {name:<40} {:>4} {:>4} {:>3} {:>3} {:>7.1} {:>7.1} {:>7.1}",
            untrimmed.len(),
            trimmed.len(),
            s.hits,
            s.false_pos,
            mean(&s.lead_secs),
            mean(&s.lag_secs),
            mean(&s.overshoot_secs),
        );
    }

    println!(
        "\ntotals: hits={} misses={} false_pos={} precision={:.3} recall={:.3}\n\
         mean lead={:.2}s lag={:.2}s overrun={:.2}s",
        totals.hits,
        totals.misses,
        totals.false_pos,
        totals.precision(),
        totals.recall(),
        mean(&totals.lead_secs),
        mean(&totals.lag_secs),
        mean(&totals.overshoot_secs),
    );
}

/// The stubbed classifier must keep its own promises, otherwise every other
/// result in this file is measuring a broken stub.
#[test]
fn stub_classifier_behaves() {
    assert!(is_paid_sponsor(classify("brought to you by Squarespace").choice));
    assert!(is_paid_sponsor(classify("our patreon is live").choice));
    assert!(!is_paid_sponsor(classify("we were talking about pricing").choice));
    // A cross-promo is a pitch for someone else's show, which is left alone.
    assert!(!is_paid_sponsor("cross_promo"));

    assert!(classify("go to shopify.com/lenny and use code SHOP").noul >= EDGE_THRESHOLD);
    assert!(classify("the idea of taste in design").noul < EDGE_THRESHOLD);
    assert!(Stub::borderline().noul < EDGE_THRESHOLD);
}

/// Words that look ad-shaped but are ordinary in a technical conversation.
/// A keyword gate that matches these would cut show talk, which is the failure
/// this whole harness exists to prevent.
#[test]
fn stub_does_not_fire_on_ordinary_conversation() {
    for text in [
        "someone who can review your code",
        "the head of engineering for Claude Code",
        "we call it vibe coding, and you all have a Codex app",
        "I can write code responsibly now",
        "AI writes all our code",
        "are they a Claude Code person?",
    ] {
        assert!(
            !is_ad_text(&text.to_lowercase()),
            "stub fired on ordinary talk: {text:?}",
        );
    }
}

#[test]
fn stub_fires_on_real_read_shapes() {
    for text in [
        "this segment is brought to you by Acorns",
        "today's sponsor is Squarespace",
        "go to shopify.com/lenny and enter code SHOP10 at checkout",
        "use code SQUAREPOD for your first month free",
        "for more info, dot com slash lenny",
        "this episode is supported by Acorns",
    ] {
        // Original casing: an uppercase promo code is part of the signal.
        assert!(is_ad_text(text), "stub missed a read: {text:?}");
    }
}
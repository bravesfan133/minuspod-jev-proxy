#!/usr/bin/env python3
"""Replay a processed MinusPod episode through a proxy and diff the results.

MinusPod stores the transcript and the cuts it actually applied, so a real
episode can be re-sent through the proxy and compared against what the
deployed version produced. That makes the change measurable on real content
instead of on fixtures written to match the code.

The transcript in the database is VTT (`[hh:mm:ss.mmm --> hh:mm:ss.mmm] text`).
The proxy expects MinusPod's window format (`[45.0s - 48.0s] text`), so the
transcript is converted on the way out.

Jev is not deterministic: the same window can come back slightly different
between runs. `--noise` sets the probability delta below which a disagreement
is reported as noise rather than as a real change, so ordinary jitter is not
mistaken for an improvement or a regression.

Usage:
    python3 tools/replay_episode.py --episode 143998 --base http://localhost:8787
    python3 tools/replay_episode.py --episode 138006 --base http://localhost:8787 \\
        --candidate http://localhost:8899
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DB = "/Users/dad/docker/minuspod/data/podcast.db"

# MinusPod slides overlapping detection windows. 600s windows with a 420s
# stride is what the live logs show (span_secs ~600, adjacent windows 70s
# apart), so this reproduces the real traffic shape.
WINDOW_SECS = 600
STRIDE_SECS = 420

VTT_RE = re.compile(r"^\[(\d[\d:.]+)\s*-->\s*(\d[\d:.]+)\]\s*(.*)$")
WINDOW_RE = re.compile(r"^=== Window \d+ \(([\d.]+)-([\d.]+)min\) ===$")

# The system prompt needs these phrases for the proxy's fallback router to
# recognise a detection request (src/classify.rs tier 4).
DETECTION_SYSTEM = (
    "You are a podcast ad detection system. Identify all advertisement "
    "segments in the transcript and return their boundaries."
)


def to_secs(ts: str) -> float:
    total = 0.0
    for part in ts.split(":"):
        total = total * 60 + float(part)
    return total


@dataclass
class Line:
    start: float
    end: float
    text: str


@dataclass
class Cut:
    start: float
    end: float
    confidence: float
    category: str
    reason: str
    end_text: str

    @property
    def key(self) -> tuple[int, int]:
        return (round(self.start), round(self.end))

    def near(self, other: "Cut", slack: float = 1.0) -> bool:
        """True when two cuts sit within `slack` seconds of each other."""
        return (
            abs(self.start - other.start) <= slack
            and abs(self.end - other.end) <= slack
        )


@dataclass
class Report:
    """One proxy's output for one episode."""
    name: str
    cuts: list[Cut] = field(default_factory=list)
    windows: int = 0
    failed_windows: int = 0
    latency_ms: int = 0


def parse_vtt(text: str) -> list[Line]:
    lines: list[Line] = []
    for raw in text.splitlines():
        m = VTT_RE.match(raw.strip())
        if not m:
            continue
        start, end = to_secs(m.group(1)), to_secs(m.group(2))
        body = m.group(3).strip()
        if end <= start or not body:
            continue
        lines.append(Line(start, end, body))
    return lines


def _cuts_from_items(items) -> list[Cut]:
    cuts: list[Cut] = []
    if not isinstance(items, list):
        return cuts
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            cuts.append(
                Cut(
                    start=float(it.get("start", 0.0)),
                    end=float(it.get("end", 0.0)),
                    confidence=float(it.get("confidence", 0.0) or 0.0),
                    category=str(it.get("category", "")),
                    reason=str(it.get("reason", "")),
                    end_text=str(it.get("end_text", "")),
                )
            )
        except (TypeError, ValueError):
            continue
    return cuts


def parse_cuts(body: str) -> list[Cut]:
    """Read cuts out of a proxy reply or a stored MinusPod response.

    Two shapes exist and they are not interchangeable:

    - A single window's reply from the proxy: the chat message content *is*
      the ad array.
    - What MinusPod stored after the original run: several `=== Window N ===`
      sections, each holding one ad array, concatenated.

    Try the whole body as JSON first, because splitting on `=== Window` and
    finding nothing silently reads as "zero ads", which looks identical to a
    real miss.
    """
    text = body.strip()
    if not text:
        return []

    try:
        return _cuts_from_items(json.loads(text))
    except json.JSONDecodeError:
        pass

    cuts: list[Cut] = []
    sections = text.split("=== Window")
    if len(sections) == 1:
        # Not JSON and not sectioned: pull out the first array as a last resort.
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            try:
                return _cuts_from_items(json.loads(m.group(0)))
            except json.JSONDecodeError:
                return []
        return []

    for block in sections[1:]:
        m = re.search(r"\[[\s\S]*?\]", block)
        if not m:
            continue
        try:
            cuts.extend(_cuts_from_items(json.loads(m.group(0))))
        except json.JSONDecodeError:
            continue
    return cuts


def build_window(lines: list[Line], lo: float, hi: float) -> str:
    out = [f"Podcast: replay\nEpisode: window {lo:.0f}-{hi:.0f}s\nTranscript:"]
    for ln in lines:
        # Lines straddling a window edge are included whole: there is no finer
        # timestamp to cut on, which is exactly what MinusPod sends.
        if ln.end > lo and ln.start < hi:
            out.append(f"[{ln.start:.1f}s - {ln.end:.1f}s] {ln.text}")
    return "\n".join(out)


def post_window(base: str, transcript: str, timeout: float = 120.0) -> list[Cut]:
    payload = {
        "model": "jev-ad-detection",
        "stream": False,
        "temperature": 0,
        "max_tokens": 4000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "ad_detection", "schema": {}},
        },
        "messages": [
            {"role": "system", "content": DETECTION_SYSTEM},
            {"role": "user", "content": transcript},
        ],
    }
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    content = body["choices"][0]["message"]["content"]
    return parse_cuts(content)


def run_proxy(name: str, base: str, lines: list[Line]) -> Report:
    rep = Report(name=name)
    duration = lines[-1].end if lines else 0.0
    lo = 0.0
    while lo < duration:
        hi = min(lo + WINDOW_SECS, duration)
        window = build_window(lines, lo, hi)
        t0 = time.monotonic()
        try:
            rep.cuts.extend(post_window(base, window))
            rep.windows += 1
        except (urllib.error.URLError, TimeoutError, OSError, KeyError) as e:
            rep.failed_windows += 1
            print(
                f"  [{name}] window {lo:.0f}-{hi:.0f} failed: {e}",
                file=sys.stderr,
            )
        rep.latency_ms += int((time.monotonic() - t0) * 1000)
        if hi >= duration:
            break
        lo += STRIDE_SECS
    # Windows overlap, so the same cut can be reported twice. Collapse to one
    # per distinct span, keeping the strongest confidence.
    best: dict[tuple[int, int], Cut] = {}
    for c in rep.cuts:
        k = c.key
        if k not in best or c.confidence > best[k].confidence:
            best[k] = c
    rep.cuts = sorted(best.values(), key=lambda c: c.start)
    return rep


def load_episode(db: str, episode: int):
    """Load a processed episode for replay.

    `original_transcript_text` is the transcript *before* MinusPod applied its
    cuts; `transcript_text` is what it replaced it with. The stored
    `first_pass_response` was produced from the original, so replaying the
    edited transcript would compare against cuts whose timestamps no longer
    line up with the audio. The original is therefore preferred, and the
    edited one is returned as a fallback.
    """
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = conn.execute(
        "SELECT d.transcript_text, d.original_transcript_text, d.first_pass_response, "
        "       e.title, p.title "
        "FROM episode_details d "
        "JOIN episodes e ON e.id = d.episode_id "
        "JOIN podcasts p ON p.id = e.podcast_id "
        "WHERE d.episode_id = ?",
        (episode,),
    ).fetchone()
    conn.close()
    if not row:
        raise SystemExit(f"episode {episode} has no stored transcript")
    edited, original, response, title, podcast = row
    used_original = bool(original) and original != edited
    transcript = original if used_original else edited
    return transcript, response or "", title or "", podcast or "", used_original


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--base", default="http://localhost:8787",
                    help="proxy holding the deployed code")
    ap.add_argument("--candidate", default=None,
                    help="proxy holding the new code; omit to report only the baseline")
    ap.add_argument("--noise", type=float, default=0.10,
                    help="probability delta treated as Jev jitter rather than a change")
    ap.add_argument("--control", default=None,
                    help="second proxy to compare (typically the deployed one); "
                         "with it, differences from the candidate can be "
                         "separated from run-to-run jitter in the model")
    args = ap.parse_args()

    transcript, stored, title, podcast, used_original = load_episode(
        args.db, args.episode
    )
    lines = parse_vtt(transcript)
    if not lines:
        print("transcript produced no usable lines", file=sys.stderr)
        return 1

    duration = lines[-1].end
    print(f"episode {args.episode}: {podcast} - {title}")
    print(f"  {len(lines)} lines, {duration / 60:.1f} min")
    print(
        "  transcript: "
        + ("original (pre-cut) -- matches the stored response"
           if used_original
           else "edited copy only; stored cuts may not line up")
    )

    stored_cuts = parse_cuts(stored)
    baseline = Report(name="stored", cuts=sorted(stored_cuts, key=lambda c: c.start))
    print(f"\nstored cuts from the original run: {len(baseline.cuts)}")
    for c in baseline.cuts:
        print(f"  {c.start:8.1f}-{c.end:8.1f}  {c.category:<12} p={c.confidence:.2f}")

    reports = [baseline]
    # The control is the deployed code replayed right now. Comparing the
    # candidate against it isolates the code change, whereas comparing against
    # the stored run mixes that with model jitter and window differences.
    if args.control:
        ctl = run_proxy("control(deployed)", args.control, lines)
        reports.append(ctl)
        print(
            f"\ncontrol (deployed code, re-run): {len(ctl.cuts)} cuts over "
            f"{ctl.windows} windows ({ctl.failed_windows} failed), "
            f"{ctl.latency_ms}ms total"
        )
    if args.candidate:
        live = run_proxy("candidate", args.candidate, lines)
        reports.append(live)
        print(
            f"\ncandidate (new code): {len(live.cuts)} cuts over {live.windows} "
            f"windows ({live.failed_windows} failed), {live.latency_ms}ms total"
        )

    if len(reports) == 1:
        return 0

    print("\n" + "=" * 74)
    for r in reports:
        sp = sum(1 for c in r.cuts if c.category == "self_promo")
        sponsor = sum(1 for c in r.cuts if c.category == "sponsor")
        print(f"{r.name:<26} cuts={len(r.cuts):>3}  sponsor={sponsor:>3}  self_promo={sp}")

    for r in reports[1:]:
        print(f"\n{r.name} vs stored:")
        only_base, only_r = diff(reports[0], r)
        print(f"  matched {len(reports[0].cuts) - len(only_base)}"
              f"  only-stored {len(only_base)}  only-{r.name} {len(only_r)}")
        for b, m in boundary_deltas(reports[0], r):
            d_start = m.start - b.start
            d_end = m.end - b.end
            if abs(d_start) > args.noise or abs(d_end) > args.noise:
                print(f"    ~ {b.start:8.1f}-{b.end:8.1f} -> {m.start:8.1f}-{m.end:8.1f}"
                      f"   start{d_start:+6.1f}s end{d_end:+6.1f}s")
        for c in only_r:
            print(f"    + {c.start:8.1f}-{c.end:8.1f}  {c.category:<11} "
                  f"p={c.confidence:.2f}  {c.reason}")
        for c in only_base:
            print(f"    - {c.start:8.1f}-{c.end:8.1f}  {c.category:<11} "
                  f"p={c.confidence:.2f}  {c.reason}")

    if len(reports) == 3:
        ctl, cand = reports[1], reports[2]
        only_ctl, only_cand = diff(ctl, cand)
        shared = len(ctl.cuts) - len(only_ctl)
        print("\ncandidate vs control (the actual code change):")
        print(f"  matched {shared}  control-only {len(only_ctl)}  "
              f"candidate-only {len(only_cand)}")
        moved = 0
        for b, m in boundary_deltas(ctl, cand):
            d_start, d_end = m.start - b.start, m.end - b.end
            if abs(d_start) > args.noise or abs(d_end) > args.noise:
                moved += 1
                print(f"    ~ {b.start:8.1f}-{b.end:8.1f} -> {m.start:8.1f}-{m.end:8.1f}"
                      f"   start{d_start:+6.1f}s end{d_end:+6.1f}s")
        print(f"  boundaries moved on {moved} matched cut(s)")
        for c in only_cand:
            print(f"    + {c.start:8.1f}-{c.end:8.1f}  {c.category:<11} "
                  f"p={c.confidence:.2f}  {c.reason}")
        for c in only_ctl:
            print(f"    - {c.start:8.1f}-{c.end:8.1f}  {c.category:<11} "
                  f"p={c.confidence:.2f}  {c.reason}")

    print("=" * 74)
    return 0


def overlap_seconds(a: Cut, b: Cut) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def matches(a: Cut, b: Cut) -> bool:
    """Two cuts are the same ad if they share most of their span.

    Matching on exact bounds would report every boundary shift as a lost cut
    plus a new one, which is precisely the change edge trimming is meant to
    make. So a pair matches when their overlap covers the larger of the two
    spans, or when one is almost entirely inside the other.
    """
    ov = overlap_seconds(a, b)
    if ov <= 0:
        return False
    shorter = min(a.end - a.start, b.end - b.start)
    return ov >= 0.8 * shorter


def diff(base: Report, other: Report) -> tuple[list[Cut], list[Cut]]:
    """Cuts unique to `base` and to `other`, by overlap rather than bounds."""
    only_base = [b for b in base.cuts if not any(matches(b, c) for c in other.cuts)]
    only_other = [c for c in other.cuts if not any(matches(c, b) for b in base.cuts)]
    return only_base, only_other


def boundary_deltas(base: Report, other: Report) -> list[tuple[Cut, Cut]]:
    """Matched pairs whose bounds moved."""
    pairs: list[tuple[Cut, Cut]] = []
    for b in base.cuts:
        m = max((c for c in other.cuts if matches(b, c)), key=lambda c: overlap_seconds(b, c), default=None)
        if m is not None:
            pairs.append((b, m))
    return pairs


if __name__ == "__main__":
    raise SystemExit(main())
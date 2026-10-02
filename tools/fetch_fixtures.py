#!/usr/bin/env python3
"""Download public podcast transcripts and convert them to MinusPod's
detection-window line format: `[12.3s - 15.9s] text`.

Source: github.com/LennysNewsletter/lennys-newsletterpodcastdata
        (free public starter pack, 50 transcripts, markdown + timestamps)

The upstream format is one block per speaker turn with only a start time:

    **Speaker** (00:02:33):
    Text of the turn, possibly several sentences.

Whisper-style lines need both a start and an end and are much shorter, so we
split each turn on sentence boundaries and distribute the turn's time span
across its sentences in proportion to word count. That approximates how
faster-whisper chunks a turn, which is what the proxy will actually see.

Usage:
    python3 tools/fetch_fixtures.py --out tests/fixtures/raw --limit 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

API_LIST = "https://api.github.com/repos/LennysNewsletter/lennys-newsletterpodcastdata/contents/podcasts"
RAW_BASE = "https://raw.githubusercontent.com/LennysNewsletter/lennys-newsletterpodcastdata/main/podcasts"

# `**Speaker Name** (00:02:33):` at the start of a line.
TURN_RE = re.compile(r"^\*\*(?P<speaker>[^*]+)\*\*\s*\((?P<ts>\d{2}:\d{2}(?::\d{2})?)\)\s*:?\s*$")
# Any `[hh:mm:ss]` or `(hh:mm:ss)` timestamp we may encounter mid-line.
INLINE_TS_RE = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)[\]\s]")

# Words per second. faster-whisper on `tiny` averages ~2.5-3 w/s for natural
# speech; used only as a fallback when a turn has no following turn to bound it.
DEFAULT_WPS = 2.8

# Guard rails: a single emitted line longer than this gets split again on
# commas so no line balloons into a multi-paragraph "segment".
MAX_LINE_SECS = 12.0
MIN_LINE_SECS = 1.0


@dataclass
class Line:
    start: float
    end: float
    text: str


def parse_ts(raw: str) -> float:
    """`02:33` -> 153.0, `01:02:33` -> 3753.0."""
    parts = [int(p) for p in raw.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"bad timestamp: {raw!r}")


def fmt_ts(secs: float) -> str:
    """Round-trip helper for debugging: 153.0 -> `00:02:33`."""
    total = int(secs)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


@dataclass
class Turn:
    speaker: str
    start: float
    text: str


def parse_markdown(body: str) -> list[Turn]:
    """Split the markdown into speaker turns."""
    turns: list[Turn] = []
    speaker: str | None = None
    start: float | None = None
    buf: list[str] = []

    def flush() -> None:
        if speaker is not None and start is not None:
            text = " ".join(" ".join(buf).split())
            if text:
                turns.append(Turn(speaker=speaker, start=start, text=text))

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        m = TURN_RE.match(line.strip())
        if m:
            flush()
            speaker = m.group("speaker").strip()
            start = parse_ts(m.group("ts"))
            buf = []
            continue
        if speaker is None:
            continue
        # Skip horizontal rules and headings between turns.
        if line.strip().startswith(("#", "---", "===")):
            continue
        buf.append(line)
    flush()
    return turns


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation, keeping the punctuation attached."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def split_long(sentence: str, max_words: int) -> list[str]:
    """Break an over-long sentence on commas/conjunction boundaries."""
    if len(sentence.split()) <= max_words:
        return [sentence]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for piece in re.split(r"(?<=[,;:])\s+", sentence):
        n = len(piece.split())
        if cur and cur_len + n > max_words:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(piece)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def distribute(start: float, span: float, weights: list[int]) -> list[tuple[float, float]]:
    """Split `span` into `len(weights)` non-overlapping, in-order intervals.

    Each interval gets a share of the span proportional to its weight, then
    every interval is floored at MIN_LINE_SECS. Flooring can overshoot the
    total, so any overflow is taken back off the longest intervals rather
    than pushing the cursor past `start + span`.
    """
    n = len(weights)
    total = sum(weights)
    durs = [max(MIN_LINE_SECS, span * (w / total)) for w in weights]

    overflow = sum(durs) - span
    if overflow > 0:
        # Shrink the widest intervals first, never below the floor.
        order = sorted(range(n), key=lambda i: -durs[i])
        i = 0
        while overflow > 1e-9 and i < n * 100:
            idx = order[i % n]
            take = min(overflow, max(0.0, durs[idx] - MIN_LINE_SECS))
            durs[idx] -= take
            overflow -= take
            i += 1
        if overflow > 1e-9:
            # Everything is at the floor; accept the overshoot and extend
            # past the turn end rather than emitting inverted intervals.
            pass

    out: list[tuple[float, float]] = []
    cursor = start
    for d in durs:
        out.append((cursor, cursor + d))
        cursor += d
    return out


def turns_to_lines(turns: list[Turn]) -> list[Line]:
    """Turn each turn into one or more timestamped lines.

    Timestamps are kept monotonic across the whole episode: each turn ends
    where the next one begins, and no line is ever emitted inverted or
    overlapping its predecessor.
    """
    lines: list[Line] = []
    prev_end = 0.0

    for i, turn in enumerate(turns):
        start = max(turn.start, prev_end)
        if i + 1 < len(turns):
            # The next turn starts when this one stops.
            end = max(turns[i + 1].start, start)
        else:
            # Final turn has no successor; estimate from speech rate.
            end = start + max(MIN_LINE_SECS, len(turn.text.split()) / DEFAULT_WPS)
        span = end - start
        if span <= 0.0:
            continue

        sents = split_sentences(turn.text)
        # Cap line length in words, derived from the time budget.
        words_per_line = max(4, int(MAX_LINE_SECS * DEFAULT_WPS))
        expanded: list[str] = []
        for s in sents:
            expanded.extend(split_long(s, words_per_line))
        if not expanded:
            continue

        weights = [max(1, len(s.split())) for s in expanded]
        for (s0, e0), text in zip(distribute(start, span, weights), expanded):
            ls, le = round(s0, 1), round(e0, 1)
            # Rounding must not create an inverted interval.
            if le <= ls:
                le = round(ls + MIN_LINE_SECS, 1)
            lines.append(Line(start=ls, end=le, text=text))
            prev_end = max(prev_end, le)
    return lines


def render(lines: list[Line], header: dict) -> str:
    out = [f"# {header.get('title', 'unknown')}"]
    if header.get("guest"):
        out[0] += f" (guest: {header['guest']})"
    out.append(f"# source: {header.get('source', 'unknown')}")
    out.append("")
    for ln in lines:
        out.append(f"[{ln.start:.1f}s - {ln.end:.1f}s] {ln.text}")
    return "\n".join(out) + "\n"


def parse_front_matter(body: str) -> dict:
    meta: dict = {}
    if not body.startswith("---"):
        return meta
    end = body.find("\n---", 3)
    if end == -1:
        return meta
    for raw in body[3:end].splitlines():
        if ":" not in raw:
            continue
        key, _, val = raw.partition(":")
        meta[key.strip()] = val.strip().strip('"')
    return meta


def fetch(url: str, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "minuspod-jev-proxy-fixtures/0.1"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # noqa: PERF203
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="tests/fixtures/raw", help="output directory")
    ap.add_argument("--limit", type=int, default=12, help="how many transcripts to convert")
    ap.add_argument("--list", action="store_true", help="list available transcripts and exit")
    args = ap.parse_args()

    try:
        listing = json.loads(fetch(API_LIST))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    names = sorted(x["name"] for x in listing if x["name"].endswith(".md"))
    if args.list:
        for n in names:
            print(n)
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for name in names[: args.limit]:
        url = f"{RAW_BASE}/{name}"
        print(f"fetching {name} ...", flush=True)
        try:
            body = fetch(url).decode("utf-8", errors="replace")
        except RuntimeError as exc:
            print(f"  skip: {exc}", file=sys.stderr)
            continue

        meta = parse_front_matter(body)
        turns = parse_markdown(body)
        if len(turns) < 20:
            print(f"  skip: only {len(turns)} turns parsed", file=sys.stderr)
            continue
        lines = turns_to_lines(turns)
        header = {
            "title": meta.get("title", name.removesuffix(".md")),
            "guest": meta.get("guest", ""),
            "source": url,
        }
        stem = name.removesuffix(".md")
        (out_dir / f"{stem}.txt").write_text(render(lines, header), encoding="utf-8")
        manifest.append(
            {
                "file": f"{stem}.txt",
                "title": header["title"],
                "source": url,
                "lines": len(lines),
                "duration_secs": round(lines[-1].end, 1) if lines else 0.0,
                "turns": len(turns),
            }
        )
        print(f"  -> {len(lines)} lines, {fmt_ts(lines[-1].end)} long", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(manifest)} fixtures + manifest.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
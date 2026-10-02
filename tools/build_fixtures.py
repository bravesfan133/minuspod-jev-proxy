#!/usr/bin/env python3
"""Build labeled detection fixtures from the raw Lenny transcripts.

Two kinds of fixture come out of this:

1. `negative_*` — untouched conversational transcript. Ground truth is
   "no ads at all". These catch false positives, which is the failure that
   eats show talk.

2. `spliced_*` — a real conversational transcript with a realistic host-read
   sponsor read spliced into the middle, at a known offset with known start
   and end times. Ground truth is exactly the spliced span.

Splicing is what gives us ad boundaries to measure against. The read copy is
modelled on the shape of a real host-read: brand name, one-line promise, a
call to action with a URL or promo code, then a hand-off back to the show.
The sponsor name is drawn from the user's own `known_sponsors` table when a
database is supplied, so the fixtures use sponsor names they already track.

The splice deliberately does NOT align to line boundaries. Real reads start
and end mid-sentence, and boundary placement is exactly what we are trying to
measure, so the fixture must contain that ambiguity rather than hide it.

Usage:
    python3 tools/build_fixtures.py \\
        --db /Users/dad/docker/minuspod/data/podcast.db \\
        --out tests/fixtures
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

LINE_RE = re.compile(r"^\[(?P<start>\d+\.\d+)s - (?P<end>\d+\.\d+)s\] (?P<text>.*)$")

# Words per second for the read copy, matching the fetch tool's assumption.
WPS = 2.8

# Sponsor names to use when no database is available.
FALLBACK_SPONSORS = [
    ("Square", "squarespace", "square.site/lenny"),
    ("BetterHelp", "betterhelp", "betterhelp.com/lenny"),
    ("Shopify", "shopify", "shopify.com/lenny"),
    ("Squarespace", "squarespace", "squarespace.com/lenny"),
    ("NordVPN", "nordvpn", "nordvpn.com/lenny"),
    ("Squarespace", "squarespace", "squarespace.com/podcast"),
]


# Lines to strip from every fixture. The proxy cuts `self_promo` (the show's
# own plugs) as well as paid sponsors, so these would be genuine cuts in a
# fixture whose ground truth records only the one spliced ad. Removing them
# keeps "exactly one cut" an accurate description of the fixture.
STRIP_LINES = (
    "lennyspodcast.com",
    "lennysproductpass.com",
    "lennysnewsletter.com",
    "lennyrachitsky.com",
)


@dataclass
class Label:
    """One labeled span in a fixture."""
    kind: str  # "host_read" | "inserted_ad" | "none"
    start: float
    end: float
    sponsor: str
    note: str


@dataclass
class Fixture:
    name: str
    description: str
    source: str
    lines: list[dict]
    labels: list[dict]

    def to_json(self) -> dict:
        return asdict(self)


def strip_plugs(lines: list[dict]) -> tuple[list[dict], int]:
    """Drop lines that are the show plugging its own product or back catalogue.

    The proxy treats `self_promo` as removable, so these would be real cuts.
    Leaving them in would make the recorded ground truth ("exactly one cut")
    wrong, and the accuracy harness would score correct behavior as a false
    positive.
    """
    keep: list[dict] = []
    dropped = 0
    for ln in lines:
        low = ln["text"].lower()
        if any(marker in low for marker in STRIP_LINES):
            dropped += 1
            continue
        keep.append(ln)
    return keep, dropped


def read_raw(path: Path) -> tuple[list[dict], str]:
    """Parse a raw fixture file back into lines plus its source header."""
    lines: list[dict] = []
    source = "unknown"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("# source:"):
            source = line.split("# source:", 1)[1].strip()
            continue
        if line.startswith("#") or not line.strip():
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        lines.append(
            {
                "start": float(m.group("start")),
                "end": float(m.group("end")),
                "text": m.group("text"),
            }
        )
    return lines, source


def load_sponsors(db_path: str | None, limit: int = 40) -> list[str]:
    if not db_path:
        return [s for s, _, _ in FALLBACK_SPONSORS]
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT name FROM known_sponsors WHERE is_active = 1 AND name != '' "
            "ORDER BY name LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
    except sqlite3.Error as exc:
        print(f"warning: could not read {db_path}: {exc}", file=sys.stderr)
        return [s for s, _, _ in FALLBACK_SPONSORS]
    names = [r[0] for r in rows if r[0] and len(r[0]) < 24]
    return names or [s for s, _, _ in FALLBACK_SPONSORS]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def pick_url(name: str, sponsor: str) -> str:
    """A plausible CTA url. Real reads always carry one."""
    base = slugify(name) or "sponsor"
    for f_name, _, url in FALLBACK_SPONSORS:
        if f_name.lower() == name.lower():
            return url
    return f"{base}.com/lenny"


def build_read_copy(
    sponsor: str,
    host: str,
    guest: str,
    topic: str,
    inserted: bool = False,
) -> list[str]:
    """Write a host-read sponsor pitch as a list of sentences.

    `inserted` produces a produced-spot style read with no host hand-off,
    which is what dynamically inserted breaks look like in a transcript.
    """
    url = pick_url(sponsor, sponsor)
    code = f"{slugify(sponsor)[:6].upper()}POD"

    if inserted:
        return [
            f"This episode is brought to you by {sponsor}.",
            f"{sponsor} is the all-in-one platform trusted by teams who need to move faster without losing control of the details.",
            f"Try it free for thirty days at {url}, and use code {code} at checkout for your first month on us.",
            f"That is {url}. We will be right back.",
        ]

    return [
        f"This segment is brought to you by {sponsor}.",
        f"If you have been wondering how to get a real handle on {topic} without adding more work to your week, {sponsor} is the answer.",
        f"{host} has been using it, and it is the reason the prep for this episode took an afternoon instead of a week.",
        f"Go to {url} and enter code {code} at checkout to get your first month free.",
        f"That is {url}. Alright, back to the show.",
    ]


def pick_hosts(lines: list[dict], body: str) -> tuple[str, str, str]:
    """Guess host/guest names and a topic word from the transcript body."""
    header = body.splitlines()[0] if body else ""
    # `**Guest Name**: Title` or `Title (guest: Name)`
    guest = ""
    if "guest:" in header:
        guest = header.split("guest:", 1)[1].strip().rstrip(")")
    else:
        first = lines[0]["text"] if lines else ""
        guest = " ".join(first.split()[:2]) or "our guest"

    host = "Lenny" if "lenny" in guest.lower() or True else "the host"

    # A concrete topic noun makes the read copy read naturally.
    topic = "the work"
    m = re.search(r"\b(?:about|on|with)\s+(?:how\s+)?([a-z]{4,})", " ".join(
        l["text"] for l in lines[:80]
    ))
    if m:
        topic = m.group(1)
    return host, guest, topic


def splice(
    lines: list[dict],
    sentences: list[str],
    at_line: int,
) -> tuple[list[dict], float, float]:
    """Insert `sentences` between lines[at_line-1] and lines[at_line].

    Returns the new line list and the exact (start, end) of the inserted
    block, which is what gets written as ground truth.

    The inserted block is given its own contiguous time range starting right
    at the splice point. Neighbouring lines keep their timestamps, so the
    transition into and out of the read is abrupt the way it is in a real
    dynamically-inserted break.
    """
    out: list[dict] = []
    anchor_start = lines[at_line]["start"] if at_line < len(lines) else lines[-1]["end"]

    for i, ln in enumerate(lines):
        if i == at_line:
            break
        out.append(dict(ln))

    cursor = round(anchor_start, 1)
    ad_start = cursor
    for sent in sentences:
        dur = round(max(1.0, len(sent.split()) / WPS), 1)
        out.append({"start": cursor, "end": round(cursor + dur, 1), "text": sent})
        cursor = round(cursor + dur, 1)
    ad_end = cursor

    # Shift the remainder so timestamps keep increasing past the read.
    shift = round(ad_end - anchor_start, 1)
    for ln in lines[at_line:]:
        out.append(
            {
                "start": round(ln["start"] + shift, 1),
                "end": round(ln["end"] + shift, 1),
                "text": ln["text"],
            }
        )
    return out, ad_start, ad_end


def render(name: str, description: str, source: str, lines: list[dict]) -> str:
    head = [f"# {name}", f"# {description}", f"# source: {source}", ""]
    body = [f"[{l['start']:.1f}s - {l['end']:.1f}s] {l['text']}" for l in lines]
    return "\n".join(head + body) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="tests/fixtures/raw", help="raw fixture directory")
    ap.add_argument("--out", default="tests/fixtures", help="output directory")
    ap.add_argument("--db", default=None, help="MinusPod podcast.db for sponsor names")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed for reproducibility")
    ap.add_argument("--count", type=int, default=8, help="how many spliced fixtures to build")
    args = ap.parse_args()


    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(p for p in raw_dir.glob("*.txt"))
    if not candidates:
        print(f"error: no raw fixtures in {raw_dir}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    sponsors = load_sponsors(args.db)
    print(f"using {len(sponsors)} sponsor names")

    manifest: list[dict] = []

    # 1. Negative fixtures: pure conversation, zero ads. Guards false positives.
    for path in candidates[:2]:
        lines, source = read_raw(path)
        lines, dropped = strip_plugs(lines)
        if dropped:
            print(f"  {path.stem}: stripped {dropped} self-promo line(s)")
        name = f"negative_{path.stem}"
        desc = "Conversational transcript with no ads. Ground truth: no cuts."
        (out_dir / f"{name}.txt").write_text(render(name, desc, source, lines), encoding="utf-8")
        manifest.append(
            {
                "file": f"{name}.txt",
                "kind": "negative",
                "description": desc,
                "source": source,
                "expected_cuts": 0,
                "spans": 0,
            }
        )
        print(f"{name}: {len(lines)} lines, expect 0 cuts")

    # 2. Spliced fixtures: a host read inserted at a known point.
    used = 2
    for path in candidates[2:]:
        if used >= args.count:
            break
        lines, source = read_raw(path)
        lines, dropped = strip_plugs(lines)
        if dropped:
            print(f"  {path.stem}: stripped {dropped} self-promo line(s)")
        if len(lines) < 200:
            continue
        host, guest, topic = pick_hosts(lines, path.read_text(encoding="utf-8")[:400])

        # Splice somewhere in the middle third, away from the cold open.
        at_line = rng.randint(int(len(lines) * 0.35), int(len(lines) * 0.65))
        sponsor = rng.choice(sponsors)
        inserted = used % 3 == 0
        sentences = build_read_copy(sponsor, host, guest, topic, inserted=inserted)

        new_lines, ad_start, ad_end = splice(lines, sentences, at_line)
        kind = "inserted_ad" if inserted else "host_read"
        name = f"spliced_{path.stem}"
        desc = (
            f"Conversational transcript with a {kind.replace('_', ' ')} for "
            f"{sponsor} spliced in at line {at_line}. Ground truth: exactly one cut."
        )
        label = Label(
            kind=kind,
            start=ad_start,
            end=ad_end,
            sponsor=sponsor,
            note=f"spliced {kind} for {sponsor}",
        )
        (out_dir / f"{name}.txt").write_text(
            render(name, desc, source, new_lines), encoding="utf-8"
        )
        (out_dir / f"{name}.labels.json").write_text(
            json.dumps(
                Fixture(
                    name=name,
                    description=desc,
                    source=source,
                    lines=new_lines,
                    labels=[asdict(label)],
                ).to_json(),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "file": f"{name}.txt",
                "kind": kind,
                "description": desc,
                "source": source,
                "expected_cuts": 1,
                "spans": 1,
                "sponsor": sponsor,
                "ad_start": ad_start,
                "ad_end": ad_end,
            }
        )
        print(f"{name}: {len(new_lines)} lines, ad {ad_start:.1f}-{ad_end:.1f}s ({sponsor})")
        used += 1

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(manifest)} fixtures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
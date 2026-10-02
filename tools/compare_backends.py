#!/usr/bin/env python3
"""A/B the same detection payloads against Jev and Cloudflare Clef.

Sends one Decisions request per fixture span set to both providers and prints
the answers side by side. This is the measurement that answers whether Clef is
actually better for this task, rather than merely compatible.

Reads credentials from the environment (JEV_PRIMARY_* / JEV_SECONDARY_* for
Jev, JEV_TERTIARY_* for Clef). Credentials are never printed.

Usage:
    set -a; . ./.env; set +a
    python3 tools/compare_backends.py
    python3 tools/compare_backends.py --limit 3 --model clef-flash
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

FIXTURE_DIR = Path("tests/fixtures")
LINE_RE = re.compile(r"^\[(\d+\.\d+)s - (\d+\.\d+)s\] (.*)$")

# Mirrors src/transcript.rs::to_segments with the shipped defaults.
SEG_TARGET, SEG_MAX, SEG_GAP = 4.0, 8.0, 1.25

# Mirrors src/handlers.rs::detection_criteria.
DETECTION_CRITERIA = {
    "content": "Editorial speech: news, interview, opinion, story, or a brand mentioned with no ask to buy, visit, or use a code.",
    "paid_ad": "A paid sponsor read or produced commercial that asks the listener to buy, visit, or use a code.",
    "host_read_sponsor": "The host reads a sponsor pitch with a call to action, URL, or promo code.",
    "inserted_ad": "A commercial break that is not the host's editorial topic.",
    "self_promo": "The show promotes its own Patreon, merch, mailing list, or live event.",
    "cross_promo": "A pitch for a different show or network.",
}

AD_LABELS = {"paid_ad", "host_read_sponsor", "inserted_ad"}

MAX_RETRIES = 5
RETRY_BASE = 2.0


@dataclass
class Backend:
    name: str
    url: str
    model: str
    key: str | None

    def call(self, state, questions):
        body = json.dumps({"model": self.model, "state": state, "questions": questions})
        req = urllib.request.Request(
            self.url,
            data=body.encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.key}"} if self.key else {}),
            },
        )
        t0 = time.monotonic()
        # Clef is documented at 2000 req/min but a sustained comparison run
        # trips it. Retry with backoff, honoring Retry-After when present, so a
        # long sweep degrades to slow rather than aborting.
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    payload = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code != 429:
                    raise
                delay = RETRY_BASE * (2**attempt)
                ra = e.headers.get("Retry-After") if e.headers else None
                if ra and ra.isdigit():
                    delay = min(int(ra), 60)
                print(
                    f"  [{self.name}] 429, retrying in {delay}s "
                    f"({attempt + 1}/{MAX_RETRIES})",
                    file=sys.stderr,
                )
                time.sleep(delay)
        else:
            raise RuntimeError(f"{self.name} rate limited after {MAX_RETRIES}: {last_err}")

        # Cloudflare Workers AI wraps the model output in an envelope:
        #   {"result": {"model":..., "answers":{...}, "usage":{...}},
        #    "success": true, "errors": [], "messages": []}
        # Jev returns the bare object. Unwrap so both look the same.
        if payload.get("success") is False:
            errs = payload.get("errors") or []
            raise RuntimeError(f"backend error: {errs}")
        body = payload.get("result", payload)
        return body.get("answers", {}), (time.monotonic() - t0) * 1000


def load_backend(prefix, name):
    url = os.environ.get(f"{prefix}_BASE_URL", "")
    model = os.environ.get(f"{prefix}_MODEL", "")
    key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not url:
        return None
    return Backend(name=name, url=url, model=model, key=key or None)


def parse_fixture(path: Path):
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(raw.strip())
        if m:
            lines.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return lines


def to_segments(lines):
    segs, cur = [], []
    for s, e, t in lines:
        if cur:
            gap = s - cur[-1][1]
            span_if = e - cur[0][0]
            if gap >= SEG_GAP or span_if > SEG_MAX:
                segs.append(cur)
                cur = []
        cur.append((s, e, t))
        if cur[-1][1] - cur[0][0] >= SEG_TARGET:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return [(g[0][0], g[-1][1], " ".join(x[2] for x in g)) for g in segs]


def build_questions(chunk, offset, all_segs):
    """Same shape as handle_detection: a noul and a choice per segment."""
    qs = {}
    for j, (start, _end, text) in enumerate(chunk):
        before = all_segs[offset + j - 1][2] if offset + j - 1 >= 0 else ""
        after = all_segs[offset + j + 1][2] if offset + j + 1 < len(all_segs) else ""
        label = f"segments[{j}]"
        qs[f"ad{j}"] = {
            "type": "noul",
            "instructions": (
                f"Is `{label}.text` a paid advertisement that should be cut? "
                f"`{label}.before` and `{label}.after` are only the adjacent speech. "
                f"Judge `{label}.text` itself."
            ),
            "criteria": {
                "true": "A paid sponsor read, host-read ad, or inserted commercial with a call to action, URL, or promo code",
                "false": "Show talk that should stay: news, interview, opinion, a sign-off, the show promoting itself, or a brand mentioned with no call to action",
            },
        }
        qs[f"seg{j}"] = {
            "type": "choice",
            "instructions": (
                f"Which label best describes `{label}.text`? "
                f"`{label}.before` and `{label}.after` are only the adjacent speech."
            ),
            "criteria": DETECTION_CRITERIA,
        }
    state = {
        "segments": [
            {
                "text": chunk[j][2],
                "before": all_segs[offset + j - 1][2] if offset + j - 1 >= 0 else "",
                "after": all_segs[offset + j + 1][2] if offset + j + 1 < len(all_segs) else "",
            }
            for j in range(len(chunk))
        ]
    }
    return state, qs


def label_of(answer):
    return (answer or {}).get("choice") or "?"


def noul_of(answer):
    v = (answer or {}).get("noul")
    return v if isinstance(v, (int, float)) else 0.0


def load_labels(path: Path):
    lp = path.with_suffix("").with_suffix("") if False else path.parent / (
        path.stem + ".labels.json"
    )
    if not lp.exists():
        return []
    return json.loads(lp.read_text()).get("labels", [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=6, help="fixtures to compare")
    ap.add_argument("--model", default=None, help="override the Clef model, e.g. clef-flash")
    ap.add_argument("--chunk", type=int, default=16, help="segments per call")
    args = ap.parse_args()

    jev = load_backend("JEV_PRIMARY", "jev") or load_backend("JEV_SECONDARY", "jev")
    clef = load_backend("JEV_TERTIARY", "clef")
    if args.model and clef:
        clef.model = args.model

    if not jev or not clef:
        print(
            "need both a Jev backend (JEV_PRIMARY_* or JEV_SECONDARY_*) and a\n"
            "Clef backend (JEV_TERTIARY_*). Load .env first:\n"
            "  set -a; . ./.env; set +a",
            file=sys.stderr,
        )
        return 1

    fixtures = sorted(p for p in FIXTURE_DIR.glob("*.txt"))
    fixtures = [f for f in fixtures if not f.stem.startswith("negative_")][
        : args.limit
    ]
    if not fixtures:
        print("no fixtures; run tools/build_fixtures.py first", file=sys.stderr)
        return 1

    agree = disagree = 0
    jev_only = clef_only = 0
    jev_ms_total = clef_ms_total = 0.0
    jev_calls = clef_calls = 0
    jev_hits = clef_hits = 0

    for fx in fixtures:
        labels = load_labels(fx)
        if not labels:
            continue
        label = labels[0]
        lines = parse_fixture(fx)
        segs = to_segments(lines)

        print(f"\n=== {fx.stem}  ad {label['start']:.1f}-{label['end']:.1f}s "
              f"({label.get('sponsor', '?')})")

        for offset in range(0, len(segs), args.chunk):
            chunk = segs[offset : offset + args.chunk]
            state, qs = build_questions(chunk, offset, segs)
            try:
                ja, jms = jev.call(state, qs)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"  jev failed: {e}", file=sys.stderr)
                break
            try:
                ca, cms = clef.call(state, qs)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"  clef failed: {e}", file=sys.stderr)
                break
            jev_ms_total += jms
            clef_ms_total += cms
            jev_calls += 1
            clef_calls += 1

            for j, (s, e, _t) in enumerate(chunk):
                jl, cl = label_of(ja.get(f"seg{j}")), label_of(ca.get(f"seg{j}"))
                jn, cn = noul_of(ja.get(f"ad{j}")), noul_of(ca.get(f"ad{j}"))
                inside = s < label["end"] and e > label["start"]
                if inside:
                    jev_hits += jl in AD_LABELS
                    clef_hits += cl in AD_LABELS
                same = jl == cl and abs(jn - cn) < 0.05
                agree += same
                disagree += not same
                jev_only += (not same) and jl in AD_LABELS and cl not in AD_LABELS
                clef_only += (not same) and cl in AD_LABELS and jl not in AD_LABELS
                flag = "AD " if inside else "   "
                print(
                    f"  {flag}[{s:7.1f}-{e:7.1f}] jev={jl:<18}{jn:.3f}  "
                    f"clef={cl:<18}{cn:.3f}{'' if same else '   <-- differs'}"
                )

    n = agree + disagree
    print("\n" + "=" * 78)
    print(f"spans compared      {n}")
    print(f"identical verdicts  {agree} ({100 * agree / n if n else 0:.1f}%)")
    print(f"disagreements       {disagree}")
    print(f"  jev-only ad       {jev_only}")
    print(f"  clef-only ad      {clef_only}")
    if jev_calls and clef_calls:
        print(f"\nlatency  jev {jev_ms_total / jev_calls:7.1f} ms/call "
              f"({jev_calls} calls)")
        print(f"         clef {clef_ms_total / clef_calls:7.1f} ms/call "
              f"({clef_calls} calls)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
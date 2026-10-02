# minuspod-jev-proxy

Thin OpenAI-compatible proxy that lets [MinusPod](https://github.com/) use
**Jev** (a decision model, not a chat model) for podcast ad detection.
Jev-only: there is deliberately **no Gemini/general-LLM fallback** inside
this proxy. MinusPod already supports Gemini natively, so if Jev doesn't
work out, point MinusPod straight back at Gemini and stop this container.

## How it works

```
MinusPod → POST /v1/chat/completions → proxy → Jev Decisions API → OpenAI-style JSON back
```

The proxy classifies each incoming chat request (by `response_format`
schema name, then stable prompt markers, then transcript structure) and:

| Request type | Handling |
|---|---|
| Primary ad detection (`ad_detection` schema, pass 1) | Group transcript lines into short spans (~4s, capped at 8s, split on a ~1.25s pause). Each Jev call gets only that batch as `{segments:[{text,before,after}]}`. One `noul` asks whether `segments[i].text` is a paid ad; one `choice` labels it. A cut is emitted only when the label is `paid_ad`, `host_read_sponsor`, or `inserted_ad` and noul is at least `JEV_RECALL_THRESHOLD`. Sign-offs, the show promoting itself, and ordinary talk are not cuts. Neighboring paid-sponsor spans within `JEV_ATTACH_GAP_SECS` join so one read is one cut. Confidence is the strongest noul in that read |
| Edge trim (after detection) | Re-judges the first and last ~6s of each cut as ~2s pieces, with both signals required to agree (`noul >= JEV_EDGE_THRESHOLD` **and** a paid-sponsor label). Bounds move inward only. This is what stops a cut from starting mid-sentence or taking show talk with it |
| Verification re-scan (pass 2, same schema) | Same Jev path, with transition-tone/orphan-URL guidance |
| Reviewer (`ad_review`) | Jev yes/no on the candidate span; confirms original bounds or rejects. Bounds are never adjusted (Jev can't emit timestamps) |
| Category repair (`segment_categories`) | Jev `choice` per listed segment |
| Trim recovery | `{"ad_start": null, "ad_end": null}` (keep original span — matches MinusPod's native null path) |
| Probes / test-connection | `{"ok": true}` |
| Chapters | HTTP 500 → MinusPod falls back to generic chapters on its own |
| Unknown | HTTP 400, fail loudly rather than guess |

Jev chain: **primary** TypeSafe direct (`jev-latest` rolling alias, needs
`JEV_PRIMARY_API_KEY`/`TYPESAFE_API_KEY`) → **secondary** OpenCode Zen
(`jev-1.13-free`, free, keyless) → **tertiary** optional, unset by default.
Paid-first because free tiers burst-limit under MinusPod's parallel windows;
Zen absorbs overflow.

The third slot exists so a new provider can be added **without displacing the
keyless free tier**. It is opt-in: with `JEV_TERTIARY_BASE_URL` empty the slot
is skipped entirely and failover behaves exactly as it did before. It is last
in the order, so it only receives traffic when both earlier backends are
exhausted.

`/health` reports all three slots, with empty `base_url` for unconfigured ones.
Credentials are never included.

### Workers AI (Clef) as a backend

Clef speaks the same Decisions wire format, so it works with no code change:

```bash
JEV_TERTIARY_BASE_URL=https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/run/@cf/cloudflare/clef
JEV_TERTIARY_MODEL=clef
JEV_TERTIARY_API_KEY=<Cloudflare API token with Workers AI access>
```

Two differences from the Jev endpoints, both handled in the client:

- **Response envelope.** Workers AI nests the output: `{"result": {"model",
  "answers", "usage"}, "success": true, "errors": [], "messages": []}`. Jev
  returns the bare object. `tools/compare_backends.py` unwraps this; note it
  if you add another client.
- **Model name.** The `model` field must match `^(clef|clef-flash)$`. A Jev
  model name returns HTTP 400, which surfaces as a 502, not as "no ads".

Watch the rate limit: Clef is documented at 2000 req/min and a sustained
sweep will hit 429s.

Rate limiting is handled, not hidden: concurrent Jev calls are capped
(`JEV_MAX_CONCURRENT_REQS`, default 4), each backend gets bounded retries
with backoff honoring `Retry-After`, and if both backends are limited the
proxy answers HTTP 429 so MinusPod defers + retries the episode instead of
dropping windows. Other total failures return 502. All other MinusPod
stages degrade gracefully on their own (reviewer keeps candidates,
verification keeps pass-1 cuts, repair defaults to `sponsor`, chapters go
generic, trim keeps the span).

### Prompts are written for a decision model, not a chat model

Jev is not an LLM. It scores the `state` against every option's text in
parallel and returns a probability per option, with no intermediate text. Two
consequences shape the prompts here:

- **`criteria` is the specification, not a hint.** Each label description says
  what belongs in that label, what does not, and how it differs from its
  neighbours. A shape missing from the description has nothing to match
  against. This was a real miss: `self_promo` originally read "Patreon, merch,
  mailing list, or live event", so "subscribe, join the community, get push
  notifications" scored as `content` at 0.97 — an app subscribe was simply not
  in the list.
- **Instructions are short and positive.** They name the subject and ask one
  question. Long negation scaffolding ("judge this segment itself", "is only
  the adjacent speech") helps a chat model and only dilutes schema scoring.
  The target text is not duplicated between `state` and `instructions`.

`tools/replay_episode.py` is how these were verified, against real processed
episodes rather than argument.

### Two thresholds, on purpose

Detection has a **recall** stage and a **precision** stage. Collapsing them
into one knob is what caused whole 10-minute windows to come back with no ads
at all: a strict gate rejects a span outright, and a rejected span never
reaches the stage that would have tightened it.

**Stage 1 — recall** (`JEV_RECALL_THRESHOLD`, default 0.35). A span becomes a
*candidate* when the choice is a paid sponsor (`paid_ad`, `host_read_sponsor`,
`inserted_ad`) **and** noul ≥ this. Both signals must agree. Show talk, a
sign-off, and the show promoting itself stay even when noul is high, because
the label gate is what excludes them. A weaker paid-sponsor neighbor (noul ≥
`JEV_ATTACH_THRESHOLD`, default 0.40, gap ≤ `JEV_ATTACH_GAP_SECS`, default 8s)
joins a candidate so one read is one cut rather than a scrap MinusPod drops as
too short. Confidence reported to MinusPod is the **strongest** noul in the
read, which is what its 80% slider compares.

**Stage 2 — precision** (`JEV_EDGE_THRESHOLD`, default 0.50). Detection spans
are ~4s, so a cut tends to begin and end mid-sentence: the read's opening
clause survives, and a little show talk gets removed with it. Edge trim
re-asks about just the first and last ~6s of each cut as ~2s pieces
(`JEV_EDGE_PIECE_SECS`) and keeps a piece only when **both** signals agree.
Bounds move **inward only** — a piece the classifier calls content is evidence
the cut was too wide, so this stage can tighten a cut but never invent or
enlarge one. A cut whose ends are uniformly rejected is left untouched rather
than collapsed, since that is not evidence the whole read is content.

`JEV_AD_THRESHOLD` is still read as a deprecated alias for
`JEV_RECALL_THRESHOLD` and logs a warning. Any unrecognized `JEV_*` or
`TYPESAFE_*` variable is warned about at startup, so config left over from an
older build cannot sit there looking active while doing nothing.

## Run

```bash
cp .env.example .env   # no keys needed for Zen free; TypeSafe key for primary
docker compose up -d --build
```

On Dockhand: create a stack from this repo (`docker-compose.yml`) and set
the variables in the stack environment UI — at minimum `JEV_PRIMARY_API_KEY`.
No `.env` file needed there.

Point MinusPod at it (Settings → provider `openai-compatible`,
Base URL `http://<host>:8787/v1`, model `jev-ad-detection`), test the
connection, process one episode. Rollback: set Base URL back to
`https://generativelanguage.googleapis.com/v1beta/openai/` and stop this
container. Nothing inside MinusPod is modified.

## Endpoints

- `GET /v1/models` — lists `jev-ad-detection`
- `POST /v1/chat/completions` — non-streaming only
- `GET /health` — status + configured backends (no keys)

## Configuration

All via environment (see `.env.example`). Never hard-code keys; keys are
never logged. Per-request logs are one line: kind, backend, transcript
span, segments, Jev latency, ads found, total time. No transcripts in logs.

## Testing

```bash
cargo test                                    # 62 tests, no network, no API key
cargo test --test accuracy -- --nocapture     # per-fixture accuracy table
```

Four layers:

- **Unit tests** (`src/*.rs`) — request classification, transcript parsing and
  segmentation, merge behavior, Jev request/response shapes, and the edge-trim
  geometry (shrink-only, no inversion, no widening, malformed lengths).
- **Accuracy harness** (`tests/accuracy.rs`) — runs the real segmentation,
  recall gate, merge, and trim code over labeled fixtures with the classifier
  **stubbed**, then reports precision, recall, and boundary error in seconds.
  Runs offline and costs nothing.
- **Failover** (`tests/failover.rs`) — all three backend slots. Pins that the
  third slot is skipped when unconfigured (so adding it can't break a working
  two-backend setup), that it *is* reached when the earlier slots fail, that
  secondary order is unchanged, and that all-slots-down returns 502 rather
  than a false "zero ads".
- **HTTP end-to-end** (`tests/http_e2e.rs`) — drives the real handler stack
  against a stub Decisions backend over HTTP. Proves the geometry is actually
  wired up: a sponsor read produces exactly one cut with the required MinusPod
  fields, ad-free conversation produces none, probes answer without spending a
  model call, unknown requests fail loudly, and a dead backend returns 502
  rather than a 200 with zero ads (which would look like "no ads" and silently
  keep everything).
- **Fixtures** (`tests/fixtures/`) — real conversational transcripts from the
  public [Lenny's Podcast starter pack](https://github.com/LennysNewsletter/lennys-newsletterpodcastdata),
  converted to MinusPod's `[12.3s - 15.9s] text` line format. `negative_*`
  fixtures are ad-free and guard against cutting show talk. `spliced_*`
  fixtures have one realistic host read inserted at a known offset, with the
  exact bounds recorded in `*.labels.json`, so recall and boundary placement
  are both measurable. Sponsor names are drawn from your own `known_sponsors`
  table.

Regenerate them with:

```bash
python3 tools/fetch_fixtures.py --limit 14
python3 tools/build_fixtures.py --db /path/to/podcast.db --count 8
```

### What the harness does and does not prove

It measures **this codebase's logic**: segmentation, the recall gate, neighbor
merging, edge trimming, boundary placement. The classifier is stubbed, so a
score here says nothing about how good Jev is — it says whether the code around
the model cuts where it should and no wider. That separation is the point:
when an ad survives or a cut lands badly, the model and this code are very
different suspects, and these fixtures let the second be checked for free.

### Comparing backends live

`tools/compare_backends.py` sends identical payloads to Jev and Clef over the
fixtures and prints both verdicts per span, so you can see where they disagree
without changing which one serves traffic:

```bash
set -a; . ./.env; set +a
python3 tools/compare_backends.py --limit 6
python3 tools/compare_backends.py --model clef-flash   # the latency-oriented variant
```

It reports how often the two agree, how many disagreements favour either side,
and average latency per call. Credentials are read from the environment and
never printed. It retries on 429, because a full sweep exceeds the documented
rate limit.

For end-to-end accuracy you still need real episodes. Process them and compare
against MinusPod-on-Gemini: same ads found, missed, false flags, boundary
deltas, detection time. **Do not claim improvement until measured.**

Starting values: `JEV_SEGMENT_TARGET_SECS=4`, `JEV_SEGMENT_MAX_SECS=8`,
`JEV_SEGMENT_GAP_SECS=1.25`, `JEV_RECALL_THRESHOLD=0.35`,
`JEV_EDGE_THRESHOLD=0.50`, `JEV_EDGE_PIECE_SECS=2`, `JEV_EDGE_CONTEXT_SECS=45`,
`JEV_ATTACH_THRESHOLD=0.40`, `JEV_ATTACH_GAP_SECS=8`.

## Notes / limits

- Jev classifies a short span of transcript text. Cut times are the
  transcript line timestamps around the spans it marks, not values Jev
  emits. MinusPod's downstream boundary snap still refines them. A single
  Whisper line that mixes show talk and a read cannot be split finer than
  that line.
- Reviewer is off by default in MinusPod (`enable_ad_review=false`); if you
  enable it, the proxy confirms/rejects but never adjusts bounds.
- `jev-1.13-free` on Zen is a limited-time offer; if it disappears, set the
  primary to TypeSafe direct or paid `jev-1.13`.

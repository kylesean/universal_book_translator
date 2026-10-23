# Configuring the semantic judge

> Checked in at the repository root (not under `docs/`, which is
> git-ignored) because this is user-facing configuration guidance, not
> operator notes.

The judge is the only gate that can see meaning: it catches a translation that
reads perfectly while dropping a word ("publicly available" → "获取",
"content interference" → "干扰"). Every zero-token gate passes those.

It is **off by default** and this page shows how to turn it on. Nothing here
assumes a particular machine: a cloud API, a company gateway and a local server
all configure the same way.

## The minimum

The judge needs a model and it needs to be *allowed* to run:

```toml
# ubt.toml, or ~/.config/ubt/config.toml
[defaults]
qe_judge_enabled = true          # the judge is enabled by this, not by qe_engine
qe_judge_pass_sample = 0.15      # also audit ~15% of clean passes
qe_judge_provider = "my-judge"   # the block declared below
```

```toml
[providers.my-judge]
base_url = "https://api.example.com/v1"
api_mode = "openai-chat"         # or openai-responses, anthropic-messages, gemini-native
api_key = "${JUDGE_API_KEY}"     # expands from the environment; never commit a key
repair_model = "some-capable-model"
```

A whole quality policy fits in the same `[defaults]` block, so it can live in a
file instead of a shell script:

```toml
[defaults]
qe_threshold = 0.80              # the bar a block must clear to ship unreviewed
max_repair_rounds = 2
consistency_enforce = "report"   # off | report | repair (terminology drift)
consistency_max_repairs = 25
```

Three things about this are easy to get wrong, and each one was measured:

### 1. The judge needs a capable model

The judge's job is to compare a sentence against its translation and notice what
is missing. A small model cannot do it, and a judge that always says "fine" is
worse than no judge — it turns a sample audit into a rubber stamp.

Measured on the same pair (a correct translation, and the same translation with
one word dropped):

| model | correct | drops one word | usable as a judge |
| --- | --- | --- | --- |
| a 4B translation model | 100 | **100** | no |
| a 27B general model (local) | 98 | 48 | yes, but 35–85 s/block |
| a hosted frontier model | 95 | 75 | yes, 1.6–3.5 s/block |

Use a general-purpose model, not a translation-specialised one: translation
models are trained to produce fluent output, which is the opposite of noticing
what is absent.

### 2. Sampling must be on, or the judge never runs

`TieredQERunner` consults the judge for the **ambiguous class** — a heuristic
rejection that no specific classifier could name (the `STRUCTURAL_OTHER` band,
0.70) — and for a **sampled share of clean passes**, and that share defaults to
`0.0`. A compressed translation scores 0.92, so with the default it never reaches
the judge at all, even with `qe_judge_enabled = true`.
`qe_judge_pass_sample = 0.15` judges roughly one clean block in seven.

There is deliberately no "score band" setting. An earlier version had
`qe_judge_gray_low`/`_high` defining a range to judge; with the heuristic engine
it was inert (the twelve emitted values are discrete, and the only one inside the
default band was already in the ambiguous class), so it was removed rather than
left looking tunable.

### 2b. What a judge verdict can and cannot do

A judge verdict never triggers an automatic rewrite. The routing rule is one
function (`defect_taxonomy.classify_defect`) with three outcomes:

| outcome | meaning | what happens |
| --- | --- | --- |
| `PASS` | nothing deterministic failed, score clears the line | ships |
| `DETERMINISTIC_DEFECT` | a named invariant failed (numeric loss, dropped term, markup corruption, empty target) | repaired |
| `SEMANTIC_DOUBT` | only the judge objected | human queue |

The asymmetry is intentional. Repairing a dropped number is **verifiable**: the
same check that rejected the draft judges the fix, so the loop can tell whether
it helped. Repairing "the judge found this awkward" is not — the judge carries
its own noise, so a rewrite cannot be shown to be better, only different. Blocks
in that class are left for the human queue instead of spending tokens on an
unmeasurable change.

One consequence is worth stating plainly, because it is easy to read the table
the wrong way: a sampled pass the judge lowers to 0.60 carries no deterministic
flag, so repair skips it (`select_repair_candidates` excludes `SEMANTIC_DOUBT`)
and triage sees a Minor block whose score is below the bar. It lands in
`NEEDS_HUMAN`: the draft still reaches the deliverable, but labelled for human
post-editing (see `ubt/adapters/unresolved.py`) rather than presented as
finished. It is not rewritten and not released as approved.


### 3. Comparison is per sentence

A whole-block score dilutes a one-word loss: the same defect measured 0.85 across
a ten-sentence block and 0.78 when the sentence carrying it is judged alone —
the difference between passing and being repaired. `qe_judge_sentences` (default
`true`) scores each sentence and takes the block's **worst** sentence, because a
gate asks whether anything is wrong, not what the average is.

If the two sides split into different sentence counts the block falls back to a
single whole-block call: a mispaired comparison would invent a defect.

## Alternative: point at the endpoint directly

For a judge endpoint that is not declared as a provider:

```bash
export UBT_QE_JUDGE_BASE_URL=https://api.example.com/v1
export UBT_QE_JUDGE_API_KEY=...
export UBT_QE_JUDGE_API_MODE=openai-chat
```

`qe_judge_provider` is preferred because it reuses the registry's credential and
header handling instead of a second mechanism invented for the judge.

## Local server

A local endpoint works exactly like a hosted one:

```toml
[providers.local-judge]
base_url = "http://127.0.0.1:8080/v1"
api_mode = "openai-chat"
api_key = "local-only"
repair_model = "your-model-name"
```

The key is not checked by a local server, and it may be left out entirely: an
empty or `mock-key` credential sends **no** `Authorization` header, so a local
server that ignores auth works as-is. Any placeholder works too, for a gateway
that insists on the header.

A local judge large enough to be useful is slow — budget 35–85 seconds per judged
block, and stop other GPU work first.

## What it costs

One model call per judged sentence of one sampled block. With
`qe_judge_pass_sample = 0.15` on a 150-block document, that is about 22 blocks,
each of a few sentences: a few cents on a hosted API, and a few minutes on a
local one.

## What it does not do

* It cannot promote a block. Its verdict can lower a score, and a lowered score
  is what routes a block to the human queue — but the routing rule
  (`classify_defect`) treats a judge-only objection as `SEMANTIC_DOUBT`, so it
  never triggers an automatic rewrite and never releases anything.
  There is no best-of-n reranking in the pipeline to feed: that track was
  deleted when every runner turned out to be uncalibrated, and
  `is_calibrated()` stays `False` for the judge leg so nothing reintroduces it
  by accident.
* It sees only what the judge model can see. A loss it does not notice is not
  caught. The report does say how far the judge actually got —
  `qe_semantic_coverage` distinguishes "judged" from "the judge was asked and
  failed" from "never requested", and `qe_judge_calls` gives the raw count.
* It does not fix anything by itself. It routes a block to repair, where a
  capable **repair model** has to notice the same loss and write it back — a 4B
  translation model cannot, which is why `repair_model` is separate from
  `draft_model` in the first place.

# Configuring the semantic judge

The judge is the only gate that can see meaning: it catches a translation that
reads perfectly while dropping a word ("publicly available" → "获取", "content
interference" → "干扰"). Every zero-token gate passes those.

Two blocks on arXiv 2606.23344 did exactly that and shipped as `MTQE_PASSED`:

| block | source | translation | what was lost |
| --- | --- | --- | --- |
| b0010 | "the model weights are **publicly available** at …" | "模型权重可在 … **获取**" | 公开 — "is publicly available" became "you can obtain it" |
| b0015 | "**content** interference from background noise" | "背景噪声或相邻元素的干扰" | 内容 — the noun the clause is about |

Both are assertions weakened by one word. No surface check can see it, because
nothing is missing from the string — only from the meaning.

It is **off by default** (`qe_judge_enabled=False`, `qe_engine=heuristic`) and
this page shows how to turn it on. Nothing here assumes a particular machine: a
cloud API, a company gateway and a local server all configure the same way.

## The minimum

The judge needs a model and it needs to be *allowed* to run:

```toml
# ubt.toml, or ~/.config/ubt/config.toml
[defaults]
qe_engine = "tiered"
qe_judge_enabled = true
qe_judge_pass_sample = 0.15      # see below — without this nothing is judged
qe_judge_provider = "my-judge"   # the block declared below
```

```toml
[providers.my-judge]
base_url = "https://api.example.com/v1"
api_mode = "openai-chat"         # or openai-responses, anthropic-messages, gemini-native
api_key = "${JUDGE_API_KEY}"     # expands from the environment; never commit a key
repair_model = "some-capable-model"
```

For a quick shell-only setup, the four settings that matter are:

```bash
export UBT_QE_ENGINE=tiered
export UBT_QE_JUDGE_ENABLED=true
export UBT_QE_JUDGE_PASS_SAMPLE=0.15
export UBT_QE_JUDGE_MODEL=some-capable-model
```

Three things about this are easy to get wrong, and each one was measured.

### 1. The judge needs a capable model

The judge's job is to compare a sentence against its translation and notice what
is missing. A small model cannot do it, and a judge that always says "fine" is
worse than no judge — it turns a sample audit into a rubber stamp.

Measured on the same pair (a correct translation, and the same translation with
one word dropped):

| model | correct | drops one word | usable as a judge |
| --- | --- | --- | --- |
| a 4B translation model (`gemma4-e4b`) | 100 | **100** | no |
| a 27B general model, local (`mirai-s-27b`) | 98 | 48 | yes, but 35–85 s/block |
| a hosted frontier model (`deepseek-v4.1-flash`) | 95 | 75 | yes, 1.6–3.5 s/block |

`gemma4-e4b` scores the broken translation 100/100 even when the rubric says in
so many words that a dropped word must score below 60. It is a good translation
model and an unusable judge. Use a general-purpose model, not a
translation-specialised one: translation models are trained to produce fluent
output, which is the opposite of noticing what is absent.

On the five measured blocks from this project's fixture (two real defects, three
correct translations), the hosted judge was right on all five in 1.6–3.5 s per
block:

| block | verdict | judge |
| --- | --- | --- |
| b0010 (drops 公开) | caught | 0.72 |
| b0015 (drops 内容) | caught | 0.78 |
| b0029 (correct) | passes | 0.92 |
| b0049 (correct) | passes | 0.97 |
| b0067 (correct) | passes | 0.87 |

The local 27B needed 35–85 s per block for the same decisions, which is the
practical argument for a hosted judge: the cost is a few seconds per sampled
block, not minutes.

### 2. Sampling must be on, or the judge never runs

`TieredQERunner` consults the judge for gray-band scores (0.70–0.80) and for a
**sampled share of clean passes** — and that share defaults to `0.0`. A
compressed translation scores 0.92, so with the default it never reaches the
judge at all, even with `qe_judge_enabled = true`. `qe_judge_pass_sample = 0.15`
judges roughly one clean block in seven.

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

A local endpoint works exactly like a hosted one, except it needs no credential:

```toml
[providers.local-judge]
base_url = "http://127.0.0.1:8080/v1"
api_mode = "openai-chat"
repair_model = "your-model-name"
```

No `api_key` is required. The engine omits the `Authorization` header entirely
when a block has no credential, so nothing is sent that a local server could
reject. (A keyless *main* provider is likewise a real run rather than a
rehearsal — but only for a loopback endpoint; a hosted one still needs a key,
because sending the request without one would just fail with a 401.)

Earlier versions required a placeholder (`api_key = "local-only"`): an empty
value produced the malformed header `Bearer ` (trailing space) and the request
died before reaching the server. That workaround is no longer needed — the
header is simply not sent. An existing placeholder is harmless, and can be
deleted.

A local judge large enough to be useful is slow — budget 35–85 seconds per judged
block, and stop other GPU work first.

## What it costs

One model call per judged sentence of one sampled block (or one call per block
when the sentence split does not align). With `qe_judge_pass_sample = 0.15` on a
150-block document, that is about 22 blocks, each of a few sentences: a few cents
on a hosted API, and a few minutes on a local one. `pass_sample` is the dial
between coverage and spend.

## What it buys you, and what it does not

* It catches a loss concentrated in a segment — the b0010 shape, judged alone.
* It does **not** catch a one-word loss diluted across a long block, unless the
  comparison is made per sentence.
* It is an *opinion*, not a measurement (`is_calibrated()` is False for the judge
  leg), so it may only lower a score, never raise one.
* It cannot rank two translations against each other — best-of-n reranking never
  uses it. Its verdict can send a block to repair and can lower a score; it
  cannot promote one.
* It sees only what the judge model can see. A loss it does not notice is not
  caught, and nothing in the report distinguishes "judged clean" from "not
  judged" unless you read `qe_judge_calls`.
* It does not fix anything by itself. It routes a block to repair, where a
  capable **repair model** has to notice the same loss and write it back — a 4B
  translation model cannot, which is why `repair_model` is separate from
  `draft_model` in the first place.

## The honest summary

The zero-token gates are the floor: they catch structure, numbers and script.
The semantic judge is the only thing that can see meaning, and it needs (a) a
large enough model, (b) sampling turned on, and (c) a comparison at sentence or
clause granularity. Without all three it silently does nothing — which is
indistinguishable, in the report, from a perfect document.

That last point is why the two defects above shipped: every signal in the report
said the document was clean.

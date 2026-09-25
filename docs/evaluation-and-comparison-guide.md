# Evaluation & Comparison Guide

Which tool answers which evaluation question, what each one costs to run, and —
honestly — what is still uncalibrated. This is the file `scripts/run_real_benchmark.sh`
(head: "Reference: docs/evaluation-and-comparison-guide.md"), `scripts/biou_score.py`
("pair it with human readability review (see …)") and
`tests/integration/test_local_model_baseline.py` ("… still lists L3 judge
calibration as 待跑") all point at, so all three references resolve here.

Status marks used below: **✅ runs today** · **⚠️ needs local services** ·
**待跑** = deliberately open, not yet executed (listed so it cannot be forgotten).

Three separate questions, three separate rulers — never substitute one for another:

| Question | Ruler | Wrong tool for the job |
| --- | --- | --- |
| Is the **text** right? | QE tiers (§2) + hard structural invariants (§4) | pixel scores prove nothing about words |
| Is the **page** right? | BIoU / pixel fidelity / render A-B (§3) | BLEU-style scores prove nothing about layout |
| What did it **cost**? | cost benchmark (§5) | quality scores say nothing about the bill |

---

## 1. Entry point: `scripts/run_real_benchmark.sh`

The driver probes the machine first, then routes to a benchmark. Everything it
finds (or does not find) is printed as a capability table before anything runs:

| Probe | Missing ⇒ |
| --- | --- |
| poppler (`pdftocairo`/`pdftotext`) | SVG vector diagrams fall back to raster |
| `typst` | PDF render legs unavailable |
| llama-swap at `127.0.0.1:9090` | local MT tests self-skip |
| `translategemma:4b` model | MT tiers self-skip (add it to llama-swap's `config.yaml`) |
| CometKiwi checkpoint in `~/.cache/huggingface/hub` | neural QE self-skips |
| cloud LLM key in the environment | mock provider is used instead |

```bash
./scripts/run_real_benchmark.sh                # --summary: probe only, print how to run each mode
./scripts/run_real_benchmark.sh --live-mt      # live TranslateGemma MT tier acceptance
./scripts/run_real_benchmark.sh --qe-calib     # L1/L2 QE consistency & calibration report
./scripts/run_real_benchmark.sh --compare-live # paired MT vs cloud-LLM comparison
./scripts/run_real_benchmark.sh --all          # every integration test (best effort)
```

All modes are plain pytest invocations over `tests/integration/`, so they inherit
the default addopts (`-m 'not slow and not legacy_drift' --timeout=600`) — and almost the whole
integration tier carries `slow`, so a bare run selects only the handful of
unmarked cases (`uv run pytest tests/integration --collect-only -q` shows the
exact split). Run the live tiers deliberately with `-o addopts=""`; note that the
override replaces addopts **wholesale**, so it also clears `--timeout=600`, which
is what you want for multi-minute local MT / CometKiwi inference (a CometKiwi
subprocess has already blown a 300s budget — see
`tests/integration/test_local_model_baseline.py`). Pass an explicit `--timeout=`
if you want a cap there.

> ✅ **Fixed 2026-09-24:** the script now passes `-o addopts=""` itself for every
> live mode. It previously did not, and because these files all carry
> `pytestmark = slow` the default marker filter deselected them — pytest then exits
> 5 ("nothing collected"), which `set -eo pipefail` turns into an abort at the
> selected-mode step. The other half of the old warning is stale too: step [3/4]
> runs `tests/unit/test_metrics.py` + `tests/unit/test_qe_score_policy.py`, both of
> which exist (the deleted `test_readme_case_count.py` is long gone).

## 2. Quality tiers: L1 / L2 / L3

| Tier | What it is | Where it runs |
| --- | --- | --- |
| **L1** | `FastPassFilter` + heuristic validators (numeric consistency, length band, punctuation topology). 0-token, deterministic. | every run, including CI |
| **L2** | CometKiwi neural score, executed in an isolated subprocess (`ubt/core/qe/comet_score_ipc.py`, packaged with the wheel) so torch/CUDA cannot take the pipeline down. The `engine` label in the reply is load-bearing: `heuristic_fallback` never masquerades as a calibrated measurement. | ⚠️ needs the HF checkpoint |
| **L3** | LLM judge for the gray zone — default band `[0.7, 0.8)` (`UBT_QE_JUDGE_GRAY_LOW` / `_GRAY_HIGH` configure it) — only blocks L1 and L2 disagree about actually reach it. | ⚠️ needs a judge-capable model |

`qe_threshold = 0.75` was set from the L1↔L2 agreement evidence; it is
**reported, never hard-gated on absolute neural values** (those are
domain-sensitive). The report itself comes from
`tests/integration/test_qe_calibration.py` (agreement rate, danger quadrant
L1-pass/L2-low, share of blocks that would hit the judge):

```bash
./scripts/run_real_benchmark.sh --qe-calib     # == uv run pytest tests/integration/test_qe_calibration.py -v -s
```

**Status board**

| Item | Owner / runner | Status |
| --- | --- | --- |
| L1↔L2 agreement report + gray-zone sizing | `tests/integration/test_qe_calibration.py`, nightly `live-local` job | ✅ runs (nightly + `--qe-calib`) |
| **L3 judge calibration** (judge verdicts vs a reviewed sample; does `[0.4,0.8)` actually catch what it claims?) | nobody yet | **待跑** |
| MT tier structural acceptance (`misrouting < 2%`) | `tests/integration/test_mt_tier_live.py` | ⚠️ needs llama-swap + CometKiwi |
| MT vs cloud-LLM paired comparison | `tests/integration/test_mt_vs_llm_compare.py` | ⚠️ needs llama-swap + `UBT_OPENCODE_SESSION_ID` |

"待跑" is a promise, not a decoration: the reference-free neural path is
uncalibrated until the L3 row is executed and its verdict recorded here.

## 3. Layout fidelity rulers

### BIoU — `scripts/biou_score.py`

Offline layout-overlap score (BabelDOC-style, at a pragmatic fidelity level):
text-row rectangles are extracted from source and translated pages **with the
same parser** (pypdfium2), normalized by page size, matched greedily in reading
order, then averaged.

```bash
uv run python scripts/biou_score.py source.pdf translated.pdf           # summary
uv run python scripts/biou_score.py source.pdf translated.pdf --json    # per-page breakdown
```

Interpretation rules (from the script's own scope notes — violating them makes
the number meaningless):

- **Not a translation-quality metric.** Pair every BIoU claim with human
  readability review; it is a calibration objective and a regression tracker.
- **Pagination-sensitive** for reflow outputs → compare *relative* (mode A vs
  mode B, or same mode across runs), not absolute thresholds.
- **Alternating dual outputs (2× pages) are out of scope** — score their
  `*_trans_stage.pdf` instead.
- `row_ratio` in the output ≈ interleave density (**~2.0 inline, ~1.0
  monolingual**) — read it next to `mean_biou`, it is the advisor-relevant
  calibration feature.
- Vector figures contribute no text rows, so figure-heavy pages score lower by
  construction (figures that move *are* layout drift for inline interleave).

### Pixel fidelity — `scripts/fidelity_baseline.py`

Aggregate rigid-render fidelity over a corpus (non-text residual + painted
coverage per document plus corpus mean), producing a diffable JSON baseline:

```bash
uv run python scripts/fidelity_baseline.py \
    --source-dir benchmarks/src --artifact-dir benchmarks/out \
    --dpi 300 --pages 8 -o benchmarks/fidelity.baseline.json
```

Pairs are either a parallel directory of the same filenames, or one directory
where each `X.pdf` has a sibling `X.translated.pdf`.

### Rasterizer A/B — `scripts/oxide_render_ab.py`

pdf-oxide vs poppler `pdftoppm` equivalence gate (page size must match, mismatch
ratio must stay inside thresholds frozen from the 2026-09-20 calibration run).
This is the stage-2 adoption evidence and is also a CI step
(`ci.yml` → *Render A/B — pdf_oxide vs pdftoppm*).

### Formula engines — `scripts/formula_matrix.sh`

Renders the same formulas through mathjax / typst / image paths for a side-by-side
comparison (needs Node for MathJax).

### In-pipeline visual gate

Independent of all of the above, every PDF render goes through the visual gate
(T0/T1/T2) with the reflow self-healing loop described in
[USER_GUIDE §四.3](USER_GUIDE.md#3-视觉门禁排版自愈-visual-gate-reflow-loop);
its findings land in `visual_report.json`.

## 4. Text quality: the zero-cost real-model baseline

`tests/integration/test_local_model_baseline.py` pushes a **fixed** EN→ZH corpus
through the real local MT tier and hard-asserts the invariants any competent
translation must hold: non-empty target, no placeholder residue
(`⟦ ⟧`), no leaked prompt scaffolding, numeric fidelity, no structural
FastPass rejection. Zero API cost (the local llama-swap gateway + cached CometKiwi),
self-skips when either is absent, and is a **hard gate** in the nightly
`live-local` job — unlike the report-only tiers above, a red here is a
regression, not missing weights.

```bash
uv run pytest tests/integration/test_local_model_baseline.py -o addopts="" -v -rs
```

Neural quality *scores* are deliberately not computed there (the CometKiwi
subprocess once failed to finish inside its own 300 s budget — paying that
again would resurrect a cost an earlier review removed); the neural profiles
stay in `test_qe_calibration.py` and remain part of the **待跑** L3 story above.

## 5. Cost — `scripts/cost_benchmark.py`

Runs the full pipeline on a real book against any OpenAI-compatible API and
reports per-stage call counts, latency, tokens and estimated USD, plus what the
0-token defense net actually saved. Everything bulky lands in `/tmp`; the one
artifact meant to be committed is the **text-free** metrics JSON
(corpus sha256, models, prices, wall clock, per-stage counts, cache hits, cost,
block completion/failure counts) written to `--metrics-dir`
(default `docs/benchmarks/`, see [its README](benchmarks/README.md)):

```bash
export DEEPSEEK_API_KEY=sk-...
uv run python scripts/cost_benchmark.py path/to/book.md --metrics-dir docs/benchmarks
```

Prices default to DeepSeek's published rates; override `--price-*` for other
providers, and treat every dollar figure as an estimate to be cross-checked
against the provider dashboard.

## 6. Where the deterministic evidence lives

Golden corpora, KPI goldens, the tier-level golden rules and how to run each
tier: [docs/golden-set.md](golden-set.md) plus the `uv run pytest` recipes in
[AGENTS.md](../AGENTS.md). Testing constraints that outrank every doc
here: [AGENTS.md](../AGENTS.md).

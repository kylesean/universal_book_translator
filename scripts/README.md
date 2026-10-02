# scripts/

Operator tooling. **Automated verification lives in `tests/`** (see
`docs/guides/TESTING_STRATEGY.md`); everything here is run by hand or by the
slow test tier, never collected by pytest (`testpaths = ["tests"]`).

## Slow-tier corpus harnesses (kept)

Six acceptance harnesses need real PDFs or render subprocesses, so they are
invoked as subprocesses by `tests/integration/test_corpus_acceptance.py`
(`pytest -m slow`) and double as on-demand acceptance tools:

| Script | What it gates |
| --- | --- |
| `shadow_reader.py` | the native PDF reader loses no text vs the legacy extractor; blocks round-trip the bridge |
| `shadow_typst.py` | the Typst backend realizes/verifies every element class over real documents |
| `shadow_rtl.py` | RTL targets: language profiles, gates, fonts, `dir: rtl` emission |
| `shadow_overlay.py` | the overlay backend places every element as an opaque source slice |
| `shadow_outputs.py` | overlay lowering composes mixed realizations losslessly |
| `shadow_delivered_pixel.py` | the pixel witness runs fail-closed on the *delivered* artifact |

All take `--corpus corpus` (default) and exit 0 on pass.

## Fixture generator

- `make_sample_corpus.py` — regenerates `tests/fixtures/synthetic-*.pdf`
  (needs `typst`; output is gitignored).

## Benchmarks & sweeps (not tests)

- `cost_benchmark.py` — writes `docs/benchmarks/` (see that README).
- `knob_sweep.py`, `oxide_render_ab.py`, `rigid_coverage_sweep.py`,
  `biou_score.py`, `fidelity_baseline.py`, `run_real_benchmark.sh`,
  `formula_matrix.sh` — experiment/benchmark harnesses for tuning and A/B runs.

## Utilities

- `dump_compiler_edges.py` — read-only static check of the core↔compiler import
  discipline (§6.1 of the evolution doc): exit 1 on any module-level
  core→{pipeline, segment, translate} edge or a `core.engine ↔ pipeline` cycle.
- `export_pdf_to_markdown.py` — one-off PDF → Markdown extraction helper.
- `mathjax/` — node renderer used by the formula pipeline (`render.mjs`).

## Removed

The other `shadow_*.py` migration-era harnesses were deleted once their
contracts were pinned as unit tests; the mapping lives in
`docs/guides/TESTING_STRATEGY.md` §P3, and git history retains the code.

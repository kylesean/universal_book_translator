# scripts/

Operator tooling. **Automated verification lives in `tests/`** — the tier rule
there is `fast` = offline and cheap enough to run on every edit, `slow` = a
long-running subprocess or a render A/B, `network` = needs outbound access.
Everything here is run by hand or by the slow tier, never collected by pytest
(`testpaths = ["tests"]`).

## Slow-tier corpus harness (kept)

One acceptance harness needs real PDFs, so it is invoked as a subprocess by
`tests/integration/test_corpus_acceptance.py` (`pytest -m slow`) and doubles as
an on-demand acceptance tool:

| Script | What it gates |
| --- | --- |
| `shadow_reader.py` | the native PDF reader loses no text vs the legacy extractor; blocks round-trip the bridge |

It takes `--corpus corpus` (default) and exits 0 on pass.

## Benchmarks & sweeps (not tests)

- `cost_benchmark.py` — writes `docs/benchmarks/` (see that README).
- `knob_sweep.py`, `biou_score.py` — experiment/benchmark harnesses for
  tuning runs.
- `probe_facsimile.py` — cheap feasibility prefilter for the `faithful`
  delivery contract (design: `docs/design/adaptive_delivery_prd.md`). For a
  source PDF it reports, per page, the largest single body size at which every
  paragraph still fits its source box. Pure geometry: no render, no model, no
  ledger, so it is safe to run before paying for a job. It is not the delivery
  acceptance criterion; the PRD requires artifact-level measurement.
  `--json` for machine reading, `--summary` to skip the page table.
- `diagnose_spills.py` — Typst layout spill and physical overflow diagnostic tool.
  Deterministic geometry scanner over rendered PDF artifacts to detect margin
  breaches, multi-column gutter overruns, table/formula clipping, and type collapse;
  emits 2D spatial terminal heatmaps, structured JSON, and actionable fix advice.
- `calibrate_measurement.py` — G2 Typst MeasurementService calibration: compiles one
  block with the requested width/style and reports real height, line boxes and
  overflow. Zero model tokens; use it to spot-check a font/width before changing
  the layout solver.
- `cels_spike.py` — G3 CELS feasibility probe on real pages: Stage 1 column
  detection (with every refused gutter and its reason) plus Stage 5 content-stream
  injection arms (`Tc`, forced `Tz`, `TL`+`T*` twin). Verifies the rewritten PDF by
  re-extracting text and char boxes with PDFium, counting graphics operators, and
  diffing full-page rasters; writes `evidence.json` to the given out dir. Zero
  model tokens; the go/no-go report cites its output.
- `cels_cjk_spike.py` — G3 Stage 5 font pipeline (AD-018): subsets a CJK TrueType
  font with `fontTools`, embeds it as `/Type0` + `/Identity-H` with a generated
  `/ToUnicode` CMap, writes a Chinese sentence at the original line position, and
  verifies with PDFium that the text extracts verbatim, stays inside the column,
  and changes no pixel outside the rewritten line. Requires a glyf-based CJK font
  (`--font`), e.g. `MapleMono-NF-CN-Regular.ttf`.
- `run_corpus_benchmark.py` — multi-scenario automated corpus benchmark runner
  and quality/cost/spill dashboard. Evaluates cases from `corpus/cases.json`
  across delivery contract integrity, physical layout spills, MTQE scores, and
  token/USD costs. Supports dry-run rehearsals, live model runs, and eval-only
  evaluations with baseline drift tracking and HTML/Markdown/JSON exports.
- `verify_sample_book.py` — end-to-end sample book pipeline verification and
  deployment smoke check runner. Ingests reference sample monograph
  (`examples/sample_book/deep_learning_primer.md`), validates multi-stage AST extraction
  (headings, math formulas, code blocks, tables), exercises container fitting, compiles
  Typst document to PDF, and confirms zero layout margin spills. Runs 100% offline.

## Utilities

- `dump_compiler_edges.py` — read-only static check of the core↔compiler import
  discipline: exit 1 on any module-level core→{segment, translate} edge or a
  `core.engine ↔ pipeline` cycle.
- `build_console_assets.py` — builds the operator console (`web/`) and syncs the
  bundle into `ubt/api/static`, the copy the wheel ships and FastAPI serves.
- `check_console_bundle.py` — the local mirror of CI's bundle check: rebuilds the
  console and exits 1 when `ubt/api/static` no longer matches the source. Runs as
  the `console-bundle-fresh` pre-push hook.
- `check_doc_refs.py` — read-only check of the `path:line` references in the
  (unversioned) `docs/` tree: exit 1 when a cited UBT file does not exist or a
  line number is past its file's end. Run it after a refactor that moves code
  the docs cite. Comparison-repo and draft references are skipped by design.
- `export_pdf_to_markdown.py` — one-off PDF → Markdown extraction helper.

## Removed

Migration-era `shadow_*.py` harnesses were deleted once their contracts were
pinned as unit tests; git history retains the code.

The `run_real_benchmark.sh` and `formula_matrix.sh` harnesses were deleted with
the test files they drove: the live-model tiers they invoked
(`test_mt_tier_live`, `test_qe_calibration`, `test_local_model_baseline`,
`test_mt_vs_llm_compare`) and the per-backend formula matrix
(`test_formula_engine_matrix`) went with the unit-suite rebuild, and
`--math-backend` is a retired knob no renderer reads. Capability probing is
`ubt doctor --probe`, which covers the same ground without a shell duplicate.
`scripts/mathjax/` went with the MathJax backend it rendered for.

`tests/unit/test_reference_resolvability.py` now fails when a harness under
`scripts/` names a pytest file that no longer exists, so this cannot recur.

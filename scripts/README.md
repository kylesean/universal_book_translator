# scripts/

Operator tooling. **Automated verification lives in `tests/`** (see
`docs/guides/TESTING_STRATEGY.md`); everything here is run by hand or by the
slow test tier, never collected by pytest (`testpaths = ["tests"]`).

## Slow-tier corpus harness (kept)

One acceptance harness needs real PDFs, so it is invoked as a subprocess by
`tests/integration/test_corpus_acceptance.py` (`pytest -m slow`) and doubles as
an on-demand acceptance tool:

| Script | What it gates |
| --- | --- |
| `shadow_reader.py` | the native PDF reader loses no text vs the legacy extractor; blocks round-trip the bridge |

It takes `--corpus corpus` (default) and exits 0 on pass.

## Fixture generator

- `make_sample_corpus.py` — regenerates `tests/fixtures/synthetic-*.pdf`
  (needs `typst`; output is gitignored).

## Benchmarks & sweeps (not tests)

- `cost_benchmark.py` — writes `docs/benchmarks/` (see that README).
- `knob_sweep.py`, `oxide_render_ab.py`,
  `biou_score.py`, `fidelity_baseline.py` — experiment/benchmark harnesses for
  tuning and A/B runs.

## Utilities

- `dump_compiler_edges.py` — read-only static check of the core↔compiler import
  discipline: exit 1 on any module-level core→{segment, translate} edge or a
  `core.engine ↔ pipeline` cycle.
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

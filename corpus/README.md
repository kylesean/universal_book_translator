# Delivery-contract verification corpus

This corpus is the **real-document gate**: it asserts the delivery *contract* —
the two axioms — on actual PDFs, and `ubt verify --corpus` fails the build when
a document loses text or assets. It complements (not replaces) the automated
suite in `tests/`, which pins the per-module contracts on synthetic fixtures
(`docs/guides/TESTING_STRATEGY.md`). Consumers: the `ubt verify` CLI, the
slow-tier corpus harnesses (`scripts/shadow_*.py` via
`tests/integration/test_corpus_acceptance.py`), and manual acceptance runs.

> The contract is engine-, language- and layout-agnostic. A regression on any
> render path shows up the same way: as an unbalanced book.

## Layout

```
corpus/
  cases.json          # the case manifest (checked in)
  documents/          # real source documents + delivered artifacts (NOT checked in)
```

Large/real documents are deliberately **not** committed (size + licensing).
Populate `corpus/documents/` locally. `cases.json` references them by relative
path; a case whose file is absent is reported `skip` (or `fail` under
`--require-all`).

## Running

```bash
# Translate every case's source document (dry-run, no API key) and verify the
# contract each run writes. This is the full gate: the pipeline itself is the
# ground truth, not a hand-written fixture. The slow test tier runs exactly
# this command automatically (pytest -m slow), skip when documents are absent.
uv run ubt verify --corpus corpus --run --require-all

# Verify only what is already on disk (no translation).
uv run ubt verify --corpus corpus

# Verify one delivered artifact (reads its <stem>_<tag>_contract.json).
uv run ubt verify /path/to/book_mono.pdf

# Re-derive the contract from a finished job's ledger (independent cross-check).
uv run ubt verify --job job_<docid>_zh --engine rigid
```

Local run for the checked-in cases (source PDFs are not committed; drop the
matching files into `corpus/documents/`):

```bash
cp ~/Downloads/2609.20519v1.pdf ~/Downloads/2608.25512v1.pdf \
   ~/Downloads/2609.22978v1.pdf ~/Downloads/chapter-3.pdf corpus/documents/
uv run ubt verify --corpus corpus --run --require-all
```

## Thresholds and CI coverage

The `expect` floors are **regression floors, not targets**: each is set a few
points below the current dry-run baseline so a real regression trips it while
normal parser/renderer churn does not. Measured with the dry-run (echo) provider
on the reference corpus:

| case | `min_delivered_ratio` | observed (dry run) |
| --- | --- | --- |
| `twocol-paper-2609` | 0.55 | 0.601 |
| `paper-2608` | 0.57 | 0.621 |
| `paper-2609-22978` | 0.66 | 0.759 |
| `book-chapter-3` | 0.60 | 0.639 |

When a real engine change moves a case, re-derive its floor and keep a margin
larger than the run-to-run noise rather than tightening the floor onto the
observed value.

Two deliberate exclusions from the automated CI gate:

- **CI does not run this gate.** The GitHub `gate` job runs `pytest -m fast`;
  this corpus is in the `slow` tier and its documents are not committed, so CI
  skips it (`_skip_without_corpus`). It runs locally, and in any CI that has the
  corpus checked out or mounted, via `pytest -m slow` or
  `ubt verify --corpus corpus --run --require-all`. The `fast`-tier ingest
  regression net that does run in CI is
  `tests/unit/adapters/test_pdf_adapter_e2e.py` (synthetic PDFs, no toolchain).
- **The `shadow_reader` coverage floor is relaxed to 0.97** in
  `tests/integration/test_corpus_acceptance.py` (`--min-coverage 0.97`) even
  though the script defaults to 0.98: the scanned-heavy `book-chapter-3` tops out
  at ~0.973 while still round-tripping losslessly. The relaxation is pinned next
  to the case in the harness, not here, so it moves with the harness.

## Case schema (`cases.json`)

```jsonc
{
  "schema_version": 1,
  "cases": [
    {
      "id": "twocol-paper-2609",
      "description": "Two-column paper: figures/tables must survive.",
      // One of:
      "document": "documents/2609.20519v1.pdf", // source to translate (needs --run)
      // "artifact": "documents/book_mono.pdf", // an already-delivered artifact
      // "job": "job_<docid>_zh",               // a finished job's ledger
      "expect": {
        "max_errors": 0,            // ERROR-severity violations allowed (default 0)
        "max_missing_assets": 0,    // lost non-text nodes allowed
        "max_source_kept": 5,       // text nodes allowed to ship source
        "min_accounted_ratio": 0.9, // (translated + verbatim) / total_text floor
        "min_delivered_ratio": 0.6  // translated / total_text floor (stricter)
      }
    }
  ]
}
```

Any key omitted is not asserted. `max_errors` defaults to `0`, so a contract
with an unaccounted text node or a missing asset fails even with no `expect`.

## Why these cases

| Case | Axis it pins |
|---|---|
| `twocol-paper-2609` | multi-column + figures/tables → rigid; no lost assets |
| `paper-2608` | figures/tables survive across a second document |
| `paper-2609-22978` | math-dense paper; tables reconstructed+verified or preserved whole |
| `book-chapter-3` | figure-heavy prose chapter; reflow delivers text, figures preserved |

Add a case whenever a bug is found; the case is the regression test. Prefer a
`document` case (the gate runs the pipeline) over a frozen `artifact` one, so the
case keeps probing the current engine rather than yesterday's output.

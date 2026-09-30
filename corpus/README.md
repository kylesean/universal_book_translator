# Delivery-contract verification corpus

The unit-test suite was removed; **this corpus is the regression gate**. It does
not test implementation details. It asserts the delivery *contract* — the two
axioms — on real documents, and `ubt verify --corpus` fails the build when a
document loses text or assets.

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
# ground truth, not a hand-written fixture.
uv run ubt verify --corpus corpus --run --require-all

# Verify only what is already on disk (no translation).
uv run ubt verify --corpus corpus

# Verify one delivered artifact (reads its <stem>_<tag>_contract.json).
uv run ubt verify /path/to/book_mono.pdf

# Re-derive the contract from a finished job's ledger (independent cross-check).
uv run ubt verify --job job_<docid>_zh --engine rigid
```

Local run for the two checked-in cases (source PDFs are not committed):

```bash
cp ~/Downloads/2609.20519v1.pdf ~/Downloads/2608.25512v1.pdf corpus/documents/
uv run ubt verify --corpus corpus --run --require-all
```

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
        "min_accounted_ratio": 0.9  // (translated + verbatim) / total_text floor
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

Add a case whenever a bug is found; the case is the regression test. Prefer a
`document` case (the gate runs the pipeline) over a frozen `artifact` one, so the
case keeps probing the current engine rather than yesterday's output.

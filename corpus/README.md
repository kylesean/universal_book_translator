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
  documents/          # real artifacts + their *_contract.json (NOT checked in)
```

Large/real documents are deliberately **not** committed. Populate
`corpus/documents/` locally (copy the delivered `*.pdf` and its `*_contract.json`
sidecar), or point a case at an existing artifact path. A case whose artifact is
absent is reported `skip` (or `fail` under `--require-all`).

## Running

```bash
# Verify every populated case. Exit 0 = all pass; 1 = a contract/expectation failed.
uv run ubt verify --corpus corpus

# CI: a missing artifact is a failure, not a skip.
uv run ubt verify --corpus corpus --require-all

# Verify one delivered artifact (reads its <stem>_contract.json).
uv run ubt verify /path/to/book_mono.pdf

# Re-derive the contract from a finished job's ledger (independent cross-check).
uv run ubt verify --job job_<docid>_zh --engine rigid
```

## Case schema (`cases.json`)

```jsonc
{
  "schema_version": 1,
  "cases": [
    {
      "id": "twocol-paper",
      "description": "Two-column academic paper: figures/tables must survive.",
      "artifact": "documents/2609.20519v1_mono.pdf", // or "job": "job_..._zh"
      "expect": {
        "max_errors": 0,            // ERROR-severity violations allowed (default 0)
        "max_missing_assets": 0,    // lost non-text nodes allowed
        "max_source_kept": 5,       // text nodes allowed to ship source
        "min_accounted_ratio": 0.95 // (translated + verbatim) / total_text floor
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
| `prose-book` | reflow prose must deliver all text; no assets to lose |
| `twocol-paper` | multi-column + figures/tables → rigid; no lost assets |
| `scanned` | scan → overlay; text layer may be sparse but must be accounted |
| `fgm-dense` | formula/figure/math-dense → rigid; formula assets preserved, not shattered |

Add a case whenever a bug is found; the case is the regression test.

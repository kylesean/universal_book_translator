# Semantic-path gold corpus (RFC-2026-02)

The **measurement set** for the semantic reflow rearchitecture. It complements
`corpus/` (the delivery-contract corpus): that one asserts "no text or asset was
lost", this one asserts "the output is actually a translation" — the dimensions
the overlay path could not move (table headers, font-size spread, untranslated
blocks).

## Layout

```
corpus/semantic_gold/
  manifest.json     # the 20 cases + their measured properties (checked in)
  documents/        # the source PDFs (NOT checked in: size + licensing)
```

Sources are fetched, not committed:

```bash
uv run python scripts/fetch_semantic_gold.py
```

## Running

```bash
# Score one delivered artifact
uv run python scripts/eval_semantic.py out.pdf --ledger .ubt/ledgers/job_*.sqlite

# Score the whole corpus (needs one artifact per case, named <case-id>.pdf)
uv run python scripts/eval_gold_corpus.py \
    --artifacts ~/Documents/UBT \
    --source-markdown /tmp/gold_srcmd \
    --json /tmp/gold_baseline.json
```

`--source-markdown` points at a directory of `<case-id>.md` files (the Docling
output the pipeline already produces). Without it the source-header leak metric
reports `0.0` and `source_tables_available` is `false`, so an unmeasured value
cannot be mistaken for a good one.

## The 20 cases

Chosen to stress one dimension each; 15 have a PDF bookmark tree (exercises the
outline-based TOC merge), 5 do not (exercises the Docling-only path).

| case | pages | bookmarks | categories |
| :--- | ---: | ---: | :--- |
| switch-transformers | 40 | 31 | formula-dense, complex-tables, toc-outline |
| attention-is-all-you-need | 15 | 22 | formula-dense, toc-outline |
| bert | 16 | 0 | complex-tables, no-outline |
| resnet | 12 | 0 | two-column-figures, complex-tables, no-outline |
| gpt3 | 75 | 32 | long-mixed, complex-tables, toc-outline |
| vit | 22 | 30 | two-column-figures, toc-outline |
| ddpm | 25 | 18 | formula-dense, two-column-figures |
| alphafold2 | 24 | 17 | formula-dense, long-mixed, complex-tables |
| flamingo | 54 | 53 | two-column-figures, complex-tables |
| instructgpt | 68 | 54 | long-mixed, complex-tables |
| llama2 | 77 | 29 | long-mixed, complex-tables, toc-outline |
| whisper | 28 | 0 | two-column-figures, complex-tables, no-outline |
| clip | 48 | 0 | two-column-figures, complex-tables, no-outline |
| stable-diffusion | 45 | 42 | two-column-figures, formula-dense |
| mamba | 36 | 57 | formula-dense, complex-tables |
| moe-mistral | 13 | 10 | formula-dense, complex-tables |
| raft | 21 | 1 | formula-dense, two-column-figures |
| neural-ode | 18 | 21 | formula-dense |
| gan | 9 | 0 | formula-dense, two-column-figures, no-outline |
| dropout | 18 | 20 | formula-dense, complex-tables |

Counts in `manifest.json` are **measured** with pypdfium2 at build time, not
copied from the papers' own claims.

## Targets vs baselines

`manifest.json` → `scoring.targets` holds the RFC §9 goals. A first baseline
needs a full pipeline run (LLM calls); until then the only measured artifact is
`switch-transformers`, whose overlay-path numbers are:

| metric | overlay artifact | target |
| :--- | ---: | :--- |
| `cjk_ratio` | 0.311 | ≥ 0.55 *(whole document; source-kept blocks count against it)* |
| `ledger.cjk_ratio_translatable` | — | ≥ 0.55 *(only the blocks meant to be translated)* |
| `untranslated_marks` | 17 | 0 |
| `p95_font_size_pt` | 11.46 | 10–11 |
| `distinct_font_sizes` | 44 | — |
| `source_header_term_leakage` | 0.96 | ≤ 0.35 |
| `ledger.blocked_ratio` | 0.059 | ≤ 0.05 |

> **Why two CJK measures.** `cjk_ratio` is a *delivery* figure: it counts the
> bibliography, the affiliations and every code listing the policy deliberately
> keeps in the source language. `ledger.cjk_ratio_translatable` is the
> *translation* figure, over exactly the blocks the renderer treated as
> translated. Measured across the 20-document corpus the gap is decisive —
> **7/20** clear 0.55 whole-document, **19/20** clear it translatable-only
> (median 0.71). Reporting only the first made a working pipeline look like it
> had barely translated anything.

Record every new baseline in the RFC's implementation log (§13) rather than
editing this table, so the history of the numbers stays visible.

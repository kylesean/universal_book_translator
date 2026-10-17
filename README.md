# Universal Book Translator (UBT)

Pre-1.0 self-hosted document/book translation compiler: ingest PDF/EPUB/DOCX/
HTML/Markdown into one IR, translate it through a ledger-backed multi-stage LLM
pipeline, and re-typeset the result as a new document laid out by Typst.

Consistency defenses: a mined book bible (terms and names) and a structural +
quality gate run on every job. Hierarchical chapter memory runs only on the
multi-chapter long route — a single-chapter document (which is how every PDF is
ingested), any document over 40 chapters, and academic profiles get no rolling
summary. Deterministic glossary enforcement is short-document only.

## Install

```bash
uv sync --extra dev                            # development (Python 3.12+)
pip install universal-book-translator          # users
pip install 'universal-book-translator[mcp]'   # users who want the MCP server
```

Extras: `pdf` (Docling), `ocr` / `academic` (RapidOCR + ONNX runtime), `mcp`
(the MCP server). PDF output needs the `typst` compiler on
`PATH` — a PDF job fails at the render stage without it. `ubt doctor` checks the
key, the ledger directory, Typst and CJK fonts, and says what breaks.

## Quick start

Set the credential and the models. Every setting has one canonical `UBT_*`
variable, and `ubt config` prints the whole list with the variable that sets
each one; a provider block in `ubt.toml` / `~/.config/ubt/config.toml` can
supply the same values instead.

```bash
export UBT_LLM_API_KEY=sk-...              # the outbound LLM key
export UBT_BASE_URL=https://api.openai.com/v1
export UBT_DRAFT_MODEL=gpt-4o-mini         # tier-1 draft model
export UBT_REPAIR_MODEL=gpt-4o             # tier-2 repair model
```

`UBT_PROVIDER` selects a named provider from that registry; without one,
`UBT_BASE_URL` is used directly. To run the whole pipeline without spending
anything, add `--dry-run`: it uses a deterministic echo provider, so no key is
needed and no tokens are billed.

```bash
ubt translate book.pdf -l zh
```

`-l` is `--target-lang`; `-s` is `--source-lang` (default `en`). Without `-o`
the deliverable is written to `<documents>/UBT/<stem>_bilingual.pdf`. The run
prints a Job ID at start, and that id is the resume handle:

```
Job ID: job_da4c06edac50_zh  (resume with: ubt translate <input> --job-id job_da4c06edac50_zh)
```

The id is derived from the run's identity, so its suffix grows with what you
selected — the same book with `--dry-run --profile paper` becomes
`job_da4c06edac50_zh_mock_prpaper`. Copy the whole id, not the prefix. Resume is
the default: re-running with the same `--job-id` picks up from the ledger's
checkpoints instead of re-paying for finished blocks, and `--fresh` discards the
ledger and re-ingests. `--fresh` keeps `tm.sqlite` (the cross-job translation
memory) and any external glossary: those are assets, not job state, so a re-run
starts from a clean ledger but still benefits from what earlier runs learned.
Other useful commands: `ubt assess book.pdf` for a
zero-token quote, `ubt status <job-id>` for checkpoint statistics, and
`ubt verify <artifact>` to re-reconcile a delivery.

A run restricted with `--pages 1-6` gets that scope in its default filename
(`book_mono_p1-6.pdf`) instead of the full document's name. Sampling a few pages
to check a gate therefore cannot overwrite the complete delivery — an explicit
`-o` is always used exactly as given.

### Operator console

```bash
ubt console                  # opens http://127.0.0.1:8000/ in a browser
```

`ubt console` serves the same FastAPI app as the `ubt-api` entry point, on
`127.0.0.1:8000` by default, and opens the browser for you; for a loopback bind
it also opens the server without an auth gate. Running `ubt-api` directly
refuses to start without one and prints the two ways to supply it: set
`UBT_API_KEY` (the inbound `X-API-Key`, not the LLM key), or opt into an open
local server for development. The console also runs the review workbench, where
a human edit is applied to the ledger under the job's writer lock.

### MCP server

```bash
pip install 'universal-book-translator[mcp]'
ubt-mcp                      # stdio transport
```

The stdio server exposes the engine's operations as `ubt_*` tools
(`ubt_translate_book`, `ubt_job_status`, `ubt_list_issues`, `ubt_edit_segment`,
`ubt_re_export`, `ubt_list_jobs`, `ubt_doctor`, …). It is local-trust — same
user, no API-key gate — so front the REST API instead when exposing it beyond
one machine.

## Rendering

There is one PDF render path. The document is rebuilt from the IR and Typst
lays it out: uniform body type (10.5pt), translated table headers, and native
math from LaTeX→Typst conversion. It is reading-grade, not DTP-grade — text
reflows and the page count changes, so the artifact is a new document rather
than a copy of the source layout.

The reader-facing choice is `dual_mode` (`mono` / `dual` / `auto`, default
`auto`): `dual` pairs each block's source line above its translation, `mono`
delivers the target only, and `auto` lets the advisor pick from the document's
structure. This is the only mode knob; effort and engine details are chosen by
a **preset** (`publication` / `standard` / `preview` / `fast`), which selects
the prompt depth, formula handling and parser. Presets never pick a model or a
render route.

Measured on the reference paper (arXiv 2101.03961): distinct font sizes drop
from 44 to 3–5, formula residue from 0.857 to 0.000, untranslated-block markers
from 17 to 0, and page count from 41 to 29. CJK coverage of the blocks meant to
be translated reaches the 0.55 target; the whole-document ratio is lower
because the bibliography is deliberately kept in the source language and
counted in the denominator.

## Known limits

Measured against the current implementation, not aspirational:

- **Best-supported input**: born-digital, white-background, single- or
  double-column text PDFs. The pdfium fast lane does not emit tables or
  figures, and the Docling path maps merged cells to the top-left value only.
- **Tables**: table cells are translated in the IR (and scored by QE), then
  re-typeset with translated headers. EPUB/DOCX/HTML/Markdown outputs carry the
  translated cells too.
- **Not supported**: vertical CJK, encrypted PDFs (rejected at assessment, not
  translated), colour/watermarked backgrounds (no background-colour probe, so
  light text on dark pages is invisible), and text drawn inside figures.
- **Text expansion**: there is no fixed box to overflow — the page reflows and
  pagination changes — so a translation is never shrunk to fit a source box.
- **Per format**: DOCX footnotes/endnotes, column settings and TOC fields are
  not read; a whole PDF is ingested as one chapter, so per-chapter resume and
  billing do not apply to it.
- **Bilingual output**: `dual` pairs each block's source line above its
  translation; headings and formulas come out monolingual. A complementary
  mono/dual companion file is written alongside the primary artifact.
- **Exchange formats**: XLIFF 2.1 and TMX import exist but are not
  standards-complete (no TBX/SRX), so hand-off to external CAT tooling is
  limited.

Design notes and guides are maintained in the working tree under `docs/`, which
is intentionally unversioned (gitignored) — a clone carries the code and tests
only, not those notes.

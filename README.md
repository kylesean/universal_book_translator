# Universal Book Translator (UBT)

Pre-1.0 self-hosted document/book translation compiler: ingest PDF/EPUB/DOCX/
HTML/Markdown into one IR, translate it through a ledger-backed multi-stage LLM
pipeline, and re-typeset the result as a new document laid out by Typst.

Consistency defenses: a mined book bible (terms and names) and a structural +
quality gate run on every job. Hierarchical chapter memory runs only on the
multi-chapter long route — a single-chapter document and academic profiles get
no rolling summary; multi-chapter books retain sliding epoch memory across chapters. A PDF up to 50 pages is
ingested as one chapter; a longer one is split into internal page chunks that
the draft stage sees as multiple chapters. Deterministic glossary enforcement is
short-document by default; the long route opts in with
`UBT_DETERMINISTIC_GLOSSARY_LONG_ROUTE=true`.

## Install

No PyPI release yet — install from a checkout:

```bash
git clone https://github.com/kylesean/universal_book_translator.git
cd universal_book_translator
pip install .                                  # users (Python 3.12+)
pip install '.[mcp]'                           # users who want the MCP server
uv sync --extra dev                            # development
```

### Docker Container (Out-of-the-Box)

Run the pre-baked, multi-arch container with **Typst 0.15.1** and **Noto CJK fonts** pre-installed:

```bash
# Using Docker Compose
docker compose up -d

# Or via Docker CLI directly
docker run -d -p 8000:8000 -v $(pwd)/data:/data universal-book-translator:latest
```

The Web Console and REST API will be available immediately at `http://localhost:8000`.
See [`deploy/docker/README.md`](file:///home/kyle/Projects/python/universal_book_translator/deploy/docker/README.md) for complete container options.

### Sample Book & Verification Runner

To verify the complete ingestion, container fit, and Typst rendering pipeline offline without external API keys:

```bash
uv run python scripts/verify_sample_book.py
```
This runs against [`examples/sample_book/deep_learning_primer.md`](file:///home/kyle/Projects/python/universal_book_translator/examples/sample_book/deep_learning_primer.md) and compiles a validated PDF with zero margin spills.

Extras: `pdf` (Docling), `ocr` / `academic` (RapidOCR + ONNX runtime), `mcp`
(the MCP server). PDF output needs the `typst` compiler on
`PATH` — a PDF job fails at the render stage without it. `ubt doctor` checks the
key, the ledger directory, Typst and CJK fonts, and says what breaks.

## Quick start

Set the credential and the models. Every config-model setting has one canonical
`UBT_*` variable, and `ubt config` prints those with the variable that sets
each one; a provider block in `ubt.toml` / `~/.config/ubt/config.toml` can
supply the same values instead. A few process-level switches
(`UBT_OUTPUT_DIR`, `UBT_ALLOW_NO_AUTH`, `UBT_UI_LANG`, `UBT_LOG_LEVEL`,
`UBT_PDF_STREAM_CHUNK_SIZE`, `UBT_VLM_DRIVER`) are read directly and are not
part of that table.

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

One entry-point difference worth knowing: with no credential configured for a
remote endpoint, the CLI refuses the run (set `UBT_LLM_API_KEY` or pass
`--dry-run`), while the REST API and MCP server auto-run the same rehearsal
and label it — the submit response carries `"rehearsal": true`, and the
console badges the job. Nothing is silently translated by a mock.

```bash
ubt translate book.pdf -l zh
```

`-l` is `--target-lang`; `-s` is `--source-lang` (default `en`). Without `-o`
the deliverable is written to `<documents>/UBT/<stem>_bilingual<suffix>` — or
`<stem>_mono<suffix>` when the run resolves monolingual (for example
`--profile paper`), with the suffix following the input format. The run
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
it also opens the server without an auth gate.

For a console that survives closing its terminal, run it as a user service:

```bash
systemctl --user enable ubt-console.service
systemctl --user start ubt-console.service
journalctl --user -u ubt-console.service -f
```

The service keeps the UI and API on `127.0.0.1:8000`; `--dev`/Vite is for frontend
development only and is not needed to use the console.

**Configuring a model in the browser.** You do not need a shell to get started.
Engine Doctor (`/system`) lists every provider block in `ubt.toml`, lets you add
one, pick which is in force, and edit its endpoint, wire protocol, model ids and
capability flags. On a loopback server the credential field is available too;
beyond loopback, or under `UBT_STRICT_AUTH`, it is refused and the panel says
where the secret belongs instead. Saves go to `ubt.toml` and take effect on the
next job — no restart. Settings bound at startup (ledger directory, job mode, auth
gate) are reported as needing a restart rather than silently ignored. Running `ubt-api` directly
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
delivers the target only, and `auto` starts bilingual and lets the advisor step
it down to target-only when the layout cannot pair cleanly. This is the only
mode knob; effort and engine details are chosen by a **preset**
(`high_fidelity` / `standard` / `vector_math` / `preview` / `fast`), which
selects the prompt depth, formula handling and parser (a preset does pick
`pdf_engine`, but never a model).

Formula enrichment is not a quality gradient: only `vector_math` sets it
to `on`, which runs the VLM math transcription to vector LaTeX (slow — roughly
15–60 s per formula on CPU). Every other preset leaves it `auto`, which resolves
to cropping the source region, so `high_fidelity` and `standard` render identical
math and differ only in prompt depth.

Measured on the reference paper (arXiv 2101.03961): distinct font sizes drop
from 44 to 3–5, formula residue from 0.857 to 0.000, untranslated-block markers
from 17 to 0, and page count from 41 to 29. CJK coverage of the blocks meant to
be translated reaches the 0.55 target; the whole-document ratio is lower
because the bibliography is deliberately kept in the source language and
counted in the denominator.

### Visual gate

Every export runs a zero-token gate over the rendered PDF and writes
`{stem}_{ext}_visual_report.json` beside the deliverable (for example
`book_mono_pdf_visual_report.json`). Tiers: `geometry` (line overlap,
out-of-bounds), `type` (glyph size), `pixel` (blank page, black blocks), and
optionally `vlm`. The report's `skipped_reason`
names the tiers that ran and the ones that did not, so a missing tier is never
mistaken for a pass.

The `type` tier exists because a layout can lose every structural check while
being unreadable. An 83-page paper was once delivered at 22 pages with 10 pages
below 4pt and the last at 0.49pt: all 137k characters were present, nothing
overflowed, and no page was blank. It measures each page's rendered glyph
heights (letters, digits and ideographs only — punctuation ink is short, and a
TOC's leader dots measure 1.29pt on a healthy page) and reports:

- `type_illegible` — the page's median glyph is under 4pt. **Fails closed**,
  even with the opt-in blocking gate off and outside the short-document window:
  a document nobody can read is broken, not styled differently.
- `type_collapsed` — a majority of one page's glyphs are under 4pt while the
  median is readable. A `major`, so a legitimately small table warns instead of
  blocking.

Sizes come from rendered rect heights, not `FPDFText_GetFontSize`: that returns
a font *state* size which a text matrix can scale (one healthy file in the
corpus reports 1.00pt on pages that render at ~5.8pt).

## Known limits

Measured against the current implementation, not aspirational:

- **Best-supported input**: born-digital, white-background, single- or
  double-column text PDFs. The pdfium fast lane does not emit tables or
  figures, and the Docling path maps merged cells to the top-left value only.
- **Tables**: table cells are translated in the IR (and scored by QE), then
  re-typeset with translated headers. EPUB/DOCX/HTML/Markdown outputs carry the
  translated cells too.
- **Not supported**: vertical CJK, encrypted PDFs (refused before extraction, at
  both assessment and translate, rather than reported as a parse failure),
  colour/watermarked backgrounds (no background-colour probe, so
  light text on dark pages is invisible), and text drawn inside figures.
- **Text expansion**: there is no fixed box to overflow — the page reflows and
  pagination changes — so a translation is never shrunk to fit a source box.
- **Per format**: DOCX footnotes/endnotes, column settings and TOC fields are
  not read; a PDF up to 50 pages is ingested as one chapter, so per-chapter
  resume and billing do not apply to it.
- **Bilingual output**: `dual` pairs each block's source line above its
  translation; headings and tables pair too, while formulas come out
  monolingual. With `--emit-both`, a complementary mono/dual companion file is
  written alongside the primary artifact (off by default).
- **Exchange formats**: XLIFF 2.1 and TMX import exist but are not
  standards-complete (no TBX/SRX), so hand-off to external CAT tooling is
  limited.

Design notes and guides are maintained in the working tree under `docs/`, which
is intentionally unversioned (gitignored) — a clone carries the code and tests
only, not those notes.

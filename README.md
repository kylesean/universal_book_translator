# Universal Book Translator (UBT)

Pre-1.0 self-hosted document/book translation compiler: ingest PDF/EPUB/DOCX/
HTML/Markdown into one IR, translate it through a ledger-backed multi-stage LLM
pipeline, and re-typeset the result onto the source PDF page layout.

Consistency defenses: a mined book bible (terms and names) and a structural +
quality gate run on every job. Hierarchical chapter memory runs only on the
multi-chapter long route — a single-chapter document (which is how every PDF is
ingested), any document over 40 chapters, and academic profiles get no rolling
summary. Deterministic glossary enforcement is short-document only.

The render path has two modes. **`overlay`** (default) shrinks type inside
each source box and reflows within a column; it does not grow boxes or
repaginate, so it is reading-grade, not DTP-grade. **`publish`** rebuilds the
document from the IR and lets Typst lay it out, which is what makes uniform
body type, translated table headers and native math possible. See "Render
modes" below.

## Render modes

`--render-mode overlay` (default) or `--render-mode publish` (env:
`UBT_RENDER_MODE`). Selectable from the CLI, the REST API, MCP and the console's
new-job wizard (Advanced options).

| | `overlay` | `publish` |
| :--- | :--- | :--- |
| Layout | each translated fragment drawn back onto its source page | document rebuilt; Typst paginates |
| Body size | fitted per box, 4pt–12pt range | one size (10.5pt) |
| Tables | left on the source layer (source-language headers) | re-typeset, headers translated |
| Formulas | kept as source geometry | converted LaTeX→Typst math |
| Use when | the layout must match the original page | the translation is the deliverable |
| Pages (arXiv 2101.03961) | 41 | 29 |

Measured on that reference paper: `publish` raises CJK coverage from 31% to
48%, removes all 17 untranslated-block markers, and cuts distinct font sizes
from 44 to 5. It also keeps the bibliography verbatim, which the overlay path
leaves in place too — the CJK ratio is lower than a naive figure because the
kept-in-source references are counted in the denominator.

Both modes are reading-grade. `publish` fixes the table and font-size limits
below; neither is a substitute for human post-editing.

## Known limits

Measured against the current implementation, not aspirational:

- **Best-supported input**: born-digital, white-background, single- or
  double-column text PDFs. The pdfium fast lane does not emit tables or
  figures, and the Docling path maps merged cells to the top-left value only.
- **Tables**: table cells are translated in the IR (and scored by QE). Under
  `--render-mode overlay` the PDF leaves tables on the source layer — the
  compositor cannot rebuild a grid, so a delivered PDF shows source-language
  tables. Under `--render-mode publish` tables are re-typeset with translated
  headers. EPUB/DOCX/HTML/Markdown outputs carry the translated cells either
  way.
- **Not supported**: vertical CJK, encrypted PDFs (rejected at assessment, not
  translated), colour/watermarked backgrounds (no background-colour probe, so
  light text on dark pages is invisible), and text drawn inside figures.
- **Text expansion**: under `overlay`, a translation that will not fit its
  source box at a readable 6pt floor is drawn smaller (down to a 4pt hard
  floor) and flagged `low_legibility_font`; below 4pt the source text is kept
  (descends). There is no box growth or page re-flow. `publish` has no such
  limit — text reflows and pagination changes — so prefer it whenever the
  translation matters more than matching the source page.
- **Per format**: DOCX footnotes/endnotes, column settings and TOC fields are
  not read; a whole PDF is ingested as one chapter, so per-chapter resume and
  billing do not apply to it.
- **Bilingual output**: in-place bilingual re-typesets the source in grey
  rather than preserving its original layout; headings and formulas come out
  monolingual. Page-facing and alternating modes are the reliable bilingual
  routes.
- **Exchange formats**: XLIFF 2.1 and TMX import exist but are not
  standards-complete (no TBX/SRX), so hand-off to external CAT tooling is
  limited.

Design notes and guides are maintained in the working tree under `docs/`, which
is intentionally unversioned (gitignored) — a clone carries the code and tests
only, not those notes.

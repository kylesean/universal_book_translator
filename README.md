# Universal Book Translator (UBT)

Pre-1.0 self-hosted document/book translation compiler: ingest PDF/EPUB/DOCX/
HTML/Markdown into one IR, translate it through a ledger-backed multi-stage LLM
pipeline, and re-typeset the result onto the source PDF page layout.

Consistency defenses: a mined book bible (terms and names) and a structural +
quality gate run on every job. Hierarchical chapter memory runs only on the
multi-chapter long route — a single-chapter document (which is how every PDF is
ingested), any document over 40 chapters, and academic profiles get no rolling
summary. Deterministic glossary enforcement is short-document only.

The render path shrinks type inside each source box and reflows within a
column; it does not grow boxes or repaginate, so it is reading-grade, not
DTP-grade. See "Known limits" below.

## Known limits

Measured against the current implementation, not aspirational:

- **Best-supported input**: born-digital, white-background, single- or
  double-column text PDFs. The pdfium fast lane does not emit tables or
  figures, and the Docling path maps merged cells to the top-left value only.
- **Tables**: table cells are translated in the IR (and scored by QE), but the
  PDF re-typeset leaves tables on the source layer — the compositor cannot
  rebuild a grid, so a delivered PDF shows source-language tables. EPUB/DOCX/
  HTML/Markdown outputs carry the translated cells.
- **Not supported**: vertical CJK, encrypted PDFs (rejected at assessment, not
  translated), colour/watermarked backgrounds (no background-colour probe, so
  light text on dark pages is invisible), and text drawn inside figures.
- **Text expansion**: a translation that will not fit its source box at a
  readable 6pt floor keeps the source text (descends); there is no box growth
  or page re-flow.
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

# Universal Book Translator (UBT)

Pre-1.0 self-hosted document/book translation compiler: ingest PDF/EPUB/DOCX/
HTML/Markdown into one IR, translate it through a ledger-backed multi-stage LLM
pipeline, and re-typeset the result as a new document laid out by Typst.

Consistency defenses: a mined book bible (terms and names) and a structural +
quality gate run on every job. Hierarchical chapter memory runs only on the
multi-chapter long route — a single-chapter document (which is how every PDF is
ingested), any document over 40 chapters, and academic profiles get no rolling
summary. Deterministic glossary enforcement is short-document only.

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

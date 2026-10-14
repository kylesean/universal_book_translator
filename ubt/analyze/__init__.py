"""``ubt.analyze`` -- turning sources into the typed Document AST.

This layer owns *source understanding*: the walkers, classification rules,
reading-order numbering and span assembly that decide a document's structure.
Two consumers build on it:

* the **format adapters** (``ubt.adapters.*``) are the production path. They
  parse a source into the pipeline's ``IRBlock`` contract by calling this
  layer's shared components -- :func:`~ubt.analyze.reader_html.walk_markup_elements`,
  :func:`~ubt.analyze.reader_docx.walk_docx_elements`, :func:`~ubt.analyze.assemble.number`
  and the classifier rules in :mod:`ubt.analyze.structure` -- so the DOM walk
  and the id space have one owner, not one per format.

* the **``read_*`` seam** (``read_pdf``/``read_md``/``read_html``/``read_epub``/
  ``read_docx``) builds a typed :class:`~ubt.model.ast.Document` *directly*, with
  real ``Span.chars`` into a :class:`~ubt.model.span.CanonicalSource`, dropping
  the adapter's DOM handles. The bridge (:mod:`ubt.analyze.bridge`) projects a
  block list into that same model and back, and the round-trip is the proof the
  model loses no document structure. This seam is exercised by the reader tests
  and by ``scripts/shadow_reader.py``'s coverage gate; the PDF reader also backs
  production through :func:`~ubt.analyze.reader_pdf.read_pdf_blocks`.

Native readers landed one at a time (incremental native reader dispatch):
``read_pdf`` for born-digital geometry, then ``read_md`` for Markdown/plain text
and ``read_html``/``read_epub``/``read_docx`` for markup and office formats.
Because the adapters and the seam share this layer's components, a format never
has to detour through a second, drifted representation to be understood.
"""

from __future__ import annotations

from ubt.analyze.bridge import (
    block_from_element,
    blocks_from_document,
    document_from_blocks,
    element_from_block,
)
from ubt.analyze.reader_docx import read_docx
from ubt.analyze.reader_epub import read_epub
from ubt.analyze.reader_html import read_html
from ubt.analyze.reader_md import read_md
from ubt.analyze.reader_pdf import read_pdf

__all__ = [
    "block_from_element",
    "blocks_from_document",
    "document_from_blocks",
    "element_from_block",
    "read_docx",
    "read_epub",
    "read_html",
    "read_md",
    "read_pdf",
]

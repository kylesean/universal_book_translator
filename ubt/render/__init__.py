"""``ubt.render`` -- lowering delivered blocks to secondary views.

:func:`compose_html` and :func:`compose_epub` lower the delivered blocks to
semantic HTML and EPUB 3 views. The readable PDF path
(``ubt/render/typst/document_renderer.py``) rebuilds the document and lets Typst
lay it out; born-digital 1:1 in-place delivery belongs to the native PDF codec
(``ubt.codecs.pdf``), whose retired Typst facsimile predecessors no longer live
in this package.
"""

from __future__ import annotations

from ubt.render.epub_view import compose_epub
from ubt.render.html_view import compose_html
from ubt.render.outputs import Composition, Placement

__all__ = [
    "Composition",
    "Placement",
    "compose_epub",
    "compose_html",
]

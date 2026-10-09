"""``ubt.render`` -- lowering translated blocks onto source pages (render compositor).

:class:`ubt.render.outputs.LayerCompositor` composes each page from three absolute
layers (source canvas, mask, typeset fragment) and records a :class:`Placement`
per element. :func:`compose_html` and :func:`compose_epub` lower the same
delivered blocks to semantic HTML and EPUB 3 views.
"""

from __future__ import annotations

from ubt.render.epub_view import compose_epub
from ubt.render.html_view import compose_html
from ubt.render.outputs import Composition, Placement, overlays_from_blocks

__all__ = [
    "Composition",
    "Placement",
    "compose_epub",
    "compose_html",
    "overlays_from_blocks",
]

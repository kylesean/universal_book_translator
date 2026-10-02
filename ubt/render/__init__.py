"""``ubt.render`` -- lowering backends and their capability declarations (render backend lowering layer).

A backend turns one element into a realization at a fidelity it declares it can
produce; :func:`ubt.pipeline.steps.realize` negotiates which. The concrete
backends (Typst, Overlay) land in later steps; this package starts with the
seam they plug into.
"""

from __future__ import annotations

from ubt.render.capability import Backend, Capabilities, Produced
from ubt.render.epub_view import compose_epub
from ubt.render.html_view import compose_html
from ubt.render.outputs import Composition, LoweringUnsupported, Placement, compose
from ubt.render.overlay_backend import OverlayBackend
from ubt.render.typst_backend import TypstBackend

__all__ = [
    "Backend",
    "Capabilities",
    "Composition",
    "LoweringUnsupported",
    "OverlayBackend",
    "Placement",
    "Produced",
    "TypstBackend",
    "compose",
    "compose_epub",
    "compose_html",
]

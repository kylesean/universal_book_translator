"""Single-page preview: re-compose one source page with the current ledger text.

The delivery render composes every PDF onto the source page as its canvas
(``LayerCompositor``; the single ``overlay`` engine). This module reuses that exact
pipeline for a *single* page, so the L3 workbench can show the effect of a human
edit without re-rendering the book: overlays are built from the current blocks,
restricted to the requested page, reflowed, composed onto the source PDF, and
rasterized to PNG.

Cost is one page's worth of Typst fragment compiles, not the whole document:
``overlays_from_blocks`` is pure Python (no compiles), and the page filter runs
before ``reflow_overlays``/``compose``, which are the parts that shell out to
Typst.

Two deliberate differences from a full delivery render, both fine for a preview
and both worth stating:

* ``realization_plan`` is not threaded through, so the default fidelity floor
  applies. A block the real run held below its floor may appear here.
* the composition is always in-place on the source page (target over source).
  For the page-zipper bilingual modes (``alternating``/``facing``) the delivered
  artifact interleaves source and target pages, so page *N* of the preview is the
  source page *N* composed with its translation — not the delivered page *N*.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ubt.adapters.pdf.oxide_render import render_page_png
from ubt.core.ir.models import IRBlock

logger = logging.getLogger(__name__)


class PagePreviewUnavailable(RuntimeError):
    """The page cannot be previewed (no source canvas, no blocks, no rasterizer)."""


def _block_page(block: IRBlock) -> int | None:
    bbox = block.bbox
    return bbox.page if bbox is not None else None


def render_source_page_png(source_pdf: Path, page: int, dpi: int = 110) -> bytes:
    """Rasterize the *source* PDF's page ``page`` (the "before" witness image).

    The pixel-witness view pairs this with :func:`render_page_preview` (the
    "after") so a reviewer can compare the original page against the composed
    translation. Raises :class:`PagePreviewUnavailable` when the file is missing
    or the rasterizer cannot run, so the caller answers 503 rather than serving a
    broken image.
    """
    if not source_pdf.exists():
        raise PagePreviewUnavailable(f"source PDF not found: {source_pdf}")
    if page < 1:
        raise PagePreviewUnavailable("page must be >= 1")
    png = render_page_png(source_pdf, page, dpi)
    if png is None:
        raise PagePreviewUnavailable("source page rasterization failed (pdf_oxide unavailable)")
    return png


def render_page_preview(
    *,
    source_pdf: Path,
    blocks: list[IRBlock],
    target_lang: str,
    page: int,
    workdir: Path,
    bilingual: bool = False,
    dpi: int = 110,
    font: str | None = None,
) -> bytes:
    """Re-compose ``page`` onto the source canvas and return it as PNG bytes.

    Raises :class:`PagePreviewUnavailable` when the page has nothing to compose
    or the rasterizer is unavailable, so the caller can answer 503 rather than
    returning a broken image.
    """
    from ubt.adapters.pdf.docling_render import _reflow_obstacles
    from ubt.layout.theme import resolve_theme
    from ubt.render.outputs import LayerCompositor, TypstFragmentTypesetter, overlays_from_blocks
    from ubt.render.reflow import reflow_overlays

    if not source_pdf.exists():
        raise PagePreviewUnavailable(f"source PDF not found: {source_pdf}")

    page_blocks = [block for block in blocks if _block_page(block) == page]
    if not page_blocks:
        raise PagePreviewUnavailable(f"no blocks on page {page}")

    # Overlay construction is pure; the page filter keeps the compile-heavy
    # reflow/compose steps to this page only.
    overlays = [
        overlay
        for overlay in overlays_from_blocks(list(blocks), None, bilingual=bilingual)
        if overlay.page == page
    ]
    if not overlays:
        raise PagePreviewUnavailable(f"page {page} has nothing to compose")

    theme = resolve_theme(target_lang=target_lang)
    typesetter = TypstFragmentTypesetter(
        font=font or theme.font_stack,
        size_pt=theme.base_size_pt,
        target_lang=target_lang,
    )
    workdir.mkdir(parents=True, exist_ok=True)
    composed_pdf = workdir / f"preview_p{page}.pdf"
    try:
        reflowed = reflow_overlays(
            overlays,
            _reflow_obstacles(page_blocks),
            measure_many=typesetter.measure_many_fixed,
            cap_size=typesetter.cap_size,
        )
        LayerCompositor(source_pdf, typesetter=typesetter).compose(reflowed, composed_pdf)
    finally:
        typesetter.close()

    png = render_page_png(composed_pdf, page, dpi)
    if png is None:
        raise PagePreviewUnavailable("page rasterization failed (pdf_oxide unavailable)")
    return png

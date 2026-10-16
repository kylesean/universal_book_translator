"""Single-page preview: render one source page's blocks with the live ledger text.

The delivery render rebuilds the whole document from the IR and lets Typst lay
it out (``ubt/render/typst/document_renderer.py``). This module reuses that
renderer for a *single page's* blocks, so the L3 workbench can show the effect of
a human edit without re-rendering the book.

Two deliberate differences from a full delivery render, both fine for a preview
and both worth stating:

* the preview renders the page's blocks as a standalone document, so pagination
  differs from the delivered book (a reflowed page is not a fixed grid). The
  reviewer is checking *wording*, and the words are the same ones the delivery
  render would typeset.
* a block the real run held as source (a fail-closed render skip) may appear
  here as a placed target, because the preview renders from the current ledger
  text rather than the recorded render decisions.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ubt.core.ir.models import IRBlock

logger = logging.getLogger(__name__)


class PagePreviewUnavailable(RuntimeError):
    """The page cannot be previewed (no blocks, no rasterizer, no such page)."""


def _block_page(block: IRBlock) -> int | None:
    bbox = block.bbox
    return bbox.page if bbox is not None else None


def render_source_page_png(source_pdf: Path, page: int, dpi: int = 110) -> bytes:
    """Rasterize the *source* PDF's page ``page`` (the "before" witness image).

    The pixel-witness view pairs this with :func:`render_page_preview` (the
    "after") so a reviewer can compare the original page against the reflowed
    translation. Raises :class:`PagePreviewUnavailable` when the file is missing
    or the rasterizer cannot run, so the caller answers 503 rather than serving a
    broken image.
    """
    if not source_pdf.exists():
        raise PagePreviewUnavailable(f"source PDF not found: {source_pdf}")
    if page < 1:
        raise PagePreviewUnavailable("page must be >= 1")
    # Function-level: the oxide rasterizer lives in the PDF adapter stack (it
    # is the page-raster leg of ingestion), and a module-level import made
    # ubt/render reach back into ubt/adapters — the layering acceptance tool
    # (dump_compiler_edges) forbids that edge. The lazily-resolved attribute
    # keeps preview testable with a monkeypatched module function.
    from ubt.adapters.pdf.oxide_render import render_page_png

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
    """Render ``page``'s blocks through the renderer as PNG bytes.

    Raises :class:`PagePreviewUnavailable` when the page has no blocks or the
    rasterizer is unavailable, so the caller can answer 503 rather than returning
    a broken image.
    """
    from ubt.adapters.pdf.typst_compile import typst_compile
    from ubt.adapters.pdf.typst_math_probe import TypstMathProbe
    from ubt.render.typst.document_renderer import TypstDocumentRenderer

    if not source_pdf.exists():
        raise PagePreviewUnavailable(f"source PDF not found: {source_pdf}")

    page_blocks = [block for block in blocks if _block_page(block) == page]
    if not page_blocks:
        raise PagePreviewUnavailable(f"no blocks on page {page}")

    workdir.mkdir(parents=True, exist_ok=True)
    build_dir = workdir / f"preview_p{page}"
    # The compile gate the delivery renderer uses (see docling_render). Without
    # it a malformed formula that the real renderer would degrade to readable
    # text instead reaches this compile verbatim and fails the *whole* preview
    # (503), so the workbench could not show a page the deliverable renders fine.
    # One probe instance: ``check_many`` pre-warms the same cache ``check`` reads.
    math_probe = TypstMathProbe()
    renderer = TypstDocumentRenderer(
        bilingual=bilingual,
        math_gate=math_probe.check,
        math_gate_many=math_probe.check_many,
        font_stack=(font,) if font else None,
    )
    source, _stats = renderer.build_source(page_blocks, build_dir, target_lang)
    typ_path = build_dir / "main.typ"
    typ_path.write_text(source, encoding="utf-8")

    composed_pdf = workdir / f"preview_p{page}.pdf"
    ok, stderr = typst_compile(str(typ_path), str(composed_pdf))
    if not ok or not composed_pdf.exists():
        raise PagePreviewUnavailable(
            f"page {page} preview failed to compile: {stderr.strip() or 'unknown Typst error'}"
        )

    # Function-level: same adapters-edge rule as render_source_page_png above.
    from ubt.adapters.pdf.oxide_render import render_page_png

    png = render_page_png(composed_pdf, 1, dpi)
    if png is None:
        raise PagePreviewUnavailable("page rasterization failed (pdf_oxide unavailable)")
    return png

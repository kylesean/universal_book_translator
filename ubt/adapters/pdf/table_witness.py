"""Deterministic visual witness for reconstructed tables (level 2 of 2).

Level 1 (``ubt.core.content.asset_verify.verify_table_structure``) judges the
Markdown/HTML *text* of a table. It catches shattering, but a structurally
plausible grid can still reflow into the wrong shape — columns collapsed, rows
dropped, the whole table stretched. The ground truth is the rendered page:
this module rasterizes the emitted Typst table and compares its structure
(aspect, connected-component magnitude) against the source crop, exactly as
:mod:`ubt.adapters.pdf.formula_witness` does for equations.

Like the formula witness it is zero-token and fail-open: any measurement
problem returns ``unwitnessable`` and the caller keeps the reconstruction. A
table that fails is swapped for its source graphic by
``TypstReconstructor._crop_table_fallback`` (lossless, Axiom A).

The tolerances are wider than the formula band on purpose: a Typst booktabs
table legitimately differs from the source's ruled grid in stroke weight,
padding and font, so only a *gross* deviation is corruption.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.adapters.pdf.formula_witness import (
    WITNESS_DPI,
    WitnessResult,
    compare_structure,
    rasterize_typst,
)

if TYPE_CHECKING:
    from ubt.core.ir.models import IRBlock

logger = logging.getLogger(__name__)

# A correct booktabs reconstruction keeps roughly the source's shape and glyph
# mass. These bands are deliberately wide: a reflowed table legitimately
# reshapes (Typst's padding/fonts run the grid taller than the source's dense
# rules), so the witness must reject a *collapsed or exploded* grid, not report
# legitimate reflow. Calibration on arXiv 2609.20519's four tables: aspect
# 0.29-0.39x, components 1.2-1.7x of source -- all inside the band, while a
# broken probe (auto page width) collapsed to 0.03x and failed.
TABLE_MIN_ASPECT_RATIO = 0.15
TABLE_MAX_ASPECT_RATIO = 8.0
TABLE_MIN_COMPONENT_RATIO = 0.10
TABLE_MAX_COMPONENT_RATIO = 8.0


def witness_table(
    table_markup: str,
    block: IRBlock,
    source_pdf: Path | str,
    typst_binary: str,
    dpi: int = WITNESS_DPI,
) -> WitnessResult:
    """Compare one emitted table against its source crop (level-2, zero-token).

    Never raises: every measurement failure — missing bbox, unavailable crop, a
    rasterizer crash, unmeasurable pixels — yields ``unwitnessable`` and the
    caller keeps its reconstruction. The witness must never break a render.
    """
    if block.bbox is None or block.bbox.page <= 0:
        return WitnessResult("unwitnessable", ["no source bounding box"])
    try:
        from ubt.adapters.pdf.visual_scalpel import crop_block_pil

        source_img = crop_block_pil(source_pdf, block.bbox.page, block.bbox, dpi=dpi, bleed_pt=0.0)
    except Exception as exc:  # witness must never break a render
        logger.debug("Table witness source crop unavailable for %s: %s", block.id, exc)
        return WitnessResult("unwitnessable", ["source crop unavailable"])
    try:
        source_width_pt = float(block.bbox.x1) - float(block.bbox.x0)
        rendered_img = rasterize_typst(
            table_markup, typst_binary, dpi=dpi, page_width_pt=source_width_pt
        )
        if rendered_img is None:
            return WitnessResult("unwitnessable", ["emitted table did not rasterize"])
        findings = compare_structure(
            rendered_img,
            source_img,
            min_aspect=TABLE_MIN_ASPECT_RATIO,
            max_aspect=TABLE_MAX_ASPECT_RATIO,
            min_component=TABLE_MIN_COMPONENT_RATIO,
            max_component=TABLE_MAX_COMPONENT_RATIO,
        )
    except Exception as exc:  # witness must never break a render
        logger.debug("Table witness rasterize/compare failed for %s: %s", block.id, exc)
        return WitnessResult("unwitnessable", ["rasterize/compare unavailable"])
    if not findings:
        return WitnessResult("pass")
    if findings == ["unwitnessable"]:
        return WitnessResult("unwitnessable", findings)
    return WitnessResult("fail", findings)


__all__ = ["witness_table"]

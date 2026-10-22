"""Overlapping formula crops merge without ever concatenating asset paths.

The merge exists because Docling emits two vertically overlapping boxes for one
equation, and rendering both duplicates the region on the page. It only ever
merged *text* correctly: an ``IMAGE`` block's ``source_text`` is an asset path,
so the old ``f"{prev}\n{cur}"`` produced a value that named no file and the
renderer drew "[missing figure]" where the equation belonged.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.docling_blocks import resolve_overlapping_formula_blocks
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element
from ubt.model.ast import Figure

pytestmark = pytest.mark.fast


def _block(block_type: BlockType, payload: str, y0: float, y1: float) -> IRBlock:
    return IRBlock(
        element=make_element(
            id="pdf_main#x",
            spine_index=1,
            block_type=block_type,
            flow_id=FlowID.MAIN_STORY,
            source_text=payload,
            bbox=BoundingBox(page=1, x0=10.0, y0=y0, x1=200.0, y1=y1),
            skip_translate=True,
        )
    )


def test_two_crops_merge_to_one_and_keep_a_real_path() -> None:
    """Both sides are crops: the survivor must name exactly one file."""
    merged = resolve_overlapping_formula_blocks(
        [
            _block(BlockType.IMAGE, "/cache/a.png", 100.0, 140.0),
            _block(BlockType.IMAGE, "/cache/b.png", 120.0, 160.0),
        ]
    )
    assert len(merged) == 1
    element = merged[0].element
    assert isinstance(element, Figure)
    asset = element.asset_id
    assert "\n" not in asset, f"merged asset names no file: {asset!r}"
    assert asset == "/cache/a.png"


def test_a_read_formula_wins_over_an_overlapping_crop() -> None:
    """Native math re-typesets; a raster of the same region cannot."""
    merged = resolve_overlapping_formula_blocks(
        [
            _block(BlockType.IMAGE, "/cache/a.png", 100.0, 140.0),
            _block(BlockType.FORMULA, "E = mc^2", 120.0, 160.0),
        ]
    )
    assert len(merged) == 1
    assert merged[0].block_type is BlockType.FORMULA
    assert merged[0].source_text == "E = mc^2"


def test_a_crop_after_a_read_formula_does_not_corrupt_the_latex() -> None:
    merged = resolve_overlapping_formula_blocks(
        [
            _block(BlockType.FORMULA, "E = mc^2", 100.0, 140.0),
            _block(BlockType.IMAGE, "/cache/b.png", 120.0, 160.0),
        ]
    )
    assert len(merged) == 1
    assert merged[0].block_type is BlockType.FORMULA
    assert merged[0].source_text == "E = mc^2"


def test_two_read_formulas_still_concatenate_their_text() -> None:
    """The original behaviour, which was right for two text formulas."""
    merged = resolve_overlapping_formula_blocks(
        [
            _block(BlockType.FORMULA, "a + b", 100.0, 140.0),
            _block(BlockType.FORMULA, "c + d", 120.0, 160.0),
        ]
    )
    assert len(merged) == 1
    assert merged[0].source_text == "a + b\nc + d"


def test_non_overlapping_blocks_are_untouched() -> None:
    merged = resolve_overlapping_formula_blocks(
        [
            _block(BlockType.IMAGE, "/cache/a.png", 100.0, 110.0),
            _block(BlockType.IMAGE, "/cache/b.png", 300.0, 310.0),
        ]
    )
    assets = []
    for block in merged:
        element = block.element
        assert isinstance(element, Figure)
        assets.append(element.asset_id)
    assert assets == ["/cache/a.png", "/cache/b.png"]

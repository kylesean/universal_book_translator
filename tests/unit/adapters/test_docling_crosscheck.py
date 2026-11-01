from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ubt.adapters.pdf.docling_crosscheck import (
    DEFAULT_MIN_IOU,
    cross_check_blocks_with_pdfium,
)
from ubt.adapters.pdf.textgeom import LineBox
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element
from ubt.model.ast import Confidence

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    text: str,
    *,
    page: int = 1,
    spine: int = 1,
    block_type: BlockType = BlockType.NARRATIVE,
    x0: float = 50.0,
    y0: float = 600.0,
    x1: float = 300.0,
    y1: float = 700.0,
) -> IRBlock:
    return IRBlock(
        element=make_element(
            block_type=block_type,
            id=block_id,
            source_text=text,
            spine_index=spine,
            bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page),
        )
    )


def test_high_iou_blocks_are_preserved() -> None:
    block = _block("b1", "Clean body paragraph text.", x0=50.0, y0=600.0, x1=300.0, y1=700.0)
    mock_lines = [
        LineBox(text="Clean body paragraph", rect=(52.0, 602.0, 298.0, 698.0)),
    ]

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        return_value=(mock_lines, (600.0, 800.0)),
    ):
        result = cross_check_blocks_with_pdfium([block], Path("dummy.pdf"))

    assert len(result) == 1
    assert result[0].id == "b1"
    assert result[0].skip_translate is False
    assert result[0].element.confidence is not Confidence.UNKNOWN


def test_phantom_block_with_no_physical_lines_is_sunk_to_opaque() -> None:
    block = _block(
        "b1",
        "A hallucinated paragraph over an empty white area.",
        x0=50.0,
        y0=600.0,
        x1=300.0,
        y1=700.0,
    )
    mock_lines = [
        LineBox(text="Header text at the very top", rect=(50.0, 750.0, 200.0, 780.0)),
    ]

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        return_value=(mock_lines, (600.0, 800.0)),
    ):
        result = cross_check_blocks_with_pdfium([block], Path("dummy.pdf"))

    assert len(result) == 1
    demoted = result[0]
    assert demoted.skip_translate is True
    assert demoted.element.confidence is Confidence.UNKNOWN
    assert "no_physical_lines" in (demoted.provenance.iou_crosscheck or "")


def test_low_iou_displaced_box_is_sunk_to_opaque() -> None:
    block = _block("b1", "Sprawling hallucinated box", x0=50.0, y0=100.0, x1=300.0, y1=600.0)
    mock_lines = [
        LineBox(text="Tiny line in corner", rect=(50.0, 100.0, 100.0, 150.0)),
    ]

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        return_value=(mock_lines, (600.0, 800.0)),
    ):
        result = cross_check_blocks_with_pdfium([block], Path("dummy.pdf"), min_iou=DEFAULT_MIN_IOU)

    assert len(result) == 1
    demoted = result[0]
    assert demoted.skip_translate is True
    assert demoted.element.confidence is Confidence.UNKNOWN
    assert "iou=" in (demoted.provenance.iou_crosscheck or "")


def test_a_caption_with_a_low_iou_box_stays_translatable() -> None:
    block = _block(
        "b1",
        "Figure 2 | The distribution of the number of sandboxes created per task.",
        x0=50.0,
        y0=100.0,
        x1=300.0,
        y1=140.0,
    )
    mock_lines = [
        LineBox(text="Tiny line in corner", rect=(50.0, 100.0, 100.0, 120.0)),
    ]

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        return_value=(mock_lines, (600.0, 800.0)),
    ):
        result = cross_check_blocks_with_pdfium([block], Path("dummy.pdf"), min_iou=DEFAULT_MIN_IOU)

    assert len(result) == 1
    assert result[0].skip_translate is False
    assert result[0].policy_translate is None


def test_a_tight_box_under_an_overshooting_witness_is_preserved() -> None:
    block = _block(
        "b1", "A caption the witness line overshoots.", x0=50.0, y0=600.0, x1=250.0, y1=700.0
    )
    mock_lines = [
        LineBox(text="A caption the witness line overshoots", rect=(0.0, 600.0, 600.0, 700.0)),
    ]

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        return_value=(mock_lines, (600.0, 800.0)),
    ):
        result = cross_check_blocks_with_pdfium([block], Path("dummy.pdf"), min_iou=DEFAULT_MIN_IOU)

    assert len(result) == 1
    assert result[0].skip_translate is False
    assert result[0].element.confidence is not Confidence.UNKNOWN


def test_a_half_covered_box_is_still_sunk_to_opaque() -> None:
    block = _block("b1", "Half the box has no ink.", x0=50.0, y0=600.0, x1=250.0, y1=700.0)
    mock_lines = [
        LineBox(text="Half the box", rect=(50.0, 650.0, 250.0, 700.0)),
    ]

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines",
        return_value=(mock_lines, (600.0, 800.0)),
    ):
        result = cross_check_blocks_with_pdfium([block], Path("dummy.pdf"), min_iou=DEFAULT_MIN_IOU)

    assert len(result) == 1
    assert result[0].skip_translate is True
    assert result[0].element.confidence is Confidence.UNKNOWN


def test_scanned_textless_pages_skip_crosscheck() -> None:
    block = _block("b1", "Transcribed OCR text", x0=50.0, y0=600.0, x1=300.0, y1=700.0)

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines", return_value=([], (600.0, 800.0))
    ):
        result = cross_check_blocks_with_pdfium([block], Path("dummy.pdf"))

    assert len(result) == 1
    assert result[0].skip_translate is False
    assert result[0].element.confidence is not Confidence.UNKNOWN


def test_non_prose_blocks_are_not_checked() -> None:
    table = _block("t1", "a | b", block_type=BlockType.TABLE)
    image = _block("i1", "", block_type=BlockType.IMAGE)

    with patch(
        "ubt.adapters.pdf.docling_crosscheck.extract_lines", return_value=([], (600.0, 800.0))
    ):
        result = cross_check_blocks_with_pdfium([table, image], Path("dummy.pdf"))

    assert len(result) == 2
    assert result[0].id == "t1"
    assert result[1].id == "i1"


def test_page_caches_are_evicted_as_the_page_advances() -> None:
    pages_seen: list[int] = []

    def fake_extract_lines(_path: object, page: int) -> tuple[list[LineBox], tuple[float, float]]:
        pages_seen.append(page)
        return [LineBox(text="body", rect=(50.0, 600.0, 300.0, 700.0))], (600.0, 800.0)

    b1 = _block("b1", "page one text", page=1)
    b2 = _block("b2", "page two text", page=2)
    b3 = _block("b3", "page one again", page=1)

    with (
        patch(
            "ubt.adapters.pdf.docling_crosscheck.extract_lines",
            side_effect=fake_extract_lines,
        ),
        patch("ubt.adapters.pdf.docling_crosscheck.extract_char_styles", return_value=[]),
    ):
        cross_check_blocks_with_pdfium([b1, b2, b3], Path("dummy.pdf"))

    assert pages_seen == [1, 2, 1]

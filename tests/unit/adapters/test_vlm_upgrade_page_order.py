"""A VLM proofread upgrade must not shove a page's table/figure to its tail."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ubt.adapters.pdf.docling_parser import vlm_fallback_missing_pages
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


def _make_block(
    block_type: BlockType,
    text: str,
    *,
    page: int = 1,
    spine_index: int = 1,
) -> IRBlock:
    elem = make_element(
        block_type=block_type,
        id=f"test#{spine_index}",
        source_text=text,
        spine_index=spine_index,
        bbox=BoundingBox(x0=10, y0=10, x1=100, y1=50, page=page),
    )
    return IRBlock(element=elem)


def test_a_leading_table_keeps_its_place_when_the_page_is_upgraded() -> None:
    # Page 1 is [table, short prose]; "weak" mode upgrades it. The transcribed
    # prose must be spliced into the prose slot, keeping the table above it —
    # not appended after it (which put the table at the page tail).
    table = _make_block(BlockType.TABLE, "a | b", page=1, spine_index=1)
    prose = _make_block(BlockType.NARRATIVE, "Short.", page=1, spine_index=2)
    fresh = _make_block(BlockType.NARRATIVE, "Transcribed prose.", page=1, spine_index=99)
    stats = MagicMock(matched=1, vlm_only=0, pdfium_only=0)
    driver = MagicMock(measured_boxes=True)

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1

    with (
        patch("pypdfium2.PdfDocument", return_value=mock_doc),
        patch(
            "ubt.adapters.pdf.vlm.registry.probe_effective_driver",
            return_value=("vlm", driver),
        ),
        patch(
            "ubt.adapters.pdf.vlm.transcribe.transcribe_page_to_blocks",
            return_value=([fresh], stats),
        ),
        patch.dict("os.environ", {"UBT_VLM_SCAN_FALLBACK": "weak"}),
    ):
        result = vlm_fallback_missing_pages(
            Path("scan.pdf"),
            [table, prose],
            ocr_mode="auto",
            allow_page_upload=True,
        )

    ids = [b.id for b in result]
    assert ids == [table.id, fresh.id]
    # The original prose was replaced, the table survived.
    assert prose.id not in ids

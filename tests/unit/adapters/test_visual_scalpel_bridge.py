from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element
from ubt.core.ports import is_visual_scalpel_applicable

pytestmark = pytest.mark.fast


def _block(
    *,
    block_type: BlockType = BlockType.NARRATIVE,
    mtqe_score: float | None = None,
    page: int = 1,
) -> IRBlock:
    return IRBlock(
        element=make_element(
            block_type=block_type,
            id="b1",
            source_text="text",
            spine_index=1,
            bbox=BoundingBox(x0=50.0, y0=100.0, x1=400.0, y1=150.0, page=page),
        ),
        mtqe_score=mtqe_score,
    )


def test_the_ports_bridge_does_not_raise_on_the_threshold_keyword(tmp_path: Path) -> None:
    pdf = tmp_path / "src.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    assert is_visual_scalpel_applicable(
        _block(block_type=BlockType.FORMULA), source_pdf_path=pdf, vision_threshold=0.6
    )


def test_the_threshold_still_gates_a_low_scoring_block(tmp_path: Path) -> None:
    pdf = tmp_path / "src.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    assert is_visual_scalpel_applicable(
        _block(mtqe_score=0.3), source_pdf_path=pdf, vision_threshold=0.6
    )
    assert not is_visual_scalpel_applicable(
        _block(mtqe_score=0.9), source_pdf_path=pdf, vision_threshold=0.6
    )


def test_a_non_pdf_source_is_never_applicable(tmp_path: Path) -> None:
    doc = tmp_path / "src.docx"
    doc.write_bytes(b"PK\x03\x04")

    assert not is_visual_scalpel_applicable(
        _block(block_type=BlockType.FORMULA), source_pdf_path=doc, vision_threshold=0.6
    )

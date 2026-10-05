"""Docling extraction fixes: ordered list markers and first-line indents."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.docling_parser import (
    annotate_first_line_indents,
    restore_ordered_markers,
)
from ubt.adapters.pdf.textgeom import LineBox
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element


def _para(eid: str, text: str, *, page: int = 1, y0: float = 0.0, y1: float = 20.0) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=eid,
            spine_index=int(eid[1:]),
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text=text,
            bbox=BoundingBox(page=page, x0=70.0, y0=y0, x1=500.0, y1=y1),
        )
    )


def test_a_run_of_numbered_paragraphs_becomes_list_items() -> None:
    blocks = [_para("b1", "(1) First item"), _para("b2", "(2) Second item")]
    out = restore_ordered_markers(blocks)
    assert [b.block_type for b in out] == [BlockType.LIST_ITEM, BlockType.LIST_ITEM]
    assert [getattr(b.element, "marker", "") for b in out] == ["(1)", "(2)"]
    assert out[0].source_text == "First item"


def test_a_lone_marker_paragraph_is_left_as_prose() -> None:
    # A bracketed year opens like a marker but a single one is not a list.
    blocks = [_para("b1", "(2024) A reference to prior work.")]
    out = restore_ordered_markers(blocks)
    assert out[0].block_type is BlockType.NARRATIVE


def test_bullet_paragraphs_are_not_retyped() -> None:
    blocks = [_para("b1", "• one"), _para("b2", "• two")]
    out = restore_ordered_markers(blocks)
    assert all(b.block_type is BlockType.NARRATIVE for b in out)


def test_a_first_line_indent_is_recorded_from_the_line_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_extract_lines(_path: Path, _page: int) -> tuple[list[LineBox], tuple[float, float]]:
        lines = [
            LineBox("indented first line", (87.8, 700.0, 500.0, 711.0), font_size=10.9),
            LineBox("second line", (70.4, 687.0, 500.0, 698.0), font_size=10.9),
            LineBox("third line", (70.4, 674.0, 500.0, 685.0), font_size=10.9),
        ]
        return lines, (612.0, 792.0)

    monkeypatch.setattr("ubt.adapters.pdf.textgeom.extract_lines", fake_extract_lines)

    block = _para("b1", "indented first line second line third line", y0=674.0, y1=711.0)
    out = annotate_first_line_indents([block], Path("/does/not/matter.pdf"))

    assert out[0].style is not None
    assert out[0].style.first_line_indent_pt == pytest.approx(17.4)


def test_a_flush_left_paragraph_gets_no_indent(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_extract_lines(_path: Path, _page: int) -> tuple[list[LineBox], tuple[float, float]]:
        lines = [
            LineBox("flush first line", (70.4, 700.0, 500.0, 711.0), font_size=10.9),
            LineBox("second line", (70.4, 687.0, 500.0, 698.0), font_size=10.9),
        ]
        return lines, (612.0, 792.0)

    monkeypatch.setattr("ubt.adapters.pdf.textgeom.extract_lines", fake_extract_lines)

    block = _para("b1", "flush first line second line", y0=687.0, y1=711.0)
    out = annotate_first_line_indents([block], Path("/does/not/matter.pdf"))
    assert out[0].style is None or out[0].style.first_line_indent_pt is None

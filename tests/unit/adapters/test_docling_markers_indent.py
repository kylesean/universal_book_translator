"""Docling extraction fixes: ordered list markers and first-line indents."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.docling_parser import (
    annotate_layout_metadata,
    restore_appendix_headings,
    restore_ordered_markers,
)
from ubt.adapters.pdf.textgeom import LineBox
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element
from ubt.model.span import CompositeSpan

pytestmark = pytest.mark.fast


def _para(
    eid: str,
    text: str,
    *,
    page: int = 1,
    y0: float = 0.0,
    y1: float = 20.0,
    x1: float = 500.0,
    block_type: BlockType = BlockType.NARRATIVE,
) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=eid,
            spine_index=int(eid[1:]),
            block_type=block_type,
            flow_id=FlowID.MAIN_STORY,
            source_text=text,
            bbox=BoundingBox(page=page, x0=70.0, y0=y0, x1=x1, y1=y1),
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
    out = annotate_layout_metadata([block], Path("/does/not/matter.pdf"))

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
    out = annotate_layout_metadata([block], Path("/does/not/matter.pdf"))
    assert out[0].style is None or out[0].style.first_line_indent_pt is None


def test_a_list_marker_column_indent_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hanging list: the marker line hangs right of the wrapped lines' margin.

    The block box starts at the wrapped lines' margin (the union), so without
    the recorded indent the compositor would draw "(1)" flush with the body
    margin where the source hangs it ~22pt to the right.
    """

    def fake_extract_lines(_path: Path, _page: int) -> tuple[list[LineBox], tuple[float, float]]:
        lines = [
            LineBox("(1) marker line text", (93.3, 700.0, 500.0, 711.0), font_size=10.9),
            LineBox("wrapped line", (71.1, 687.0, 500.0, 698.0), font_size=10.9),
            LineBox("wrapped line two", (71.2, 674.0, 500.0, 685.0), font_size=10.9),
        ]
        return lines, (612.0, 792.0)

    monkeypatch.setattr("ubt.adapters.pdf.textgeom.extract_lines", fake_extract_lines)

    block = _para(
        "b1",
        "(1) marker line text wrapped line",
        y0=674.0,
        y1=711.0,
        block_type=BlockType.LIST_ITEM,
    )
    out = annotate_layout_metadata([block], Path("/does/not/matter.pdf"))

    assert out[0].style is not None
    assert out[0].style.first_line_indent_pt == pytest.approx(22.2)


def test_a_centered_heading_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A centered title's short line hangs symmetrically inside its box."""

    def fake_extract_lines(_path: Path, _page: int) -> tuple[list[LineBox], tuple[float, float]]:
        lines = [
            LineBox("Short Title", (172.1, 710.0, 424.9, 738.0), font_size=17.0),
            LineBox(
                "the long full-width second title line", (71.0, 690.0, 524.0, 705.0), font_size=17.0
            ),
        ]
        return lines, (595.0, 842.0)

    monkeypatch.setattr("ubt.adapters.pdf.textgeom.extract_lines", fake_extract_lines)

    heading = _para(
        "b1",
        "Short Title the long full-width second title line",
        y0=690.0,
        y1=738.0,
        x1=524.0,
        block_type=BlockType.HEADING,
    )
    out = annotate_layout_metadata([heading], Path("/does/not/matter.pdf"))

    assert out[0].style is not None
    assert out[0].style.alignment == "center"
    assert out[0].style.first_line_indent_pt is None


def test_a_flush_left_heading_is_not_marked_centered(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_extract_lines(_path: Path, _page: int) -> tuple[list[LineBox], tuple[float, float]]:
        lines = [
            LineBox("Section header", (71.0, 710.0, 200.0, 725.0), font_size=12.0),
        ]
        return lines, (595.0, 842.0)

    monkeypatch.setattr("ubt.adapters.pdf.textgeom.extract_lines", fake_extract_lines)

    heading = _para("b1", "Section header", y0=710.0, y1=725.0, block_type=BlockType.HEADING)
    out = annotate_layout_metadata([heading], Path("/does/not/matter.pdf"))

    assert out[0].style is None or out[0].style.alignment is None


def test_a_multi_line_heading_records_its_line_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The title's own two line boxes are recorded on the block's span.

    The extractor keeps the source's line structure, so a later stage can see
    that the title breaks at the subtitle colon rather than at an arbitrary
    width-fill point.
    """

    def fake_extract_lines(_path: Path, _page: int) -> tuple[list[LineBox], tuple[float, float]]:
        lines = [
            LineBox("Short Title", (172.1, 710.0, 424.9, 738.0), font_size=17.0),
            LineBox(
                "the long full-width second title line",
                (71.0, 690.0, 524.0, 705.0),
                font_size=17.0,
            ),
        ]
        return lines, (595.0, 842.0)

    monkeypatch.setattr("ubt.adapters.pdf.textgeom.extract_lines", fake_extract_lines)

    heading = _para(
        "b1",
        "Short Title the long full-width second title line",
        y0=690.0,
        y1=738.0,
        x1=524.0,
        block_type=BlockType.HEADING,
    )
    out = annotate_layout_metadata([heading], Path("/does/not/matter.pdf"))

    span = out[0].element.span
    assert isinstance(span, CompositeSpan)
    assert len(span.boxes) == 2
    # Reading order is top-down: the first box is the higher line (larger y).
    assert span.boxes[0].bbox[1] > span.boxes[1].bbox[1]
    # The narrow first line (172→425) is preserved, not normalized to the box.
    assert span.boxes[0].bbox[0] == pytest.approx(172.1)
    assert span.boxes[1].bbox[2] == pytest.approx(524.0)
    assert out[0].style is not None and out[0].style.alignment == "center"
    # The chain is mirrored into the one provenance key the ledger rebuilds it
    # from; without it the export stage sees a single first-line span.
    assert [box["bbox"][1] for box in out[0].provenance.physical_boxes] == [
        pytest.approx(710.0),
        pytest.approx(690.0),
    ]


def test_a_multi_line_heading_chain_survives_the_ledger_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The export stage reloads blocks from the ledger; the chain must come back.

    A heading read back as a single first-line box renders its translation
    squeezed into that one line and leaves the rest of the source title in the
    source language, so the round trip is part of the contract.
    """
    import tempfile

    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.ledger_base import _upsert_blocks_batch

    def fake_extract_lines(_path: Path, _page: int) -> tuple[list[LineBox], tuple[float, float]]:
        return [
            LineBox("Short Title", (172.1, 710.0, 424.9, 738.0), font_size=17.0),
            LineBox(
                "a long full-width second title line", (71.0, 690.0, 524.0, 705.0), font_size=17.0
            ),
        ], (595.0, 842.0)

    monkeypatch.setattr("ubt.adapters.pdf.textgeom.extract_lines", fake_extract_lines)

    heading = _para(
        "b1",
        "Short Title a long full-width second title line",
        y0=690.0,
        y1=738.0,
        x1=524.0,
        block_type=BlockType.HEADING,
    )
    out = annotate_layout_metadata([heading], Path("/does/not/matter.pdf"))

    ledger = SQLiteJobLedger(Path(tempfile.mkdtemp()) / "l.sqlite")
    with ledger._get_conn() as conn:
        conn.execute(
            "INSERT INTO job_meta(job_id,doc_id,source_path,target_lang,total_blocks,status)"
            " VALUES(?,?,?,?,?,?)",
            ("job_x", "doc", "/x.pdf", "zh", 1, "running"),
        )
        _upsert_blocks_batch(conn.cursor(), "job_x", out)
    (reloaded,) = ledger.get_all_blocks("job_x")

    span = reloaded.element.span
    assert isinstance(span, CompositeSpan)
    assert [box.bbox[1] for box in span.boxes] == [pytest.approx(710.0), pytest.approx(690.0)]


def test_a_single_line_heading_keeps_its_simple_span(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_extract_lines(_path: Path, _page: int) -> tuple[list[LineBox], tuple[float, float]]:
        lines = [
            LineBox("Section header", (71.0, 710.0, 200.0, 725.0), font_size=12.0),
        ]
        return lines, (595.0, 842.0)

    monkeypatch.setattr("ubt.adapters.pdf.textgeom.extract_lines", fake_extract_lines)

    heading = _para("b1", "Section header", y0=710.0, y1=725.0, block_type=BlockType.HEADING)
    out = annotate_layout_metadata([heading], Path("/does/not/matter.pdf"))

    assert not isinstance(out[0].element.span, CompositeSpan)


# --- appendix headings the layout model missed ------------------------------


def test_an_appendix_heading_matching_typed_siblings_is_retitled() -> None:
    """``A ...`` ships a heading where Docling typed a paragraph.

    Measured on the ATC'25 paper: Docling read "B Complete results ..." and
    "C Complete results ..." as headings but "A Another Example for OSTD APIs"
    as a body paragraph, which then shipped set as body text. The document's
    own letter-marked headings corroborate the shape.
    """
    heading_b = _para(
        "b293",
        "B Complete results for Redis benchmark",
        block_type=BlockType.HEADING,
    )
    narrative_a = _para("b290", "A Another Example for OSTD APIs")
    prose = _para("b291", "A new approach to memory isolation is proposed here.")

    out = restore_appendix_headings([narrative_a, prose, heading_b])

    assert out[0].block_type is BlockType.HEADING
    assert out[0].source_text == "A Another Example for OSTD APIs"
    # Prose keeps its type even though it opens with an article: the word after
    # the marker is lowercase, so the shape does not match at all.
    assert out[1].block_type is BlockType.NARRATIVE
    assert out[2].block_type is BlockType.HEADING


def test_a_letter_shape_without_typed_siblings_stays_prose() -> None:
    # A paper with no letter-marked headings offers no corroboration: "A CPU
    # pipeline ..." as a one-line paragraph stays prose.
    prose = _para("p1", "A CPU pipeline flush costs microseconds here")
    assert restore_appendix_headings([prose])[0].block_type is BlockType.NARRATIVE


def test_a_wrapped_or_terminated_paragraph_is_never_a_heading() -> None:
    heading_b = _para("b293", "B Complete results", block_type=BlockType.HEADING)
    # Terminal punctuation reads as a sentence even with the marker shape.
    terminated = _para("b294", "A Second observation follows.")
    # A 60pt-tall block is two printed lines of prose, not a title.
    two_lines = _para("b295", "A Example with a very long appendix title", y1=60.0)

    out = restore_appendix_headings([heading_b, terminated, two_lines])

    assert out[1].block_type is BlockType.NARRATIVE
    assert out[2].block_type is BlockType.NARRATIVE

"""Unit tests for multi-page table continuity inspection and stitching."""

from __future__ import annotations

import pytest

from ubt.core.cleaners.table_stitcher import (
    can_stitch_tables,
    fuse_table_markups,
    header_row_similarity,
    is_continuation_label,
    is_data_like_row,
    stitch_multipage_tables,
)
from ubt.core.ir.models import BlockStatus, BlockType, IRBlock
from ubt.model.ast import Heading, Paragraph, RegionKind, Table
from ubt.model.span import Span
from ubt.render.typst.document_renderer import render_table

pytestmark = pytest.mark.fast


def _make_table_block(
    block_id: str,
    markup: str,
    page: int = 1,
    y0: float = 100.0,
    y1: float = 300.0,
    x0: float = 50.0,
    x1: float = 500.0,
) -> IRBlock:
    return IRBlock(
        element=Table(
            id=block_id,
            spine_index=0,
            span=Span(page=page, bbox=(x0, y0, x1, y1)),
            markup=markup,
        ),
        style=None,
        status=BlockStatus.PENDING,
    )


def _make_narrative_block(
    block_id: str,
    text: str,
    page: int = 1,
    region: RegionKind = RegionKind.BODY,
) -> IRBlock:
    return IRBlock(
        element=Paragraph(
            id=block_id,
            spine_index=0,
            span=Span(page=page, bbox=(50.0, 400.0, 500.0, 420.0)),
            text=text,
            region=region,
        ),
        style=None,
        status=BlockStatus.PENDING,
    )


def test_is_continuation_label() -> None:
    assert is_continuation_label("Table 1 (continued)")
    assert is_continuation_label("Table 3.1: Continued")
    assert is_continuation_label("表 2（续）")
    assert is_continuation_label("表 4 续表")
    assert is_continuation_label("Table 5 (cont'd)")
    assert is_continuation_label("(cont.)")
    assert is_continuation_label("接上页")
    assert not is_continuation_label("This is a standard paragraph discussing Table 1.")
    assert not is_continuation_label("Section 2: Related Work")


def test_header_row_similarity() -> None:
    h1 = ["Model", "Accuracy (%)", "Latency (ms)", "Cost ($)"]
    h2 = ["Model", "Accuracy (%)", "Latency (ms)", "Cost ($)"]
    assert header_row_similarity(h1, h2) == 1.0

    # Minor normalization diffs
    h3 = ["**Model**", "accuracy", "latency (ms)", "cost"]
    assert header_row_similarity(h1, h3) > 0.85

    # Column count mismatch
    assert header_row_similarity(h1, ["Model", "Accuracy"]) == 0.0

    # Completely different
    assert header_row_similarity(h1, ["A", "B", "C", "D"]) < 0.40


def test_is_data_like_row() -> None:
    data_row = ["ResNet-50", "76.1%", "12.4ms", "$0.02"]
    assert is_data_like_row(data_row)

    header_row = ["Model Architecture", "Top-1 Accuracy", "Inference Time", "Per-query Cost"]
    assert not is_data_like_row(header_row)


def test_can_stitch_adjacent_pages_matched_header() -> None:
    tbl_a_markup = (
        "| Architecture | Accuracy | Latency |\n"
        "| :--- | :---: | ---: |\n"
        "| ResNet-18 | 69.8% | 8ms |\n"
        "| ResNet-50 | 76.1% | 15ms |"
    )
    tbl_b_markup = (
        "| Architecture | Accuracy | Latency |\n"
        "| :--- | :---: | ---: |\n"
        "| ResNet-101 | 77.4% | 24ms |\n"
        "| ResNet-152 | 78.3% | 35ms |"
    )
    tbl_a = _make_table_block("tbl_a", tbl_a_markup, page=1, y0=50.0, y1=250.0)
    tbl_b = _make_table_block("tbl_b", tbl_b_markup, page=2, y0=600.0, y1=750.0)

    can_stitch, reason = can_stitch_tables(tbl_a, tbl_b)
    assert can_stitch
    assert "matched_header" in reason


def test_can_stitch_continuation_label() -> None:
    tbl_a_markup = (
        "| City | Population | Area |\n"
        "| --- | --- | --- |\n"
        "| Tokyo | 37M | 2194 |\n"
        "| Delhi | 32M | 1484 |"
    )
    tbl_b_markup = (
        "| Table 1 (continued) | | |\n"
        "| --- | --- | --- |\n"
        "| Shanghai | 29M | 6340 |\n"
        "| São Paulo | 22M | 1521 |"
    )
    tbl_a = _make_table_block("tbl_a", tbl_a_markup, page=3)
    tbl_b = _make_table_block("tbl_b", tbl_b_markup, page=4)

    can_stitch, reason = can_stitch_tables(tbl_a, tbl_b)
    assert can_stitch
    assert "continuation_label" in reason


def test_can_stitch_headless_continuation() -> None:
    tbl_a_markup = (
        "| Method | Score | Ratio |\n"
        "| --- | --- | --- |\n"
        "| Baseline | 82.4 | 1.0x |\n"
        "| Ours-Lite | 85.1 | 1.4x |"
    )
    # Fragment on page 2 parsed with first data row as header
    tbl_b_markup = (
        "| 88.7 | 2.1x | Ours-Full |\n"  # data row treated as header
        "| --- | --- | --- |\n"
        "| Ours-XL | 90.3 | 2.8x |"
    )
    tbl_a = _make_table_block("tbl_a", tbl_a_markup, page=5)
    tbl_b = _make_table_block("tbl_b", tbl_b_markup, page=6)

    can_stitch, reason = can_stitch_tables(tbl_a, tbl_b)
    assert can_stitch


def test_cannot_stitch_mismatched_columns() -> None:
    tbl_a = _make_table_block("tbl_a", "| A | B |\n| --- | --- |\n| 1 | 2 |", page=1)
    tbl_b = _make_table_block("tbl_b", "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |", page=2)

    can_stitch, reason = can_stitch_tables(tbl_a, tbl_b)
    assert not can_stitch
    assert "column_count_mismatch" in reason


def test_cannot_stitch_non_adjacent_pages() -> None:
    tbl_a = _make_table_block("tbl_a", "| A | B |\n| --- | --- |\n| 1 | 2 |", page=1)
    tbl_b = _make_table_block("tbl_b", "| A | B |\n| --- | --- |\n| 3 | 4 |", page=5)

    can_stitch, reason = can_stitch_tables(tbl_a, tbl_b)
    assert not can_stitch
    assert "non_adjacent_pages" in reason


def test_cannot_stitch_intervening_heading() -> None:
    tbl_a = _make_table_block("tbl_a", "| A | B |\n| --- | --- |\n| 1 | 2 |", page=1)
    tbl_b = _make_table_block("tbl_b", "| A | B |\n| --- | --- |\n| 3 | 4 |", page=2)
    heading = IRBlock(
        element=Heading(
            id="h1",
            spine_index=0,
            span=Span(page=1),
            level=2,
            text="3. Experimental Setup",
        ),
        style=None,
        status=BlockStatus.PENDING,
    )

    can_stitch, reason = can_stitch_tables(tbl_a, tbl_b, intervening=[heading])
    assert not can_stitch


def test_fuse_table_markups_duplicate_header() -> None:
    markup_a = (
        "| Architecture | Accuracy |\n| :--- | ---: |\n| Model A | 91.2% |\n| Model B | 92.5% |"
    )
    markup_b = (
        "| Architecture | Accuracy |\n| :--- | ---: |\n| Model C | 93.1% |\n| Model D | 94.0% |"
    )
    fused = fuse_table_markups(markup_a, markup_b, match_type="matched_header")
    lines = [line.strip() for line in fused.strip().splitlines() if line.strip()]
    assert len(lines) == 6  # 1 header + 1 separator + 4 rows
    assert "Model A" in lines[2]
    assert "Model B" in lines[3]
    assert "Model C" in lines[4]
    assert "Model D" in lines[5]


def test_fuse_table_markups_headless() -> None:
    markup_a = "| Col1 | Col2 |\n| --- | --- |\n| Val1 | Val2 |"
    markup_b = "| Val3 | Val4 |\n| --- | --- |\n| Val5 | Val6 |"
    fused = fuse_table_markups(markup_a, markup_b, match_type="headless_continuation")
    lines = [line.strip() for line in fused.strip().splitlines() if line.strip()]
    assert len(lines) == 5  # 1 header + 1 separator + 3 data rows (Val1, Val3, Val5)
    assert "Val3" in lines[3]
    assert "Val5" in lines[4]


def test_stitch_multipage_tables_cascading() -> None:
    # A table that spans 3 consecutive pages!
    t1 = _make_table_block("t1", "| Param | Value |\n| --- | --- |\n| P1 | 10 |", page=1)
    footer = _make_narrative_block("foot", "Page 1 of 50", page=1, region=RegionKind.FOOTER)
    header = _make_narrative_block("head", "Chapter 2: Methods", page=2, region=RegionKind.HEADER)
    t2 = _make_table_block("t2", "| Param | Value |\n| --- | --- |\n| P2 | 20 |", page=2)
    cont_cap = _make_narrative_block("cont", "Table 1 (continued)", page=3)
    t3 = _make_table_block("t3", "| Param | Value |\n| --- | --- |\n| P3 | 30 |", page=3)

    blocks = [t1, footer, header, t2, cont_cap, t3]
    stitched = stitch_multipage_tables(blocks, drop_absorbed=True)

    # t2, cont_cap, and t3 should be absorbed into t1!
    table_blocks = [b for b in stitched if b.block_type is BlockType.TABLE]
    assert len(table_blocks) == 1
    fused_tbl = table_blocks[0]
    assert fused_tbl.id == "t1"
    assert "P1" in fused_tbl.source_text
    assert "P2" in fused_tbl.source_text
    assert "P3" in fused_tbl.source_text
    assert "t2" in fused_tbl.provenance.fused_block_ids
    assert "t3" in fused_tbl.provenance.fused_block_ids
    assert len(fused_tbl.provenance.physical_boxes) >= 3


def test_typst_rendering_for_stitched_multipage_table() -> None:
    rows = "\n".join(f"| Item {i} | Category {i % 3} | {i * 10} |" for i in range(25))
    markup = f"| Item Name | Category | Score |\n| :--- | :---: | ---: |\n{rows}"

    # Multi-page table rendered with is_multipage=True
    rendered = render_table(markup, is_multipage=True, page_columns=2)

    # Must NOT contain #place(..., float: true) because floats cannot split across pages in Typst!
    assert "#place" not in rendered
    # Must contain native table.header(repeat: true)
    assert "table.header(" in rendered
    assert "repeat: true" in rendered

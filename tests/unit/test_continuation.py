"""Continuation runs and the block-based overlay builder.

A paragraph the PDF extractor split across a page/column boundary should render
as *one* element flowed across its boxes, not two independent overlays. These pin
the conservative detection (same flow, no terminal punctuation, same-or-next
page) and the overlay builder that turns a run into a multi-box overlay and a
formula into a math overlay.
"""

from __future__ import annotations

import pytest

from ubt.core.ir.continuation import find_continuation_runs
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element
from ubt.model.ast import RegionKind
from ubt.model.fidelity import Fidelity
from ubt.render.outputs import overlays_from_blocks

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    text: str,
    *,
    page: int = 1,
    spine: int = 1,
    block_type: BlockType = BlockType.NARRATIVE,
    region: RegionKind | None = None,
    x0: float = 54.0,
    y0: float = 700.0,
    x1: float = 354.0,
    y1: float = 712.0,
    target: str | None = None,
) -> IRBlock:
    block = IRBlock(
        element=make_element(
            block_type=block_type,
            id=block_id,
            source_text=text,
            spine_index=spine,
            region=region,
            bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page),
        )
    )
    if target is not None:
        block.target_text = target
    return block


def test_a_sentence_split_across_a_page_break_is_one_run() -> None:
    first = _block("b1", "The machine relies on attention", page=1, spine=1)
    second = _block("b2", "and runs a forward pass.", page=2, spine=2)
    (run,) = find_continuation_runs([first, second])
    assert run.block_ids == ("b1", "b2")
    assert [box.page for box in run.boxes] == [1, 2]
    assert run.is_composite


def test_a_block_ending_a_sentence_does_not_continue() -> None:
    first = _block("b1", "A complete sentence.", page=1, spine=1)
    second = _block("b2", "Another sentence.", page=2, spine=2)
    assert find_continuation_runs([first, second]) == ()


def test_a_block_ending_with_curly_quotes_does_not_continue() -> None:
    # Sentences ending on curly quotes or CJK closers must be recognized as finished.
    first = _block("b1", "He said, “A complete sentence.”", page=1, spine=1)
    second = _block("b2", "another sentence begins.", page=2, spine=2)
    assert find_continuation_runs([first, second]) == ()

    first_cjk = _block("b3", "这是完整的结论。”", page=1, spine=1)
    second_cjk = _block("b4", "下一段开始。", page=2, spine=2)
    assert find_continuation_runs([first_cjk, second_cjk]) == ()


def test_a_gap_of_more_than_one_page_does_not_continue() -> None:
    first = _block("b1", "Dangling clause", page=1, spine=1)
    second = _block("b2", "far away", page=3, spine=2)
    assert find_continuation_runs([first, second]) == ()


def test_a_same_page_pair_does_not_merge() -> None:
    # Regression: a heading and the paragraph under it are both NARRATIVE to the
    # extractor, and the heading has no terminal punctuation -- same-page merging
    # fused them into one overlay and drew the heading into the paragraph's flow.
    first = _block("b1", "The Attention Machine", page=1, spine=1, y0=737.0, y1=746.0)
    second = _block("b2", "The machine relies on attention.", page=1, spine=2, y0=719.0, y1=730.0)
    assert find_continuation_runs([first, second]) == ()


def test_an_uppercase_start_does_not_continue() -> None:
    first = _block("b1", "Dangling clause", page=1, spine=1)
    second = _block("b2", "Another paragraph begins.", page=2, spine=2)
    assert find_continuation_runs([first, second]) == ()


def test_a_sentence_split_across_a_column_break_is_one_run() -> None:
    # Left column ends mid-sentence; reading order bubbles to the top of the
    # right column, which starts lowercase and sits beside (not below) the left.
    left = _block(
        "b1", "The machine relies on attention and", page=1, spine=1, x0=55, x1=295, y0=626, y1=635
    )
    right = _block(
        "b2", "and runs a forward pass.", page=1, spine=2, x0=333, x1=485, y0=623, y1=747
    )
    (run,) = find_continuation_runs([left, right])
    assert run.block_ids == ("b1", "b2")
    assert [box.page for box in run.boxes] == [1, 1]


def test_a_column_jump_from_bottom_of_col1_to_top_of_col2_is_one_run() -> None:
    # A realistic multi-column split: left column ends at the bottom of the page
    # (small y), right column starts at the top of the page (large y), with zero
    # vertical overlap between the two bounding boxes.
    left = _block(
        "b1", "The machine relies on attention and", page=1, spine=1, x0=55, x1=295, y0=80, y1=95
    )
    right = _block(
        "b2", "runs a forward pass smoothly.", page=1, spine=2, x0=333, x1=485, y0=720, y1=750
    )
    (run,) = find_continuation_runs([left, right])
    assert run.block_ids == ("b1", "b2")
    assert [box.page for box in run.boxes] == [1, 1]


def test_a_same_column_block_below_does_not_continue() -> None:
    first = _block(
        "b1", "The machine relies on attention and", page=1, spine=1, x0=55, x1=295, y0=640, y1=649
    )
    second = _block("b2", "runs a forward pass.", page=1, spine=2, x0=55, x1=295, y0=624, y1=633)
    assert find_continuation_runs([first, second]) == ()


def test_a_non_prose_block_is_never_part_of_a_run() -> None:
    first = _block("b1", "Dangling clause", page=1, spine=1)
    code = _block("b2", "x = 1", page=1, spine=2, block_type=BlockType.CODE)
    assert find_continuation_runs([first, code]) == ()


def test_the_overlay_builder_merges_a_run_into_one_box_chain() -> None:
    first = _block("b1", "The machine relies on attention", page=1, spine=1, target="T-one")
    second = _block("b2", "and runs a forward pass.", page=2, spine=2, target="T-two")

    (overlay,) = overlays_from_blocks([first, second], None)
    assert overlay.element_id == "b1"
    assert [box.page for box in overlay.flow_boxes] == [1, 2]
    assert overlay.text == "T-one T-two"
    assert overlay.kind == "text"


def test_the_overlay_builder_marks_a_formula_as_math() -> None:
    formula = _block(
        "f1", "$e^{i\\pi}+1=0$", block_type=BlockType.FORMULA, spine=1, target="e^{i\\pi}+1=0"
    )
    (overlay,) = overlays_from_blocks([formula], None)
    assert overlay.kind == "math"
    assert overlay.text == "e^{i\\pi}+1=0"


def test_the_overlay_builder_keeps_tables_and_figures_on_the_canvas() -> None:
    table = _block("t1", "a b c", block_type=BlockType.TABLE, spine=1, target="X Y Z")
    figure = _block("i1", "", block_type=BlockType.IMAGE, spine=2, target="caption")
    assert overlays_from_blocks([table, figure], None) == ()


def test_the_overlay_builder_skips_blocks_kept_at_the_floor() -> None:
    block = _block("b1", "Source kept", spine=1, target="T")
    plan = {"b1": Fidelity.PRESERVED_OPAQUE}
    assert overlays_from_blocks([block], plan) == ()


def test_bare_numbers_and_page_numbers_are_not_continuation_candidates() -> None:
    # Standalone numbers (e.g. page numbers mistakenly classed as narrative) must not merge.
    first = _block("b1", "Dangling sentence without period", page=1, spine=1)
    page_num = _block("b2", "2", page=1, spine=2)
    next_item = _block("b3", "(3) Next numbered item.", page=2, spine=3)
    assert find_continuation_runs([first, page_num, next_item]) == ()


def test_list_item_markers_do_not_continue_previous_paragraph() -> None:
    # A block starting with a list bullet/number marker is starting a new item, not continuing.
    first = _block("b1", "Some ongoing sentence without period", page=1, spine=1)
    second = _block("b2", "(1) First list item starts here.", page=2, spine=2)
    third = _block("b3", "1. Numbered list item.", page=2, spine=3)
    assert find_continuation_runs([first, second]) == ()
    assert find_continuation_runs([first, third]) == ()


def test_intervening_page_furniture_does_not_break_continuation() -> None:
    # A footer or page number on page 1, followed by a header on page 2, does not
    # break the continuation between the real body text across pages.
    from ubt.model.ast import RegionKind

    first = _block("b1", "The machine relies on attention", page=1, spine=1)
    footer = _block("b2", "1", page=1, spine=2)  # bare number page furniture
    header = _block("b3", "JOURNAL OF COMPUTING", page=2, spine=3, region=RegionKind.HEADER)
    second = _block("b4", "and runs a forward pass.", page=2, spine=4)

    (run,) = find_continuation_runs([first, footer, header, second])
    assert run.block_ids == ("b1", "b4")
    assert [box.page for box in run.boxes] == [1, 2]


def test_join_continuous_text_cjk_and_latin() -> None:
    from ubt.core.ir.continuation import join_continuous_text

    # Latin text is joined with a space.
    assert (
        join_continuous_text(["The machine", "relies on attention"])
        == "The machine relies on attention"
    )

    # CJK text without boundary space is joined seamlessly without extra spaces.
    assert join_continuous_text(["这是前半句，", "这是后半句。"]) == "这是前半句，这是后半句。"

    # Mixed Latin and CJK gets spaced at the script boundary.
    assert join_continuous_text(["Attention is all", "你所需要的"]) == "Attention is all 你所需要的"


def test_fuse_continuation_blocks_creates_composite_span() -> None:
    from ubt.core.ir.continuation import fuse_continuation_blocks
    from ubt.model.span import CompositeSpan

    first = _block("b1", "The machine relies on attention", page=1, spine=1)
    second = _block("b2", "and runs a forward pass.", page=2, spine=2)

    fused_list = fuse_continuation_blocks([first, second])
    assert len(fused_list) == 1
    fused = fused_list[0]
    assert fused.id == "b1"
    assert fused.source_text == "The machine relies on attention and runs a forward pass."
    assert isinstance(fused.element.span, CompositeSpan)
    assert len(fused.element.span.boxes) == 2
    assert [b.page for b in fused.element.span.boxes] == [1, 2]
    assert fused.provenance.get("fused_block_ids") == ["b1", "b2"]


def test_fused_span_survives_the_ledger_round_trip() -> None:
    # The ledger rebuilds a block's span from provenance["physical_boxes"], so a
    # fused block must record them or its box chain is lost and the whole target
    # is squeezed into the first (single-line) box.
    import tempfile
    from pathlib import Path

    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.ledger_base import _upsert_blocks_batch
    from ubt.core.ir.continuation import fuse_continuation_blocks
    from ubt.model.span import CompositeSpan

    first = _block("b1", "The machine relies on attention", page=1, spine=1)
    second = _block("b2", "and runs a forward pass.", page=2, spine=2)
    fused = fuse_continuation_blocks([first, second])[0]
    assert [box["page"] for box in fused.provenance["physical_boxes"]] == [1, 2]

    ledger = SQLiteJobLedger(Path(tempfile.mkdtemp()) / "l.sqlite")
    with ledger._get_conn() as conn:
        conn.execute(
            "INSERT INTO job_meta(job_id,doc_id,source_path,target_lang,total_blocks,status)"
            " VALUES(?,?,?,?,?,?)",
            ("job_x", "doc", "/x.pdf", "zh", 1, "running"),
        )
        _upsert_blocks_batch(conn.cursor(), "job_x", [fused])
    (reloaded,) = ledger.get_all_blocks("job_x")
    assert isinstance(reloaded.element.span, CompositeSpan)
    assert [box.page for box in reloaded.element.span.boxes] == [1, 2]


def test_fused_block_flows_through_overlay_builder() -> None:
    from ubt.core.ir.continuation import fuse_continuation_blocks

    first = _block("b1", "The machine relies on attention", page=1, spine=1)
    second = _block("b2", "and runs a forward pass.", page=2, spine=2)

    fused_list = fuse_continuation_blocks([first, second])
    fused = fused_list[0]
    fused.target_text = "机器依赖注意力并运行前向传播。"

    (overlay,) = overlays_from_blocks([fused], None)
    assert overlay.element_id == "b1"
    assert overlay.text == "机器依赖注意力并运行前向传播。"
    assert len(overlay.flow_boxes) == 2
    assert [b.page for b in overlay.flow_boxes] == [1, 2]

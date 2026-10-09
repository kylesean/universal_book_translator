"""Unit tests for AST and IRBlock bifurcation on layout collision / semantic break markers."""

from __future__ import annotations

import pytest

from ubt.core.ir.bifurcation import (
    SEMANTIC_BREAK_TOKEN,
    bifurcate_block,
    bifurcate_blocks,
)
from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.model.ast import Paragraph
from ubt.model.span import CompositeSpan, PhysicalBox, Span
from ubt.render.outputs import overlays_from_blocks

pytestmark = pytest.mark.fast


def _make_block(
    block_id: str,
    page: int,
    bbox: tuple[float, float, float, float],
    source: str,
    target: str = "",
) -> IRBlock:
    elem = Paragraph(
        id=block_id,
        span=Span(page=page, bbox=bbox),
        text=source,
        spine_index=0,
    )
    block = IRBlock(element=elem)
    block.target_text = target
    block.draft_text = target
    block.status = BlockStatus.DRAFTED
    return block


def test_bifurcate_block_no_marker_returns_unchanged() -> None:
    block = _make_block("b1", 1, (10, 10, 100, 50), "Hello world.", "你好世界。")
    result = bifurcate_block(block)
    assert len(result) == 1
    assert result[0] is block


def test_bifurcate_block_single_chunk_strips_token() -> None:
    block = _make_block(
        "b1", 1, (10, 10, 100, 50), "Hello world.", f"你好世界。{SEMANTIC_BREAK_TOKEN}"
    )
    result = bifurcate_block(block)
    assert len(result) == 1
    assert result[0].target_text == "你好世界。"
    assert result[0].draft_text == "你好世界。"


def test_bifurcate_multi_box_composite_span() -> None:
    box1 = PhysicalBox.of(1, (50, 100, 200, 300))
    box2 = PhysicalBox.of(2, (50, 500, 200, 700))
    elem = Paragraph(
        id="fused_001",
        span=CompositeSpan(boxes=(box1, box2)),
        text="The attention mechanism was proposed. It enables seq2seq learning.",
        spine_index=0,
    )
    block = IRBlock(element=elem)
    block.provenance = {
        "fused_block_ids": ["src_b1", "src_b2"],
        "fused_sources": [
            "The attention mechanism was proposed.",
            "It enables seq2seq learning.",
        ],
    }
    block.target_text = f"注意力机制被提出。{SEMANTIC_BREAK_TOKEN}它支持序列到序列学习。"
    block.draft_text = block.target_text
    block.status = BlockStatus.DRAFTED

    bifurcated = bifurcate_block(block)
    assert len(bifurcated) == 2

    part1, part2 = bifurcated
    assert part1.id == "src_b1"
    assert part1.target_text == "注意力机制被提出。"
    assert part1.source_text == "The attention mechanism was proposed."
    assert isinstance(part1.element.span, Span)
    assert part1.element.span.page == 1
    assert part1.element.span.bbox == (50, 100, 200, 300)
    assert "bifurcated:semantic_break" in part1.error_flags

    assert part2.id == "src_b2"
    assert part2.target_text == "它支持序列到序列学习。"
    assert part2.source_text == "It enables seq2seq learning."
    assert isinstance(part2.element.span, Span)
    assert part2.element.span.page == 2
    assert part2.element.span.bbox == (50, 500, 200, 700)
    assert "bifurcated:semantic_break" in part2.error_flags


def test_bifurcate_single_box_prose() -> None:
    source = "First sentence across column. Second sentence in other column."
    target = f"跨栏第一句。{SEMANTIC_BREAK_TOKEN}另一栏第二句。"
    block = _make_block("b_col", 1, (50, 50, 200, 150), source, target)

    bifurcated = bifurcate_block(block)
    assert len(bifurcated) == 2

    part1, part2 = bifurcated
    assert part1.id == "b_col_s0"
    assert part1.target_text == "跨栏第一句。"
    assert part1.source_text == "First sentence across column."
    assert part1.element.span.bbox == (50, 50, 200, 150)
    assert "bifurcated:semantic_break" in part1.error_flags

    assert part2.id == "b_col_s1"
    assert part2.target_text == "另一栏第二句。"
    assert part2.source_text == "Second sentence in other column."
    assert part2.element.span.bbox == (50, 50, 200, 150)
    assert "bifurcated:semantic_break" in part2.error_flags


def test_bifurcate_blocks_sequence() -> None:
    b1 = _make_block("b1", 1, (10, 10, 100, 50), "Hello", "你好")
    b2 = _make_block(
        "b2", 1, (10, 60, 100, 100), "Part1. Part2.", f"段落1{SEMANTIC_BREAK_TOKEN}段落2"
    )
    b3 = _make_block("b3", 1, (10, 110, 100, 150), "World", "世界")

    results = bifurcate_blocks([b1, b2, b3])
    assert len(results) == 4
    assert [b.id for b in results] == ["b1", "b2_s0", "b2_s1", "b3"]


def test_overlays_from_blocks_merges_in_box_bifurcated_siblings() -> None:
    # When single-box blocks share the exact same bounding box, overlays_from_blocks
    # merges them with double newlines so the typesetter formats distinct paragraphs.
    source = "Sentence 1. Sentence 2."
    target = f"第一句。{SEMANTIC_BREAK_TOKEN}第二句。"
    block = _make_block("b_shared", 1, (50, 50, 200, 150), source, target)

    overlays = overlays_from_blocks([block])
    assert len(overlays) == 1
    overlay = overlays[0]
    assert overlay.page == 1
    assert overlay.bbox == (50, 50, 200, 150)
    assert overlay.text == "第一句。\n\n第二句。"


def test_overlays_from_blocks_multi_box_bifurcation_emits_separate_overlays() -> None:
    box1 = PhysicalBox.of(1, (50, 100, 200, 300))
    box2 = PhysicalBox.of(2, (50, 500, 200, 700))
    elem = Paragraph(
        id="fused_p",
        span=CompositeSpan(boxes=(box1, box2)),
        text="Part 1. Part 2.",
        spine_index=0,
    )
    block = IRBlock(element=elem)
    block.provenance = {
        "fused_block_ids": ["p1_b", "p2_b"],
        "fused_sources": ["Part 1.", "Part 2."],
    }
    block.target_text = f"第一页。{SEMANTIC_BREAK_TOKEN}第二页。"

    overlays = overlays_from_blocks([block])
    assert len(overlays) == 2
    assert overlays[0].page == 1
    assert overlays[0].text == "第一页。"
    assert overlays[0].bbox == (50, 100, 200, 300)

    assert overlays[1].page == 2
    assert overlays[1].text == "第二页。"
    assert overlays[1].bbox == (50, 500, 200, 700)


def test_overlays_from_blocks_multi_box_bifurcation_with_lowercase_continuation() -> None:
    # The second box of a *real* continuation starts lowercase -- that is exactly
    # what fused the blocks. The continuation heuristic must not re-fuse the
    # blocks a semantic break split out of that fused block.
    box1 = PhysicalBox.of(1, (50, 100, 200, 300))
    box2 = PhysicalBox.of(2, (50, 500, 200, 700))
    elem = Paragraph(
        id="fused_p",
        span=CompositeSpan(boxes=(box1, box2)),
        text="x",
        spine_index=0,
    )
    block = IRBlock(element=elem)
    block.provenance = {
        "fused_block_ids": ["p1_b", "p2_b"],
        "fused_sources": ["The machine relies on", "attention and runs a forward pass."],
    }
    block.target_text = f"第一部分。{SEMANTIC_BREAK_TOKEN}第二部分。"

    overlays = overlays_from_blocks([block])
    assert len(overlays) == 2
    assert [overlay.text for overlay in overlays] == ["第一部分。", "第二部分。"]

"""The overlay carries the source layout facts the fragment needs.

Two geometry facts live outside the block's bounding box and are recorded by
the analyzer from the page's own line rects (``annotate_layout_metadata``):

- a list item's marker-column indent — the block box starts at the wrapped
  lines' margin, so the compositor would otherwise draw "(1)" flush with the
  body margin where the source hangs it to the right;
- a centered heading — without the flag the title fragment hugs the left
  margin where the source centered it, and a multi-line title records its
  own line boxes so the translation breaks where the source breaks.

These tests pin the overlay lowering for both facts, the Typst source the
typesetter emits, and the flow's sizing rule for mixed-width chains.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

from ubt.core.ir.models import (
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    StyleMeta,
    make_element,
)
from ubt.render.outputs import overlays_from_blocks

pytestmark = pytest.mark.fast


def _block(block_type: BlockType, *, style: StyleMeta | None = None) -> IRBlock:
    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=block_type,
            flow_id=FlowID.MAIN_STORY,
            source_text="source",
            bbox=BoundingBox(x0=54.0, y0=700.0, x1=354.0, y1=712.0, page=1),
        )
    )
    block.style = style
    block.target_text = "target"
    return block


def test_a_list_items_marker_indent_reaches_the_overlay() -> None:
    style = StyleMeta(first_line_indent_pt=22.2)
    block = _block(BlockType.LIST_ITEM, style=style)
    (overlay,) = overlays_from_blocks([block], None)
    assert overlay.indent_pt == pytest.approx(22.2)
    assert overlay.align_center is False


def test_a_narrative_paragraph_keeps_its_own_indent_rule() -> None:
    style = StyleMeta(first_line_indent_pt=17.4)
    block = _block(BlockType.NARRATIVE, style=style)
    (overlay,) = overlays_from_blocks([block], None)
    assert overlay.indent_pt == pytest.approx(17.4)


def test_a_centered_heading_reaches_the_overlay() -> None:
    style = StyleMeta(alignment="center")
    block = _block(BlockType.HEADING, style=style)
    (overlay,) = overlays_from_blocks([block], None)
    assert overlay.align_center is True
    assert overlay.indent_pt is None


def test_a_left_aligned_heading_is_not_marked_centered() -> None:
    block = _block(BlockType.HEADING, style=StyleMeta())
    (overlay,) = overlays_from_blocks([block], None)
    assert overlay.align_center is False


def test_centering_is_heading_only() -> None:
    # A narrative block with a stray alignment value stays left-aligned: the
    # flag is a heading fact, not a general indent substitute.
    style = StyleMeta(alignment="center")
    block = _block(BlockType.NARRATIVE, style=style)
    (overlay,) = overlays_from_blocks([block], None)
    assert overlay.align_center is False


def test_the_typesetter_emits_the_centering_rule() -> None:
    from ubt.render.outputs import TypstFragmentTypesetter

    ts = TypstFragmentTypesetter(cache_dir=":temp:")
    centered = ts._text_source("标题", 200.0, 40.0, 12.0, align_center=True)
    flush = ts._text_source("标题", 200.0, 40.0, 12.0)
    assert "#set align(center)" in centered
    assert "#set align(center)" not in flush


def test_a_heading_with_line_boxes_builds_a_box_chain() -> None:
    """A multi-line heading's CompositeSpan becomes the overlay's flow chain."""
    from ubt.model.span import CompositeSpan, PhysicalBox

    style = StyleMeta(alignment="center")
    block = _block(BlockType.HEADING, style=style)
    chain = (
        PhysicalBox.of(1, (172.0, 710.0, 425.0, 738.0)),
        PhysicalBox.of(1, (71.0, 690.0, 524.0, 705.0)),
    )
    block.element = replace(block.element, span=CompositeSpan(boxes=chain))
    (overlay,) = overlays_from_blocks([block], None)
    assert overlay.boxes == chain
    assert overlay.align_center is True


class _SizeSpy:
    """A typesetter stub whose wrapped-line height shrinks as width grows."""

    name: ClassVar[str] = "size-spy"

    def __init__(self) -> None:
        self.measured_widths: list[float] = []

    def cap_size(self, kind: str, font_size: float | None) -> float:
        return 12.0

    def measure_fixed(
        self,
        text: str,
        width: float,
        size: float,
        *,
        kind: str = "text",
        is_bold: bool = False,
        indent_pt: float | None = None,
        runs: tuple[Any, ...] = (),
    ) -> float:
        self.measured_widths.append(width)
        chars_per_line = max(1, int(width / size))
        lines = -(-len(text) // chars_per_line)
        return lines * 10.0

    def typeset(self, *_args: Any, **_kwargs: Any) -> Path | None:
        return None

    def typeset_math(self, _latex: str, _width: float, _height: float) -> Path | None:
        return None

    def measure(self, _text: str, _width: float) -> float:
        return 0.0


def test_flow_plan_sizes_against_the_widest_box(tmp_path: Path) -> None:
    """A mixed-width chain (narrow title line 1 + full-width rest) must not
    size the whole heading by the narrow first box — that inflates the height
    and shrinks every line. The sizing measurement runs at the widest width,
    and the flow splits the text at the source's own line structure.
    """
    from pdf_builders import write_text_pdf

    from ubt.model.span import PhysicalBox
    from ubt.render.outputs import LayerCompositor, Overlay

    spy = _SizeSpy()
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE_TEXT])
    compositor = LayerCompositor(source, typesetter=spy)
    overlay = Overlay(
        "e1",
        1,
        (71.0, 690.0, 524.0, 738.0),
        "DeepSeek 弹性计算（DSec）：用于高效大规模代理训练的 A 沙箱基础设施",
        kind="heading",
        align_center=True,
    )
    boxes = (
        PhysicalBox(1, (172.0, 724.0, 425.0, 738.0), 253.0, 14.0),
        PhysicalBox(1, (71.0, 700.0, 524.0, 714.0), 453.0, 14.0),
    )
    parts, draw_size = compositor._flow_plan(overlay, boxes)

    assert measured_widths_first_is_widest(spy)
    assert draw_size == pytest.approx(12.0)
    assert [part.box for part in parts] == list(boxes)
    # The narrow first box takes what fits; the full-width box takes the rest.
    assert 0 < len(parts[0].text) < len(overlay.text)
    assert parts[1].text == overlay.text[len(parts[0].text) :].lstrip()


def measured_widths_first_is_widest(spy: _SizeSpy) -> bool:
    return spy.measured_widths[0] == pytest.approx(453.0)


_PAGE_TEXT = ["DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure"]

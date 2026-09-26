"""Overlay coverage fixes (arXiv 2609.20519 page-by-page review).

Three render-layer defects, all confirmed to be *coverage* problems (the ledger
held a translation for every one; the rigid overlay simply failed to erase or
paint it):

* an untranslated ghost line survived because the erase rect was the *painted*
  zone, and a compact CJK translation left the block's continuation zone
  unpainted (:func:`_erase_rects_by_page` now erases every zone of a rendered
  block);
* sibling list bullets rendered at different sizes because each block was
  fitted to its own maximum (:meth:`RigidTypesetter._unify_list_sizes`);
* a running head was paginated *across* its per-page zones (so only one page
  got text and the rest kept the source head) instead of *repeated* on each
  (:meth:`RigidTypesetter._is_repeating_chrome`).
"""

from __future__ import annotations

from ubt.adapters.pdf.rigid.typesetter import (
    RigidTypesetter,
    ZonePlan,
    _erase_rects_by_page,
)
from ubt.adapters.pdf.rigid.zones import Zone
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock


def _rect_zone(block_id: str, page: int, y0: float, y1: float, rows: tuple[str, ...] = ()) -> Zone:
    return Zone(
        block_id=block_id, page=page, x0=60.0, y0=y0, x1=520.0, y1=y1, base_size=10.0, rows=rows
    )


# --- Fix A: erase every zone of a rendered block ---------------------------


def test_erase_covers_unpainted_continuation_zone() -> None:
    """A rendered block's continuation zone is erased even though it painted nothing.

    This is the ghost-line bug: the translation fit entirely in the first zone,
    so the continuation zone received no text, yet its source line still has to
    go or it survives untranslated.
    """
    own = _rect_zone("b1", 3, 84.0, 119.0)
    cont = _rect_zone("b1", 4, 743.0, 755.0)
    zone_map: dict[str, tuple[Zone, ...]] = {"b1": (own, cont)}
    rects = _erase_rects_by_page(zone_map, {"b1"})
    assert rects[3] == [own.rect]
    assert rects[4] == [cont.rect]  # the unpainted continuation zone is still erased


def test_erase_skips_zones_of_unrendered_blocks() -> None:
    """A spilled block keeps its whole source; erasing its zones would punch a hole."""
    z = _rect_zone("b1", 4, 743.0, 755.0)
    zone_map: dict[str, tuple[Zone, ...]] = {"b1": (z,)}
    assert _erase_rects_by_page(zone_map, set()) == {}


# --- Fix B: unify sibling list-item sizes ----------------------------------


def _planned_item(
    block_id: str, flow: FlowID, size: float
) -> tuple[IRBlock, str, tuple[Zone, ...], list[ZonePlan]]:
    block = IRBlock(
        id=block_id,
        flow_id=flow,
        spine_index=1,
        block_type=BlockType.LIST_ITEM,
        source_text="x",
        target_text="内容",
        bbox=BoundingBox(page=1, x0=60, y0=100, x1=520, y1=130),
    )
    zone = _rect_zone(block_id, 1, 100.0, 130.0)
    plan = ZonePlan(zone=zone, size=size, lines=["内容"], text="内容")
    return block, "内容", (zone,), [plan]


def _stub_typesetter() -> RigidTypesetter:
    ts = RigidTypesetter(target_lang="zh")
    # Re-plan without the Typst/font kernel: echo the requested size back.
    ts._plan_at_size = lambda text, zones, size: [  # type: ignore[method-assign]
        ZonePlan(zone=zones[0], size=size, lines=[text], text=text)
    ]
    return ts


def test_list_items_shrink_to_the_common_minimum() -> None:
    ts = _stub_typesetter()
    planned = [
        _planned_item("i1", FlowID.MAIN_STORY, 12.0),
        _planned_item("i2", FlowID.MAIN_STORY, 9.0),
        _planned_item("i3", FlowID.MAIN_STORY, 11.0),
    ]
    ts._unify_list_sizes(planned)
    assert [p[3][0].size for p in planned] == [9.0, 9.0, 9.0]


def test_list_items_in_different_flows_are_not_merged() -> None:
    ts = _stub_typesetter()
    planned = [
        _planned_item("i1", FlowID.MAIN_STORY, 12.0),
        _planned_item("i2", FlowID.FOOTNOTE, 9.0),
    ]
    ts._unify_list_sizes(planned)
    assert [p[3][0].size for p in planned] == [12.0, 9.0]


def test_list_items_separated_by_narrative_do_not_bleed_font_size() -> None:
    """A cramped list item on page 75 must not shrink unrelated list items on page 4 to 7.5pt."""
    ts = _stub_typesetter()
    b1, t1, z1, p1 = _planned_item("p4_i1", FlowID.MAIN_STORY, 11.0)
    b2, t2, z2, p2 = _planned_item("p4_i2", FlowID.MAIN_STORY, 11.0)
    b1.spine_index = 76
    b2.spine_index = 77

    narr = IRBlock(
        id="p4_narr",
        flow_id=FlowID.MAIN_STORY,
        spine_index=78,
        block_type=BlockType.NARRATIVE,
        source_text="prose",
        target_text="正文",
        bbox=BoundingBox(page=4, x0=60, y0=50, x1=520, y1=80),
    )
    zn = _rect_zone("p4_narr", 4, 50.0, 80.0)
    pn = [ZonePlan(zone=zn, size=11.0, lines=["正文"], text="正文")]

    b_far, t_far, z_far, p_far = _planned_item("p75_i1", FlowID.MAIN_STORY, 7.5)
    b_far.spine_index = 750
    b_far.bbox = BoundingBox(page=75, x0=60, y0=100, x1=520, y1=130)

    planned = [
        (b1, t1, z1, p1),
        (b2, t2, z2, p2),
        (narr, "正文", (zn,), pn),
        (b_far, t_far, z_far, p_far),
    ]
    ts._unify_list_sizes(planned)
    assert planned[0][3][0].size == 11.0
    assert planned[1][3][0].size == 11.0
    assert planned[3][3][0].size == 7.5


def test_footnote_leading_number_formatted_as_superscript() -> None:
    """A translated footnote starting with '1 2026...' formats its leading index as superscript."""
    from ubt.adapters.pdf.rigid.typesetter import _with_list_marker

    fn_block = IRBlock(
        id="pdf_main#b0090",
        flow_id=FlowID.FOOTNOTE,
        spine_index=90,
        block_type=BlockType.NARRATIVE,
        source_text="1 Data retrieved from the Visual Studio Code Marketplace on June 9, 2026.",
        target_text="1 2026 年 6 月 9 日检索自 Visual Studio Code Marketplace 的数据。",
        bbox=BoundingBox(page=5, x0=79.9, y0=82.6, x1=371.0, y1=93.1),
    )
    rendered = _with_list_marker(fn_block, fn_block.target_text or "")
    assert rendered.startswith("¹ 2026")


# --- Fix C: detect a page-repeating running head ---------------------------


def _head_block() -> IRBlock:
    return IRBlock(
        id="h",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.HEADING,
        source_text="SoL-Pi: Recursively Scaling Auto-Research Loops",
        target_text="递归扩展",
        bbox=BoundingBox(page=1, x0=60, y0=720, x1=520, y1=757),
    )


def test_running_head_is_detected_as_repeating() -> None:
    ts = RigidTypesetter(target_lang="zh")
    src = "SoL-Pi: Recursively Scaling Auto-Research Loops"
    zones = (
        _rect_zone("h", 1, 720.0, 757.0, rows=(src,)),
        _rect_zone("h", 2, 777.0, 787.0, rows=(src,)),
        _rect_zone("h", 3, 777.0, 787.0, rows=(src,)),
    )
    assert ts._is_repeating_chrome(_head_block(), zones)


def test_paragraph_continuation_is_not_repeating() -> None:
    """A block split across pages holds different fragments, not the whole line."""
    ts = RigidTypesetter(target_lang="zh")
    zones = (
        _rect_zone("h", 1, 700.0, 757.0, rows=("SoL-Pi: Recursively Scaling",)),
        _rect_zone("h", 2, 100.0, 130.0, rows=("Auto-Research Loops",)),
    )
    assert not ts._is_repeating_chrome(_head_block(), zones)


def test_single_zone_block_is_not_repeating() -> None:
    ts = RigidTypesetter(target_lang="zh")
    src = "SoL-Pi: Recursively Scaling Auto-Research Loops"
    assert not ts._is_repeating_chrome(_head_block(), (_rect_zone("h", 1, 720.0, 757.0, (src,)),))


def test_list_item_is_never_misclassified_as_center_aligned() -> None:
    """Page 8 bug: an indented LIST_ITEM whose 3 lines span to the right margin must stay justify at x0=86.75, never center."""
    from ubt.adapters.pdf.rigid.extract import LineBox
    from ubt.adapters.pdf.rigid.zones import PageFacts, _resolve_horizontal_span_and_align

    block = IRBlock(
        id="pdf_main#b0116",
        spine_index=1,
        block_type=BlockType.LIST_ITEM,
        flow_id=FlowID.MAIN_STORY,
        source_text="Spatial composability demands that inter-component dependencies be declared and reactively managed.",
    )
    lines = [
        LineBox(
            rect=(86.75, 480.0, 528.66, 491.0),
            text="• Spatial composability demands that inter-component",
        ),
        LineBox(
            rect=(86.75, 466.0, 528.66, 477.0),
            text="dependencies be declared and reactively managed.",
        ),
        LineBox(
            rect=(86.75, 452.0, 508.50, 463.0),
            text="against environmental supply as context evolves.",
        ),
    ]
    all_rows = [LineBox(rect=(69.21, 510.0, 528.66, 521.0), text="Body paragraph")] + lines
    facts = PageFacts(
        page=8, width=595.28, height=841.89, bg="ffffff", lines=tuple(all_rows), images=()
    )
    x0, x1, align = _resolve_horizontal_span_and_align(
        block, facts, lines, all_rows, (), 86.75, 452.0, 528.66, 491.0
    )
    assert align == "justify"
    assert abs(x0 - 86.75) < 1.0


def test_single_word_proof_block_matches_zone_row() -> None:
    """Page 11 bug: standalone 'Proof.' block has only 1 word (5 letters) and must match its PDF line."""
    from ubt.adapters.pdf.rigid.zones import _row_matches

    assert _row_matches("Proof.", "Proof.") is True


def test_ordered_list_item_restores_number_marker_from_zone_rows() -> None:
    """Page 11 bug: '1. The unit...' should restore '1. ' instead of '• ' when zone.rows starts with '1. '."""
    from ubt.adapters.pdf.rigid.typesetter import _with_list_marker

    block = IRBlock(
        id="pdf_main#b0146",
        spine_index=1,
        block_type=BlockType.LIST_ITEM,
        flow_id=FlowID.MAIN_STORY,
        source_text="The unit is carried to the unit.",
        target_text="单位元对应到单位元。",
    )
    res = _with_list_marker(
        block,
        "单位元对应到单位元。",
        zone_rows=("1. The unit is carried to the unit.",),
    )
    assert res == "1. 单位元对应到单位元。"


def test_build_zones_does_not_claim_centered_display_equation_as_continuation() -> None:
    """Page 10 bug: orphan centered display equation (10) below Theorem 7 must not be claimed as a continuation zone."""
    from ubt.adapters.pdf.rigid.extract import LineBox
    from ubt.adapters.pdf.rigid.zones import PageFacts, build_zones
    from ubt.core.ir.models import BoundingBox

    block = IRBlock(
        id="pdf_main#b0152_thm7",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        source_text="Theorem 7. For every (gamma, varphi) and every pair (f, g) with g(f(gamma)) = gamma,",
        target_text="定理 7。对任意 (gamma, varphi) 及满足 g(f(gamma)) = gamma 的任意函数对 (f, g)，",
        bbox=BoundingBox(page=10, x0=69.21, y0=310.0, x1=420.5, y1=323.0),
    )
    lines = (
        LineBox(
            rect=(69.21, 312.0, 420.5, 323.0),
            text="Theorem 7. For every (gamma, varphi) and every pair (f, g) with g(f(gamma)) = gamma,",
        ),
        LineBox(
            rect=(163.57, 290.16, 434.53, 301.81),
            text="recover (track (f, g)(gamma, varphi)) = recover (gamma, varphi)",
        ),
    )
    facts = PageFacts(page=10, width=595.28, height=841.89, bg="ffffff", lines=lines, images=())
    zones = build_zones({10: facts}, [block])
    assert len(zones["pdf_main#b0152_thm7"]) == 1
    assert zones["pdf_main#b0152_thm7"][0].y0 >= 310.0

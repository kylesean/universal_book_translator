"""Anchored zone assembly: bbox seeds, growth guards, tail adoption."""

import pytest

from ubt.adapters.pdf.rigid.typesetter import (
    RigidTypesetter,
    ZonePlan,
    _boxes_for,
    _erase_rects_by_page,
)
from ubt.adapters.pdf.rigid.zones import (
    PageFacts,
    Zone,
    _run_zone,
    build_zones,
    content_window,
    own_zone,
)
from ubt.adapters.pdf.textgeom import LineBox
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock


def _block(
    bid: str,
    source: str,
    page: int = 1,
    y0: float = 100.0,
    y1: float = 130.0,
    btype: BlockType = BlockType.NARRATIVE,
    x0: float = 100.0,
    x1: float = 400.0,
) -> IRBlock:
    return IRBlock(
        id=bid,
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=btype,
        source_text=source,
        target_text="译文",
        bbox=BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1),
    )


def _line(
    text: str, y0: float, y1: float | None = None, x0: float = 100.0, x1: float = 400.0
) -> LineBox:
    return LineBox(text, (x0, y0, x1, y1 if y1 is not None else y0 + 9.0))


_LONG = (
    "The surface potential model used in BSIM-CMG is based on a solution of "
    "Poisson's equation for a long-channel double-gate FinFET with finite "
    "doping in the channel to mimic the doped channel used in fabrication."
)
_NEXT = "A.1 Continuous starting function"


def test_own_zone_paints_a_table_cell_inside_its_own_bbox() -> None:
    """A table row extracted as prose sits on ``table_band`` lines, which
    ``visual_rows`` drops, so ``own_zone`` found no seed and the cell failed
    closed (source-visible). Its bbox is authoritative: paint strictly inside it,
    never expanding into the next column."""
    from ubt.adapters.pdf.rigid.zones import own_zone

    facts = PageFacts(
        page=1,
        width=460.0,
        height=660.0,
        lines=(
            # A non-table row keeps ``rows`` non-empty, so the empty-page
            # ``_bbox_zone`` shortcut does not mask the fallback under test.
            LineBox("unrelated body prose on the page", (100.0, 100.0, 400.0, 109.0)),
            LineBox("ℓ screening length", (193.0, 512.0, 280.0, 521.0), table_band=True),
            LineBox("22 nm", (318.0, 514.0, 350.0, 521.0), table_band=True),
            LineBox("assumed", (380.0, 514.0, 413.0, 521.0), table_band=True),
        ),
    )
    block = _block(
        "cell",
        "ℓ screening length 22 nm assumed",
        y0=512.0,
        y1=521.0,
        btype=BlockType.HEADING,
        x0=193.0,
        x1=413.0,
    )
    zone = own_zone(block, facts)
    assert zone is not None
    assert (zone.x0, zone.x1) == (193.0, 413.0), "cell must not expand into the next column"
    assert zone.y0 <= 512.0 and zone.y1 >= 521.0

    # A block whose text matches no table row still fails closed.
    stranger = _block(
        "cell2",
        "completely different words entirely",
        y0=512.0,
        y1=521.0,
        btype=BlockType.HEADING,
        x0=193.0,
        x1=413.0,
    )
    assert own_zone(stranger, facts) is None


def test_own_zone_grows_over_paragraph_and_stops_at_foreign_row() -> None:
    facts = PageFacts(
        page=1,
        width=460.0,
        height=660.0,
        lines=(
            _line(_LONG[:70], 120.0),
            _line(_LONG[70:140], 108.0),
            _line(_LONG[140:], 96.0),
            _line("Metal Gate", 200.0),
        ),
    )
    block = _block("b1", _LONG, y0=115.0, y1=132.0)
    zone = own_zone(block, facts)
    assert zone is not None
    assert zone.y0 <= 96.0 and zone.y1 >= 132.0
    # The figure label above the seed is never absorbed.
    assert zone.y1 < 200.0


def test_own_zone_does_not_cross_another_blocks_bbox() -> None:
    facts = PageFacts(
        page=1,
        width=460.0,
        height=660.0,
        lines=(_line(_LONG[:70], 120.0), _line(_LONG[70:140], 108.0), _line(_NEXT, 60.0)),
    )
    block = _block("b1", _LONG, y0=115.0, y1=132.0)
    other = _block("b2", _NEXT, y0=55.0, y1=65.0, btype=BlockType.HEADING)
    zone = own_zone(block, facts, (block, other))
    assert zone is not None
    assert zone.y0 > 65.0  # never swallows the next heading row


def test_own_zone_absorbs_one_garbled_edge_row() -> None:
    facts = PageFacts(
        page=1,
        width=460.0,
        height=660.0,
        lines=(
            _line(_LONG[:70], 120.0),
            _line(_LONG[70:140], 108.0),
            _line("dependence, Nc Nch is the channel doping and the temperature", 96.0),
        ),
    )
    block = _block("b1", _LONG, y0=115.0, y1=132.0)
    zone = own_zone(block, facts)
    assert zone is not None and zone.y0 <= 96.0


def test_build_zones_adopts_matching_tail_on_later_page() -> None:
    tail = "starting function in the following steps and then compute it"
    pages = {
        1: PageFacts(page=1, width=460.0, height=660.0, lines=(_line(_LONG[:70], 120.0),)),
        2: PageFacts(page=2, width=460.0, height=660.0, lines=(_line(tail, 500.0),)),
    }
    block = _block("b1", _LONG + " " + tail, y0=115.0, y1=132.0)
    zones = build_zones(pages, [block])
    kinds = {(z.page, z.kind) for z in zones["b1"]}
    assert (1, "own") in kinds and (2, "continuation") in kinds


def test_build_zones_adopts_short_word_tail_on_later_page() -> None:
    """Regression (arXiv 2609.20519 p6): the last word of a paragraph that
    wrapped to the next page ("...main agent." -> "agent.") is far below the
    8-char substring floor, yet the exact-end tail rung must claim it —
    otherwise the bare source word stays visible under the translated page."""
    src = _LONG + " The reducer verifies the receipt before passing it to the main agent."
    pages = {
        1: PageFacts(page=1, width=460.0, height=660.0, lines=(_line(src[:70], 120.0),)),
        2: PageFacts(page=2, width=460.0, height=660.0, lines=(_line("agent.", 500.0),)),
    }
    block = _block("b1", src, y0=115.0, y1=132.0)
    zones = build_zones(pages, [block])
    kinds = {(z.page, z.kind) for z in zones["b1"]}
    assert (2, "continuation") in kinds


def test_short_row_that_is_not_the_source_tail_is_not_claimed() -> None:
    """The tail rung is anchored to the END of the source; a short orphan
    from elsewhere ("mimic") must stay unclaimed."""
    src = _LONG + " The reducer verifies the receipt before passing it to the main agent."
    pages = {
        1: PageFacts(page=1, width=460.0, height=660.0, lines=(_line(src[:70], 120.0),)),
        2: PageFacts(page=2, width=460.0, height=660.0, lines=(_line("mimic", 500.0),)),
    }
    block = _block("b1", src, y0=115.0, y1=132.0)
    zones = build_zones(pages, [block])
    kinds = {(z.page, z.kind) for z in zones["b1"]}
    assert (2, "continuation") not in kinds


def test_build_zones_keeps_contacting_zones() -> None:
    """Pad-level zone contact must not drop a one-line list item."""
    pages = {
        1: PageFacts(
            page=1,
            width=460.0,
            height=660.0,
            lines=(_line(_LONG[:70], 120.0), _line("1. Calculate the function", 100.0)),
        )
    }
    para = _block("b1", _LONG, y0=115.0, y1=132.0)
    item = _block("b2", "Calculate the function", y0=95.0, y1=106.0, btype=BlockType.LIST_ITEM)
    zones = build_zones(pages, [para, item])
    assert "b2" in zones


def test_content_window_clamps_below_page_number_band() -> None:
    facts = PageFacts(page=1, width=460.0, height=660.0)
    bottom, top = content_window(facts)
    assert bottom == 30.0 and top < 660.0


def test_own_zone_falls_back_to_bbox_on_textless_page() -> None:
    """Scanned pages have no rows: the OCR bbox itself is the zone."""
    facts = PageFacts(page=1, width=460.0, height=660.0, lines=(), bg="#f5f5f0")
    block = _block("b1", _LONG, y0=100.0, y1=130.0)
    zone = own_zone(block, facts)
    assert zone is not None
    assert zone.kind == "own"
    assert zone.y0 <= 100.0 and zone.y1 >= 130.0
    assert zone.x0 <= 100.0 and zone.x1 >= 400.0


def test_build_zones_claims_side_column_continuation() -> None:
    """A paragraph flowing into the next column is claimed by x-disjoint evidence."""
    left = _LONG[:120]
    right = _LONG[120:]
    pages = {
        1: PageFacts(
            page=1,
            width=600.0,
            height=660.0,
            lines=(
                _line(left, 500.0, x0=60.0, x1=280.0),
                _line(right, 500.0, x0=320.0, x1=540.0),
            ),
        )
    }
    block = _block("b1", _LONG, y0=495.0, y1=520.0, x0=60.0, x1=280.0)
    zones = build_zones(pages, [block])
    block_zones = zones["b1"]
    assert len(block_zones) == 2
    # Reading order: left column first, then the side-column continuation.
    assert block_zones[0].x0 < block_zones[1].x0
    assert block_zones[1].kind == "continuation"


def test_boxes_for_respects_zone_height() -> None:
    zone = Zone("b1", 1, 100.0, 100.0, 400.0, 160.0, 10.0)
    boxes = _boxes_for(zone, 10.0)
    assert len(boxes) >= 3
    assert all(zone.y0 - 2 <= b[1] < b[3] <= zone.y1 + 2 for b in boxes)
    assert _boxes_for(Zone("b1", 1, 100.0, 100.0, 400.0, 104.0, 10.0), 18.0) == []


def test_run_zone_keeps_a_wrapped_fragment_row() -> None:
    """A run whose only weak row is a wrapped fragment is still continuation."""
    source = _LONG[:70] + " expressed by"
    run = (_line(_LONG[:70], 120.0), _line("expressed by", 108.0))
    block = _block("b1", source, y0=115.0, y1=132.0)
    zone = _run_zone(block, run, 1, strong=False)
    assert zone is not None
    assert len(zone.rows) == 2


def test_run_zone_rejects_a_run_with_a_foreign_row() -> None:
    """Judgeable foreign rows still veto: one match out of two is not enough."""
    source = _LONG[:70]
    run = (_line(_LONG[:70], 120.0), _line("Metal Gate Voltage", 108.0))
    block = _block("b1", source, y0=115.0, y1=132.0)
    assert _run_zone(block, run, 1, strong=False) is None


def test_build_zones_claims_every_run_between_formula_islands() -> None:
    """Fragments separated by two display formulas join as one paragraph."""
    first, second, third = _LONG[:70], _LONG[70:140], _LONG[140:]
    pages = {
        1: PageFacts(
            page=1,
            width=460.0,
            height=660.0,
            lines=(
                _line(first, 500.0),  # seed row
                _line(second, 430.0),
                _line(third, 390.0),
            ),
        )
    }
    block = _block("b1", _LONG, y0=495.0, y1=520.0)
    formula_a = _block("f1", "E = m c^2", y0=440.0, y1=470.0, btype=BlockType.FORMULA)
    formula_b = _block("f2", "F = m a", y0=400.0, y1=425.0, btype=BlockType.FORMULA)
    zones = build_zones(pages, [block, formula_a, formula_b])
    block_zones = zones["b1"]
    assert [z.kind for z in block_zones] == ["own", "continuation", "continuation"]
    assert block_zones[0].y1 > block_zones[1].y1 > block_zones[2].y1
    assert block_zones[2].rows == (third,)


def test_flow_prefix_reports_the_text_it_actually_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zone's text is the exact prefix the fitter proofed.

    ``split_clauses`` now reconstructs the source exactly (zero-width
    separators), so the consumed prefix is a true slice of the original and no
    un-measured characters ride along at a zone boundary.
    """
    from ubt.adapters.pdf.rigid.rows import split_clauses
    from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter

    measured: list[str] = []

    def _fake_flow(
        self: RigidTypesetter, text: str, boxes: object, size: float
    ) -> list[str] | None:
        measured.append(text)
        return [text] if len(text) <= 40 else None

    monkeypatch.setattr(RigidTypesetter, "_flow", _fake_flow)
    source = "First clause here,  second clause here,  third clause here."
    ts = RigidTypesetter()
    result = ts._flow_prefix(source, [(0.0, 0.0, 100.0, 10.0)], 11.0)

    assert result is not None
    lines, consumed, remaining = result
    assert consumed and remaining
    assert "".join(split_clauses(source)) == source
    assert consumed + remaining == source
    assert consumed in measured, "the plan must carry exactly the proofed text"
    assert source[: len(consumed)] == consumed


def test_row_is_continuation_robust_to_math_ocr_discrepancy() -> None:
    """A cross-page continuation row must match its block source even when
    different PDF extraction engines produce minor character-level OCR differences
    in inline math symbols or subscripts (e.g. Docling vs pdfium).

    Regression: In 2608.25512v1 page 39, the top line had '| d' while source had '| d_n',
    and page 47 had superscript/subscript ordering flipped in math. Strict substring
    matching failed, causing top lines of page 39 and 47 to remain unrendered English.
    """
    from ubt.adapters.pdf.rigid.rows import _row_is_continuation

    # Page 39 case: dropped tiny subscript in math symbol '| d' vs '| d n'
    row_text = "at its key and its presence, both determined by 𝜎 𝑛 together with the 𝜎 𝑚 | 𝑑 , and writes what it"
    block_source = (
        "Proof. For confinement, by induction on the construction. For clause (2), "
        "a stage reads the binding at its key and its presence, both determined by 𝜎 𝑛 "
        "together with the 𝜎 𝑚 | 𝑑 𝑛 , and writes what it read into the same two parts."
    )
    assert _row_is_continuation(row_text, block_source)

    # Page 47 case: subscript/superscript ordering in math symbol
    row_text2 = "In particular 𝜎 𝑛 𝑢⁺¹ = ⌀ , which is the premise an O-Remove of 𝑛 carries."
    block_source2 = (
        "Corollary 69. (Terminal recovery.) Let an episode of 𝑛 open at 𝑏 and close at 𝑢 . "
        "Then, with 𝑡 1 < ⋯ < 𝑡 𝑙 as in Theorem 68, In particular 𝜎 𝑢+1 𝑛 = ⌀ , which is the "
        "premise an O-Remove of 𝑛 carries."
    )
    assert _row_is_continuation(row_text2, block_source2)

    # Negative case: completely unrelated text must NOT match
    unrelated = "This is a completely different theorem about something else entirely."
    assert not _row_is_continuation(unrelated, block_source)


@pytest.mark.fast
def test_latin_splits_reconstruct_the_source() -> None:
    from ubt.adapters.pdf.rigid.rows import split_clauses, split_sentences

    src = "Bonjour le monde. Ceci est un test, avec des virgules; et des deux-points: oui!"
    assert "".join(split_clauses(src)) == src
    assert "".join(split_sentences(src)) == src

    cjk = "第一句。第二句，第三句；第四句：第五句。"
    assert "".join(split_clauses(cjk)) == cjk


def _rect_zone(block_id: str, page: int, y0: float, y1: float, rows: tuple[str, ...] = ()) -> Zone:
    return Zone(
        block_id=block_id, page=page, x0=60.0, y0=y0, x1=520.0, y1=y1, base_size=10.0, rows=rows
    )


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


def test_region_floor_lets_a_small_print_block_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """M2: a block that fits only below the body floor still renders in a
    small-print region, instead of failing closed to source-visible.

    ``_flow`` is the fit kernel; here it accepts only sizes <= 6.8pt, so the
    block is un-fittable at the 7.0 body floor and fittable at the 6.5 caption/
    footnote floor. ``_paginate`` must honour the floor it is handed.
    """
    from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter

    def _fake_flow(
        self: RigidTypesetter, text: str, boxes: object, size: float
    ) -> list[str] | None:
        return [text] if size <= 6.8 else None

    monkeypatch.setattr(RigidTypesetter, "_flow", _fake_flow)
    ts = RigidTypesetter(target_lang="zh")
    zone = _rect_zone("f1", 1, 700.0, 715.0)

    assert ts._paginate("脚注内容", (zone,), min_font_pt=7.0) is None
    fitted = ts._paginate("脚注内容", (zone,), min_font_pt=6.5)
    assert fitted is not None
    assert fitted[0].size < 7.0


def test_plan_blocks_applies_the_region_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """M2 end-to-end: ``_plan_blocks`` resolves the floor from ``layout_role``.

    A caption whose text only fits below the body floor renders (the caption
    floor applies) while the identical body block spills — the evidence the
    synthetic corpora cannot provide, because captions/footnotes are classified
    only by docling, which the ``dev`` extra does not install.
    """
    from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter
    from ubt.adapters.pdf.rigid.zones import build_zones
    from ubt.core.ir.models import LayoutRole

    def _fake_flow(
        self: RigidTypesetter, text: str, boxes: object, size: float
    ) -> list[str] | None:
        return [text] if size <= 6.8 else None

    monkeypatch.setattr(RigidTypesetter, "_flow", _fake_flow)
    src = "Figure 1: a caption long enough to need shrinking below the body floor."
    facts = PageFacts(
        page=1, width=460.0, height=660.0, lines=(LineBox(src, (100.0, 400.0, 400.0, 409.0)),)
    )
    caption = _block("cap", src, page=1, y0=400.0, y1=409.0).model_copy(
        update={"layout_role": LayoutRole.CAPTION}
    )
    body = _block("body", src, page=1, y0=400.0, y1=409.0).model_copy(
        update={"layout_role": LayoutRole.BODY}
    )

    ts = RigidTypesetter(target_lang="zh")
    _paints, cap_report = ts._plan_blocks([caption], build_zones({1: facts}, [caption]), {1: 660.0})
    _paints2, body_report = ts._plan_blocks([body], build_zones({1: facts}, [body]), {1: 660.0})
    assert "cap" in cap_report.rendered_blocks
    assert "body" not in body_report.rendered_blocks
    assert ("body", "spill") in body_report.skipped


def test_plan_blocks_stamps_the_region_floor_for_the_emitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-region floor must reach the emitted ``#ubt-fit`` min-sz.

    Regression: planning used the caption/footnote floor (6.5) but ``_zone_typst``
    emitted the *global* ``self.min_font_pt`` (7.0) as ``min-sz``, so
    ``min-sz > base-sz`` and Typst's fit loop never shrank — the safety net was
    inert exactly where it was needed and overflow was clipped.
    """
    from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter
    from ubt.adapters.pdf.rigid.zones import build_zones
    from ubt.core.ir.models import LayoutRole

    def _fake_flow(
        self: RigidTypesetter, text: str, boxes: object, size: float
    ) -> list[str] | None:
        return [text] if size <= 6.8 else None

    monkeypatch.setattr(RigidTypesetter, "_flow", _fake_flow)
    src = "Figure 1: a caption long enough to need shrinking below the body floor."
    facts = PageFacts(
        page=1, width=460.0, height=660.0, lines=(LineBox(src, (100.0, 400.0, 400.0, 409.0)),)
    )
    caption = _block("cap", src, page=1, y0=400.0, y1=409.0).model_copy(
        update={"layout_role": LayoutRole.CAPTION}
    )
    ts = RigidTypesetter(target_lang="zh")
    paints, _ = ts._plan_blocks([caption], build_zones({1: facts}, [caption]), {1: 660.0})
    plan = paints[1][0]
    assert plan.min_font_pt == 6.5  # caption region floor, not the 7.0 body default
    assert plan.min_font_pt <= plan.size  # min-sz must not exceed base-sz


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
    from ubt.adapters.pdf.rigid.zones import PageFacts, _resolve_horizontal_span_and_align
    from ubt.adapters.pdf.textgeom import LineBox

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
    from ubt.adapters.pdf.rigid.zones import PageFacts, build_zones
    from ubt.adapters.pdf.textgeom import LineBox
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


@pytest.mark.fast
def test_glue_run_and_extract_lines_cache_fast_on_dense_fragments() -> None:
    """_glue_run must precompute dehyph() in O(K) instead of O(K^2) inner-loop regex calls."""
    import time

    from ubt.adapters.pdf.textgeom import LineBox, _glue_run

    run = [
        LineBox(f"fragment_{i}_with_some_text", (float(i * 2), 100.0, float(i * 2 + 10), 110.0))
        for i in range(600)
    ]
    t0 = time.perf_counter()
    glued = _glue_run(run)
    elapsed = time.perf_counter() - t0
    assert glued.text
    assert elapsed < 0.25, f"_glue_run took {elapsed:.3f}s on 600 fragments (expected < 0.25s)"

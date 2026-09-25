"""Anchored zone assembly: bbox seeds, growth guards, tail adoption."""

import pytest

from ubt.adapters.pdf.rigid.typesetter import _boxes_for
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

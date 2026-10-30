"""Stage 3A style-group solve, against a deterministic fake measure.

The fake measure is a pure character model — no Typst, no subprocess — which
is what keeps this tier fast and offline: ``capacity_chars = width / (0.5 x
size)`` (CJK glyphs are ~1em wide, so 0.5em is a deliberately *wrong* average
that still orders candidates correctly), ``lines = ceil(len / capacity)``,
``height = lines x leading``. What the tests pin is the *search policy*, not
wrapping fidelity: knob preference order (P10 — size is the last resort), the
role floors, the 1.10x cap, and the Stage 4 budget handoff. Real wrapping is
the measurement backend's problem, and the Typst probe is calibrated
elsewhere (``tests/unit/adapters/test_typst_measurement.py``).

Every overflow scenario below is built by arithmetic on that model — at
10pt/350pt one line holds 70 chars and costs one leading — so each test names
the exact candidate it expects the search to land on.
"""

from __future__ import annotations

import math

import pytest

from ubt.core.layout.solver import (
    BODY_FLOOR_PT,
    NON_BODY_FLOOR_PT,
    SOURCE_STYLE_FITS,
    StyleBlockInput,
    condensation_char_budget,
    solve_style_group,
)

pytestmark = pytest.mark.fast


class _Measured:
    __slots__ = ("line_count", "height_pt")

    def __init__(self, line_count: int, height_pt: float) -> None:
        self.line_count = line_count
        self.height_pt = height_pt


def fake_measure(
    text: str, size_pt: float, tracking_em: float, leading_pt: float, width_pt: float
) -> _Measured:
    """Pure character model: 0.5em per char, hard wrap at the width."""
    if not text.strip():
        return _Measured(0, 0.0)
    capacity = max(1, int(width_pt / (0.5 * size_pt)))
    lines = math.ceil(len(text) / capacity)
    return _Measured(lines, lines * leading_pt)


def make_block(
    index: int,
    *,
    text: str = "翻译后的正文文本",
    size: float = 10.0,
    leading: float = 13.0,
    width: float = 350.0,
    height: float = 100.0,
    borrowable: float = 0.0,
    role: str = "body",
) -> StyleBlockInput:
    return StyleBlockInput(
        index=index,
        text=text,
        role=role,  # type: ignore[arg-type]
        source_size_pt=size,
        source_leading_pt=leading,
        source_width_pt=width,
        source_height_pt=height,
        borrowable_pt=borrowable,
    )


def test_source_style_that_fits_is_returned_untouched() -> None:
    block = make_block(0, text="短文本")
    solution = solve_style_group([block], available_height_pt=200.0, measure=fake_measure)
    assert solution.fits
    assert solution.reason == SOURCE_STYLE_FITS
    assert solution.size_pt == 10.0
    assert solution.tracking_em == 0.0
    assert solution.blocks[0].line_count == 1
    assert solution.blocks[0].char_budget is None, "a fitting group owes Stage 4 nothing"


def test_leading_is_tightened_before_size_is_reduced() -> None:
    """P10 in action: 6 lines cost 78pt at the source leading; 70pt are available.

    Tracking does not change a character count in this model, so the knob that
    saves the page is leading: the search must land on the source *size* with a
    tighter leading (6 x 11.5 = 69pt at λ=1.15), never on a smaller font.
    """
    block = make_block(0, text="字" * 360, leading=13.0)  # 6 lines x 13pt = 78pt
    solution = solve_style_group([block], available_height_pt=70.0, measure=fake_measure)
    assert solution.fits
    assert solution.size_pt == 10.0, "size is the last resort; micro knobs fit first"
    assert solution.leading_ratio == pytest.approx(1.15), (
        "leading tightened (1.30 -> 1.15) instead of shrinking the font"
    )


def test_size_reduces_only_after_micro_knobs_are_exhausted() -> None:
    """11 lines cannot fit 106pt at any leading >= 1.08 (min 118.8pt).

    One size step (10 -> 9.75pt) lifts the line capacity 70 -> 71, so 710 chars
    wrap to 10 lines and 10 x 9.75 x 1.08 = 105.3pt fits. The solve must stop
    exactly there, not walk to the floor.
    """
    block = make_block(0, text="字" * 710)
    solution = solve_style_group([block], available_height_pt=106.0, measure=fake_measure)
    assert solution.fits
    assert solution.size_pt == pytest.approx(9.75), "one step, not a walk to the floor"


def test_borrow_is_used_before_size_reduces() -> None:
    """Δw (right-borrow) outranks the font size: 380 chars wrap 6->5 lines at +50pt.

    6 lines x 13pt = 78pt overflows the 70pt track at the source width; the
    full borrow (400pt measure, 80 chars/line) wraps to 5 lines = 65pt.
    """
    block = make_block(0, text="字" * 380, borrowable=50.0)
    solution = solve_style_group([block], available_height_pt=70.0, measure=fake_measure)
    assert solution.fits
    assert solution.size_pt == 10.0, "borrow absorbed the overflow; the font stayed"
    assert solution.blocks[0].width_pt == pytest.approx(400.0)


def test_body_floor_is_max_of_eighty_percent_and_absolute_seven_point_five() -> None:
    """A 9pt source bottoms out at the absolute 7.5pt, not 0.80 x 9 = 7.2pt."""
    block = make_block(0, text="字" * 2000, size=9.0, leading=12.0, height=300.0)
    solution = solve_style_group([block], available_height_pt=300.0, measure=fake_measure)
    if solution.fits:
        assert solution.size_pt >= BODY_FLOOR_PT
    else:
        assert solution.size_pt == BODY_FLOOR_PT


def test_non_body_role_may_go_lower_but_never_below_five_point_five() -> None:
    block = make_block(0, text="字" * 3000, size=9.0, leading=12.0, height=200.0, role="caption")
    solution = solve_style_group([block], available_height_pt=200.0, measure=fake_measure)
    assert solution.size_pt >= NON_BODY_FLOOR_PT


def test_a_source_below_the_floor_still_returns_the_floor_candidate() -> None:
    """A 7pt body source is below its own floor: the floor is the only candidate."""
    block = make_block(0, text="字" * 50, size=7.0, leading=9.0, height=40.0)
    solution = solve_style_group([block], available_height_pt=400.0, measure=fake_measure)
    assert solution.fits
    assert solution.size_pt == BODY_FLOOR_PT


def test_size_never_exceeds_the_one_ten_expansion_cap() -> None:
    """The search starts at the source size and only descends; 1.10x is a ceiling."""
    block = make_block(0, text="短", height=500.0)
    solution = solve_style_group([block], available_height_pt=500.0, measure=fake_measure)
    assert solution.size_pt <= 1.10 * 10.0


def test_group_shares_one_size_even_when_blocks_differ_in_source_size() -> None:
    """§8.2: 字号差异只能来自 StyleRole — one group, one size."""
    small = make_block(0, text="字" * 30, size=9.5, leading=13.0, height=60.0)
    large = make_block(1, text="字" * 30, size=10.5, leading=14.5, height=60.0)
    solution = solve_style_group([small, large], available_height_pt=200.0, measure=fake_measure)
    assert solution.fits
    # The group baseline is the largest source size; both blocks get that size.
    assert solution.size_pt == 10.5
    assert all(fit.char_budget is None for fit in solution.blocks)


def test_mixed_roles_are_refused_not_silently_merged() -> None:
    body = make_block(0, role="body")
    caption = make_block(1, role="caption")
    with pytest.raises(ValueError, match="one role"):
        solve_style_group([body, caption], available_height_pt=100.0, measure=fake_measure)


def test_no_fit_returns_floor_style_with_measured_char_budgets() -> None:
    """Stage 4 handoff: fits=False, floor style, per-block budget from real measure calls."""
    text = "字" * 5000
    block = make_block(0, text=text, height=50.0)
    solution = solve_style_group([block], available_height_pt=50.0, measure=fake_measure)
    assert not solution.fits
    assert solution.reason == "floor_exhausted"
    assert solution.size_pt == 8.0, "reported at the body floor, not the failed source style"
    budget = solution.blocks[0].char_budget
    assert budget is not None and 0 < budget < len(text), "the budget must be a real prefix bound"
    # The budget is honest: re-measuring the prefix at the floor style fits.
    at_budget = fake_measure(
        text[:budget],
        solution.size_pt,
        solution.tracking_em,
        solution.leading_pt,
        solution.blocks[0].width_pt,
    )
    assert at_budget.height_pt <= 50.0 + 1e-9
    over_budget = fake_measure(
        text[: budget + 1],
        solution.size_pt,
        solution.tracking_em,
        solution.leading_pt,
        solution.blocks[0].width_pt,
    )
    assert over_budget.height_pt > 50.0


def test_budget_is_balanced_across_blocks_by_source_height_share() -> None:
    """Two starving blocks: each budget is measured against its own share, not the page."""
    tall = make_block(0, text="字" * 900, height=180.0)
    short = make_block(1, text="字" * 900, height=60.0)
    solution = solve_style_group([tall, short], available_height_pt=100.0, measure=fake_measure)
    assert not solution.fits
    budgets = [fit.char_budget for fit in solution.blocks]
    first, second = budgets[0], budgets[1]
    assert first is not None and second is not None
    assert first > second, "the taller source block earns the larger share"


def test_empty_text_needs_no_lines_and_never_blocks_the_group() -> None:
    blank = make_block(0, text="", height=0.0)
    body = make_block(1, text="短文本", height=200.0)
    solution = solve_style_group([blank, body], available_height_pt=200.0, measure=fake_measure)
    assert solution.fits
    assert solution.blocks[0].line_count == 0
    assert solution.blocks[0].height_pt == 0.0


def test_condensation_budget_bounds_are_exact() -> None:
    """The budget binary search returns the largest prefix that measurably fits."""
    text = "字" * 100
    # capacity at 10pt/350pt = 70 chars/line; 2 lines need 2 x leading.
    budget = condensation_char_budget(
        text,
        fake_measure,
        size_pt=10.0,
        tracking_em=0.0,
        leading_pt=13.0,
        width_pt=350.0,
        available_height_pt=26.0,
    )
    assert budget == 100
    tighter = condensation_char_budget(
        text,
        fake_measure,
        size_pt=10.0,
        tracking_em=0.0,
        leading_pt=13.0,
        width_pt=350.0,
        available_height_pt=13.0,
    )
    assert tighter == 70
    none_fits = condensation_char_budget(
        "字",
        fake_measure,
        size_pt=10.0,
        tracking_em=0.0,
        leading_pt=13.0,
        width_pt=350.0,
        available_height_pt=0.0,
    )
    assert none_fits == 0

"""The BreakageSolver: flowing one target across a chain of physical boxes.

Pure logic -- the measurement is injected -- so these pin the break policy
(punctuation-preferring, capacity-bounded) without a typesetter.
"""

from __future__ import annotations

import pytest

from ubt.model.span import PhysicalBox
from ubt.render.flow import solve_flow

pytestmark = pytest.mark.fast


def _measure(text: str, width_pt: float) -> float:
    """Height model: one point per character, independent of width."""
    return float(len(text))


def _box(height: float, width: float = 1000.0) -> PhysicalBox:
    return PhysicalBox.of(1, (0.0, 0.0, width, height))


def test_no_boxes_places_nothing() -> None:
    assert solve_flow("text", (), _measure) == ()


def test_a_single_box_holds_the_whole_text_when_it_fits() -> None:
    (placement,) = solve_flow("short", (_box(1000.0),), _measure)
    assert placement.text == "short"


def test_text_breaks_at_the_largest_fitting_punctuation_boundary() -> None:
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta."
    first, second = solve_flow(text, (_box(20.0), _box(1000.0)), _measure)
    # 20pt fits up to "Alpha beta gamma. " -> the break lands after the period.
    assert first.text == "Alpha beta gamma."
    assert second.text == "Delta epsilon zeta. Eta theta."


def test_a_cjk_subtitle_break_keeps_the_colon_on_the_first_line() -> None:
    # A CJK title has no spaces, so its only break candidates are punctuation.
    # Breaking after the closing bracket instead of after the colon would lead
    # the second line with "：".
    text = "弹性计算（DSec）：用于大规模训练的沙箱基础设施"
    first, second = solve_flow(text, (_box(11.0), _box(1000.0)), _measure)
    assert first.text == "弹性计算（DSec）："
    assert second.text == "用于大规模训练的沙箱基础设施"


def test_a_box_too_small_for_any_candidate_is_left_empty() -> None:
    first, second = solve_flow("Alpha beta gamma.", (_box(3.0), _box(1000.0)), _measure)
    assert first.text == ""
    assert second.text == "Alpha beta gamma."


def test_the_last_box_receives_the_remainder_even_when_it_overflows() -> None:
    text = "Alpha beta gamma. Delta epsilon zeta."
    first, second = solve_flow(text, (_box(12.0), _box(1.0)), _measure)
    assert first.text == "Alpha beta"
    # The tail is not dropped just because the final box is too small.
    assert second.text == "gamma. Delta epsilon zeta."


def test_flow_preserves_the_words_in_order() -> None:
    text = "One two three. Four five six. Seven eight nine."
    boxes = (_box(15.0), _box(15.0), _box(1000.0))
    joined = " ".join(
        placement.text for placement in solve_flow(text, boxes, _measure) if placement.text
    )
    assert joined.split() == text.split()

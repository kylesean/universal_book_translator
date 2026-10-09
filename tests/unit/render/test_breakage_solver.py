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


# --------------------------------------------------------------------------- #
# CJK without punctuation: character-level fallback (避头尾).
# --------------------------------------------------------------------------- #


def test_an_unpunctuated_cjk_run_fills_the_first_box() -> None:
    # No punctuation cut exists, so the old solver returned nothing for the
    # first box: it was left empty and the whole run overflowed the last box and
    # descended to source. The fallback breaks between characters.
    text = "这是一个没有任何标点符号的长中文句子需要被断流到多个物理框里"
    first, second = solve_flow(text, (_box(20.0), _box(1000.0)), _measure)
    assert first.text == text[:20]
    assert second.text == text[20:]


def test_the_cjk_fallback_does_not_start_a_line_with_closing_punctuation() -> None:
    # The box fits 5 characters; the punctuation cut (after "。") is at offset 6
    # and does not fit. The fallback would happily cut at offset 5, leading the
    # next line with "。"; 避头尾 forbids it, so the break lands at offset 4.
    text = "甲乙丙丁戊。己庚辛壬癸"
    first, second = solve_flow(text, (_box(5.0), _box(1000.0)), _measure)
    assert first.text == "甲乙丙丁"
    assert second.text == "戊。己庚辛壬癸"
    assert first.text + second.text == text


def test_a_latin_run_is_never_split_mid_word_by_the_fallback() -> None:
    # The fallback offers cuts only next to a CJK char, so an embedded Latin
    # word keeps its space boundaries. A box too small for the whole Latin run
    # breaks at the surrounding spaces, never inside the word.
    text = "中文甲乙丙 thequickbrownfox 丁戊己庚辛"
    parts = solve_flow(text, (_box(10.0), _box(1000.0)), _measure)
    placed = [p.text for p in parts if p.text]
    # The Latin token survives whole in exactly one fragment.
    assert sum("thequickbrownfox" in chunk for chunk in placed) == 1


def test_a_box_too_small_for_one_cjk_char_still_descends() -> None:
    # Even the fallback needs one character to fit; a zero-capacity box yields
    # nothing, so the last-box remainder rule still applies.
    text = "甲乙丙丁"
    first, second = solve_flow(text, (_box(0.0), _box(1000.0)), _measure)
    assert first.text == ""
    assert second.text == text

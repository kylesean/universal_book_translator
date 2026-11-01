from __future__ import annotations

import pytest

from ubt.adapters.pdf.docling_crosscheck import reinsert_inline_math

pytestmark = pytest.mark.fast


def test_an_inline_variable_is_restored_to_its_slot() -> None:
    text = "The coefficient controls how far guidance extrapolates. ω"
    witness = "The coefficient ω controls how far guidance extrapolates."
    assert (
        reinsert_inline_math(text, witness)
        == "The coefficient ω controls how far guidance extrapolates."
    )


def test_a_whole_pile_of_inline_math_is_restored() -> None:
    text = (
        "For a fixed input prefix, let be the initial state of the recurrent "
        "block and its state after iterations. h 0 h k k f R h R"
    )
    witness = (
        "For a fixed input prefix let h0 be the initial state of the recurrent "
        "block and hk its state after k iterations."
    )
    assert reinsert_inline_math(text, witness) == (
        "For a fixed input prefix, let h0 be the initial state of the recurrent "
        "block and hk its state after k iterations."
    )


def test_a_witness_hyphenated_word_does_not_block_the_repair() -> None:
    text = "The reference and final state enter the interface. hR"
    witness = "The reference hR and final state en- ter the interface."
    assert reinsert_inline_math(text, witness) == (
        "The reference hR and final state enter the interface."
    )


def test_a_bare_number_is_not_moved() -> None:
    assert (
        reinsert_inline_math("See the results in Figure 4.", "See the results in Figure 4.") is None
    )


def test_a_number_inside_a_pile_is_still_pile() -> None:
    text = "Let be the state after steps. h 0 k"
    witness = "Let h0 be the state after k steps."
    assert reinsert_inline_math(text, witness) == "Let h0 be the state after k steps."


def test_prose_disagreement_is_rejected() -> None:
    text = "The coefficient controls how far guidance extrapolates. ω"
    witness = "A completely different sentence about something else."
    assert reinsert_inline_math(text, witness) is None


def test_text_without_a_trailing_pile_is_untouched() -> None:
    assert (
        reinsert_inline_math(
            "An ordinary sentence with no math.", "An ordinary sentence with no math."
        )
        is None
    )


def test_a_witness_with_no_math_is_rejected() -> None:
    text = "The final state matters here. hR"
    witness = "The final state matters here."
    assert reinsert_inline_math(text, witness) is None

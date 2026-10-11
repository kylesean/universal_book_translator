"""Put back the inline math Docling strips out of a sentence.

IBM Docling's PDF reader pulls inline math (``h0``, ``ω``, ``hR``) out of a
paragraph and concatenates the dropped glyphs at the paragraph's end. The
sentence ships with a hole where the math was ("let be the initial state ... :
h 0 h k k f R h R"), and the model translates around the hole. The page's own
pdfium lines carry the sentence with the math in place, so the cross-check pass
restores it before anything is translated.

The pure function is pinned here; the pdfium-witness wiring lives in the adapter.
"""

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
    """The real p3 shape: four math tokens stripped and piled at the end."""
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
    """pdfium breaks a word across a line ("represen-" + "tation"); the tokens
    still have to align, or every long paragraph would be skipped."""
    text = "The reference and final state enter the interface. hR"
    witness = "The reference hR and final state en- ter the interface."
    assert reinsert_inline_math(text, witness) == (
        "The reference hR and final state enter the interface."
    )


def test_a_bare_number_is_not_moved() -> None:
    """A paragraph ending in a number ("in Figure 4.") is prose, not a pile."""
    assert (
        reinsert_inline_math("See the results in Figure 4.", "See the results in Figure 4.") is None
    )


def test_a_number_inside_a_pile_is_still_pile() -> None:
    """``h0`` reaches Docling as ``h`` ``0``: the trailing number is part of the
    pile, so the pile guard must not stop at it."""
    text = "Let be the state after steps. h 0 k"
    witness = "Let h0 be the state after k steps."
    assert reinsert_inline_math(text, witness) == "Let h0 be the state after k steps."


def test_prose_disagreement_is_rejected() -> None:
    """A witness for a different sentence must not be spliced in."""
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
    """Nothing to restore: the doc's pile must not be dropped for nothing."""
    text = "The final state matters here. hR"
    witness = "The final state matters here."
    assert reinsert_inline_math(text, witness) is None

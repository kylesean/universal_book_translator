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


def test_a_glued_pile_with_primes_and_calls_is_restored() -> None:
    """The pile Docling emits is glyph soup: ``z = ′ lm _ head(coda(h ))``.

    Token-by-token math classification stops at ``′``/``_``/``))`` and leaves the
    whole pile in the sentence (the ``ω ω = 0`` placeholder readers saw).
    """
    text = (
        "The coefficient sets the strength of the contrast, with recovering the "
        "unguided state . The combined state then passes through the coda layers "
        "and language modeling head: . ω ω = 0 hR z = ′ lm _ head(coda(h )) ′ "
        "lm _ head"
    )
    witness = (
        "The coefficient ω sets the strength of the contrast with ω = 0 recovering "
        "the unguided state hR. The , combined state then passes through the coda "
        "layers and language modeling head: z′ = lm_head(coda(h′)) ."
    )
    assert reinsert_inline_math(text, witness) == (
        "The coefficient ω sets the strength of the contrast, with ω = 0 recovering "
        "the unguided state hR. The combined state then passes through the coda "
        "layers and language modeling head: z′ = lm_head(coda(h′))."
    )


def test_a_pile_that_is_a_display_formula_is_dropped() -> None:
    """``We set <equation>``: the equation is a separate formula block, and the
    witness line stops at ``We set``. Keeping the pile would leave it dangling."""
    text = (
        "Let and be the two largest token probabilities in . We set "
        "ω pR,(1) pR,(2) p = R softmax(z ) R"
    )
    witness = (
        "Let pR,(1) and pR,(2) be the two largest token probabilities in pR = softmax(zR). We set"
    )
    assert reinsert_inline_math(text, witness) == (
        "Let pR,(1) and pR,(2) be the two largest token probabilities in pR = softmax(zR). We set"
    )


def test_a_lone_prime_from_line_segmentation_is_not_inserted() -> None:
    text = "Applying the same rule to LoopCD-Hidden would require a pass on to obtain . zR hR pR h′"
    witness = "Applying the same rule ′ to LoopCD-Hidden would require a pass on hR to obtain pR"
    assert reinsert_inline_math(text, witness) == (
        "Applying the same rule to LoopCD-Hidden would require a pass on hR to obtain pR."
    )


def test_a_word_the_doc_kept_is_not_inserted_twice() -> None:
    """``lm`` in ``lm head`` is prose the doc kept, not inline math to splice."""
    text = "LoopCD-Hidden combines and (Eq. (2)) and evaluates the coda and lm _ head once. h1 hR"
    witness = "LoopCD-Hidden combines h1 and hR (Eq. (2)) and evaluates the coda and lm head once."
    assert reinsert_inline_math(text, witness) == (
        "LoopCD-Hidden combines h1 and hR (Eq. (2)) and evaluates the coda and lm _ head once."
    )

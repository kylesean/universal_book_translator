"""Restore a heading's section number that Docling glued to the wrong end.

IBM Docling's PDF reader emits ``"Guidance Space 3.1"`` for a page that reads
``"3.1 Guidance Space"`` -- it reads the heading text run and the number run out
of order. The translation is then faithful to its (wrong) input, so the number
ships at the end of the CJK heading (``引导空间 3.1``). The page's own pdfium
line witness spells the heading correctly, so the cross-check pass restores the
order before anything is translated.

The pure function is pinned here; the pdfium-witness wiring lives in the adapter.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.docling_crosscheck import restore_heading_number_order

pytestmark = pytest.mark.fast


def test_a_trailing_section_number_is_moved_to_the_front() -> None:
    assert (
        restore_heading_number_order("Guidance Space 3.1", "3.1 Guidance Space")
        == "3.1 Guidance Space"
    )


def test_a_single_level_number_is_restored() -> None:
    assert restore_heading_number_order("Conclusion 6", "6 Conclusion") == "6 Conclusion"


def test_a_three_level_number_is_restored() -> None:
    assert restore_heading_number_order("The probability form A.3", "A.3 The probability form") == (
        "A.3 The probability form"
    )


def test_a_heading_without_a_trailing_number_is_untouched() -> None:
    assert restore_heading_number_order("Guidance Space", "Guidance Space") is None


def test_a_witness_that_does_not_lead_with_the_number_is_rejected() -> None:
    """No witness confirmation: the reorder would be a guess, so refuse it."""
    assert restore_heading_number_order("Model 2", "Model 2") is None


def test_a_mismatched_number_is_rejected() -> None:
    assert restore_heading_number_order("Guidance Space 3.1", "3.2 Guidance Space") is None


def test_a_body_that_does_not_appear_in_the_witness_is_rejected() -> None:
    assert restore_heading_number_order("Something Else 3.1", "3.1 Guidance Space") is None


def test_a_heading_that_genuinely_ends_in_a_number_is_left_alone() -> None:
    """``AIME 2024`` ends in a number, but its witness leads with a word: refuse."""
    assert restore_heading_number_order("AIME 2024", "AIME 2024") is None

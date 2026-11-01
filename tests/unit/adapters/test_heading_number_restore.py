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
    assert restore_heading_number_order("Model 2", "Model 2") is None


def test_a_mismatched_number_is_rejected() -> None:
    assert restore_heading_number_order("Guidance Space 3.1", "3.2 Guidance Space") is None


def test_a_body_that_does_not_appear_in_the_witness_is_rejected() -> None:
    assert restore_heading_number_order("Something Else 3.1", "3.1 Guidance Space") is None


def test_a_heading_that_genuinely_ends_in_a_number_is_left_alone() -> None:
    assert restore_heading_number_order("AIME 2024", "AIME 2024") is None

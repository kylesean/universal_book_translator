"""LNDS page-number pruning, Calibre cleanup, academic-PDF math repair, block dedup.

The cleaner is a bag of pure line/text transforms plus two IR-level entry
points. The load-bearing facts pinned here:

* the ASCII-digit gate is deliberately narrower than ``str.isdigit()`` (Unicode
  digits crash ``int()`` and, historically, the whole chapter's ingestion);
* a monotonic page sequence is only recognised when it is both long enough and
  covers a large enough share of the digit lines (LNDS length / ratio gates);
* ordered-list markers introduced by a colon are protected *as a run*, so a
  four-item list of full sentences is not mistaken for page numbers;
* the CMap ``¼ -> =`` repair is evidence-gated, so genuine prose fractions
  (``¼ cup``) and currency (``paid $5``) survive;
* textbook cleanup strips boilerplate/page/header noise while a math/protected
  span only disables the CMap normalizer, never the boilerplate stripping;
* duplicate extraction fragments are removed only on identical normalized text
  *and* near-coincident boxes, keeping the larger fragment.
"""

from __future__ import annotations

import pytest

from ubt.core.cleaners.lnds_pruner import (
    _MAX_INT_DIGITS,
    LNDSPageCleaner,
    _area,
    _has_cmap_equation_evidence,
    _has_dollar_math,
    _iou,
    _norm_text,
    clean_calibre_and_lnds_pages,
    collect_dropped_line_indices,
    dedup_duplicate_blocks,
    detect_page_number_lines,
    is_ascii_digit_line,
    is_calibre_noise_line,
    is_likely_heading_or_list_item,
    normalize_academic_pdf_math,
    strip_calibre_markup,
    strip_textbook_ocr_artifacts,
)
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BoundingBox,
    IRBlock,
    make_element,
)

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    text: str,
    *,
    block_type: BlockType = BlockType.NARRATIVE,
    bbox: BoundingBox | None = None,
) -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=0,
        block_type=block_type,
        source_text=text,
        bbox=bbox,
    )
    return IRBlock(element=element)


def _bbox(page: int, x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
    return BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1)


# --------------------------------------------------------------------------- #
# _has_dollar_math
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", False),
        ("no dollars here", False),
        ("$$E = mc^2$$", True),
        ("the value $x^2$ grows", True),
        ("$5 and $10", False),  # unpaired currency, not math
        ("price $5", False),
    ],
)
def test_has_dollar_math(content: str, expected: bool) -> None:
    assert _has_dollar_math(content) is expected


# --------------------------------------------------------------------------- #
# is_ascii_digit_line
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("12", True),
        ("  007  ", True),
        ("0", True),
        ("", False),
        ("   ", False),
        ("1 2", False),
        ("12.3", False),
        ("-1", False),
        ("12a", False),
        ("\u00b2", False),  # superscript two: isdigit() True, int() raises
        ("\uff11", False),  # full-width one: isdigit() True
        ("\u0661\u0662", False),  # Arabic-Indic digits
    ],
)
def test_is_ascii_digit_line(line: str, expected: bool) -> None:
    assert is_ascii_digit_line(line) is expected


@pytest.mark.skipif(_MAX_INT_DIGITS == 0, reason="int()-digit limit disabled on this interpreter")
def test_the_int_digit_limit_is_enforced() -> None:
    assert is_ascii_digit_line("1" * _MAX_INT_DIGITS) is True
    assert is_ascii_digit_line("1" * (_MAX_INT_DIGITS + 1)) is False


# --------------------------------------------------------------------------- #
# detect_page_number_lines (LNDS)
# --------------------------------------------------------------------------- #


def test_a_short_run_is_not_a_page_sequence() -> None:
    assert detect_page_number_lines(["1", "2", "3"]) == set()


def test_an_empty_input_yields_nothing() -> None:
    assert detect_page_number_lines([]) == set()
    assert detect_page_number_lines(["alpha", "beta"]) == set()


def test_a_monotonic_run_is_kept() -> None:
    assert detect_page_number_lines(["1", "2", "3", "4"]) == {0, 1, 2, 3}


def test_non_digit_lines_are_skipped_in_the_sequence() -> None:
    assert detect_page_number_lines(["1", "x", "2", "x", "3", "x", "4"]) == {0, 2, 4, 6}


def test_min_length_is_honoured() -> None:
    assert detect_page_number_lines(["1", "2"], min_length=2) == {0, 1}


def test_a_mostly_monotonic_run_still_qualifies() -> None:
    # LNDS = [3, 4, 5, 6] over 5 digit lines -> ratio 0.8 >= 0.5.
    assert detect_page_number_lines(["10", "3", "4", "5", "6"]) == {1, 2, 3, 4}


def test_a_sequence_shorter_than_half_the_digit_lines_is_rejected() -> None:
    # Values 5,4,3,2,1,2,3,4,0: LNDS = [1,2,3,4], n = 9 -> ratio 0.444 < 0.5.
    assert detect_page_number_lines(["5", "4", "3", "2", "1", "2", "3", "4", "0"]) == set()


def test_the_ratio_gate_is_configurable() -> None:
    lines = ["5", "4", "3", "2", "1", "2", "3", "4", "0"]
    assert detect_page_number_lines(lines, min_ratio=0.4) == {4, 5, 6, 7}


# --------------------------------------------------------------------------- #
# strip_calibre_markup
# --------------------------------------------------------------------------- #


def test_calibre_pseudo_classes_are_removed() -> None:
    assert strip_calibre_markup("Hello {.calibre9}world") == "Hello world"


def test_calibre_link_anchors_are_removed() -> None:
    assert strip_calibre_markup("See (#calibre_link-12) here") == "See  here"
    assert strip_calibre_markup("text {#calibre_link-5 .foo}") == "text"


def test_redundant_bold_bracket_markers_are_unwrapped() -> None:
    assert strip_calibre_markup("[**bold**] and [**more**]") == "**bold** and **more**"


def test_plain_markdown_bold_is_untouched() -> None:
    assert strip_calibre_markup("**bold**") == "**bold**"


# --------------------------------------------------------------------------- #
# is_calibre_noise_line
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", False),
        ("   ", False),
        ("::: callout", True),
        (":::", True),
        ("some text .ct}", True),
        ("some text .cn}", True),
        ("some text .ct", False),
        ("ordinary prose", False),
    ],
)
def test_is_calibre_noise_line(line: str, expected: bool) -> None:
    assert is_calibre_noise_line(line) is expected


# --------------------------------------------------------------------------- #
# is_likely_heading_or_list_item
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", False),
        ("   ", False),
        ("# Heading", True),
        ("- bullet", True),
        ("* bullet", True),
        ("\u2022 bullet", True),
        ("Chapter 1", True),
        ("Section 2", True),
        ("Part Three", True),
        ("Step 4", True),
        ("A short title", True),
        ("This is a full sentence.", False),
        ("Is this a question?", False),
        ("Exclamation!", False),
        ("\u53e5\u53f7\u3002", False),
        ("\u95ee\u53e5\uff1f", False),
        ("\u611f\u53f9\uff01", False),
        ("clause;", False),
        ("clause\uff1b", False),
        ("x" * 100, False),
    ],
)
def test_is_likely_heading_or_list_item(line: str, expected: bool) -> None:
    assert is_likely_heading_or_list_item(line) is expected


def test_calibre_noise_is_never_a_heading() -> None:
    assert is_likely_heading_or_list_item("::: heading") is False
    assert is_likely_heading_or_list_item("title .ct}") is False


# --------------------------------------------------------------------------- #
# collect_dropped_line_indices
# --------------------------------------------------------------------------- #


def test_a_monotonic_page_run_is_dropped() -> None:
    lines = ["Intro text.", "1", "Body text.", "2", "More text.", "3", "End.", "4"]
    assert collect_dropped_line_indices(lines) == {1, 3, 5, 7}


def test_an_isolated_digit_line_is_not_a_page_number() -> None:
    lines = ["Hello", "5", "This is a full sentence."]
    assert collect_dropped_line_indices(lines) == set()


def test_strip_all_page_numbers_drops_even_a_lone_number() -> None:
    lines = ["Hello", "5", "This is a full sentence."]
    assert collect_dropped_line_indices(lines, strip_all_page_numbers=True) == {1}


def test_a_digit_before_a_heading_is_protected() -> None:
    lines = ["1", "Introduction", "2", "Body.", "3", "More.", "4", "End."]
    dropped = collect_dropped_line_indices(lines)
    assert 0 not in dropped
    assert {2, 4, 6} <= dropped


def test_blank_lines_do_not_break_heading_protection() -> None:
    assert collect_dropped_line_indices(["1", "", "Introduction"]) == set()


def test_a_colon_introduced_ordered_list_keeps_every_marker() -> None:
    lines = [
        "Steps:",
        "1",
        "Install the thing.",
        "2",
        "Run the thing.",
        "3",
        "Deploy the thing.",
        "4",
        "Verify the thing.",
    ]
    assert collect_dropped_line_indices(lines) == set()


def test_a_broken_list_run_is_no_longer_protected() -> None:
    lines = [
        "Steps:",
        "1",
        "Do it.",
        "7",
        "Body text.",
        "8",
        "More.",
        "9",
        "End.",
        "10",
    ]
    dropped = collect_dropped_line_indices(lines)
    assert 1 not in dropped
    assert {3, 5, 7, 9} <= dropped


def test_calibre_noise_lines_are_dropped() -> None:
    assert collect_dropped_line_indices(["::: a", "text", "b .ct}"]) == {0, 2}


def test_a_digit_next_to_calibre_noise_is_dropped() -> None:
    assert collect_dropped_line_indices(["Hello", "5", "::: footer"]) == {1, 2}


def test_a_digit_after_calibre_noise_is_dropped() -> None:
    assert collect_dropped_line_indices(["bar .ct}", "5", "A sentence."]) == {0, 1}


# --------------------------------------------------------------------------- #
# _has_cmap_equation_evidence / normalize_academic_pdf_math
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain text", False),
        ("x \u00bc 5", True),
        ("vG \u00bc VG", True),
        ("the \u00bc value", False),  # stopword guard
        ("\u00bc cup", False),  # no flanking identifier at all
        ("x \u00bc cup", False),  # measure-noun guard
    ],
)
def test_cmap_equation_evidence(content: str, expected: bool) -> None:
    assert _has_cmap_equation_evidence(content) is expected


def test_normalize_handles_empty_input() -> None:
    assert normalize_academic_pdf_math("") == ""


def test_cmap_fraction_before_a_number_becomes_equals() -> None:
    assert normalize_academic_pdf_math("x \u00bc 5") == "x = 5"


def test_cmap_fraction_between_identifiers_becomes_equals() -> None:
    assert normalize_academic_pdf_math("vG \u00bc VG") == "vG = VG"


def test_a_stopword_fraction_is_left_alone() -> None:
    assert normalize_academic_pdf_math("the \u00bc value") == "the \u00bc value"


def test_a_measure_noun_fraction_is_left_alone() -> None:
    assert normalize_academic_pdf_math("x \u00bc cup") == "x \u00bc cup"


def test_pdf_stream_glyph_escapes_are_repaired() -> None:
    assert normalize_academic_pdf_math("a /C2 b") == "a \u00d7 b"
    assert normalize_academic_pdf_math("/C0 x") == "- x"
    assert normalize_academic_pdf_math("5 /uni03BC m") == "5 \u03bc m"


def test_null_bytes_are_stripped() -> None:
    assert normalize_academic_pdf_math("a\x00b") == "ab"


def test_span_sentinels_survive_null_byte_stripping() -> None:
    assert normalize_academic_pdf_math("\x00SPAN0\x00text\x00") == "\x00SPAN0\x00text"


def test_soft_hyphen_line_breaks_rejoin_words() -> None:
    assert normalize_academic_pdf_math("compos\xad ability") == "composability"
    assert normalize_academic_pdf_math("a\xadb") == "ab"
    assert normalize_academic_pdf_math("solo\xad") == "solo"


def test_kerning_split_subscripts_are_rejoined() -> None:
    assert normalize_academic_pdf_math("t ox =") == "t_ox ="
    assert normalize_academic_pdf_math("N ch \u2248") == "N_ch \u2248"


def test_a_prose_comma_is_not_a_subscript_operator() -> None:
    assert normalize_academic_pdf_math("I am, and then") == "I am, and then"


def test_flattened_scientific_notation_is_rebuilt() -> None:
    assert normalize_academic_pdf_math("1 - 10 15 cm 3") == "1 \u00d7 10^15 cm^-3"


def test_a_power_without_a_unit_is_rebuilt() -> None:
    assert normalize_academic_pdf_math("3 \u00d7 10 8 m") == "3 \u00d7 10^8 m"


def test_the_currency_gate_requires_cmap_evidence() -> None:
    # Without equation-shaped ¼, a currency amount must survive untouched.
    assert normalize_academic_pdf_math("paid $5") == "paid $5"
    assert normalize_academic_pdf_math("x \u00bc 5 paid $5") == "x = 5 paid -5"


def test_the_parenthesis_cmap_repair_requires_evidence() -> None:
    assert normalize_academic_pdf_math("\u00f0x\u00de") == "\u00f0x\u00de"
    assert normalize_academic_pdf_math("x \u00bc 5 \u00f0a\u00de") == "x = 5 (a)"


# --------------------------------------------------------------------------- #
# strip_textbook_ocr_artifacts
# --------------------------------------------------------------------------- #


def test_strip_textbook_handles_empty_input() -> None:
    assert strip_textbook_ocr_artifacts("") == ""


def test_control_debris_is_always_stripped() -> None:
    assert strip_textbook_ocr_artifacts("a\x01b\x7fc") == "abc"


def test_math_protection_disables_the_cmap_normalizer() -> None:
    assert strip_textbook_ocr_artifacts("x /C2 y") == "x \u00d7 y"
    assert strip_textbook_ocr_artifacts("`x /C2 y`") == "`x /C2 y`"


def test_legal_disclaimers_are_stripped() -> None:
    text = "All rights reserved. No part of this publication may be reproduced."
    assert strip_textbook_ocr_artifacts(text) == ""


def test_photo_credits_are_stripped() -> None:
    assert strip_textbook_ocr_artifacts("A valley at dawn Getty Images") == "A valley at dawn"


def test_page_markers_are_stripped() -> None:
    assert strip_textbook_ocr_artifacts("Page 12\nBody text.") == "Body text."


def test_running_headers_are_stripped() -> None:
    assert strip_textbook_ocr_artifacts("5 CHAPTER 3") == ""


def test_isolated_backslashes_are_stripped() -> None:
    assert strip_textbook_ocr_artifacts("foo \\ bar") == "foo bar"


def test_decorative_separator_lines_are_stripped() -> None:
    assert strip_textbook_ocr_artifacts("Title\n=====\nBody") == "Title\n\nBody"


def test_in_prose_equals_is_preserved() -> None:
    assert strip_textbook_ocr_artifacts("a = b") == "a = b"


def test_horizontal_whitespace_is_collapsed() -> None:
    assert strip_textbook_ocr_artifacts("a   b\tc") == "a b c"


# --------------------------------------------------------------------------- #
# clean_calibre_and_lnds_pages
# --------------------------------------------------------------------------- #


def test_clean_removes_calibre_markup() -> None:
    assert clean_calibre_and_lnds_pages("Hello {.calibre9}world") == "Hello world"


def test_clean_drops_a_monotonic_page_run() -> None:
    text = "Intro text.\n1\nBody text.\n2\nMore text.\n3\nEnd.\n4"
    assert clean_calibre_and_lnds_pages(text) == "Intro text.\nBody text.\nMore text.\nEnd."


def test_clean_collapses_blank_line_runs() -> None:
    assert clean_calibre_and_lnds_pages("A\n\n\n\nB") == "A\n\nB"


# --------------------------------------------------------------------------- #
# LNDSPageCleaner
# --------------------------------------------------------------------------- #


def test_cleaner_clean_delegates_to_the_pure_cleaner() -> None:
    assert LNDSPageCleaner().clean("Hello {.calibre9}world") == "Hello world"


def test_cleaner_honours_strip_all_page_numbers() -> None:
    text = "Hello\n5\nThis is a full sentence."
    assert LNDSPageCleaner(strip_all_page_numbers=True).clean(text) == (
        "Hello\nThis is a full sentence."
    )


def test_clean_block_returns_a_cleaned_copy() -> None:
    block = _block("b1", "Hello {.calibre9}world")
    cleaned = LNDSPageCleaner().clean_block(block)
    assert cleaned is not block
    assert cleaned.source_text == "Hello world"
    assert block.source_text == "Hello {.calibre9}world"


def test_clean_chapter_blocks_handles_no_blocks() -> None:
    assert LNDSPageCleaner().clean_chapter_blocks([]) == []


def test_clean_chapter_blocks_drops_cross_block_page_numbers() -> None:
    blocks = [
        _block("b0", "Intro text."),
        _block("b1", "1"),
        _block("b2", "Body text."),
        _block("b3", "2"),
        _block("b4", "More text."),
        _block("b5", "3"),
        _block("b6", "End."),
        _block("b7", "4"),
    ]
    cleaned = LNDSPageCleaner().clean_chapter_blocks(blocks)
    assert cleaned[0].source_text == "Intro text."
    for idx in (1, 3, 5, 7):
        page = cleaned[idx]
        assert page.source_text == ""
        assert page.target_text == ""
        assert page.status is BlockStatus.MTQE_PASSED
        assert page.mtqe_score == 1.0
        assert page.skip_translate is True


def test_clean_chapter_blocks_never_blanks_a_protected_block_type() -> None:
    blocks = [
        _block("b0", "Intro text."),
        _block("b1", "1"),
        _block("b2", "Body text."),
        _block("b3", "2"),
        _block("b4", "More text."),
        _block("b5", "3"),
        _block("b6", "End."),
        _block("b7", "4"),
        _block("b8", "5", block_type=BlockType.HEADING),
    ]
    cleaned = LNDSPageCleaner().clean_chapter_blocks(blocks)
    # b8's lone digit is part of the global sequence but its block is a heading.
    assert cleaned[8].source_text == "5"
    assert cleaned[1].source_text == ""


# --------------------------------------------------------------------------- #
# _norm_text / _area / _iou
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("  Hello   World  ", "hello world"),
        ("A\tB", "a b"),
        ("Panel", "panel"),
    ],
)
def test_norm_text(text: str, expected: str) -> None:
    assert _norm_text(text) == expected


def test_area() -> None:
    assert _area((0.0, 0.0, 2.0, 3.0)) == 6.0
    assert _area((3.0, 3.0, 1.0, 1.0)) == 0.0


def test_iou() -> None:
    assert _iou((0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 2.0, 2.0)) == 1.0
    assert _iou((0.0, 0.0, 2.0, 2.0), (5.0, 5.0, 7.0, 7.0)) == 0.0
    assert _iou((0.0, 0.0, 2.0, 2.0), (1.0, 0.0, 3.0, 2.0)) == pytest.approx(2.0 / 6.0)


# --------------------------------------------------------------------------- #
# dedup_duplicate_blocks
# --------------------------------------------------------------------------- #


def test_dedup_leaves_short_input_untouched() -> None:
    only = [_block("b", "Panel")]
    assert dedup_duplicate_blocks(only) is only
    empty: list[IRBlock] = []
    assert dedup_duplicate_blocks(empty) is empty


def test_dedup_keeps_the_largest_overlapping_duplicate() -> None:
    blocks = [
        _block("a", "Panel", bbox=_bbox(1, 0, 0, 10, 10)),
        _block("b", "Panel", bbox=_bbox(1, 0, 0, 9, 9)),
        _block("c", "Panel", bbox=_bbox(1, 0, 0, 8, 8)),
    ]
    kept = dedup_duplicate_blocks(blocks)
    assert [b.id for b in kept] == ["a"]


def test_dedup_keeps_disjoint_same_text_blocks() -> None:
    blocks = [
        _block("a", "Panel", bbox=_bbox(1, 0, 0, 10, 10)),
        _block("b", "Panel", bbox=_bbox(1, 20, 0, 30, 10)),
    ]
    assert [b.id for b in dedup_duplicate_blocks(blocks)] == ["a", "b"]


def test_dedup_keeps_overlap_below_the_bar() -> None:
    blocks = [
        _block("a", "Panel", bbox=_bbox(1, 0, 0, 10, 10)),
        _block("b", "Panel", bbox=_bbox(1, 9, 9, 19, 19)),
    ]
    assert [b.id for b in dedup_duplicate_blocks(blocks)] == ["a", "b"]


def test_dedup_buckets_by_normalized_text() -> None:
    blocks = [
        _block("a", "Panel", bbox=_bbox(1, 0, 0, 10, 10)),
        _block("b", "  PANEL  ", bbox=_bbox(1, 0, 0, 9, 9)),
    ]
    assert [b.id for b in dedup_duplicate_blocks(blocks)] == ["a"]


def test_dedup_keeps_distinct_text_in_the_same_box() -> None:
    blocks = [
        _block("a", "Panel A", bbox=_bbox(1, 0, 0, 10, 10)),
        _block("b", "Panel B", bbox=_bbox(1, 0, 0, 10, 10)),
    ]
    assert [b.id for b in dedup_duplicate_blocks(blocks)] == ["a", "b"]


def test_dedup_keeps_identical_text_on_different_pages() -> None:
    blocks = [
        _block("a", "Panel", bbox=_bbox(1, 0, 0, 10, 10)),
        _block("b", "Panel", bbox=_bbox(2, 0, 0, 10, 10)),
    ]
    assert [b.id for b in dedup_duplicate_blocks(blocks)] == ["a", "b"]


def test_dedup_skips_blocks_without_geometry() -> None:
    blocks = [_block("a", "Panel"), _block("b", "Panel")]
    assert [b.id for b in dedup_duplicate_blocks(blocks)] == ["a", "b"]


def test_dedup_skips_empty_text() -> None:
    blocks = [
        _block("a", "", bbox=_bbox(1, 0, 0, 10, 10)),
        _block("b", "", bbox=_bbox(1, 0, 0, 10, 10)),
    ]
    assert [b.id for b in dedup_duplicate_blocks(blocks)] == ["a", "b"]

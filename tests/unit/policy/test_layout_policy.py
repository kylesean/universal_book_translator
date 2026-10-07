"""Contract tests for the unified policy/knob registry (``layout_policy``)."""

from __future__ import annotations

import pytest

import ubt.core.policy.layout_policy as lp
from ubt.core.ir.models import BlockType, FlowID
from ubt.core.policy.layout_policy import (
    CALIBRATION,
    CONJUNCTIONS,
    NON_PROSE_FLOWS,
    NON_TEXT_BLOCK_TYPES,
    PAIR_TERMINAL_PUNCT,
    PDF_PATH_OPS,
    PDF_TEXT_OPS,
    PROSE_BLOCK_TYPES,
    Calibration,
    KnobMeta,
    calibration_summary,
    formula_debris_share,
    is_non_prose_degradable,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# Calibration registry
# --------------------------------------------------------------------------- #


def test_calibration_enum_values() -> None:
    assert [c.value for c in Calibration] == ["proven", "single_doc", "hypothesis"]


def test_status_aliases_match_enum_members() -> None:
    assert lp.P is Calibration.PROVEN
    assert lp.S is Calibration.SINGLE_DOC
    assert lp.H is Calibration.HYPOTHESIS


def test_every_calibration_entry_is_a_knobmeta_with_valid_status() -> None:
    assert CALIBRATION
    for name, meta in CALIBRATION.items():
        assert isinstance(meta, KnobMeta), name
        assert isinstance(meta.status, Calibration), name
        assert meta.rationale.strip(), name


def test_every_calibration_key_resolves_to_a_module_attribute() -> None:
    missing = [name for name in CALIBRATION if not hasattr(lp, name)]
    assert missing == []


def test_calibration_summary_matches_registry() -> None:
    summary = calibration_summary()
    assert summary == {"proven": 28, "single_doc": 30, "hypothesis": 16}
    assert sum(summary.values()) == len(CALIBRATION)


# --------------------------------------------------------------------------- #
# formula_debris_share
# --------------------------------------------------------------------------- #


def test_formula_debris_no_tokens_is_zero() -> None:
    assert formula_debris_share("") == 0.0
    assert formula_debris_share("123 456") == 0.0


def test_formula_debris_counts_isolated_letters() -> None:
    # 2 singles (K, x) among 3 ASCII word tokens ("=" is not a word).
    assert formula_debris_share("K = x plus") == pytest.approx(2 / 3)


def test_formula_debris_excludes_prose_single_letters() -> None:
    # a / A / I are legitimate English words, not equation debris.
    assert formula_debris_share("a A I cat") == 0.0


def test_formula_debris_prose_stays_low() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    assert formula_debris_share(text) == 0.0


def test_formula_debris_ignores_non_ascii_tokens() -> None:
    assert formula_debris_share("x 中文 中") == 1.0  # only "x" is an ASCII word token


# --------------------------------------------------------------------------- #
# is_non_prose_degradable
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "block_type", [BlockType.TABLE, BlockType.CODE, BlockType.FORMULA, BlockType.IMAGE]
)
def test_non_prose_blocks_degrade(block_type: BlockType) -> None:
    assert is_non_prose_degradable(block_type, []) is True


@pytest.mark.parametrize(
    "block_type", [BlockType.NARRATIVE, BlockType.HEADING, BlockType.LIST_ITEM]
)
def test_prose_blocks_never_degrade(block_type: BlockType) -> None:
    assert is_non_prose_degradable(block_type, ["x"]) is False


@pytest.mark.parametrize("marker", ["Prompt template XML artifacts", "Prompt scaffold"])
def test_fatal_leak_markers_block_degradation(marker: str) -> None:
    flags = ["render_skip:non_prose", marker]
    assert is_non_prose_degradable(BlockType.TABLE, flags) is False


def test_unknown_block_type_is_treated_as_non_prose() -> None:
    assert is_non_prose_degradable("mystery", []) is True
    assert is_non_prose_degradable(None, []) is True


def test_block_type_accepts_bare_strings() -> None:
    assert is_non_prose_degradable("table", []) is True
    assert is_non_prose_degradable("narrative", []) is False


# --------------------------------------------------------------------------- #
# Vocabulary sets
# --------------------------------------------------------------------------- #


def test_block_type_vocabularies_are_pinned() -> None:
    assert {BlockType.HEADING, BlockType.NARRATIVE, BlockType.LIST_ITEM} == PROSE_BLOCK_TYPES
    assert {BlockType.FORMULA, BlockType.CODE, BlockType.IMAGE} == NON_TEXT_BLOCK_TYPES
    assert {FlowID.FOOTNOTE, FlowID.CAPTION, FlowID.TABLE_GRID} == NON_PROSE_FLOWS


def test_table_is_not_verbatim_ship() -> None:
    assert BlockType.TABLE not in NON_TEXT_BLOCK_TYPES


def test_conjunctions_cover_english_and_cjk() -> None:
    assert {"and", "or", "but"} <= CONJUNCTIONS
    assert {"以及", "但是", "然而"} <= CONJUNCTIONS


def test_pdf_operator_inventories_are_pinned() -> None:
    assert {"m", "l", "c", "v", "y", "h", "re"} == PDF_PATH_OPS
    assert {"Tj", "TJ", "'", '"'} == PDF_TEXT_OPS


def test_pair_terminal_punct_membership() -> None:
    assert {"。", "！", "？", ".", "!", "?"} <= PAIR_TERMINAL_PUNCT
    assert "," not in PAIR_TERMINAL_PUNCT


# --------------------------------------------------------------------------- #
# Text-normalization mechanics
# --------------------------------------------------------------------------- #


def test_ws_re_collapses_runs() -> None:
    assert lp.WS_RE.sub(" ", "a\t  b\n c") == "a b c"


def test_fold_map_folds_quotes_dashes_ligatures_and_nbsp() -> None:
    # Only fi/fl ligatures are folded (ﬀ is deliberately not in FOLD_MAP).
    folded = "‘a’ “b” c–d—e ﬁ ﬂ\u00a0g".translate(lp.FOLD_MAP)
    assert folded == "'a' \"b\" c-d-e fi fl g"


def test_control_re_strips_glyph_artifacts_but_keeps_tab_and_newline() -> None:
    raw = "a\x02b\tc\nd\x7fe"
    assert lp.CONTROL_RE.sub("", raw) == "ab\tc\nde"


def test_ascii_word_re_matches_letters_only() -> None:
    assert lp.ASCII_WORD_RE.findall("a1 b_c d") == ["a", "b", "c", "d"]


def test_unicode_word_re_matches_cjk() -> None:
    assert lp.UNICODE_WORD_RE.findall("中 文 abc") == ["中", "文", "abc"]


# --------------------------------------------------------------------------- #
# Verdict / continuation regexes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "matches"),
    [
        ("0306406152", True),  # ISBN-10
        ("9780306406157", True),  # ISBN-13
        ("978030640615X", True),  # check digit X
        ("12345", False),
        ("97803064061570", False),
    ],
)
def test_isbn_digits_re(value: str, matches: bool) -> None:
    assert (lp.ISBN_DIGITS_RE.match(value) is not None) is matches


@pytest.mark.parametrize(
    ("value", "matches"),
    [
        ("https://example.org/x", True),
        ("http://a.b", True),
        ("doi: 10.1000/xyz", True),
        ("DOI:10.1/a", True),
        ("example.org", False),
    ],
)
def test_url_doi_re(value: str, matches: bool) -> None:
    assert (lp.URL_DOI_RE.match(value) is not None) is matches


@pytest.mark.parametrize(
    ("value", "matches"),
    [("af", True), ("AF12", True), ("a", False), ("xyz", False), ("12g", False)],
)
def test_byte_word_re(value: str, matches: bool) -> None:
    assert (lp.BYTE_WORD_RE.match(value) is not None) is matches


@pytest.mark.parametrize(
    ("value", "matches"),
    [("word", True), ("word-1/2", True), ("two words", False), ("_x", False), ("!x", False)],
)
def test_single_token_line_re(value: str, matches: bool) -> None:
    assert (lp.SINGLE_TOKEN_LINE_RE.match(value) is not None) is matches


@pytest.mark.parametrize(
    ("value", "matches"),
    [("A", True), ("Fig 1", True), ("a/b", True), ("!bad", False), ("", False)],
)
def test_short_label_re(value: str, matches: bool) -> None:
    assert (lp.SHORT_LABEL_RE.match(value) is not None) is matches


@pytest.mark.parametrize(
    ("value", "matches"),
    [
        ("Figure 1", True),
        ("Fig. 2", True),
        ("Table 3", True),
        ("TAB.4", True),
        ("图1", True),
        ("表 2", True),
        ("The table shows", False),
    ],
)
def test_caption_re(value: str, matches: bool) -> None:
    assert (lp.CAPTION_RE.match(value) is not None) is matches


@pytest.mark.parametrize(
    ("value", "matches"),
    [
        ("Copyright 2020", True),
        ("© 2020", True),
        ("see doi.org/x", True),
        ("All rights reserved", True),
        ("Published by Elsevier", True),
        ("just a sentence", False),
    ],
)
def test_footer_patterns(value: str, matches: bool) -> None:
    assert (lp.FOOTER_PATTERNS.search(value) is not None) is matches


@pytest.mark.parametrize(
    ("value", "group"),
    [("1.1 Intro", "1.1"), ("2.2.1 x", "2.2.1"), ("21 century", "21"), ("21世纪", None)],
)
def test_heading_number_re(value: str, group: str | None) -> None:
    match = lp.HEADING_NUMBER_RE.match(value)
    assert (match.group(1) if match else None) == group


@pytest.mark.parametrize(
    ("value", "matches"),
    [
        ("1.2 Intro", True),
        ("3 x", True),
        ("第一章", True),
        ("第2章", False),
        ("21世纪", False),
    ],
)
def test_target_numbered_re(value: str, matches: bool) -> None:
    assert (lp.TARGET_NUMBERED_RE.match(value) is not None) is matches


@pytest.mark.parametrize(
    ("value", "matches"),
    [("1.2 foo", True), ("1 foo", False), ("x", False)],
)
def test_cont_chapter_number_re(value: str, matches: bool) -> None:
    assert (lp.CONT_CHAPTER_NUMBER_RE.match(value) is not None) is matches


@pytest.mark.parametrize(
    ("value", "matches"),
    [("end.", True), ("end。", True), ("end…", True), ("end", False)],
)
def test_cont_sentence_end_re(value: str, matches: bool) -> None:
    assert (lp.CONT_SENTENCE_END_RE.search(value) is not None) is matches


@pytest.mark.parametrize(
    ("value", "matches"),
    [
        ("Table 1", True),
        ("Figure 2", True),
        ("Scheme 1", True),
        ("Equation 2", True),
        ("Section 4", True),
        ("Chapter 5", True),
        # NOTE: the ``fig\.``/``eq\.`` alternatives are unreachable — the
        # trailing ``\b`` sits between two non-word chars ("." then " ") and
        # never matches, so the abbreviated forms fall through to no-match.
        ("Fig. 3", False),
        ("Fig 3", False),
        ("Eq. 3", False),
        ("Tables are fun", False),
    ],
)
def test_cont_label_barrier_re(value: str, matches: bool) -> None:
    assert (lp.CONT_LABEL_BARRIER_RE.match(value) is not None) is matches


@pytest.mark.parametrize(
    ("value", "matches"),
    [
        ("1. item", True),
        ("2) item", True),
        ("3、 item", True),
        ("• item", True),
        ("- item", True),
        ("(a) item", True),
        ("3、item", False),  # the marker requires following whitespace
        ("plain", False),
    ],
)
def test_cont_list_marker_re(value: str, matches: bool) -> None:
    assert (lp.CONT_LIST_MARKER_RE.match(value) is not None) is matches


def test_cont_upper_start_re() -> None:
    assert lp.CONT_UPPER_START_RE.match("Word") is not None
    assert lp.CONT_UPPER_START_RE.match("word") is None


# --------------------------------------------------------------------------- #

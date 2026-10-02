"""Level-1 structural asset verification (the cheap half of the two-level check).

A reconstructed formula/table is only trustworthy if it survives a structural
check: the formula is non-empty, brace-balanced and free of engine error
markers; the table did not shatter into single-character cells. A FAIL is
corruption (Axiom A) and must never be softened into a pass, while "not a
recognized table grammar" is a deliberate SKIP -- the pixel level covers it,
and treating it as corruption would fail every non-markdown table.
"""

from __future__ import annotations

import pytest

from ubt.core.content.asset_verify import (
    StructuralVerdict,
    VerifyResult,
    _balanced,
    _cells_from_html,
    _cells_from_markdown,
    verify_asset_structure,
    verify_formula_structure,
    verify_table_structure,
)
from ubt.core.content.nodes import AssetKind

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# VerifyResult
# --------------------------------------------------------------------------- #


def test_verified_and_corrupt_are_disjoint_and_only_pass_verifies() -> None:
    assert VerifyResult(StructuralVerdict.PASS).verified is True
    assert VerifyResult(StructuralVerdict.PASS).corrupt is False
    assert VerifyResult(StructuralVerdict.FAIL).verified is False
    assert VerifyResult(StructuralVerdict.FAIL).corrupt is True
    # SKIP is neither verified nor corruption.
    assert VerifyResult(StructuralVerdict.SKIP).verified is False
    assert VerifyResult(StructuralVerdict.SKIP).corrupt is False


# --------------------------------------------------------------------------- #
# _balanced
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    ["", "()", "[]", "{}", "a(b)c", "()[]{}", r"\{", r"\}", r"{a\}b}"],
)
def test_balanced_text(text: str) -> None:
    assert _balanced(text) is True


@pytest.mark.parametrize("text", ["a(b", "a)b", "a[b)", "(a]", "{a]", "([)]"])
def test_unbalanced_text(text: str) -> None:
    assert _balanced(text) is False


# --------------------------------------------------------------------------- #
# verify_formula_structure
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_an_empty_formula_fails(text: str) -> None:
    result = verify_formula_structure(text)
    assert result.verdict is StructuralVerdict.FAIL
    assert result.detail == "empty formula"


@pytest.mark.parametrize(
    "text",
    ["data-mjx-error here", "MATHJAX ERROR", "Unsupported command", "an unsupported feature"],
)
def test_an_engine_error_marker_fails_case_insensitively(text: str) -> None:
    result = verify_formula_structure(text)
    assert result.verdict is StructuralVerdict.FAIL
    assert result.detail == "engine error marker in emitted math"


def test_unbalanced_delimiters_fail() -> None:
    result = verify_formula_structure("{unbalanced")
    assert result.verdict is StructuralVerdict.FAIL
    assert result.detail == "unbalanced delimiters"


def test_a_clean_formula_passes() -> None:
    result = verify_formula_structure("x^2 + y")
    assert result.verdict is StructuralVerdict.PASS
    assert result.detail == "structural"


def test_escaped_braces_do_not_break_balance() -> None:
    assert verify_formula_structure(r"x \{ y").verdict is StructuralVerdict.PASS


# --------------------------------------------------------------------------- #
# cell extraction
# --------------------------------------------------------------------------- #


def test_markdown_cells_skip_separator_and_non_pipe_lines() -> None:
    text = "| a | b |\n| --- | --- |\nnot a row\n| 1 | 2 |"
    assert _cells_from_markdown(text) == ["a", "b", "1", "2"]


def test_a_separator_only_row_yields_no_cells() -> None:
    assert _cells_from_markdown("| --- | :---: |") == []


def test_an_all_blank_pipe_row_is_treated_as_a_separator() -> None:
    assert _cells_from_markdown("| | |") == []


def test_html_cells_extract_td_and_th_and_strip_tags() -> None:
    assert _cells_from_html("<table><tr><td>a</td><th>B</th></tr></table>") == ["a", "B"]
    assert _cells_from_html("<td><b>x</b> y</td>") == ["x y"]


# --------------------------------------------------------------------------- #
# verify_table_structure
# --------------------------------------------------------------------------- #


def test_an_empty_table_fails() -> None:
    result = verify_table_structure("")
    assert result.verdict is StructuralVerdict.FAIL
    assert result.detail == "empty table"


def test_an_unrecognized_grammar_is_a_skip_not_a_failure() -> None:
    result = verify_table_structure("just prose, no table here")
    assert result.verdict is StructuralVerdict.SKIP
    assert result.detail == "unrecognized table grammar"


def test_a_shattered_table_fails() -> None:
    result = verify_table_structure("| a | b |\n| c | d |")
    assert result.verdict is StructuralVerdict.FAIL
    assert result.detail == "4/4 cells are single-character (shattered)"


def test_too_few_cells_to_shatter_passes() -> None:
    # Two single-character cells are below the four-cell floor.
    assert verify_table_structure("| a | b |").verdict is StructuralVerdict.PASS


def test_a_minority_of_short_cells_passes() -> None:
    # Four cells, two single-character: the majority rule is not met.
    assert verify_table_structure("| ab | c |\n| de | f |").verdict is StructuralVerdict.PASS


def test_a_bare_majority_of_short_cells_passes() -> None:
    # 4/8 == 0.5 is not strictly greater than the 0.5 share.
    assert (
        verify_table_structure("| a | b | c | d |\n| ee | ff | gg | hh |").verdict
        is StructuralVerdict.PASS
    )


def test_a_strict_majority_of_short_cells_fails() -> None:
    result = verify_table_structure("| a | b | c | d |\n| e | ff | gg | hh |")
    assert result.verdict is StructuralVerdict.FAIL
    assert result.detail == "5/8 cells are single-character (shattered)"


def test_an_html_table_is_recognized_and_checked() -> None:
    result = verify_table_structure(
        "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
    )
    assert result.verdict is StructuralVerdict.FAIL


# --------------------------------------------------------------------------- #
# verify_asset_structure dispatch
# --------------------------------------------------------------------------- #


def test_dispatch_routes_formula_and_table() -> None:
    assert verify_asset_structure(AssetKind.FORMULA, "x^2").verdict is StructuralVerdict.PASS
    assert verify_asset_structure(AssetKind.TABLE, "| a |").verdict is StructuralVerdict.PASS


@pytest.mark.parametrize("kind", [AssetKind.FIGURE, AssetKind.DIAGRAM])
def test_other_asset_kinds_have_no_structural_policy(kind: AssetKind) -> None:
    result = verify_asset_structure(kind, "anything")
    assert result.verdict is StructuralVerdict.SKIP
    assert result.detail == "no structural policy"

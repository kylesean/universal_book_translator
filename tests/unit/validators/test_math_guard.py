"""Contract tests for the shared math-debris / formula-invariant guards (Gate 1/3)."""

from __future__ import annotations

from typing import Any

import pytest

from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    FlowID,
    IRBlock,
    make_element,
)
from ubt.core.validators.math_guard import (
    RENDERABLE_LATEX_COMMANDS,
    _has_math_soup_residue,
    _is_math_subscript_residue,
    _source_command_set,
    _strip_code_snake_case,
    _unicode_math_style_command,
    apply_math_guards,
    formula_skeleton_intact,
    formula_target_intact,
    looks_like_math_debris,
    normalize_math,
    novel_unsupported_latex_commands,
    target_missing_math_delimiters,
)

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    source: str,
    target: str,
    *,
    block_type: BlockType = BlockType.FORMULA,
    status: BlockStatus = BlockStatus.PENDING,
) -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=0,
        block_type=block_type,
        flow_id=FlowID.MAIN_STORY,
        source_text=source,
        skip_translate=False,
    )
    return IRBlock(element=element, status=status, target_text=target)


# --------------------------------------------------------------------------- #
# Subscript residue
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("x_1", True),  # single-letter base
        ("x_{i}", True),  # brace subscript
        ("psi_pert", True),  # greek name base
        ("alpha_2", True),
        ("x_", True),  # empty subscript still splits into two parts
        ("ab_1", True),  # short latin base + numeric sub
        ("create_task", False),  # snake_case identifier
        ("code_change", False),
        ("ab_cd", False),  # multi-letter latin base, non-numeric sub
        ("_x", False),  # no base
        ("F1", False),  # no underscore at all
    ],
)
def test_is_math_subscript_residue(token: str, expected: bool) -> None:
    assert _is_math_subscript_residue(token) is expected


def test_has_math_soup_residue_scans_tokens() -> None:
    assert _has_math_soup_residue("a x_1 b") is True
    assert _has_math_soup_residue("a create_task b") is False
    assert _has_math_soup_residue("no underscore") is False


def test_strip_code_snake_case_keeps_math_but_drops_identifiers() -> None:
    assert _strip_code_snake_case("use create_task here") == "use   here"
    assert _strip_code_snake_case("a x_1 b") == "a x_1 b"


# --------------------------------------------------------------------------- #
# Unicode math style
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("char", "expected"),
    [
        ("\U0001d44e", "mathit"),  # MATHEMATICAL ITALIC SMALL A
        ("\U0001d400", "mathbf"),  # MATHEMATICAL BOLD CAPITAL A
        ("\U0001d538", "mathbb"),  # MATHEMATICAL DOUBLE-STRUCK CAPITAL A
        ("\u211d", None),  # DOUBLE-STRUCK CAPITAL R (not "MATHEMATICAL ...")
        ("A", None),
        ("1", None),
        ("\u03b1", None),
    ],
)
def test_unicode_math_style_command(char: str, expected: str | None) -> None:
    assert _unicode_math_style_command(char) == expected


# --------------------------------------------------------------------------- #
# LaTeX command sets
# --------------------------------------------------------------------------- #


def test_source_command_set_reads_control_sequences() -> None:
    assert _source_command_set(r"\frac{a}{b} x \in S") == {"frac", "in"}


def test_source_command_set_expands_unicode_symbols() -> None:
    assert _source_command_set("x \u2208 S") == {"in"}
    assert _source_command_set("\U0001d538 \u211d") == {"mathbb"}
    # A styled letter that is not in the unicode map is read from its name.
    assert _source_command_set("\U0001d538") == {"mathbb"}


def test_source_command_set_is_empty_for_plain_text() -> None:
    assert _source_command_set("plain prose") == set()


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("x", r"\mathrm{2D}", []),  # \mathrm is renderable
        (r"\frac{a}{b}", r"\frac{a}{b}", []),
        ("x \u2208 S", r"x \in S", []),  # unicode symbol excuses its LaTeX spelling
        ("Z", r"\mathbb{R}", ["mathbb"]),  # no source blackboard letter: bare command novel
    ],
)
def test_novel_unsupported_latex_commands(source: str, target: str, expected: list[str]) -> None:
    assert novel_unsupported_latex_commands(source, target) == expected


def test_blackboard_letter_identity_is_compared() -> None:
    assert novel_unsupported_latex_commands("\u2124", r"\mathbb{R}") == ["mathbb{R}"]
    assert novel_unsupported_latex_commands("\u2124", r"\mathbb{Z}") == []


def test_renderable_command_set_contains_the_documented_members() -> None:
    assert {"text", "mathrm", "frac", "in", "alpha"} <= RENDERABLE_LATEX_COMMANDS


# --------------------------------------------------------------------------- #
# normalize_math
# --------------------------------------------------------------------------- #


def test_normalize_math_collapses_all_whitespace() -> None:
    assert normalize_math("a b\n c") == "abc"
    assert normalize_math("") == ""
    assert normalize_math("  ") == ""


# --------------------------------------------------------------------------- #
# looks_like_math_debris
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("K , V , x", True),  # isolated letters + no running word
        ("K V x", True),
        ("K , V =", True),  # strong math symbol rung (only 2 isolated letters)
        ("K V yz ab", False),  # 2 isolated letters fall below the 3-letter rung
        (". . . .", True),  # punctuation-only run
        ("(1) Layer 1: K, V", False),  # running word vetoes
        ("hello world this is prose", False),
        ("C OVER T ITLE P AGE", False),
        ("", False),
        ("a", False),  # fewer than min_tokens
        ("\u4e2d\u6587 K V x", False),  # CJK is never debris
    ],
)
def test_looks_like_math_debris(text: str, expected: bool) -> None:
    assert looks_like_math_debris(text) is expected


def test_debris_respects_its_thresholds() -> None:
    assert looks_like_math_debris(".", min_tokens=1) is True
    assert looks_like_math_debris(".", min_tokens=2) is False
    assert looks_like_math_debris("K V x yz", single_ratio=0.9) is False
    assert looks_like_math_debris("K V x yz", single_ratio=0.5) is True
    assert looks_like_math_debris("K , V , x" + " " * 200, max_chars=5) is False


# --------------------------------------------------------------------------- #
# target_missing_math_delimiters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("$x$", "x", True),  # delimited source, flat target
        ("F 1 , K , V , x", "F 1 K V x", True),  # debris source with a candidate
        ("Ag_0 x", "Ag 0 x", True),
        ("F 1 , K , V , x", "\u6ca1\u6709", False),  # no ASCII run to delimit
        ("F 1 , K , V , x", "\u7eaf\u4e2d\u6587", False),
        ("$x$", "$y$", False),  # target already delimited
        ("hello world", "hello world", False),  # no math candidate
        ("K , V , x", "K V x", False),  # debris but nothing concrete to redelimit
        ("F1 hello world", "F1 hello world", False),
        ("", "x", False),
        ("x", "", False),
    ],
)
def test_target_missing_math_delimiters(source: str, target: str, expected: bool) -> None:
    assert target_missing_math_delimiters(source, target) is expected


# --------------------------------------------------------------------------- #
# Formula invariants
# --------------------------------------------------------------------------- #


def test_formula_target_intact_ignores_whitespace_only() -> None:
    assert formula_target_intact(_block("b", "$x + y$", "$x  +  y$")) is True
    assert formula_target_intact(_block("b", "$x + y$", "$x - y$")) is False


def test_formula_invariants_skip_non_formula_and_empty_targets() -> None:
    narrative = _block("n", "src", "tgt", block_type=BlockType.NARRATIVE)
    assert formula_target_intact(narrative) is True
    assert formula_skeleton_intact(narrative) is True
    assert formula_target_intact(_block("e", "$x$", "")) is True
    assert formula_skeleton_intact(_block("e", "$x$", "")) is True


def test_formula_skeleton_accepts_text_span_translation() -> None:
    block = _block("b", r"$a \text{hello} b$", r"$a \text{你好} b$")
    assert formula_target_intact(block) is False
    assert formula_skeleton_intact(block) is True


def test_formula_skeleton_rejects_changed_math() -> None:
    block = _block("b", r"$a \text{hello} b$", r"$a \text{你好} c$")
    assert formula_target_intact(block) is False
    assert formula_skeleton_intact(block) is False


# --------------------------------------------------------------------------- #
# apply_math_guards
# --------------------------------------------------------------------------- #


def _flags(block: IRBlock) -> list[str]:
    return list(block.error_flags)


def test_apply_math_guards_resets_a_drifted_formula() -> None:
    drifted = _block("d1", "$x + y$", "$x - y$")
    checkpoints, counts = apply_math_guards([drifted])
    assert drifted.target_text == "$x + y$"
    assert counts == {
        "formula_invariant_repairs": 1,
        "math_debris_fallbacks": 0,
        "c_text_accepted": 0,
    }
    assert checkpoints == [
        {
            "block_id": "d1",
            "target_text": "$x + y$",
            "status": BlockStatus.PENDING,
            "error_flags": _flags(drifted),
        }
    ]
    assert any(f.startswith("math_invariant_repair:") for f in _flags(drifted))


def test_apply_math_guards_accepts_a_text_span_translation() -> None:
    accepted = _block("d2", r"$a \text{hello} b$", r"$a \text{你好} b$")
    checkpoints, counts = apply_math_guards([accepted])
    assert accepted.target_text == r"$a \text{你好} b$"  # untouched
    assert counts["c_text_accepted"] == 1
    assert counts["formula_invariant_repairs"] == 0
    assert len(checkpoints) == 1
    assert any(f.startswith("c_text_span_translation:") for f in _flags(accepted))


def test_apply_math_guards_falls_back_a_failed_debris_block() -> None:
    debris = _block(
        "d3",
        "F 1 , K , V , x",
        "garbage draft",
        block_type=BlockType.NARRATIVE,
        status=BlockStatus.FAILED,
    )
    checkpoints, counts = apply_math_guards([debris])
    assert debris.target_text == "F 1 , K , V , x"
    assert counts["math_debris_fallbacks"] == 1
    assert len(checkpoints) == 1


def test_apply_math_guards_ignores_a_passing_debris_block() -> None:
    passing = _block(
        "d4",
        "F 1 , K , V , x",
        "garbage draft",
        block_type=BlockType.NARRATIVE,
        status=BlockStatus.PENDING,
    )
    checkpoints, counts = apply_math_guards([passing])
    assert passing.target_text == "garbage draft"
    assert counts == {
        "formula_invariant_repairs": 0,
        "math_debris_fallbacks": 0,
        "c_text_accepted": 0,
    }
    assert checkpoints == []


def test_apply_math_guards_ignores_a_failed_non_debris_block() -> None:
    prose = _block(
        "d6",
        "hello world",
        "translated prose",
        block_type=BlockType.NARRATIVE,
        status=BlockStatus.FAILED,
    )
    checkpoints, counts = apply_math_guards([prose])
    assert prose.target_text == "translated prose"
    assert prose.error_flags == []
    assert counts["math_debris_fallbacks"] == 0
    assert checkpoints == []


def test_apply_math_guards_is_idempotent() -> None:
    blocks = [
        _block("d1", "$x + y$", "$x - y$"),
        _block("d2", r"$a \text{hello} b$", r"$a \text{你好} b$"),
        _block(
            "d3",
            "F 1 , K , V , x",
            "garbage draft",
            block_type=BlockType.NARRATIVE,
            status=BlockStatus.FAILED,
        ),
    ]
    apply_math_guards(blocks)
    flags_after_first = {b.id: len(b.error_flags) for b in blocks}
    checkpoints, counts = apply_math_guards(blocks)
    assert counts == {
        "formula_invariant_repairs": 0,
        "math_debris_fallbacks": 0,
        "c_text_accepted": 0,
    }
    assert checkpoints == []
    assert {b.id: len(b.error_flags) for b in blocks} == flags_after_first


def test_apply_math_guards_leaves_a_clean_formula_alone() -> None:
    clean = _block("d5", "$x + y$", "$x + y$")
    checkpoints, counts = apply_math_guards([clean])
    assert clean.error_flags == []
    assert counts == {
        "formula_invariant_repairs": 0,
        "math_debris_fallbacks": 0,
        "c_text_accepted": 0,
    }
    assert checkpoints == []


def test_apply_math_guards_never_touches_block_status() -> None:
    blocks: list[IRBlock] = [
        _block("d1", "$x + y$", "$x - y$"),
        _block(
            "d3",
            "F 1 , K , V , x",
            "garbage draft",
            block_type=BlockType.NARRATIVE,
            status=BlockStatus.FAILED,
        ),
    ]
    statuses: dict[str, Any] = {b.id: b.status for b in blocks}
    apply_math_guards(blocks)
    assert {b.id: b.status for b in blocks} == statuses

"""C-track math-text protection: the skeleton invariant.

Inside a formula, only the natural-language bodies of ``\\text{…}``-family
commands may be translated; the math skeleton must survive byte-for-byte. The
whole module exists to enforce one invariant -- ``skeleton(source) ==
skeleton(target)`` -- and to fail closed (keep the source verbatim) whenever a
model returns something that would break it. These tests pin the invariant, the
conservative translatability gate, and the fail-closed paths.
"""

from __future__ import annotations

import pytest

from ubt.core.cleaners.math_text import (
    extract_text_spans,
    get_math_text_system_prompt,
    is_translatable_text,
    reassemble,
    skeleton,
    skeleton_holds,
    translatable_spans,
    translate_math_text,
)

pytestmark = pytest.mark.fast


async def _upper(text: str, target_lang: str) -> str:
    return text.upper()


async def _returns_command(text: str, target_lang: str) -> str:
    return r"\text{oops}"


async def _unchanged(text: str, target_lang: str) -> str:
    return text


async def _provider_down(text: str, target_lang: str) -> str:
    raise RuntimeError("provider down")


# --------------------------------------------------------------------------- #
# Span extraction.
# --------------------------------------------------------------------------- #


def test_extract_text_spans_covers_the_command_family() -> None:
    spans = extract_text_spans(r"\text{a} \mathrm{b} \operatorname{c} \textrm{d} \mbox{e}")
    assert [s.cmd for s in spans] == ["text", "mathrm", "operatorname", "textrm", "mbox"]
    assert [s.inner for s in spans] == ["a", "b", "c", "d", "e"]


def test_extract_text_spans_respects_balanced_braces() -> None:
    spans = extract_text_spans(r"\text{a {b} c}")
    assert len(spans) == 1
    assert spans[0].inner == "a {b} c"


def test_extract_text_spans_skips_unbalanced_input() -> None:
    assert extract_text_spans(r"\text{oops") == []


def test_extract_text_spans_does_not_descend_into_a_matched_span() -> None:
    spans = extract_text_spans(r"\text{\text{x}}")
    assert len(spans) == 1
    assert spans[0].inner == r"\text{x}"


# --------------------------------------------------------------------------- #
# Translatability gate (conservative by design).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("inner", "expected"),
    [
        ("Hit Rate", True),
        ("by terms", True),
        ("softmax", False),  # single token
        ("Figure 1", False),  # digits are Gate-4 territory
        (r"a\b", False),  # command
        ("a$b$", False),  # math operator
        ("", False),
        ("x" * 121, False),  # over the length cap
    ],
)
def test_is_translatable_text_only_accepts_multiword_plain_phrases(
    inner: str, expected: bool
) -> None:
    assert is_translatable_text(inner) is expected


def test_translatable_spans_filters_untranslatable_ones() -> None:
    spans = translatable_spans(r"\mathrm{softmax} + \text{by terms}")
    assert [(s.cmd, s.inner) for s in spans] == [("text", "by terms")]


# --------------------------------------------------------------------------- #
# The skeleton invariant.
# --------------------------------------------------------------------------- #


def test_skeleton_is_identity_without_spans() -> None:
    assert skeleton(r"a+b=c") == r"a+b=c"


def test_skeleton_blanks_only_the_braced_body() -> None:
    assert skeleton(r"Z(\text{Hit Rate})") == "Z(\\text{\x00SPAN0\x00})"


def test_skeleton_distinguishes_command_names() -> None:
    # The command is part of the skeleton: a translation that rewrote \text as
    # \mathrm must not compare equal.
    assert skeleton(r"\text{x}") != skeleton(r"\mathrm{x}")


def test_skeleton_normalizes_command_brace_whitespace() -> None:
    assert skeleton(r"\text {x}") == skeleton(r"\text{x}")


def test_skeleton_holds_when_only_span_contents_change() -> None:
    assert skeleton_holds(r"\text{Hit Rate}", r"\text{命中率}")


def test_skeleton_holds_rejects_a_changed_command() -> None:
    assert not skeleton_holds(r"\text{x}", r"\mathrm{x}")


def test_skeleton_holds_rejects_a_changed_math_operator() -> None:
    # \mathrm / \operatorname carry symbols (softmax, units), not prose.
    assert not skeleton_holds(r"\mathrm{softmax}", r"\mathrm{softmaxx}")


def test_skeleton_holds_rejects_a_changed_span_count() -> None:
    assert not skeleton_holds(r"\text{a} \text{b}", r"\text{a}")


def test_skeleton_holds_rejects_a_change_outside_spans() -> None:
    assert not skeleton_holds(r"x+\text{a}", r"y+\text{a}")


# --------------------------------------------------------------------------- #
# Reassembly.
# --------------------------------------------------------------------------- #


def test_reassemble_splices_by_index_and_keeps_the_rest() -> None:
    assert reassemble(r"\text{a} \text{b}", {0: "A"}) == r"\text{A} \text{b}"


def test_reassemble_normalizes_command_brace_whitespace() -> None:
    assert reassemble(r"\text {x}", {}) == r"\text{x}"


def test_reassemble_of_a_translation_satisfies_the_skeleton() -> None:
    source = r"Z(\text{Hit Rate})"
    target = reassemble(source, {0: "命中率"})
    assert skeleton_holds(source, target)


# --------------------------------------------------------------------------- #
# The async driver: success and fail-closed paths.
# --------------------------------------------------------------------------- #


async def test_translate_math_text_returns_none_without_translatable_spans() -> None:
    assert await translate_math_text(r"a+b", "zh", _upper) is None
    assert await translate_math_text(r"\mathrm{softmax}", "zh", _upper) is None


async def test_translate_math_text_reassembles_on_success() -> None:
    result = await translate_math_text(r"Z(\text{Hit Rate})", "zh", _upper)
    assert result == r"Z(\text{HIT RATE})"


async def test_translate_math_text_fails_closed_on_a_returned_command() -> None:
    assert await translate_math_text(r"\text{Hit Rate}", "zh", _returns_command) is None


async def test_translate_math_text_fails_closed_on_provider_error() -> None:
    assert await translate_math_text(r"\text{Hit Rate}", "zh", _provider_down) is None


async def test_translate_math_text_returns_none_when_nothing_changed() -> None:
    assert await translate_math_text(r"\text{Hit Rate}", "zh", _unchanged) is None


# --------------------------------------------------------------------------- #
# Prompt construction.
# --------------------------------------------------------------------------- #


def test_math_text_prompt_names_both_languages() -> None:
    assert "English" in get_math_text_system_prompt()
    assert "Chinese" in get_math_text_system_prompt()
    prompt = get_math_text_system_prompt("ja", "ko")
    assert "Japanese" in prompt
    assert "Korean" in prompt

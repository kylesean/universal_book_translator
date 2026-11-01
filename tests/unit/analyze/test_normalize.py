from __future__ import annotations

import pytest

from ubt.analyze.normalize import normalize_text

pytestmark = pytest.mark.fast

_LIGATURES: tuple[tuple[str, str], ...] = (
    ("\ufb00", "ff"),
    ("\ufb01", "fi"),
    ("\ufb02", "fl"),
    ("\ufb03", "ffi"),
    ("\ufb04", "ffl"),
    ("\ufb05", "ft"),
    ("\ufb06", "st"),
)

_INVISIBLE: tuple[str, ...] = (
    "\u00ad",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
)


@pytest.mark.parametrize(("ligature", "expansion"), _LIGATURES)
def test_presentation_ligatures_expand_to_their_letters(ligature: str, expansion: str) -> None:
    assert normalize_text(ligature) == expansion


@pytest.mark.parametrize(("ligature", "expansion"), _LIGATURES)
def test_ligatures_expand_inside_a_word(ligature: str, expansion: str) -> None:
    assert normalize_text(f"a{ligature}b") == f"a{expansion}b"


@pytest.mark.parametrize("invisible", _INVISIBLE)
def test_invisible_formatting_characters_are_removed(invisible: str) -> None:
    assert normalize_text(f"a{invisible}b") == "ab"


def test_non_breaking_space_becomes_an_ordinary_space() -> None:
    assert normalize_text("a\u00a0b") == "a b"


def test_empty_input_is_returned_unchanged() -> None:
    assert normalize_text("") == ""


@pytest.mark.parametrize(
    "text",
    [
        "plain text",
        "\ufb01n",
        "\ufb02ow",
        "co\u00adoperate",
        "a\u200bb",
        "a\u00a0b",
        "a\ufb01b\u200bc\u00a0d",
        "mixed \ufb03 and \u2060 invisible \ufeff",
    ],
)
def test_normalization_is_idempotent(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_a_string_of_every_normalized_form_collapses_in_one_pass() -> None:
    raw = "a\ufb00b\ufb01c\ufb02d\u00ade\u200bf\u200cg\u200dh\u2060i\ufeffj\u00a0k"
    assert normalize_text(raw) == "affbficfldefghij k"
    assert normalize_text(normalize_text(raw)) == "affbficfldefghij k"


@pytest.mark.parametrize(
    "text",
    [
        "ABC",
        "\uff21\uff22",
        "café",
        "\u4e2d\u6587",
        "\U0001f600",
    ],
)
def test_visible_text_is_left_byte_faithful(text: str) -> None:
    assert normalize_text(text) == text

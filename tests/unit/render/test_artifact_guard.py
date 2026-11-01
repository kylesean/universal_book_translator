from __future__ import annotations

import pytest

from ubt.render.typst.artifact_guard import (
    Sentinel,
    find_sentinels,
    has_sentinel,
)

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    ("token", "kind"),
    [
        ("⟦MATH_MASK_0001-abc⟧", "math"),
        ("⟦SOUP_MASK_0002-def⟧", "soup"),
        ("⟦CITE_MASK_0003-012⟧", "citation"),
        ("⟦CODE_MASK_0004-345⟧", "code"),
        ("⟦EMAIL_MASK_0005-678⟧", "email"),
        ("⟦MASK_0006-9ab⟧", "mask"),
        ("⟦B⟧", "emphasis"),
        ("⟦/B⟧", "emphasis"),
        ("⟦SEMANTIC_BREAK⟧", "semantic_break"),
    ],
)
def test_every_family_is_recognised(token: str, kind: str) -> None:
    assert has_sentinel(token)
    assert find_sentinels(token) == [Sentinel(kind=kind, text=token)]


def test_the_registry_covers_every_masker_family() -> None:
    from ubt.segment.placeholders import default_placeholder_engine

    engine = default_placeholder_engine()
    maskers = {
        "email": engine._email,
        "code": engine._code,
        "math": engine._math,
        "soup": engine._soup,
        "citation": engine._citation,
    }
    for name, masker in maskers.items():
        prefix = masker.default_prefix
        token = f"{prefix}0001-abc⟧"
        assert has_sentinel(token), f"{name} prefix {prefix!r} is not covered by the guard"


def test_the_emphasis_sentinels_come_from_the_emphasis_module() -> None:
    from ubt.core.ir.emphasis import BOLD_CLOSE, BOLD_OPEN, ITALIC_CLOSE, ITALIC_OPEN

    for token in (BOLD_OPEN, BOLD_CLOSE, ITALIC_OPEN, ITALIC_CLOSE):
        assert has_sentinel(token), token
    assert [s.kind for s in find_sentinels(ITALIC_OPEN)] == ["emphasis"]


def test_a_latex_transcribed_italic_marker_is_a_sentinel() -> None:
    assert has_sentinel(r"\llbracket I \rrbracket")
    assert has_sentinel(r"\llbracket /I \rrbracket")
    assert has_sentinel(r"\\llbracket i \\rrbracket")


def test_the_semantic_break_token_is_covered() -> None:
    from ubt.core.ir.bifurcation import SEMANTIC_BREAK_TOKEN

    assert has_sentinel(SEMANTIC_BREAK_TOKEN)


def test_the_latex_transcription_of_the_emphasis_marker_is_recognised() -> None:
    for text in (
        r"\llbracket B \rrbracket",
        r"\llbracket /B \rrbracket",
        r"\llbracket / B \rrbracket",
        r"\llbracket b \rrbracket",
    ):
        assert has_sentinel(text), text
        assert [s.kind for s in find_sentinels(text)] == ["emphasis"], text


@pytest.mark.parametrize(
    "text",
    [
        "普通段落，没有任何标记。",
        r"$x^2 + y^2 = z^2$",
        r"see [21] and [32]",
        r"$[a, b]$",
        r"a \frac{1}{2} b",
        r"${\rm I\!R}$",
        "an unmatched ⟦ open bracket alone",
    ],
)
def test_ordinary_text_is_not_flagged(text: str) -> None:
    assert not has_sentinel(text), text
    assert find_sentinels(text) == []


def test_a_stray_open_bracket_does_not_swallow_the_paragraph() -> None:
    text = "before ⟦ here\nlots of ordinary prose\nafter ⟧ end"
    assert find_sentinels(text) == []


def test_sentinels_are_reported_in_order_with_their_text() -> None:
    text = "a ⟦CITE_MASK_0001-abc⟧ b ⟦SOUP_MASK_0002-def⟧ c"
    found = find_sentinels(text)

    assert [s.kind for s in found] == ["citation", "soup"]
    assert [s.text for s in found] == ["⟦CITE_MASK_0001-abc⟧", "⟦SOUP_MASK_0002-def⟧"]


def test_both_spellings_of_the_emphasis_marker_are_found_together() -> None:
    text = r"⟦B⟧甲⟦/B⟧ 与 \llbracket B \rrbracket 乙 \llbracket /B \rrbracket"

    assert find_sentinels(text) == [
        Sentinel(kind="emphasis", text="⟦B⟧"),
        Sentinel(kind="emphasis", text="⟦/B⟧"),
        Sentinel(kind="emphasis", text=r"\llbracket B \rrbracket"),
        Sentinel(kind="emphasis", text=r"\llbracket /B \rrbracket"),
    ]


def test_a_token_inside_a_formula_is_found_by_scanning_the_inner_text() -> None:
    assert has_sentinel(r"(\Delta, ⟦ B ⟧ A ⟦ /B ⟧)")
    assert has_sentinel("a_i : ⟦SOUP_MASK_0001-551⟧ b_j")
    assert has_sentinel(r"(\Delta, \llbracket B \rrbracket A)")

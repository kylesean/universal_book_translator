"""Body metrics follow the target's own typesetting convention, not one rule.

The renderer emitted ``first-line-indent: 2em`` for every target. That is the
Chinese rule (首行缩进两格, GB/T 15834-2011); Japanese is 一字下げ — one
character — and Korean inherits one character from 원고지 practice. So two of
the three languages this engine renders most were indented exactly twice as far
as their own tradition, and English received a CJK paragraph indent it does not
use. Nothing owned the value, so nothing could catch it.

These tests pin the *values against their sources*, and — just as important —
pin that a language whose convention is not recorded keeps the previous
literal rather than receiving a plausible-looking guess.
"""

from __future__ import annotations

import pytest

from ubt.layout.theme import TypographicConvention, convention_for, resolve_theme

pytestmark = pytest.mark.fast


def test_chinese_keeps_the_two_character_indent() -> None:
    """The one target whose output must not move.

    Chinese led with two characters before this table existed and still does;
    a change here would silently restyle every Chinese delivery.
    """
    zh = convention_for("zh")
    assert zh.first_line_indent_em == 2.0
    assert zh.leading_em == pytest.approx(0.78)


def test_japanese_indents_one_character() -> None:
    """一字下げ: one character, the post-war Japanese publishing convention."""
    assert convention_for("ja").first_line_indent_em == 1.0


def test_korean_indents_one_character() -> None:
    """원고지 practice leaves one 칸 blank; W3C klreq §4.2."""
    assert convention_for("ko").first_line_indent_em == 1.0


def test_a_regional_tag_inherits_its_base_language_convention() -> None:
    """zh-TW/zh-HK follow the Chinese rule; ja-JP follows Japanese."""
    assert convention_for("zh-TW").first_line_indent_em == 2.0
    assert convention_for("zh_HK").first_line_indent_em == 2.0
    assert convention_for("ja-JP").first_line_indent_em == 1.0
    assert convention_for("ko-KR").first_line_indent_em == 1.0


def test_a_latin_target_takes_a_latin_indent_and_tighter_leading() -> None:
    """One em, and ~1.35x line height — not a CJK paragraph rule."""
    en = convention_for("en")
    assert en.first_line_indent_em == 1.0
    # Typst's `leading` is the gap; measured advance ≈ 1 + leading, so 0.35
    # yields ~1.35x, inside the 1.2-1.35x band Latin body text uses.
    assert en.leading_em < 0.5


def test_an_unrecorded_convention_is_not_guessed() -> None:
    """A language with no recorded convention keeps what was emitted before.

    Arabic and Hebrew book typography do not inherit the Latin indent rule, and
    Thai line-breaking conventions are their own subject. Inventing values would
    make the delivery look researched while being wrong; keeping the previous
    literal is the honest default, and it is what this asserts.
    """
    for lang in ("ar", "he", "th", "lo", "my", "fa", "ur"):
        conv = convention_for(lang)
        assert conv.first_line_indent_em == 2.0, (
            f"{lang} has no recorded convention; it must keep the previous literal "
            f"rather than be restyled on a guess"
        )
        assert "pre-table" in conv.source, f"{lang} should be marked as a legacy default"


def test_every_recorded_convention_names_its_source() -> None:
    """A metric without a citation is a guess wearing a number's clothes."""
    for lang in ("zh", "ja", "ko", "en"):
        conv = convention_for(lang)
        assert isinstance(conv, TypographicConvention)
        assert conv.source and len(conv.source) > 10, f"{lang} has no usable source"
        assert "pre-table" not in conv.source


def test_the_unknown_language_default_stays_conservative() -> None:
    """An unrecognised tag must not receive the Chinese indent by accident."""
    conv = convention_for("xx-YY")
    # Not Chinese: falls to the Latin convention.
    assert conv.first_line_indent_em == 1.0


def test_resolve_theme_carries_the_convention() -> None:
    """The theme is the owner a renderer reads, so it must expose the value."""
    ja = resolve_theme("en", "ja")
    assert ja.first_line_indent_em == 1.0
    # An explicit leading still wins over the convention's default.
    custom = resolve_theme("en", "ja", leading_em=0.5)
    assert custom.leading_em == pytest.approx(0.5)


def test_the_preamble_emits_the_convention() -> None:
    """The emitted Typst must carry the value, not merely compute it."""
    from ubt.render.typst.document_renderer import TypstDocumentRenderer

    for lang, indent in (("zh", "2em"), ("ja", "1em"), ("ko", "1em"), ("en", "1em")):
        source = TypstDocumentRenderer().preamble(lang)
        par_line = next(line for line in source.splitlines() if line.startswith("#set par"))
        assert f"first-line-indent: (amount: {indent}, all: true)" in par_line, (
            f"{lang} preamble does not emit a {indent} indent: {par_line}"
        )


def test_the_academic_template_follows_the_target_too() -> None:
    """A paper has the same problem: it hard-coded a single indent amount."""
    from ubt.render.typst.academic import academic_preamble

    for lang, indent in (("zh", "2em"), ("ja", "1em"), ("ko", "1em")):
        for columns in (1, 2):
            source = academic_preamble(target_lang=lang, columns=columns)
            par_line = next(line for line in source.splitlines() if line.startswith("#set par"))
            assert f"amount: {indent}" in par_line, f"{lang}/{columns}col: {par_line}"

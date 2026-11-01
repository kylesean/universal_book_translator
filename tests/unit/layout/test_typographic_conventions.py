from __future__ import annotations

import pytest

from ubt.layout.theme import TypographicConvention, convention_for, resolve_theme

pytestmark = pytest.mark.fast


def test_chinese_keeps_the_two_character_indent() -> None:
    zh = convention_for("zh")
    assert zh.first_line_indent_em == 2.0
    assert zh.leading_em == pytest.approx(0.78)


def test_japanese_indents_one_character() -> None:
    assert convention_for("ja").first_line_indent_em == 1.0


def test_korean_indents_one_character() -> None:
    assert convention_for("ko").first_line_indent_em == 1.0


def test_a_regional_tag_inherits_its_base_language_convention() -> None:
    assert convention_for("zh-TW").first_line_indent_em == 2.0
    assert convention_for("zh_HK").first_line_indent_em == 2.0
    assert convention_for("ja-JP").first_line_indent_em == 1.0
    assert convention_for("ko-KR").first_line_indent_em == 1.0


def test_a_latin_target_takes_a_latin_indent_and_tighter_leading() -> None:
    en = convention_for("en")
    assert en.first_line_indent_em == 1.0
    assert en.leading_em < 0.5


def test_an_unrecorded_convention_is_not_guessed() -> None:
    for lang in ("ar", "he", "th", "lo", "my", "fa", "ur"):
        conv = convention_for(lang)
        assert conv.first_line_indent_em == 2.0, (
            f"{lang} has no recorded convention; it must keep the previous literal "
            f"rather than be restyled on a guess"
        )
        assert "pre-table" in conv.source, f"{lang} should be marked as a legacy default"


def test_every_recorded_convention_names_its_source() -> None:
    for lang in ("zh", "ja", "ko", "en"):
        conv = convention_for(lang)
        assert isinstance(conv, TypographicConvention)
        assert conv.source and len(conv.source) > 10, f"{lang} has no usable source"
        assert "pre-table" not in conv.source


def test_the_unknown_language_default_stays_conservative() -> None:
    conv = convention_for("xx-YY")
    assert conv.first_line_indent_em == 1.0


def test_resolve_theme_carries_the_convention() -> None:
    ja = resolve_theme("en", "ja")
    assert ja.first_line_indent_em == 1.0
    custom = resolve_theme("en", "ja", leading_em=0.5)
    assert custom.leading_em == pytest.approx(0.5)


def test_the_preamble_emits_the_convention() -> None:
    from ubt.render.typst.document_renderer import TypstDocumentRenderer

    for lang, indent in (("zh", "2em"), ("ja", "1em"), ("ko", "1em"), ("en", "1em")):
        source = TypstDocumentRenderer().preamble(lang)
        par_line = next(line for line in source.splitlines() if line.startswith("#set par"))
        assert f"first-line-indent: (amount: {indent}, all: true)" in par_line, (
            f"{lang} preamble does not emit a {indent} indent: {par_line}"
        )


def test_the_academic_template_follows_the_target_too() -> None:
    from ubt.render.typst.academic import academic_preamble

    for lang, indent in (("zh", "2em"), ("ja", "1em"), ("ko", "1em")):
        for columns in (1, 2):
            source = academic_preamble(target_lang=lang, columns=columns)
            par_line = next(line for line in source.splitlines() if line.startswith("#set par"))
            assert f"amount: {indent}" in par_line, f"{lang}/{columns}col: {par_line}"


def test_the_source_indent_is_never_used_as_the_target_indent() -> None:
    from ubt.core.ir.models import StyleMeta
    from ubt.render.typst.document_renderer import TypstDocumentRenderer

    source = TypstDocumentRenderer().preamble("zh")
    par_line = next(line for line in source.splitlines() if line.startswith("#set par"))
    assert "amount: 2em" in par_line

    style = StyleMeta(first_line_indent_pt=17.4)
    assert style.model_dump()["first_line_indent_pt"] == pytest.approx(17.4)


def test_the_renderer_does_not_read_the_source_indent() -> None:
    import re
    from pathlib import Path

    import ubt.render as render_pkg

    render_root = Path(render_pkg.__file__).resolve().parent
    offenders: list[str] = []
    for path in render_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"first_line_indent_pt", text):
            line_no = text.count("\n", 0, match.start()) + 1
            line = text.splitlines()[line_no - 1]
            if line.strip().startswith("#"):
                continue
            offenders.append(f"{path.relative_to(render_root)}:{line_no} {line.strip()}")
    assert not offenders, (
        "the render layer must take its indent from the target-language convention, "
        "not from the source block: " + "; ".join(offenders)
    )

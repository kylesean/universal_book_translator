from __future__ import annotations

import pytest

from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.fast


def test_a_japanese_target_does_not_lead_with_the_sc_face() -> None:
    source = TypstDocumentRenderer().preamble("ja")
    assert "Noto Serif CJK JP" in source
    assert 'font: ("Libertinus Serif", "Noto Serif CJK SC")' not in source


def test_default_stack_is_unchanged_for_zh() -> None:
    source = TypstDocumentRenderer().preamble("zh")
    assert '#set text(font: ("Noto Serif CJK SC",' in source
    en = TypstDocumentRenderer().preamble("en")
    assert en.startswith("#set page(")
    assert "CJK" not in en.split("#set text(font: ")[1].split(")")[0]


def test_the_latin_default_names_a_face_the_bundled_set_carries() -> None:
    from ubt.core.language_profile import resolve_font_config
    from ubt.render.typst.fonts import BUNDLED_FONTS

    bundled_latin = {
        face.family
        for face in BUNDLED_FONTS
        if "latin" in face.scripts and "cjk" not in face.scripts
    }
    assert bundled_latin, "the set must ship at least one Latin face"

    for lang in ("en", "fr", "de", "es", "it", "pt", "nl"):
        configured = set(resolve_font_config(lang).typst_fonts)
        assert configured & bundled_latin, (
            f"{lang} resolves to {sorted(configured)}, none of which the bundled set carries; "
            f"an isolated render would silently use Typst's builtin instead"
        )


def test_a_supplied_font_stack_leads_the_body() -> None:
    renderer = TypstDocumentRenderer(font_stack=("Noto Serif CJK JP", "Liberation Serif"))
    source = renderer.preamble("ja")
    assert (
        '#set text(font: ("Noto Serif CJK JP", '
        '(name: "Liberation Serif", covers: "latin-in-cjk")),' in source
    )


def test_headings_reverse_the_stack_so_the_cjk_face_leads() -> None:
    renderer = TypstDocumentRenderer(font_stack=("Libertinus Serif", "Noto Serif CJK SC"))
    source = renderer.preamble("zh")
    assert (
        'font: ("Noto Serif CJK SC", (name: "Libertinus Serif", covers: "latin-in-cjk"))' in source
    )

"""The preamble's font stack follows the target language (W3-5).

Two dead-capability regressions in one wiring: the preamble hard-coded
("Libertinus Serif", "Noto Serif CJK SC") for every target, so a Japanese
book typeset kanji with Simplified-Chinese glyphs (JK-different forms for
骨/直/与...), and ``config.font_family`` — documented as "Override body font
family for typesetter" — was stored on the strategy and never read by
anything on the delivery path.
"""

from __future__ import annotations

import pytest

from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.fast


def test_a_japanese_target_does_not_lead_with_the_sc_face() -> None:
    source = TypstDocumentRenderer().preamble("ja")
    # The body stack resolves through language_profile: JP faces lead.
    assert "Noto Serif CJK JP" in source
    assert 'font: ("Libertinus Serif", "Noto Serif CJK SC")' not in source


def test_default_stack_is_unchanged_for_zh() -> None:
    """zh resolves through language_profile to the SC-first stack.

    The order differs from the pre-wiring hard-code (Latin face led): the
    language stack leads with the real CJK face so Chinese text does not
    fall through to the Latin serif, which is the point of the wiring.
    English still resolves Latin-first.

    The assertion is about *shape*, not about a specific family name: which
    Latin face survives is whatever the host (or the bundled set) provides, and
    a test that pinned ``Liberation Serif`` passed only on machines where that
    face was installed. What must hold everywhere is that an English preamble
    never leads with a CJK face — the CJK-first ordering exists for Chinese.
    """
    source = TypstDocumentRenderer().preamble("zh")
    assert '#set text(font: ("Noto Serif CJK SC",' in source
    en = TypstDocumentRenderer().preamble("en")
    assert en.startswith("#set page(")
    assert "CJK" not in en.split("#set text(font: ")[1].split(")")[0]


def test_the_latin_default_names_a_face_the_bundled_set_carries() -> None:
    """A Latin target must be able to isolate onto a face it actually names.

    ``can_isolate`` seals system fonts for every Latin target. If the configured
    Latin stack names only host faces, the sealed compiler cannot resolve any of
    them and falls through to Typst's built-in Libertinus — while the quality
    report still prints the host face it was asked for. The stack therefore has
    to include the bundled Latin family, and that family has to be the one the
    set declares.

    This is deterministic: it reads the configured data, not the host's fonts.
    """
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
    """Order is preserved; the Latin tail is scoped rather than emitted bare.

    A bare Latin entry in a CJK document claims the shared punctuation
    (curly quotes, dashes, ellipsis), so it carries ``covers: "latin-in-cjk"``.
    The *preference order* the caller supplied is unchanged.
    """
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

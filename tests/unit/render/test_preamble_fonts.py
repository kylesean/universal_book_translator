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
    """
    source = TypstDocumentRenderer().preamble("zh")
    assert '#set text(font: ("Noto Serif CJK SC",' in source
    en = TypstDocumentRenderer().preamble("en")
    assert en.startswith("#set page(")
    assert '"Liberation Serif"' in en


def test_a_supplied_font_stack_leads_the_body() -> None:
    renderer = TypstDocumentRenderer(font_stack=("Noto Serif CJK JP", "Liberation Serif"))
    source = renderer.preamble("ja")
    assert '#set text(font: ("Noto Serif CJK JP", "Liberation Serif"),' in source


def test_headings_reverse_the_stack_so_the_cjk_face_leads() -> None:
    renderer = TypstDocumentRenderer(font_stack=("Libertinus Serif", "Noto Serif CJK SC"))
    source = renderer.preamble("zh")
    assert 'font: ("Noto Serif CJK SC", "Libertinus Serif")' in source

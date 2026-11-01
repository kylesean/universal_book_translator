from __future__ import annotations

import pytest

from ubt.layout.theme import Direction, direction_for, resolve_theme
from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.fast

_CASES = (
    ("en", Direction.LTR),
    ("zh", Direction.LTR),
    ("ja", Direction.LTR),
    ("ar", Direction.RTL),
    ("he", Direction.RTL),
    ("fa", Direction.RTL),
    ("ur", Direction.RTL),
    ("ckb", Direction.RTL),
    ("ar-Arab", Direction.RTL),
    ("en-Arab", Direction.RTL),
    ("ar-Latn", Direction.LTR),
    ("zh-Hans", Direction.LTR),
    ("", Direction.LTR),
    (None, Direction.LTR),
)


@pytest.mark.parametrize(("tag", "want"), _CASES, ids=lambda v: str(v))
def test_direction_for_the_language_table(tag: str | None, want: Direction) -> None:
    assert direction_for(tag) is want


def test_an_rtl_target_resolves_an_rtl_theme_with_fonts() -> None:
    ar = resolve_theme("en", "ar")
    assert direction_for(ar.target_lang) is Direction.RTL
    assert ar.fonts, "an RTL target must still resolve a font stack"


def test_the_preamble_emits_the_rtl_direction_line_only_for_rtl() -> None:
    ltr = TypstDocumentRenderer().preamble("zh")
    assert 'lang: "zh"' in ltr
    assert "dir: rtl" not in ltr

    rtl = TypstDocumentRenderer().preamble("ar")
    assert 'lang: "ar"' in rtl
    assert "dir: rtl" in rtl


def test_the_preamble_uses_the_target_language_not_a_hardcoded_one() -> None:
    for lang in ("en", "ja", "de", "he"):
        assert f'lang: "{lang}"' in TypstDocumentRenderer().preamble(lang)

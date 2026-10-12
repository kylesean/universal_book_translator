"""Direction detection and the typesetter's RTL emission seam.

A :class:`~ubt.layout.theme.Theme` owns direction, direction is detected from
the language tag (a script subtag wins over the base language), and the fragment
typesetter emits ``dir: rtl`` for an RTL target while keeping the text in logical
order (Typst runs UAX #9 itself).

Pinned here:

- the detection table: RTL base languages, script subtags overriding the base
  (``ar-Latn`` is LTR, ``en-Arab`` is RTL), unknown/empty tags default LTR;
- an RTL target resolves a theme whose target direction is RTL with a
  non-empty font stack, while ``en``->``zh`` stays LTR;
- the fragment typesetter's direction line is empty for an LTR target and
  ``#set text(dir: rtl)`` for an RTL one.
"""

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
    ("en-Arab", Direction.RTL),  # a script subtag wins over the base language
    ("ar-Latn", Direction.LTR),
    ("zh-Hans", Direction.LTR),
    ("", Direction.LTR),
    (None, Direction.LTR),
)


@pytest.mark.parametrize(("tag", "want"), _CASES, ids=lambda v: str(v))
def test_direction_for_the_language_table(tag: str | None, want: Direction) -> None:
    assert direction_for(tag) is want


def test_an_rtl_target_resolves_an_rtl_theme_with_fonts() -> None:
    zh = resolve_theme("en", "zh")
    ar = resolve_theme("en", "ar")
    assert zh.source_direction is Direction.LTR
    assert zh.target_direction is Direction.LTR
    assert ar.source_direction is Direction.LTR
    assert ar.target_direction is Direction.RTL
    assert ar.fonts, "an RTL target must still resolve a font stack"


def test_the_preamble_emits_the_rtl_direction_line_only_for_rtl() -> None:
    """An RTL target must declare its direction, and an LTR one must not.

    The reflow renderer used to hardcode ``lang: "zh"``, so an Arabic or Hebrew
    book got Chinese line-breaking and left-to-right layout on every page. The
    language and direction now come from the target.
    """
    ltr = TypstDocumentRenderer.preamble("zh")
    assert 'lang: "zh"' in ltr
    assert "dir: rtl" not in ltr

    rtl = TypstDocumentRenderer.preamble("ar")
    assert 'lang: "ar"' in rtl
    assert "dir: rtl" in rtl


def test_the_preamble_uses_the_target_language_not_a_hardcoded_one() -> None:
    """Every target gets its own ``lang``: hyphenation and line-breaking follow it."""
    for lang in ("en", "ja", "de", "he"):
        assert f'lang: "{lang}"' in TypstDocumentRenderer.preamble(lang)

"""Direction detection and the theme's RTL emission seam.

The ADR's bidi decision landed as: a :class:`~ubt.layout.theme.Theme` owns
direction, direction is detected from the language tag (a script subtag wins
over the base language), and the L6 Typst seam emits ``dir: rtl`` for an RTL
target while keeping the text in logical order (Typst runs UAX #9 itself).

Pinned here:

- the detection table: RTL base languages, script subtags overriding the base
  (``ar-Latn`` is LTR, ``en-Arab`` is RTL), unknown/empty tags default LTR;
- an RTL target resolves a theme whose target direction is RTL with a
  non-empty font stack, while ``en``→``zh`` stays LTR;
- the Typst backend's fragment for a text element is unchanged for LTR (and
  for a theme-less backend, the legacy behaviour) and wrapped in
  ``#text(dir: rtl)[...]`` for RTL -- while the *payload* the verifier judges
  is the same text in every case: direction is presentation only.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from ubt.adapters.pdf.overlay_text import typst_escape
from ubt.layout.theme import Direction, Theme, direction_for, resolve_theme
from ubt.model.ast import Paragraph
from ubt.model.fidelity import Fidelity
from ubt.model.span import CanonicalSource, Span
from ubt.render.typst_backend import TypstBackend

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


class _Fragment(NamedTuple):
    label: str
    fragment: str
    payload: str


def _produce(theme: Theme | None) -> _Fragment:
    element = Paragraph(id="e1", spine_index=0, span=Span(), text="hello")
    backend = (
        TypstBackend({"e1": "hello"})
        if theme is None
        else TypstBackend({"e1": "hello"}, theme=theme)
    )
    produced = backend.produce(element, Fidelity.RECONSTRUCTED_ADAPTED, CanonicalSource(doc_id="t"))
    assert produced is not None
    return _Fragment(
        "no-theme" if theme is None else type(theme).__name__,
        produced.fragment,
        produced.payload,
    )


def test_the_ltr_fragment_is_unchanged_and_rtl_is_wrapped() -> None:
    zh = _produce(resolve_theme("en", "zh"))
    ar = _produce(resolve_theme("en", "ar"))
    plain = _produce(None)

    expected = typst_escape("hello")
    assert zh.fragment == expected
    assert plain.fragment == expected, "a theme-less backend must keep LTR behaviour"
    assert ar.fragment.startswith("#text(dir: rtl)[")


def test_direction_never_leaks_into_the_verifiers_payload() -> None:
    zh = _produce(resolve_theme("en", "zh"))
    ar = _produce(resolve_theme("en", "ar"))
    assert zh.payload == "hello"
    assert ar.payload == "hello"

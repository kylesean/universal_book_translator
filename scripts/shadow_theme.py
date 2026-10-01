#!/usr/bin/env python
"""§12 Q4 acceptance: direction detection and the theme's RTL emission seam.

The ADR asks whether bidi/RTL should be supported from day one. The decision
landed here: a :class:`~ubt.layout.theme.Theme` owns direction, direction is
detected from the language tag (a script subtag wins over the base language), and
the L6 Typst seam emits ``dir: rtl`` for an RTL target while keeping the text in
logical order (Typst runs UAX #9 itself). End-to-end RTL jobs still need language
profiles and RTL fonts, which are a separate, documented step.

Checks the pure detection table, theme resolution for an RTL target, and that the
Typst backend's fragment is unchanged for LTR and wrapped for RTL.
"""

from __future__ import annotations

from ubt.adapters.pdf.overlay_text import typst_escape
from ubt.layout.theme import Direction, resolve_theme
from ubt.model.ast import Paragraph
from ubt.model.fidelity import Fidelity
from ubt.model.span import CanonicalSource, Span
from ubt.render.typst_backend import TypstBackend

_DIRECTION_CASES: tuple[tuple[str, Direction], ...] = (
    ("en", Direction.LTR),
    ("zh", Direction.LTR),
    ("ja", Direction.LTR),
    ("ar", Direction.RTL),
    ("he", Direction.RTL),
    ("fa", Direction.RTL),
    ("ur", Direction.RTL),
    ("ckb", Direction.RTL),
    ("ar-Arab", Direction.RTL),
    ("en-Arab", Direction.RTL),  # script subtag wins over the base language
    ("ar-Latn", Direction.LTR),
    ("zh-Hans", Direction.LTR),
    ("", Direction.LTR),
    (None, Direction.LTR),
)


def main() -> int:
    from ubt.layout.theme import direction_for

    problems = [
        f"direction_for({tag!r}) != {want}"
        for tag, want in _DIRECTION_CASES
        if direction_for(tag) is not want
    ]

    zh_theme = resolve_theme("en", "zh")
    ar_theme = resolve_theme("en", "ar")
    if zh_theme.target_direction is not Direction.LTR:
        problems.append("zh theme is not LTR")
    if ar_theme.target_direction is not Direction.RTL:
        problems.append("ar theme is not RTL")
    if not ar_theme.fonts:
        problems.append("ar theme has no font stack")

    source = CanonicalSource(doc_id="t")
    element = Paragraph(id="e1", spine_index=0, span=Span(), text="hello")
    ltr = TypstBackend({"e1": "hello"}, theme=zh_theme)
    rtl = TypstBackend({"e1": "hello"}, theme=ar_theme)
    plain = TypstBackend({"e1": "hello"})  # no theme: legacy LTR behaviour

    ltr_fragment = ltr.produce(element, Fidelity.RECONSTRUCTED_ADAPTED, source)
    rtl_fragment = rtl.produce(element, Fidelity.RECONSTRUCTED_ADAPTED, source)
    plain_fragment = plain.produce(element, Fidelity.RECONSTRUCTED_ADAPTED, source)
    assert ltr_fragment is not None and rtl_fragment is not None and plain_fragment is not None

    expected_ltr = typst_escape("hello")
    if ltr_fragment.fragment != expected_ltr:
        problems.append(f"LTR fragment {ltr_fragment.fragment!r} != {expected_ltr!r}")
    if plain_fragment.fragment != expected_ltr:
        problems.append("theme-less backend changed LTR behaviour")
    if not rtl_fragment.fragment.startswith("#text(dir: rtl)["):
        problems.append(f"RTL fragment not wrapped: {rtl_fragment.fragment!r}")
    # Direction is presentation only: the payload the verifier judges is untouched.
    if ltr_fragment.payload != "hello" or rtl_fragment.payload != "hello":
        problems.append("direction leaked into the verifier's payload")

    print(f"\nTheme / direction acceptance — {len(_DIRECTION_CASES)} tags")
    print(f"  zh target: {zh_theme.target_direction}  ar target: {ar_theme.target_direction}")
    print(f"  LTR fragment: {ltr_fragment.fragment!r}")
    print(f"  RTL fragment: {rtl_fragment.fragment!r}")
    print(f"\n  problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"    {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

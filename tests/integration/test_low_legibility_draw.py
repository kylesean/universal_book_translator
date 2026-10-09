"""A box too short for its target at 6pt is still drawn, and flagged.

The readable floor is 6pt. A source line the extractor split into fragments (or
a code/formula stub) is narrower than its own translation's line, so at 6pt the
target wraps to a second line its box has no room for. Keeping the source there
loses a translation that *could* have been delivered at 5.4pt, so the compositor
descends to ``_MIN_DRAW_PT`` and records ``low_legibility_font``. These tests pin
both halves of the trade and the search that decides the size.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_text_pdf

from ubt.adapters.pdf import pdf_struct
from ubt.render.outputs import (
    _DEFAULT_SLACK_FONT_PT,
    _FIT_TOL,
    _LINE_SLACK_RATIO,
    _MIN_DRAW_PT,
    _MIN_FONT_PT,
    LayerCompositor,
    Overlay,
    TypstFragmentTypesetter,
)

pytestmark = pytest.mark.slow

#: The line slack ``_compile_overlay`` adds below an extracted ink box when the
#: block carries no source font size.
_SLACK = _LINE_SLACK_RATIO * _DEFAULT_SLACK_FONT_PT
#: ``(title, body)`` for :func:`write_text_pdf`; the boxes below are fragments
#: of a source line, which is where the below-floor draw happens in practice.
_PAGE = ("The Attention Machine", "The machine relies on attention and runs a forward pass.")


def _fits(
    typesetter: TypstFragmentTypesetter, text: str, width: float, limit: float, size: float
) -> bool:
    return typesetter.measure_fixed(text, width, size) <= limit + _FIT_TOL


def test_the_two_stage_search_lands_on_the_largest_size_that_fits() -> None:
    """The bisection returns the readable maximum, not the proportional overshoot.

    A proportional step scaled by ``height / natural`` jumps straight past a
    line-count change: measured against this box it landed near 4pt where 5.375
    fits. Bisecting ``[_MIN_DRAW_PT, _MIN_FONT_PT]`` lands on the boundary.
    """
    typesetter = TypstFragmentTypesetter()
    try:
        text, width, height = "[模拟翻译] effect.", 27.6, 8.3
        limit = height + _SLACK
        size = typesetter._fit_size(text, width, limit)
        assert size is not None
        assert _MIN_DRAW_PT <= size < _MIN_FONT_PT, size
        assert _fits(typesetter, text, width, limit, size)
        assert not _fits(typesetter, text, width, limit, size + 0.2), "not the largest that fits"
    finally:
        typesetter.close()


def test_a_box_that_no_size_in_the_range_holds_still_keeps_the_source() -> None:
    """A 12pt-wide box cannot hold a 17-character line at any drawn size."""
    typesetter = TypstFragmentTypesetter()
    try:
        assert typesetter._fit_size("[模拟翻译] of", 12.1, 8.3 + _SLACK) is None
    finally:
        typesetter.close()


def test_a_box_that_fits_at_the_floor_is_never_dragged_below_it() -> None:
    """Lowering the floor must not change a box that already fit above it.

    Regression against the naive change: with the proportional step allowed past
    6pt, a box the target fits at the readable floor kept the same step -- which
    overshoots across the line-count change -- and landed near 4.4pt. That is a
    worse deliverable than the size it should have drawn, for no reason at all.
    """
    typesetter = TypstFragmentTypesetter()
    try:
        text, width, height = "[模拟翻译] Abstract", 57.0, 11.2
        limit = height + _SLACK
        assert _fits(typesetter, text, width, limit, _MIN_FONT_PT), "fixture must fit at the floor"
        size = typesetter._fit_size(text, width, limit)
        assert size is not None and size >= _MIN_FONT_PT, size
    finally:
        typesetter.close()


def test_the_compositor_records_a_below_floor_draw_on_the_placement(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    typesetter = TypstFragmentTypesetter()
    try:
        composition = LayerCompositor(source, typesetter=typesetter).compose(
            [Overlay("e1", 1, (54.0, 700.0, 81.6, 708.3), "[模拟翻译] effect.")], output
        )
    finally:
        typesetter.close()

    (placement,) = composition.placements
    assert placement.drawn
    assert placement.drawn_pt is not None and placement.drawn_pt < _MIN_FONT_PT
    assert composition.low_legibility_fonts == (("e1", placement.drawn_pt),)
    assert pdf_struct.page_sizes(output) == pdf_struct.page_sizes(source)


def test_a_roomy_box_is_not_reported_as_low_legibility(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    typesetter = TypstFragmentTypesetter()
    try:
        composition = LayerCompositor(source, typesetter=typesetter).compose(
            [Overlay("e1", 1, (54.0, 620.0, 500.0, 700.0), "Translated body text.")], output
        )
    finally:
        typesetter.close()

    assert composition.low_legibility_fonts == ()

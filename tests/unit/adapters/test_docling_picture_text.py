"""A figure's own text is dropped; a caption just below it is not."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ubt.adapters.pdf.docling_blocks import is_inside_picture

pytestmark = pytest.mark.fast

#: A plot area, in Docling's bottom-left coordinates.
_PLOT = SimpleNamespace(l=141.7, b=211.1, r=453.4, t=355.2)

#: Figure 6's plot area in the two-column corpus (Docling bottom-left coords),
#: the picture whose legend escaped at the old 2pt containment tolerance.
_FIGURE_6 = SimpleNamespace(l=63.47, b=554.50, r=531.82, t=744.74)


def _box(left: float, bottom: float, right: float, top: float) -> SimpleNamespace:
    return SimpleNamespace(l=left, b=bottom, r=right, t=top)


def _in_picture(
    box: SimpleNamespace, page: int = 9, pictures: list[tuple[int, Any]] | None = None
) -> bool:
    return is_inside_picture(page, box, pictures if pictures is not None else [(9, _PLOT)])


def test_a_label_fully_inside_the_plot_area_is_the_figures_own_text() -> None:
    assert _in_picture(_box(200.0, 260.0, 300.0, 280.0))


def test_an_axis_title_hanging_past_the_plot_edge_is_the_figures_own_text() -> None:
    # An x-axis title sits just below the plot area: most of its box overlaps the
    # picture (and its centre is inside), so it belongs to the figure even though
    # it is not fully contained. This is the "Sandbox count per task" case: it was
    # translated and painted into its ~100pt box at 15pt, wrecking the figure.
    axis_title = _box(262.3, 207.6, 361.6, 218.2)
    assert _in_picture(axis_title)


def test_a_legend_floating_above_the_plot_area_is_the_figures_own_text() -> None:
    # The real b0084 case: a two-series legend ("Enabled Alone" / "All Mechanisms
    # Enabled") whose top sits 6.8pt above the picture bbox -- no overlap, centre
    # outside, so only the (widened) containment test can claim it. At the old 2pt
    # tolerance it leaked through as body prose, kept source in its 41pt column,
    # and tripped the delivery gate.
    legend = _box(234.9, 744.78, 379.1, 751.51)
    assert _in_picture(legend, page=10, pictures=[(10, _FIGURE_6)])


def test_body_text_above_the_figure_is_not_the_figures_own_text() -> None:
    # The nearest genuine prose sits ~22pt above the picture (the corpus
    # "SoL-Pi : Recursively Scaling..." heading); the widened margin must stop
    # well short of it, or a heading above a figure would be swallowed as its text.
    prose_above = _box(141.7, 755.0, 453.4, 766.7)
    assert not _in_picture(prose_above, page=10, pictures=[(10, _FIGURE_6)])


def test_a_caption_below_the_picture_is_not_the_figures_own_text() -> None:
    # "Figure 2 | The distribution ...": no overlap at all, so it stays
    # translatable.
    caption = _box(70.9, 173.1, 524.4, 197.3)
    assert not _in_picture(caption)


def test_a_footnote_below_the_picture_is_not_the_figures_own_text() -> None:
    assert not _in_picture(_box(70.9, 52.7, 360.6, 63.1))


def test_a_block_mostly_outside_the_picture_is_not_the_figures_own_text() -> None:
    # A body paragraph that only clips the picture's corner keeps its text.
    clipped = _box(440.0, 200.0, 530.0, 240.0)
    assert not _in_picture(clipped)


def test_a_picture_on_another_page_does_not_claim_the_text() -> None:
    assert not _in_picture(_box(200.0, 260.0, 300.0, 280.0), page=10)

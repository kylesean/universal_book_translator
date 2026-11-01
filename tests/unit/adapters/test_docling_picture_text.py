from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ubt.adapters.pdf.docling_blocks import is_inside_picture

pytestmark = pytest.mark.fast

_PLOT = SimpleNamespace(l=141.7, b=211.1, r=453.4, t=355.2)

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
    axis_title = _box(262.3, 207.6, 361.6, 218.2)
    assert _in_picture(axis_title)


def test_a_legend_floating_above_the_plot_area_is_the_figures_own_text() -> None:
    legend = _box(234.9, 744.78, 379.1, 751.51)
    assert _in_picture(legend, page=10, pictures=[(10, _FIGURE_6)])


def test_body_text_above_the_figure_is_not_the_figures_own_text() -> None:
    prose_above = _box(141.7, 755.0, 453.4, 766.7)
    assert not _in_picture(prose_above, page=10, pictures=[(10, _FIGURE_6)])


def test_a_caption_below_the_picture_is_not_the_figures_own_text() -> None:
    caption = _box(70.9, 173.1, 524.4, 197.3)
    assert not _in_picture(caption)


def test_a_footnote_below_the_picture_is_not_the_figures_own_text() -> None:
    assert not _in_picture(_box(70.9, 52.7, 360.6, 63.1))


def test_a_block_mostly_outside_the_picture_is_not_the_figures_own_text() -> None:
    clipped = _box(440.0, 200.0, 530.0, 240.0)
    assert not _in_picture(clipped)


def test_a_picture_on_another_page_does_not_claim_the_text() -> None:
    assert not _in_picture(_box(200.0, 260.0, 300.0, 280.0), page=10)

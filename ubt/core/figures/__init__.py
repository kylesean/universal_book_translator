"""Publication-grade multi-panel figure grid aligner and subcaption parser."""

from __future__ import annotations

from ubt.core.figures.detector import (
    calculate_grid_columns,
    detect_figure_panels,
    extract_main_and_subcaptions,
    parse_subcaptions,
)
from ubt.core.figures.generator import (
    render_figure_grid_html,
    render_figure_grid_typst,
)
from ubt.core.figures.models import (
    FigureLayoutConfig,
    FigureLayoutMode,
    FigurePanel,
    SubFigure,
)

__all__ = [
    "FigureLayoutConfig",
    "FigureLayoutMode",
    "FigurePanel",
    "SubFigure",
    "calculate_grid_columns",
    "detect_figure_panels",
    "extract_main_and_subcaptions",
    "parse_subcaptions",
    "render_figure_grid_html",
    "render_figure_grid_typst",
]

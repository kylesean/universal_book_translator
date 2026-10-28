"""Data models for multi-panel publication figures, subfigures, and grid alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class FigureLayoutMode(StrEnum):
    """Layout arrangement for multi-panel figures."""

    AUTO = "auto"
    GRID = "grid"
    STACK = "stack"


@dataclass(slots=True)
class SubFigure:
    """A single subpanel within a composite figure panel."""

    asset: str
    label: str = ""  # e.g., "(a)", "(b)", "A", "1"
    caption: str = ""  # e.g., "Architecture overview"
    width_pt: float | None = None
    block_id: str = ""
    aspect_ratio: float | None = None


@dataclass(slots=True)
class FigurePanel:
    """A publication-grade multi-panel figure composed of one or more subfigures."""

    panel_id: str = ""
    main_caption: str = ""
    subfigures: list[SubFigure] = field(default_factory=list)
    columns: int = 2
    gutter_pt: float = 10.0
    span_full_width: bool = False
    anchor_id: str = ""
    layout_mode: FigureLayoutMode = FigureLayoutMode.GRID

    @property
    def is_composite(self) -> bool:
        """True if panel contains multiple subfigures or has explicit subfigure labels."""
        return len(self.subfigures) > 1 or any(sf.label for sf in self.subfigures)


@dataclass(slots=True)
class FigureLayoutConfig:
    """Configuration options for figure detection and multi-panel grid layout."""

    columns: int = 2
    gutter_pt: float = 10.0
    auto_subfigures: bool = True
    max_full_width_threshold_pt: float = 400.0
    enable_subcaption_labels: bool = True
    default_subcaption_format: str = "({char})"
    max_columns: int = 4

"""Data models for publication chapter header decoration and drop caps."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class HeaderStyle(StrEnum):
    """Visual theme and layout style for chapter title headings."""

    CLASSIC = "classic"  # Centered, Roman/Chinese numerals, floral fleuron ornaments, double rule
    MODERN = "modern"  # Left-aligned, bold color-bar, clean geometric accents
    ACADEMIC = "academic"  # Formal hierarchical numbering with sleek horizontal rule
    MINIMAL = "minimal"  # Quiet elegance, understated small-caps tracking and whitespace


@dataclass(slots=True)
class DropCapConfig:
    """Configuration for paragraph initial drop caps."""

    enabled: bool = True
    lines: int = 2
    color: str = "#1e3a8a"  # Deep publishing navy
    hanging_quote: bool = True  # Hang opening quotes outside or group with initial
    cjk_scale: float = 1.35
    latin_scale: float = 1.50

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "lines": self.lines,
            "color": self.color,
            "hanging_quote": self.hanging_quote,
            "cjk_scale": self.cjk_scale,
            "latin_scale": self.latin_scale,
        }


@dataclass(slots=True)
class ChapterHeaderConfig:
    """Configuration for publication chapter opening headers."""

    enabled: bool = True
    style: HeaderStyle = HeaderStyle.CLASSIC
    accent_color: str = "#1e3a8a"
    secondary_color: str = "#64748b"
    show_fleuron: bool = True
    fleuron_char: str = "❦"
    top_spacing_cm: float = 1.2
    bottom_spacing_cm: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "style": str(self.style),
            "accent_color": self.accent_color,
            "secondary_color": self.secondary_color,
            "show_fleuron": self.show_fleuron,
            "fleuron_char": self.fleuron_char,
            "top_spacing_cm": self.top_spacing_cm,
            "bottom_spacing_cm": self.bottom_spacing_cm,
        }


@dataclass(slots=True)
class DecorSpec:
    """Full decorative specification for book layout rendering."""

    header: ChapterHeaderConfig = field(default_factory=ChapterHeaderConfig)
    drop_cap: DropCapConfig = field(default_factory=DropCapConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "drop_cap": self.drop_cap.to_dict(),
        }

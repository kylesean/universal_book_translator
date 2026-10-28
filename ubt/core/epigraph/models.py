"""Data models for chapter epigraphs, quotations, and abstract summary callouts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EpigraphStyle(StrEnum):
    """Visual presentation style for chapter epigraphs."""

    CLASSIC_RIGHT = "classic_right"  # Classical publishing: 75% width right-aligned block
    BORDER_LEFT = "border_left"  # Modern technical book: subtle left vertical accent stroke
    CALLOUT_BOX = "callout_box"  # Light rounded shaded container
    MINIMAL = "minimal"  # Centered clean italic quotation


@dataclass(slots=True)
class Epigraph:
    """A chapter-opening quotation or motto (题头引言 / 名人名言 / 题词)."""

    text: str
    source: str = ""
    chapter_index: int = 1
    block_id: str = ""


@dataclass(slots=True)
class ChapterAbstract:
    """A chapter-opening summary or introductory briefing (本章导读 / 内容提要)."""

    title: str = "本章导读"
    text: str = ""
    chapter_index: int = 1
    block_id: str = ""


@dataclass(slots=True)
class EpigraphConfig:
    """Configuration options for epigraph and chapter abstract formatting."""

    style: EpigraphStyle = EpigraphStyle.CLASSIC_RIGHT
    font_size_pt: float = 9.0
    author_prefix_zh: str = "—— "
    author_prefix_en: str = "— "
    enable_quotes_decoration: bool = True
    abstract_box_shaded: bool = True

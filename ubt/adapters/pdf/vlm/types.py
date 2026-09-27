"""Driver contract: recognition without geometry ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VlmLine:
    """One recognized line, in reading order.

    ``measured_box`` is PDF-point geometry ONLY when the driver measured it
    with a detector (rapidocr-style). LLM-VLM drivers MUST leave it None:
    hallucinated coordinates are worse than none.
    The adapter — never the driver — decides whether the box is used.
    """

    text: str
    reading_index: int
    confidence: float = 1.0
    measured_box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class PageTranscript:
    """Whole-page recognition result from one driver."""

    lines: tuple[VlmLine, ...] = ()
    engine: str = ""
    measured_boxes: bool = False


class VlmDriver(Protocol):
    """Recognition driver: image in, ordered lines out, no geometry promises."""

    name: str
    measured_boxes: bool

    def recognize(
        self,
        image: object,
        page_size_pt: tuple[float, float],
        scale: float,
    ) -> PageTranscript:
        """Recognize ``image`` (PIL) rendered at ``scale`` px/pt.

        ``page_size_pt`` + ``scale`` let detector-drivers convert pixel boxes
        to PDF points for ``measured_box``. LLM-drivers ignore both.
        """
        ...

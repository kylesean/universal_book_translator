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
    #: A vision-LLM driver that hit its output token limit (``finish_reason ==
    #: "length"``) sets this so the truncated page tail is not silently
    #: accepted as a complete transcription.
    truncated: bool = False


class VlmDriver(Protocol):
    """Recognition driver: image in, ordered lines out, no geometry promises."""

    name: str
    measured_boxes: bool

    def recognize(
        self,
        image: object,
        page_size_pt: tuple[float, float],
        scale: float,
        rotation: int = 0,
    ) -> PageTranscript:
        """Recognize ``image`` (PIL) rendered at ``scale`` px/pt.

        ``page_size_pt`` + ``scale`` let detector-drivers convert pixel boxes
        to PDF points for ``measured_box``. LLM-drivers ignore both.

        ``page_size_pt`` is the DISPLAYED page's size and ``rotation`` its
        ``/Rotate``: the bitmap is what a viewer shows, so a box measured on it
        is in the display frame. ``measured_box`` must come back in the
        unrotated user-space frame -- see
        :func:`~ubt.adapters.pdf.coordinate_resolver.undo_page_rotation` -- or
        every rotated page's geometry lands somewhere the renderer does not
        paint. LLM-drivers ignore this too.
        """
        ...

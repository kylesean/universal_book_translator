"""Build segments from a Document (the native reader's output).

The reader produces structure (elements); translation needs units. This is the
join: one source-side :class:`~ubt.model.segment.Segment` per translatable
element, its protected spans masked into placeholders so a translator or CAT tool
preserves them. The element id becomes the segment id, so a translation maps
straight back to the element it belongs to.
"""

from __future__ import annotations

from ubt.model.ast import Document, TextElement
from ubt.model.segment import Segment, SegmentState
from ubt.segment.placeholders import PlaceholderEngine
from ubt.segment.xliff import xml_safe


def segments_from_document(document: Document, *, engine: PlaceholderEngine) -> list[Segment]:
    """One source-side segment per translatable element, in reading order.

    Non-text elements (figures/formulas/tables) are skipped: they are carried
    as immutable assets by the delivery, not handed to a translator as prose.
    """
    segments: list[Segment] = []
    for element in document.elements:
        if not isinstance(element, TextElement):
            continue
        text = element.text
        if not text.strip():
            continue
        masked = engine.mask(xml_safe(text))
        segments.append(
            Segment(
                id=element.id,
                source=masked.text,
                placeholders=masked.placeholders,
                state=SegmentState.NEW,
            )
        )
    return segments


__all__ = ["segments_from_document"]

"""Source geometry and the canonical text stream (core document model).

A :class:`Span` is the stable back-reference from an element to *where it came
from*: a page, a bounding box, and (once the canonical stream exists) a
character range into :class:`CanonicalSource.text`. Making it first-class is
what lets every later decision -- reading order, verification, cropping a
source asset for fallback -- point at checkable evidence instead of guessing.

Nothing here imports the pipeline, a PDF library, or pydantic: the model layer
is pure data so it can be reasoned about (and serialized) without a document.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: Axis-aligned source rectangle in page points: ``(x0, y0, x1, y1)``.
BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Span:
    """Where an element came from.

    ``page`` is 1-based; ``0`` means "no page" (a format without pagination,
    e.g. Markdown or HTML). ``bbox`` and ``chars`` are optional because not
    every extractor can supply both; a span with neither still identifies the
    element by its position in reading order.
    """

    page: int = 0
    bbox: BBox | None = None
    chars: tuple[int, int] | None = None

    @property
    def on_page(self) -> bool:
        return self.page > 0

    @property
    def placed(self) -> bool:
        """True when the element has a page and a box -- i.e. can be cropped."""
        return self.on_page and self.bbox is not None


@dataclass(frozen=True, slots=True)
class PhysicalBox:
    """One rectangular container along an element's reading-order box chain."""

    page: int
    bbox: BBox
    available_width: float
    available_height: float

    @classmethod
    def of(cls, page: int, bbox: BBox) -> PhysicalBox:
        return cls(page, bbox, bbox[2] - bbox[0], bbox[3] - bbox[1])


@dataclass(frozen=True, slots=True)
class CompositeSpan:
    """A semantic element spread across several physical boxes.

    The geometry layer's answer to the tension between PDF's discrete 2D boxes
    and the 1D reading-order flow: one element (say, a paragraph crossing a page
    boundary) owns a chain of boxes in reading order, and the BreakageSolver
    flows its target text across them.
    """

    boxes: tuple[PhysicalBox, ...]
    chars: tuple[int, int] | None = None

    @property
    def is_composite(self) -> bool:
        return len(self.boxes) > 1

    @property
    def page(self) -> int:
        return self.boxes[0].page if self.boxes else 0

    @property
    def bbox(self) -> BBox | None:
        return self.boxes[0].bbox if self.boxes else None

    @property
    def on_page(self) -> bool:
        return self.page > 0

    @property
    def placed(self) -> bool:
        return self.on_page and bool(self.boxes)


def boxes_to_provenance(boxes: Sequence[PhysicalBox]) -> list[dict[str, Any]]:
    """Serialize a box chain to the ``physical_boxes`` ledger form.

    The ledger round-trips an element's box chain through ``block.provenance``
    as an untyped list of ``{"page", "bbox"}`` dicts (it cannot store the typed
    ``CompositeSpan`` directly). Centralizing the shape here is what keeps the
    writer and the reader in agreement: a typo in a hand-rolled dict silently
    collapsed a multi-box element to its first box on reload, squeezing a whole
    translation into one line.
    """
    return [{"page": box.page, "bbox": list(box.bbox)} for box in boxes]


def boxes_from_provenance(raw: object) -> tuple[PhysicalBox, ...]:
    """Rebuild a box chain from ``provenance["physical_boxes"]`` (empty if absent).

    Returns an empty tuple for a missing, malformed or single-box value, so the
    caller falls back to the element's own span rather than trusting a partial
    chain. Only a well-formed chain of two or more boxes is usable -- a single
    box adds nothing over the element's ``Span``.
    """
    if not isinstance(raw, list):
        return ()
    boxes: list[PhysicalBox] = []
    for item in raw:
        if not isinstance(item, dict):
            return ()
        try:
            page = int(item["page"])
            bbox = item["bbox"]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return ()
            boxes.append(
                PhysicalBox.of(
                    page, (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                )
            )
        except (KeyError, TypeError, ValueError):
            return ()
    return tuple(boxes) if len(boxes) > 1 else ()


@dataclass(frozen=True, slots=True)
class PageGeometry:
    """Physical size of one source page, in points."""

    index: int
    width_pt: float
    height_pt: float
    rotation: int = 0


@dataclass(frozen=True, slots=True)
class CanonicalSource:
    """The normalized input a document was understood from.

    ``doc_id`` is the content hash of the source file, so a derivation keyed on
    it is stable across paths and re-runs. ``text`` is the concatenated,
    offset-stable text stream that :class:`Span.chars` indexes into; it is empty
    until an analyzer builds it.
    """

    doc_id: str
    path: str = ""
    text: str = ""
    pages: tuple[PageGeometry, ...] = ()

    def page_geometry(self, index: int) -> PageGeometry | None:
        for page in self.pages:
            if page.index == index:
                return page
        return None


__all__ = [
    "BBox",
    "CanonicalSource",
    "CompositeSpan",
    "PageGeometry",
    "PhysicalBox",
    "Span",
    "boxes_from_provenance",
    "boxes_to_provenance",
]

"""The overlay backend: keep the source region whole (render backend lowering layer).

This is the floor of the fidelity lattice as a backend. Where a reflowing
backend re-typesets an element, this one places it as an *opaque source slice*
-- the original bytes, unchanged -- which is lossless by construction. It is what
:func:`ubt.pipeline.steps.realize` falls back to whenever no backend can
reconstruct an element, and the backend a document can always be realized with
when reconstruction is unavailable.

``produce`` returns the exact canonical slice for the element (its
:attr:`~ubt.model.span.Span.chars` into the document text, or the element's own
markup for an asset), so "preserved" is a *measurable* claim rather than a
label: the shadow checks the slice against the source element by element.
"""

from __future__ import annotations

from typing import ClassVar

from ubt.model.ast import ELEMENT_CLASSES, Element, Figure, Formula, Table, TextElement
from ubt.model.fidelity import Fidelity
from ubt.model.span import CanonicalSource
from ubt.render.capability import Capabilities, Produced

#: Every element class, preserved opaque -- the whole of what this backend can do.
_OPAQUE_ONLY: frozenset[tuple[type[Element], Fidelity]] = frozenset(
    (cls, Fidelity.PRESERVED_OPAQUE) for cls in ELEMENT_CLASSES
)


def _carried_source(element: Element) -> str:
    """The source an element carries itself, when the canonical stream cannot index it."""
    if isinstance(element, TextElement):
        return element.text
    if isinstance(element, Formula):
        return element.source
    if isinstance(element, Table):
        return element.markup
    if isinstance(element, Figure):
        return element.asset_id
    return ""


def source_slice(element: Element, source: CanonicalSource) -> str:
    """The exact source text one element preserves, from its character span.

    Falls back to the element's own carried source when the document has no
    canonical stream or the span does not index it -- the slice is then coarser,
    but still the source bytes rather than a reconstruction.
    """
    chars = element.span.chars
    if chars is not None:
        start, end = chars
        if 0 <= start <= end <= len(source.text):
            return source.text[start:end]
    return _carried_source(element)


class OverlayBackend:
    """Place-only backend: every element kept as an opaque source slice."""

    name: ClassVar[str] = "overlay"

    def capabilities(self) -> Capabilities:
        return Capabilities(supported=_OPAQUE_ONLY, reflows=False)

    def produce(
        self, element: Element, fidelity: Fidelity, source: CanonicalSource
    ) -> Produced | None:
        if fidelity is not Fidelity.PRESERVED_OPAQUE:
            return None
        return Produced(
            payload=source_slice(element, source),
            note=f"opaque:{element.kind}:p{element.span.page}",
        )


__all__ = ["OverlayBackend", "source_slice"]

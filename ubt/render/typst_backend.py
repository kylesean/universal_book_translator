"""The Typst backend: re-typeset translated text into Typst source (ADR-0001 L6).

The first reflowing backend, deliberately narrow: it declares
``RECONSTRUCTED_ADAPTED`` for **Paragraph** alone and places everything else
opaque. A paragraph is lowered by escaping its *translation* into Typst content.
The translation comes from the run (L4), so the backend is constructed with the
element -> target map; a paragraph with no translation has nothing to reflow and
returns ``None``, and :func:`ubt.pipeline.steps.realize` descends to the opaque
slice -- the lattice rule that keeps a document from getting worse than its
source.

The other fragment types (headings, lists, formulas, tables) and the drawing of
these fragments are later steps. Until they land, this backend's only *produced*
output is the paragraph fragment, and ``compose`` still descends every element to
its opaque slice.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ubt.adapters.pdf.overlay_text import typst_escape
from ubt.model.ast import ELEMENT_CLASSES, Element, Paragraph
from ubt.model.fidelity import Fidelity
from ubt.model.span import CanonicalSource
from ubt.render.capability import Capabilities, Produced
from ubt.render.overlay_backend import source_slice


def paragraph_fragment(text: str) -> str | None:
    """One paragraph's Typst content, or ``None`` when there is nothing to reflow."""
    body = text.strip()
    if not body:
        return None
    return typst_escape(body)


class TypstBackend:
    """Reflowing backend: paragraphs are re-typeset from their translation."""

    name: ClassVar[str] = "typst"

    def __init__(self, translations: Mapping[str, str] | None = None) -> None:
        self._translations = dict(translations or {})

    def capabilities(self) -> Capabilities:
        supported = {(cls, Fidelity.PRESERVED_OPAQUE) for cls in ELEMENT_CLASSES}
        supported.add((Paragraph, Fidelity.RECONSTRUCTED_ADAPTED))
        return Capabilities(supported=frozenset(supported), reflows=True)

    def produce(
        self, element: Element, fidelity: Fidelity, source: CanonicalSource
    ) -> Produced | None:
        if fidelity is Fidelity.PRESERVED_OPAQUE:
            return Produced(payload=source_slice(element, source), note=f"opaque:{element.kind}")
        if fidelity is Fidelity.RECONSTRUCTED_ADAPTED and isinstance(element, Paragraph):
            target = self._translations.get(element.id, "")
            fragment = paragraph_fragment(target)
            if fragment is None:
                return None
            return Produced(payload=target, note="typst:paragraph", fragment=fragment)
        return None


__all__ = ["TypstBackend", "paragraph_fragment"]

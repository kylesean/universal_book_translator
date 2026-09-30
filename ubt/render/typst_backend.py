"""The Typst backend: re-typeset translated text into Typst source (ADR-0001 L6).

The reflowing backend. It declares ``RECONSTRUCTED_ADAPTED`` for the text
classes that can be re-typeset (heading, paragraph, dialogue, list item, caption)
and places everything else opaque -- code, formulas, tables and figures are
preserved, never reflowed. An element is lowered by escaping its *translation*
into Typst content; the translation comes from the run (L4), so the backend is
constructed with the element -> target map. An element with no translation has
nothing to reflow and returns ``None``, and :func:`ubt.pipeline.steps.realize`
descends to the opaque slice -- the lattice rule that keeps a document from
getting worse than its source.

``Produced`` keeps the two halves apart: ``payload`` is the delivered text the
verifier judges, ``fragment`` is the Typst source the compositor will draw. They
must not be conflated -- escaping turns Typst-special characters into markup, and
a verifier reading escaped text would judge the markup instead of the translation.

Drawing these fragments into an artifact is the compositor's job and a later
step; until it lands, ``compose`` still descends every element to its opaque
slice.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ubt.adapters.pdf.overlay_text import typst_escape
from ubt.model.ast import (
    ELEMENT_CLASSES,
    Caption,
    Dialogue,
    Element,
    Heading,
    ListItem,
    Paragraph,
    TextElement,
)
from ubt.model.fidelity import Fidelity
from ubt.model.span import CanonicalSource
from ubt.render.capability import Capabilities, Produced
from ubt.render.overlay_backend import source_slice

#: Text classes this backend can re-typeset. Code is deliberately absent: its
#: descent has no reconstructed rung, so it is only ever placed.
REFLOW_CLASSES: tuple[type[TextElement], ...] = (
    Heading,
    Paragraph,
    Dialogue,
    ListItem,
    Caption,
)


def text_fragment(text: str) -> str | None:
    """One text element's Typst content, or ``None`` when there is nothing to reflow."""
    body = text.strip()
    if not body:
        return None
    return typst_escape(body)


class TypstBackend:
    """Reflowing backend: translatable text is re-typeset from its translation."""

    name: ClassVar[str] = "typst"

    def __init__(self, translations: Mapping[str, str] | None = None) -> None:
        self._translations = dict(translations or {})

    def capabilities(self) -> Capabilities:
        supported = {(cls, Fidelity.PRESERVED_OPAQUE) for cls in ELEMENT_CLASSES}
        supported |= {(cls, Fidelity.RECONSTRUCTED_ADAPTED) for cls in REFLOW_CLASSES}
        return Capabilities(supported=frozenset(supported), reflows=True)

    def produce(
        self, element: Element, fidelity: Fidelity, source: CanonicalSource
    ) -> Produced | None:
        if fidelity is Fidelity.PRESERVED_OPAQUE:
            return Produced(payload=source_slice(element, source), note=f"opaque:{element.kind}")
        if fidelity is Fidelity.RECONSTRUCTED_ADAPTED and isinstance(element, REFLOW_CLASSES):
            if element.skip_translate:
                # Deliberately kept in the source (a listing, a proper noun): the
                # contract calls this VERBATIM, so it is placed opaque, not reflowed.
                return None
            target = self._translations.get(element.id, "")
            fragment = text_fragment(target)
            if fragment is None:
                return None
            return Produced(payload=target, note=f"typst:{element.kind}", fragment=fragment)
        return None


__all__ = ["REFLOW_CLASSES", "TypstBackend", "text_fragment"]

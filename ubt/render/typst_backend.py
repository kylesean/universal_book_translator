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
from typing import TYPE_CHECKING, ClassVar

from ubt.adapters.pdf.overlay_text import typst_escape
from ubt.layout.theme import Direction
from ubt.model.ast import (
    ELEMENT_CLASSES,
    Caption,
    Dialogue,
    Element,
    Formula,
    Heading,
    ListItem,
    Paragraph,
    Table,
    TextElement,
)
from ubt.model.fidelity import Fidelity
from ubt.model.span import CanonicalSource
from ubt.render.capability import Capabilities, Produced
from ubt.render.overlay_backend import source_slice

if TYPE_CHECKING:
    from ubt.layout.theme import Theme

#: Text classes this backend can re-typeset. Code is deliberately absent: its
#: descent has no reconstructed rung, so it is only ever placed.
REFLOW_CLASSES: tuple[type[TextElement], ...] = (
    Heading,
    Paragraph,
    Dialogue,
    ListItem,
    Caption,
)


def text_fragment(text: str, *, direction: Direction = Direction.LTR) -> str | None:
    """One text element's Typst content, or ``None`` when there is nothing to reflow.

    A right-to-left target is wrapped so Typst lays the run out right-to-left;
    the text stays in logical order (Typst runs UAX #9 itself), so nothing is
    reordered here.
    """
    body = text.strip()
    if not body:
        return None
    escaped = typst_escape(body)
    if direction is Direction.RTL:
        return f"#text(dir: rtl)[{escaped}]"
    return escaped


class TypstBackend:
    """Reflowing backend: translatable text is re-typeset from its translation."""

    name: ClassVar[str] = "typst"

    def __init__(
        self, translations: Mapping[str, str] | None = None, *, theme: Theme | None = None
    ) -> None:
        self._translations = dict(translations or {})
        self._theme = theme

    @property
    def _direction(self) -> Direction:
        return self._theme.target_direction if self._theme is not None else Direction.LTR

    def capabilities(self) -> Capabilities:
        supported = {(cls, Fidelity.PRESERVED_OPAQUE) for cls in ELEMENT_CLASSES}
        supported |= {(cls, Fidelity.RECONSTRUCTED_ADAPTED) for cls in REFLOW_CLASSES}
        supported |= {(Formula, Fidelity.RECONSTRUCTED_VERIFIED)}
        supported |= {(Table, Fidelity.RECONSTRUCTED_VERIFIED)}
        return Capabilities(supported=frozenset(supported), reflows=True)

    def produce(
        self, element: Element, fidelity: Fidelity, source: CanonicalSource
    ) -> Produced | None:
        if fidelity is Fidelity.PRESERVED_OPAQUE:
            return Produced(payload=source_slice(element, source), note=f"opaque:{element.kind}")
        target = self._translations.get(element.id, "")
        if fidelity is Fidelity.RECONSTRUCTED_ADAPTED and isinstance(element, REFLOW_CLASSES):
            if element.skip_translate:
                # Deliberately kept in the source (a listing, a proper noun): the
                # contract calls this VERBATIM, so it is placed opaque, not reflowed.
                return None
            fragment = text_fragment(target, direction=self._direction)
            if fragment is None:
                return None
            return Produced(payload=target, note=f"typst:{element.kind}", fragment=fragment)
        if fidelity is Fidelity.RECONSTRUCTED_VERIFIED:
            # The delivered markup is the reconstruction the existing renderer
            # produced; the structural verifier judges it, and a failure descends
            # to the opaque slice. With no delivered markup the element's own
            # source markup is the reconstruction -- the same fallback the
            # contract's asset policy uses, so the two agree.
            if isinstance(element, Formula):
                markup = target or element.source
            elif isinstance(element, Table):
                markup = target or element.markup
            else:
                return None
            if not markup.strip():
                return None
            return Produced(payload=markup, note=f"typst:{element.kind}")
        return None


__all__ = ["REFLOW_CLASSES", "TypstBackend", "text_fragment"]

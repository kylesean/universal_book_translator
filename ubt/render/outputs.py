"""Placement records for the secondary views (HTML / EPUB).

The PDF path does not lower onto the source page: it rebuilds the document from
the IR and lets Typst lay it out (``ubt/render/typst/document_renderer.py``).
These records are what the HTML and EPUB views use to say *where* each delivered
element ended up and whether it was drawn or kept as source. They are
deliberately plain data, so a second producer cannot disagree with the first
about what a placement means.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Placement:
    """One element's position in the artifact: whether it was drawn or kept."""

    element_id: str
    page: int
    drawn: bool
    detail: str = ""

    @property
    def kept_source(self) -> bool:
        """True when the element was kept as source rather than delivered."""
        return not self.drawn


@dataclass(frozen=True, slots=True)
class Composition:
    """A lowered artifact plus, per element, how it was placed."""

    output_path: Path
    placements: tuple[Placement, ...]

    @property
    def kept_source_ids(self) -> tuple[str, ...]:
        """Elements kept as source rather than delivered (no drawing for their target)."""
        return tuple(placement.element_id for placement in self.placements if placement.kept_source)


__all__ = ["Composition", "Placement"]

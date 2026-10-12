"""Placement records for the secondary views (HTML / EPUB).

The PDF path no longer lowers onto the source page: it rebuilds the document
from the IR and lets Typst lay it out (``ubt/render/typst/document_renderer.py``).
What remains here are the two records the HTML and EPUB views still use to say
*where* each delivered element ended up and whether it was drawn or kept as
source. They are deliberately plain data: the composer that used to produce them
is gone, and a second producer must not be able to disagree with the first about
what a placement means.

``_MIN_FONT_PT`` is the readability floor the placement records are compared
against; it is defined here rather than in the deleted composer so a future
lowering path has one place to read it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Below this drawn size a fragment is no longer reading-grade. Recorded as a
#: ``low_legibility_font`` defect rather than silently shipped.
_MIN_FONT_PT = 6.0


@dataclass(frozen=True, slots=True)
class Placement:
    """One element's position in the artifact: whether it was drawn or kept."""

    element_id: str
    page: int
    drawn: bool
    detail: str = ""
    #: The size (pt) the target was drawn at, when the lowering chose it by
    #: fitting (``None`` for a mask-only part, a math/TOC fragment, or a
    #: source-kept element). Below ``_MIN_FONT_PT`` the drawn fragment is no
    #: longer reading-grade and is reported as a ``low_legibility_font`` defect.
    drawn_pt: float | None = None

    @property
    def kept_source(self) -> bool:
        """True when the lowering kept the source because it cannot draw the target."""
        return not self.drawn


@dataclass(frozen=True, slots=True)
class Composition:
    """A lowered artifact plus, per element, how it was placed."""

    output_path: Path
    placements: tuple[Placement, ...]

    @property
    def by_page(self) -> dict[int, tuple[Placement, ...]]:
        """The placements grouped by source page -- the per-page composition."""
        grouped: dict[int, list[Placement]] = {}
        for placement in self.placements:
            grouped.setdefault(placement.page, []).append(placement)
        return {page: tuple(items) for page, items in grouped.items()}

    @property
    def kept_source_ids(self) -> tuple[str, ...]:
        """Elements the lowering had to keep as source (no drawing for their target)."""
        return tuple(placement.element_id for placement in self.placements if placement.kept_source)

    @property
    def low_legibility_fonts(self) -> tuple[tuple[str, float], ...]:
        """Drawn elements whose fragment had to drop below the readable floor.

        ``(element_id, size_pt)`` pairs for every placement that was drawn at
        less than ``_MIN_FONT_PT`` -- a translation delivered at a size no reader
        can comfortably read. Recorded rather than silent: the adapter turns each
        into a ``low_legibility_font`` defect flag on the block.
        """
        return tuple(
            (placement.element_id, placement.drawn_pt)
            for placement in self.placements
            if placement.drawn
            and placement.drawn_pt is not None
            and placement.drawn_pt < _MIN_FONT_PT
        )


__all__ = ["Composition", "Placement"]

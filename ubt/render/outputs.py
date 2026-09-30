"""Lowering a realized document to an artifact (ADR-0001 L6).

The lowering has exactly one capability today: place an element as its opaque
source slice. Every element therefore descends to that slice, every page is
carried over whole, and the artifact *is* the source -- the lattice floor made
concrete, and the strongest fidelity claim available.

A realization above the floor has no drawing yet, so it descends to the slice and
is **recorded** as descended (:class:`Placement`), never silently approximated:
keeping the source is the lattice's guaranteed lower bound, and the record is
what makes the descent explicit rather than a quiet loss. When a backend can
actually draw a reconstructed fragment, this is where a real per-element
composition grows; until then the placements are the seam it will fill.

An element with no attestation is not a realization at all but a coverage gap,
and is refused.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pikepdf

from ubt.adapters.pdf import pdf_struct
from ubt.model.ast import Document
from ubt.model.fidelity import Attestation, Fidelity


class LoweringUnsupported(Exception):
    """The document cannot be lowered: an element was never judged."""


@dataclass(frozen=True, slots=True)
class Placement:
    """One element's position in the artifact: what it was attested at, what was drawn."""

    element_id: str
    page: int
    fidelity: Fidelity  # what realize() attested
    placed_as: Fidelity  # what the lowering actually drew
    detail: str = ""

    @property
    def descended(self) -> bool:
        """True when the lowering kept the source because it cannot draw the realization."""
        return self.placed_as < self.fidelity


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
    def descended_ids(self) -> tuple[str, ...]:
        """Elements the lowering had to keep as source (no drawing for their rung)."""
        return tuple(placement.element_id for placement in self.placements if placement.descended)


def _place(element_id: str, page: int, attestation: Attestation) -> Placement:
    """The lowering's verdict for one element: the opaque slice is all it can draw."""
    placed_as = Fidelity.PRESERVED_OPAQUE
    detail = (
        f"{attestation.fidelity.name} has no lowering yet; source kept"
        if attestation.fidelity > placed_as
        else "opaque source slice"
    )
    return Placement(element_id, page, attestation.fidelity, placed_as, detail)


def compose(
    document: Document,
    attestations: Sequence[Attestation],
    source_pdf: str | Path,
    output_path: str | Path,
) -> Composition:
    """Lower a realized document to a PDF, recording how each element was placed.

    Every element must carry an attestation; a missing one is a coverage gap, not
    a realization, and raises. Every attested element is placed as its opaque
    source slice -- the only lowering wired -- and a realization above the floor
    is recorded as descended rather than approximated.
    """
    by_id = {attestation.element_id: attestation for attestation in attestations}
    unjudged = [element.id for element in document.elements if element.id not in by_id]
    if unjudged:
        raise LoweringUnsupported(
            f"{len(unjudged)} element(s) have no attestation: {', '.join(unjudged[:5])}"
        )
    placements = tuple(
        _place(element.id, element.span.page, by_id[element.id]) for element in document.elements
    )

    source = Path(source_pdf)
    output = Path(output_path)
    with pdf_struct.open_pdf(source) as src, pikepdf.new() as composed:
        for page in src.pages:
            composed.pages.append(page)
        output.parent.mkdir(parents=True, exist_ok=True)
        composed.save(str(output))
    return Composition(output_path=output, placements=placements)


__all__ = ["Composition", "LoweringUnsupported", "Placement", "compose"]

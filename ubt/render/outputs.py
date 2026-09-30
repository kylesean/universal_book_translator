"""Lowering a realized document to an artifact (ADR-0001 L6).

The floor is exact. When every element is ``PRESERVED_OPAQUE``, placing its
opaque slices reproduces the source page it came from, so the composed artifact
is the source pages carried over whole. That is the only lowering wired so far:
a realization above the floor has no lowering yet and is *refused*, not
approximated -- the same fail-closed rule :func:`ubt.pipeline.steps.realize`
applies one level up.

Composition goes through the pikepdf page-copy path the bilingual alternator
uses, so the output is a real PDF document -- the seam a mixed-fidelity lowering
extends -- rather than a raw file copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pikepdf

from ubt.adapters.pdf import pdf_struct
from ubt.model.ast import Document
from ubt.model.fidelity import Attestation, Fidelity


class LoweringUnsupported(Exception):
    """A realization (or a gap in coverage) has no artifact lowering yet."""


def compose(
    document: Document,
    attestations: Sequence[Attestation],
    source_pdf: str | Path,
    output_path: str | Path,
) -> Path:
    """Lower a realized document to a PDF, fail-closed on anything but the floor.

    Every element must carry an attestation and be ``PRESERVED_OPAQUE``: the
    opaque slices tile their source page, so copying the source pages is the
    exact lowering. Any other fidelity has no lowering yet and raises rather
    than approximating; a missing attestation means an element was never judged,
    which is a coverage gap, not a realization.
    """
    covered = {attestation.element_id for attestation in attestations}
    unjudged = [element.id for element in document.elements if element.id not in covered]
    if unjudged:
        raise LoweringUnsupported(
            f"{len(unjudged)} element(s) have no attestation: {', '.join(unjudged[:5])}"
        )
    above_floor = [
        attestation.element_id
        for attestation in attestations
        if attestation.fidelity is not Fidelity.PRESERVED_OPAQUE
    ]
    if above_floor:
        raise LoweringUnsupported(
            f"{len(above_floor)} element(s) are not PRESERVED_OPAQUE and have no lowering: "
            f"{', '.join(above_floor[:5])}"
        )

    source = Path(source_pdf)
    output = Path(output_path)
    with pdf_struct.open_pdf(source) as src, pikepdf.new() as composed:
        for page in src.pages:
            composed.pages.append(page)
        output.parent.mkdir(parents=True, exist_ok=True)
        composed.save(str(output))
    return output


__all__ = ["LoweringUnsupported", "compose"]

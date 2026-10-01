"""Delivered-artifact pixel witness (ADR-0001 Phase 1 acceptance).

The reconstruction witnesses (``formula_witness`` / ``table_witness``) run at
render time, on the markup they are about to place. This is their delivered-
artifact twin: after a document has been lowered to a PDF, does each element's
region in the artifact still carry the ink its source region carries?

It does not invent a second comparator. It crops the same region from the source
and the artifact (``visual_scalpel.crop_block_pil``) and judges the pair with the
one structural metric the witnesses already use
(``formula_witness.compare_structure``), so "an asset survived" means the same
thing at render time and at delivery time.

Page alignment is the precondition: the lowering must keep source geometry (the
overlay/compose lowering, and rigid renders) for page N of the artifact to
correspond to page N of the source. When it does not -- a reflowed or bilingual
artifact -- a crop cannot be located, so the witness returns ``UNVERIFIABLE``
rather than a false verdict, the same fail-open discipline as the render-time
witnesses.
"""

from __future__ import annotations

from pathlib import Path

from ubt.model.fidelity import Proof, ProofKind

#: Rasterization DPI for the crop comparison. Matches the scalpel default; high
#: enough to see component structure, cheap enough to run per element.
DEFAULT_PIXEL_DPI = 150


def witness_region(
    source_pdf: str | Path,
    artifact_pdf: str | Path,
    page: int,
    bbox: tuple[float, float, float, float],
    *,
    dpi: int = DEFAULT_PIXEL_DPI,
) -> Proof:
    """Compare one element's region in the source against the delivered artifact.

    Never raises: an unavailable crop or an unmeasurable region is
    ``UNVERIFIABLE`` (there is nothing to compare), while a measurable source
    whose artifact region lost its ink is ``FAILED``.
    """
    if page <= 0:
        return Proof.unknown(ProofKind.PIXEL, "no page for element")
    try:
        from ubt.adapters.pdf.formula_witness import compare_structure
        from ubt.adapters.pdf.visual_scalpel import crop_block_pil
    except Exception as exc:  # pragma: no cover - witness must never break a run
        return Proof.unknown(ProofKind.PIXEL, f"witness unavailable: {exc}")

    try:
        source_img = crop_block_pil(source_pdf, page, bbox, dpi=dpi, bleed_pt=0.0)
    except Exception as exc:
        return Proof.unknown(ProofKind.PIXEL, f"source crop unavailable: {exc}")
    try:
        artifact_img = crop_block_pil(artifact_pdf, page, bbox, dpi=dpi, bleed_pt=0.0)
    except Exception as exc:
        # The source region was measurable but the artifact region is not even
        # croppable: that is loss, not an unverifiable probe.
        return Proof.fail(ProofKind.PIXEL, f"artifact crop unavailable: {exc}")
    try:
        findings = compare_structure(artifact_img, source_img)
    finally:
        source_img.close()
        artifact_img.close()

    if findings == ["unwitnessable"]:
        return Proof.unknown(ProofKind.PIXEL, "source region has no measurable ink")
    if findings:
        return Proof.fail(ProofKind.PIXEL, "; ".join(findings), findings=tuple(findings))
    return Proof.ok(ProofKind.PIXEL, "artifact region matches source")


__all__ = ["DEFAULT_PIXEL_DPI", "witness_region"]

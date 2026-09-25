"""Rigid-render fidelity: prove non-text regions survive the render, unchanged.

Route ``rigid`` keeps the source page as its canvas and paints the translated
text back into the original bounding boxes, so everything that is *not* painted
prose — figures, vector art, equations, tables, page chrome — is supposed to be
byte-identical to the source. ``artifact_parity`` checks this only by proxy
(image counts, page geometry); it cannot see a strip that clipped vector ink or
an overlay that slipped under a cover. This module measures it directly:

render both pages with the *same* pdfium rasterizer at a high DPI, mask out the
translated-text rectangles, and compare what remains. A nonzero residual means
something outside the text boxes changed — the exact failure self-attestation
misses. A low masked-coverage ratio means most of the page was *not* painted
(such as low overlay-coverage observed on dense multi-column academic texts).

Findings are advisory (``severity="info"``): this is the ruler, not a gate — the
delivery is never blocked by it. Pure-Pillow (no numpy / no new dependency); the
raster step holds ``PDFIUM_LOCK`` and runs both pages through pdfium so renderer
differences cannot masquerade as fidelity loss.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from ubt.adapters.pdf.artifact_parity import ParityFinding
from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK
from ubt.adapters.pdf.visual_scalpel import _compute_crop_coords
from ubt.core.policy.layout_policy import PROSE_BLOCK_TYPES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

    from PIL import Image as PILImage

    from ubt.core.ir.models import IRBlock

logger = logging.getLogger(__name__)

# Publication-grade raster sampling. 150 DPI (the scalpel default) is too coarse
# to tell a clipped vector edge from antialiasing; 300 keeps the residual honest.
DEFAULT_FIDELITY_DPI: int = 300
# A kept pixel differing by more than this (0-255 luminance) counts as changed.
# Comfortably above pdfium antialiasing jitter at 300 DPI, below any real ink loss.
_RESIDUAL_PIXEL_THRESHOLD: int = 8
# Mask-rect bleed (pt): swallow the glyph antialiasing halo at box edges so it is
# treated as "text" and excluded, instead of inflating the non-text residual.
_MASK_BLEED_PT: float = 4.0
# Advisory thresholds (first calibration; revisit after ForMaT baseline).
_RESIDUAL_WARN = 0.005  # >0.5% of non-text pixels changed
_COVERAGE_WARN = 0.06  # <6% of the page area was painted text


def render_page_to_pil(
    doc: Any,
    page_index: int,
    dpi: int,
) -> PILImage.Image:
    """Rasterize one page (0-based) of an open pdfium doc to an RGB PIL image.

    Caller must already hold ``PDFIUM_LOCK``. Kept separate from the diff so the
    measurement core can be unit-tested with synthetic images.
    """
    page = doc[page_index]
    try:
        bitmap = page.render(scale=dpi / 72.0)
        return cast("PILImage.Image", bitmap.to_pil().convert("RGB"))
    finally:
        page.close()


def diff_outside_masks(
    source_img: PILImage.Image,
    artifact_img: PILImage.Image,
    mask_rects: Sequence[tuple[int, int, int, int]],
) -> tuple[float, float]:
    """Compare two same-size images ignoring the given rectangles.

    ``mask_rects`` are ``(x0, y0, x1, y1)`` pixel boxes covering the painted
    text; everything outside them should be identical between source and
    artifact. Returns ``(non_text_residual_ratio, masked_coverage_ratio)``:
    the fraction of *kept* (non-masked) pixels that changed, and the fraction
    of the page that was masked out as text. Size mismatch yields ``(1.0, 0.0)``
    (maximally suspect), which the caller reports rather than trusts.
    """
    from PIL import Image, ImageChops, ImageDraw  # noqa: PLC0415

    if source_img.size != artifact_img.size:
        return 1.0, 0.0
    width, height = source_img.size
    total_pixels = width * height
    if total_pixels == 0:
        return 0.0, 0.0

    src = source_img.convert("L")
    art = artifact_img.convert("L")

    # 255 = keep (non-text), 0 = mask out (painted text).
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)
    for box in mask_rects:
        draw.rectangle(list(box), fill=0)

    # Per-pixel luminance difference, then zero it inside the masked boxes so
    # only the non-text area contributes to the residual.
    diff = ImageChops.difference(src, art)
    kept_diff = ImageChops.multiply(diff, mask)

    keep_total = mask.histogram()[255]
    if keep_total == 0:
        residual = 0.0
    else:
        changed = sum(kept_diff.histogram()[_RESIDUAL_PIXEL_THRESHOLD + 1 :])
        residual = changed / keep_total

    masked_histogram_total = sum(mask.histogram())
    coverage = (
        (masked_histogram_total - keep_total) / total_pixels if masked_histogram_total else 0.0
    )
    return residual, coverage


def compute_render_fidelity(
    source_pdf: Path,
    artifact_pdf: Path,
    blocks: Sequence[IRBlock],
    *,
    dpi: int = DEFAULT_FIDELITY_DPI,
    max_pages: int = 8,
) -> dict[str, Any]:
    """Measure rigid-render fidelity over sampled pages.

    Returns a stats dict (empty-ish with ``pages_measured=0`` when the probe
    cannot run, so it never breaks delivery). Only prose blocks (the ones the
    engine paints) become mask rectangles; guarded FORMULA/IMAGE/TABLE blocks are
    deliberately left in the compared region because they must stay untouched.
    """
    stats: dict[str, Any] = {
        "non_text_diff_ratio": 0.0,
        "masked_coverage_ratio": 0.0,
        "pages_measured": 0,
        "skipped_reason": None,
    }
    try:
        import pypdfium2 as pdfium  # noqa: PLC0415
    except ImportError:  # pragma: no cover - pdfium is a hard PDF dep
        stats["skipped_reason"] = "pypdfium2_unavailable"
        return stats

    pages_by_no: dict[int, list[IRBlock]] = {}
    for block in blocks:
        bbox = getattr(block, "bbox", None)
        if bbox is None or block.block_type not in PROSE_BLOCK_TYPES:
            continue
        pages_by_no.setdefault(int(bbox.page), []).append(block)

    try:
        with PDFIUM_LOCK:
            src_doc = pdfium.PdfDocument(str(source_pdf))
            art_doc = pdfium.PdfDocument(str(artifact_pdf))
            try:
                common = min(len(src_doc), len(art_doc))
                candidates = sorted(
                    (pno for pno in pages_by_no if 1 <= pno <= common),
                )[: max(1, max_pages)]
                residuals: list[float] = []
                coverages: list[float] = []
                for page_no in candidates:
                    idx = page_no - 1
                    src_page = src_doc[idx]
                    try:
                        page_w_pt, page_h_pt = src_page.get_size()
                    finally:
                        src_page.close()
                    src_img = render_page_to_pil(src_doc, idx, dpi)
                    art_img = render_page_to_pil(art_doc, idx, dpi)
                    scale = dpi / 72.0
                    rects: list[tuple[int, int, int, int]] = []
                    for block in pages_by_no[page_no]:
                        bbox = block.bbox
                        if bbox is None:
                            continue
                        try:
                            rects.append(
                                _compute_crop_coords(
                                    bbox,
                                    page_w_pt,
                                    page_h_pt,
                                    scale,
                                    bleed_pt=_MASK_BLEED_PT,
                                )
                            )
                        except ValueError:
                            continue  # out-of-page phantom block: not evidence
                    residual, coverage = diff_outside_masks(src_img, art_img, rects)
                    residuals.append(residual)
                    coverages.append(coverage)
                    src_img.close()
                    art_img.close()
            finally:
                src_doc.close()
                art_doc.close()
    except Exception as exc:  # pragma: no cover - probe must never break delivery
        logger.debug("render fidelity skipped (%s)", exc)
        stats["skipped_reason"] = "probe_error"
        return stats

    if not residuals:
        stats["skipped_reason"] = "no_measurable_pages"
        return stats

    stats["non_text_diff_ratio"] = round(sum(residuals) / len(residuals), 6)
    stats["masked_coverage_ratio"] = round(sum(coverages) / len(coverages), 6)
    stats["pages_measured"] = len(residuals)
    return stats


def fidelity_findings(stats: dict[str, Any]) -> list[ParityFinding]:
    """Advisory ``info`` findings from a fidelity stats dict (never blocking)."""
    findings: list[ParityFinding] = []
    if stats.get("pages_measured", 0) <= 0:
        return findings
    residual = float(stats.get("non_text_diff_ratio", 0.0))
    coverage = float(stats.get("masked_coverage_ratio", 0.0))
    if residual > _RESIDUAL_WARN:
        findings.append(
            ParityFinding(
                "info",
                "fidelity_non_text_residual",
                f"{residual:.4%} of non-text pixels differ from the source (rigid "
                "render should leave them pixel-intact).",
            )
        )
    if coverage < _COVERAGE_WARN:
        findings.append(
            ParityFinding(
                "info",
                "fidelity_low_coverage",
                f"only {coverage:.2%} of the page area was painted prose; most "
                "content may still be showing source text.",
            )
        )
    return findings

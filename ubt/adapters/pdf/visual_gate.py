"""Post-render visual gate: deterministic checks + structural checks + sampled VLM review.

Design (warn-only, ROI-first):
- T0 deterministic, zero model tokens, always safe to run: banned unicode scan
  of the Typst source (U+2011), PDF page count, text-level blank-page
  candidates via pdf_oxide in-process extraction.
- T1 structural, zero model tokens: block bbox overlap / out-of-bounds on the
  same page (pure Python over IR blocks + pikepdf mediabox dimensions).
- Pixel confirmation + T2 VLM are *optional*: page PNGs are rendered
  in-process by ``pdf_oxide`` (base dependency; no poppler ``pdftoppm``
  subprocess) and inspected with Pillow. PyMuPDF is deliberately NOT used
  (AGPL licensing).
  Missing tools degrade gracefully to ``skipped_reason`` instead of failing.
- T0.5 artifact parity is not here: it compares the delivered PDF against
  the *source file* as physical evidence (target-language coverage, geometry
  and asset-count preservation) and lives in ``artifact_parity.py``; the reflow
  loop merges its findings into this gate's result. ``target_language_absent``
  is fail-closed in ``blocking_gate_tripped`` alongside the unreadable-PDF
  case.

The gate never blocks export: findings are warnings written to
``{stem}_visual_report.json`` next to the quality report. Two exceptions:
the opt-in blocking gate (``blocking_gate_tripped``) — short docs
with CRITICAL findings refuse export when explicitly enabled, off by default,
see ``UBTConfig.visual_blocking_gate_enabled`` — and an unreadable rendered
PDF, which fails closed even when the blocking gate is off because the
exported artifact itself cannot be read.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ubt.adapters.pdf import oxide_render, pdf_struct
from ubt.core.policy.layout_policy import (
    QA_FULL_GATE_MAX_PAGES,
    QA_LONG_BOOK_SAMPLE,
    QA_TIERED_BAND_PAGES,
    QA_TIERED_BAND_SAMPLE,
)

logger = logging.getLogger(__name__)

Severity = Literal["info", "major", "critical"]

BANNED_NON_BREAKING_HYPHEN = "\u2011"

# Artifact-parity codes (injected by the reflow loop from artifact_parity.py)
# that refuse export on the same reasoning as the unreadable-PDF case, even
# when the blocking gate is off:
# an artifact carrying none of the commissioned target language is unshippable.
_ALWAYS_FAIL_CLOSED_CODES = frozenset({"target_language_absent"})
# Delivery-breaking parity MAJORS. The geometry ones only fire on
# geometry-preserving renders (artifact_parity gates them on
# keeps_source_geometry), so a change here is not "different layout by
# design" but a broken artifact: rigid must keep the page count, and losing
# or duplicating image XObjects under the overlay means assets were eaten.
# ``target_language_sparse`` (>=92% of the artifact is not the commissioned
# language) is unshippable on the same reasoning. A gate reporting
# passed=False on exactly these codes must not ship the artifact.
_ALWAYS_FAIL_CLOSED_MAJORS = frozenset(
    {
        "page_count_changed",
        "image_count_changed",
        "target_language_sparse",
        # A content figure/table the reflow render could not stage is gone from
        # the delivered artifact while the render still reports full coverage.
        "content_asset_missing",
    }
)
BLANK_TEXT_THRESHOLD_CHARS = 20
OVERLAP_IOU_THRESHOLD = 0.6
MIN_OVERLAP_AREA_PT2 = 500.0
OUT_OF_BOUNDS_TOLERANCE_PT = 5.0
DEFAULT_DPI = 110
# Pillow pixel heuristics (white-paper academic PDFs only).
BLANK_MEAN_THRESHOLD = 250.0
BLANK_STD_THRESHOLD = 6.0
BLACK_PIXEL_VALUE = 12
BLACK_FRACTION_THRESHOLD = 0.05
# Adaptive zero-token sampling ceiling. VLM spend is capped separately
# by max_vlm_pages (default 3) and never scales with book size.
ADAPTIVE_SAMPLE_CAP = 12

VLM_PROMPT = """You are a print QA inspector. Look at this rendered PDF page image.
Reply in exactly two lines:
verdict: PASS or FAIL
issues: <short comma-separated list, or 'none'>
FAIL only for: clipped/truncated text, overlapping text blocks, solid black \
squares replacing glyphs, broken tables, unreadable text, or a fully blank page \
where content is expected. Ignore language and translation quality."""

# Type of the injected vision judge: (images_b64_png, prompt) -> raw text verdict.
VlmJudgeFn = Callable[[list[str], str], Awaitable[str]]


@dataclass(frozen=True)
class VisualFinding:
    """One visual QA observation (warning-grade, never blocks export)."""

    severity: Severity
    code: str
    message: str
    page: int | None = None


@dataclass(frozen=True)
class VisualGateResult:
    """Outcome of one visual-gate run over a rendered PDF."""

    passed: bool
    findings: tuple[VisualFinding, ...] = ()
    sampled_pages: tuple[int, ...] = ()
    vlm_pages: tuple[int, ...] = ()
    skipped_reason: str | None = None
    stats: dict[str, str | int | float] = field(default_factory=dict)


def scan_typ_source(typ_text: str) -> list[VisualFinding]:
    """T0: scan Typst source for banned unicode dashes (skill rule)."""
    findings: list[VisualFinding] = []
    banned = typ_text.count(BANNED_NON_BREAKING_HYPHEN)
    if banned:
        findings.append(
            VisualFinding(
                severity="major",
                code="banned_unicode_dash",
                message=(
                    f"Typst source contains {banned} U+2011 non-breaking "
                    "hyphen(s); they render as missing glyphs"
                ),
            )
        )
    return findings


def pdf_page_count(pdf_path: Path) -> int:
    """Return PDF page count via pikepdf, or -1 when unreadable."""
    try:
        return pdf_struct.page_count(pdf_path)
    except Exception as exc:
        logger.debug("Visual gate: cannot read %s: %s", pdf_path, exc)
        return -1


def page_dimensions(pdf_path: Path) -> dict[int, tuple[float, float]]:
    """Return 1-indexed {page_no: (width_pt, height_pt)} via pikepdf."""
    try:
        return pdf_struct.page_sizes(pdf_path)
    except Exception as exc:
        logger.debug("Visual gate: cannot read dimensions of %s: %s", pdf_path, exc)
        return {}


def blank_page_candidates(pdf_path: Path, min_chars: int = BLANK_TEXT_THRESHOLD_CHARS) -> list[int]:
    """T0: pages whose extracted text is nearly empty (needs pixel confirm)."""
    texts = oxide_render.extract_page_texts(pdf_path)
    if not texts:
        logger.debug("Visual gate: text scan failed for %s", pdf_path)
        return []
    return [idx for idx, text in enumerate(texts, start=1) if len(text.strip()) < min_chars]


def _box_area(x0: float, y0: float, x1: float, y1: float) -> float:
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = _box_area(ix0, iy0, ix1, iy1)
    if inter <= 0:
        return 0.0
    union = _box_area(*a) + _box_area(*b) - inter
    return inter / union if union > 0 else 0.0


def block_overlap_findings(blocks: Sequence[object]) -> list[VisualFinding]:
    """T1: flag heavily overlapping text blocks on the same page.

    Accepts IRBlock-like objects (duck-typed ``bbox``) to avoid importing IR
    models here; anything without a usable bbox is skipped.
    """
    findings: list[VisualFinding] = []
    by_page: dict[int, list[tuple[str, tuple[float, float, float, float]]]] = {}
    for block in blocks:
        bbox = getattr(block, "bbox", None)
        if bbox is None:
            continue
        try:
            page = int(bbox.page)
            rect = (float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1))
        except (AttributeError, TypeError, ValueError):
            continue
        if _box_area(*rect) < MIN_OVERLAP_AREA_PT2:
            continue
        block_id = str(getattr(block, "id", "?"))
        by_page.setdefault(page, []).append((block_id, rect))
    for page, items in by_page.items():
        capped = items[:200]  # bound O(n^2) on pathological pages
        for i in range(len(capped)):
            for j in range(i + 1, len(capped)):
                if _box_iou(capped[i][1], capped[j][1]) >= OVERLAP_IOU_THRESHOLD:
                    findings.append(
                        VisualFinding(
                            severity="major",
                            code="block_overlap",
                            message=(
                                f"Blocks {capped[i][0]} and {capped[j][0]} "
                                f"overlap heavily (IoU>=0.6)"
                            ),
                            page=page,
                        )
                    )
    return findings


def blocks_out_of_bounds_findings(
    blocks: Sequence[object], dims: dict[int, tuple[float, float]]
) -> list[VisualFinding]:
    """T1: flag blocks placed outside the page mediabox (clipped text)."""
    findings: list[VisualFinding] = []
    for block in blocks:
        bbox = getattr(block, "bbox", None)
        if bbox is None:
            continue
        try:
            page = int(bbox.page)
            x0, y0, x1, y1 = (
                float(bbox.x0),
                float(bbox.y0),
                float(bbox.x1),
                float(bbox.y1),
            )
        except (AttributeError, TypeError, ValueError):
            continue
        size = dims.get(page)
        if size is None:
            continue
        width, height = size
        tol = OUT_OF_BOUNDS_TOLERANCE_PT
        if x0 < -tol or y0 < -tol or x1 > width + tol or y1 > height + tol:
            findings.append(
                VisualFinding(
                    severity="major",
                    code="block_out_of_bounds",
                    message=(
                        f"Block {getattr(block, 'id', '?')} exceeds page "
                        f"{page} mediabox {width:.0f}x{height:.0f}pt"
                    ),
                    page=page,
                )
            )
    return findings


def adaptive_sample_budget(total_pages: int, configured: int) -> int:
    """Size-aware zero-token sampling budget.

    ``configured`` (from ``UBT_VISUAL_SAMPLE_PAGES``) is a floor, never a
    ceiling: short books get full inspection, medium books ~20%, long books
    a fixed anchor set. ``configured <= 0`` still disables rendering.
    VLM spend is unaffected (capped by ``max_vlm_pages`` downstream).
    """
    if total_pages <= 0 or configured <= 0:
        return 0
    if total_pages <= QA_FULL_GATE_MAX_PAGES:
        return total_pages  # full inspection, uncapped (<=20 local renders)
    target = (
        min(QA_TIERED_BAND_SAMPLE, max(6, -(-total_pages // 5)))
        if total_pages <= QA_TIERED_BAND_PAGES
        else QA_LONG_BOOK_SAMPLE
    )
    return min(ADAPTIVE_SAMPLE_CAP, max(configured, target))


def select_sample_pages(
    total_pages: int, flagged_pages: Sequence[int], max_sample: int
) -> list[int]:
    """Stratified sample: flagged pages first, then even spread over the book."""
    if total_pages <= 0 or max_sample <= 0:
        return []
    flagged = sorted({p for p in flagged_pages if 1 <= p <= total_pages})
    chosen: list[int] = list(flagged[:max_sample])
    if len(chosen) >= max_sample:
        return chosen
    if total_pages > 100:
        # Long books: anchor cover / TOC / tail before the even spread so a
        # fixed budget always covers the highest-signal pages.
        for anchor in (1, 2, total_pages):
            if len(chosen) >= max_sample:
                return sorted(chosen)
            if anchor not in chosen:
                chosen.append(anchor)
    remaining = max_sample - len(chosen)
    # Even spread across the book, skipping already chosen pages.
    step = max(1, total_pages // max(remaining + 1, 1))
    page = 1
    while len(chosen) < max_sample and page <= total_pages:
        if page not in chosen:
            chosen.append(page)
        page += step
    # Fill any gap from the tail (short books).
    page = total_pages
    while len(chosen) < max_sample and page >= 1:
        if page not in chosen:
            chosen.append(page)
        page -= 1
    return sorted(chosen)


def render_pages_to_png(
    pdf_path: Path, pages: Sequence[int], dpi: int = DEFAULT_DPI, work_dir: Path | None = None
) -> dict[int, Path]:
    """Render selected pages in-process via ``pdf_oxide`` (base dependency;
    there is no poppler ``pdftoppm`` fallback branch).

    Returns {page_no: png_path}; empty dict when every render fails.
    Never raises.
    """
    if not pages:
        return {}
    out: dict[int, Path] = {}
    owns_tmp = work_dir is None
    tmp = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="ubt_visual_"))
    try:
        tmp.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.debug("Visual gate: cannot create tmpdir: %s", exc)
        if owns_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        return {}
    for page in pages:
        png = oxide_render.write_page_png(pdf_path, page, dpi, tmp)
        if png is not None:
            out[page] = png
    # When the caller supplies work_dir it owns cleanup; clean up
    # self-created temporary directory if nothing was produced.
    if owns_tmp and not out:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def pixel_findings(png_path: Path, page: int) -> list[VisualFinding]:
    """Pixel heuristics via Pillow (optional dep): blank + black-square check."""
    try:
        from PIL import Image, ImageStat
    except ImportError:
        return []
    findings: list[VisualFinding] = []
    try:
        with Image.open(png_path) as img:
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            mean = float(stat.mean[0]) if stat.mean else 255.0
            stddev = float(stat.stddev[0]) if stat.stddev else 0.0
            if mean >= BLANK_MEAN_THRESHOLD and stddev <= BLANK_STD_THRESHOLD:
                findings.append(
                    VisualFinding(
                        severity="critical",
                        code="blank_page",
                        message=f"Page {page} renders blank (mean={mean:.1f})",
                        page=page,
                    )
                )
                return findings
            small = gray.copy()
            small.thumbnail((400, 400))
            hist = small.histogram()
            dark = sum(hist[: BLACK_PIXEL_VALUE + 1])
            total = max(1, sum(hist))
            if dark / total >= BLACK_FRACTION_THRESHOLD and mean < 200.0:
                findings.append(
                    VisualFinding(
                        severity="major",
                        code="black_block_candidate",
                        message=(
                            f"Page {page} has {dark / total:.1%} near-black pixels "
                            "(possible missing-glyph squares; verify visually)"
                        ),
                        page=page,
                    )
                )
    except OSError as exc:
        logger.debug("Visual gate: cannot inspect %s: %s", png_path, exc)
    return findings


def parse_vlm_verdict(raw_text: str) -> tuple[bool, str]:
    """Parse the documented ``verdict: PASS|FAIL`` line; unparseable counts as pass+note.

    Only the documented prefix commits a verdict: free-form prose mentioning
    "pass" must not be silently read as an explicit PASS (it is reported as an
    unparseable note so the operator can see the judge misbehaved).
    """
    lowered = (raw_text or "").lower()
    if "verdict: fail" in lowered:
        return False, raw_text.strip()[:300]
    if "verdict: pass" in lowered:
        return True, ""
    return True, f"unparseable verdict: {raw_text.strip()[:200]}"


async def judge_pages_with_vlm(
    images_b64: dict[int, str], judge_fn: VlmJudgeFn
) -> list[VisualFinding]:
    """T2: one bounded VLM call per sampled page (caller caps the count)."""
    findings: list[VisualFinding] = []
    for page in sorted(images_b64):
        try:
            raw = await judge_fn([images_b64[page]], VLM_PROMPT)
        except Exception as exc:
            findings.append(
                VisualFinding(
                    severity="info",
                    code="vlm_judge_error",
                    message=f"Page {page} VLM judge failed (non-fatal): {exc}",
                    page=page,
                )
            )
            continue
        passed, detail = parse_vlm_verdict(raw)
        if not passed:
            findings.append(
                VisualFinding(
                    severity="major",
                    code="vlm_visual_fail",
                    message=f"Page {page} VLM visual FAIL: {detail}",
                    page=page,
                )
            )
        elif detail:
            findings.append(
                VisualFinding(severity="info", code="vlm_note", message=detail, page=page)
            )
    return findings


def _encode_png_b64(png_path: Path) -> str | None:
    try:
        return base64.b64encode(png_path.read_bytes()).decode("ascii")
    except OSError as exc:
        logger.debug("Visual gate: cannot read %s: %s", png_path, exc)
        return None


# Page kinds whose VLM slots outrank plain-text pages: a
# chart/diagram page hides defects (clipped labels, broken tables) that the
# zero-token gates cannot see; a flagged text page still outranks an
# unflagged chart (known suspect beats suspected value).
CHART_PAGE_KINDS = frozenset({"vector_heavy", "mixed_complex"})


def chart_pages_from_blocks(blocks: Sequence[object]) -> set[int]:
    """1-based pages carrying chart/diagram画像 (via block provenance)."""
    pages: set[int] = set()
    for block in blocks:
        provenance = getattr(block, "provenance", None)
        if not isinstance(provenance, dict):
            continue
        if str(provenance.get("page_kind", "")) in CHART_PAGE_KINDS:
            bbox = getattr(block, "bbox", None)
            page = getattr(bbox, "page", None)
            try:
                if page is not None and int(page) >= 1:
                    pages.add(int(page))
            except (TypeError, ValueError):
                continue
    return pages


def blocking_gate_tripped(
    findings: Sequence[VisualFinding],
    total_pages: int,
    enabled: bool,
) -> list[VisualFinding]:
    """Blocking gate (opt-in, short docs only).

    Returns the CRITICAL findings that refuse export when ``enabled`` and
    the document is within ``QA_FULL_GATE_MAX_PAGES``; otherwise []. MAJOR
    findings never block (warn-only). Pure function, unit-tested.

    An unreadable rendered PDF (``total_pages < 0``, the
    ``pdf_page_count`` sentinel) fails closed regardless of ``enabled`` and
    of the page window — a gate that reports ``passed=False`` must not be
    silently ignored, and an artifact nobody can read must not ship.
    ``total_pages == 0`` still means "no document measured" and stays
    non-blocking.
    """
    if total_pages < 0:
        return [f for f in findings if getattr(f, "severity", "") == "critical"]
    # An artifact carrying none of the commissioned target language is
    # unshippable on the same reasoning as the unreadable-PDF case,
    # regardless of ``enabled`` or the
    # page window — there is nothing for a reader to consume. The parity
    # majors in _ALWAYS_FAIL_CLOSED_MAJORS join that reasoning: they only
    # fire when the render promises preserved geometry/assets and the
    # measurement says otherwise, which is a broken artifact, not a style
    # preference.
    always_fail = [
        f
        for f in findings
        if (
            getattr(f, "code", "") in _ALWAYS_FAIL_CLOSED_CODES
            and getattr(f, "severity", "") == "critical"
        )
        or (
            getattr(f, "code", "") in _ALWAYS_FAIL_CLOSED_MAJORS
            and getattr(f, "severity", "") == "major"
        )
    ]
    if always_fail:
        return always_fail
    if not enabled or total_pages == 0 or total_pages > QA_FULL_GATE_MAX_PAGES:
        return []
    return [f for f in findings if getattr(f, "severity", "") == "critical"]


async def run_visual_gate(
    pdf_path: Path,
    blocks: Sequence[object] = (),
    typ_text: str | None = None,
    sample_pages: int = 6,
    max_vlm_pages: int = 3,
    vlm_judge: VlmJudgeFn | None = None,
    dpi: int = DEFAULT_DPI,
    facing_spread: bool = False,
    padding_pages: Sequence[int] = (),
) -> VisualGateResult:
    """Run T0+T1 always, pixel+T2 only when tools/judge are available.

    ``padding_pages`` names the 1-based pages the bilingual alternator filled
    with intentional blanks (facing flyleaf plus page-count padding). They are
    exempt from the blank-page checks — the gate cannot tell a deliberate blank
    from a failed render, so the renderer has to declare them.
    """
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return VisualGateResult(passed=True, skipped_reason="not a rendered pdf")
    total = pdf_page_count(pdf_path)
    if total <= 0:
        return VisualGateResult(
            passed=False,
            findings=(
                VisualFinding(
                    severity="critical",
                    code="unreadable_pdf",
                    message=f"Cannot read rendered PDF: {pdf_path.name}",
                ),
            ),
            # Report the unreadable sentinel instead of 0: export
            # refuses on total_pages < 0 via blocking_gate_tripped, while 0
            # means "not measured" there and would let a failed gate slip by.
            stats={"total_pages": -1},
        )
    findings: list[VisualFinding] = []
    if typ_text is not None:
        findings.extend(scan_typ_source(typ_text))
    # Intentional blanks: the facing flyleaf plus every page the alternator
    # padded to square up unequal page counts. The gate cannot tell a
    # deliberate blank from a failed render, so the renderer declares them and
    # the blank-page checks skip them.
    exempt_blank_pages = set(padding_pages)
    if facing_spread:
        exempt_blank_pages.add(1)
    # blank_page_candidates extracts every page's text in-process — CPU
    # bound; keep it off the loop so concurrent SSE streams and lease
    # renewals are never frozen.
    for page in await asyncio.to_thread(blank_page_candidates, pdf_path):
        if page in exempt_blank_pages:
            continue
        findings.append(
            VisualFinding(
                severity="info",
                code="blank_candidate",
                message=f"Page {page} has almost no extractable text (confirm visually)",
                page=page,
            )
        )
    if blocks:
        findings.extend(block_overlap_findings(blocks))
        dims = await asyncio.to_thread(page_dimensions, pdf_path)
        findings.extend(blocks_out_of_bounds_findings(blocks, dims))
    flagged = [f.page for f in findings if f.page is not None]
    sampled = select_sample_pages(total, flagged, adaptive_sample_budget(total, sample_pages))
    vlm_pages: list[int] = []
    skip_notes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ubt_visual_") as tmp_dir:
        # render_pages_to_png rasterizes each sampled page in-process via
        # pdf_oxide — this is the hot path that can freeze the loop for
        # minutes; offload it.
        pngs = await asyncio.to_thread(
            render_pages_to_png, pdf_path, sampled, dpi=dpi, work_dir=Path(tmp_dir)
        )
        if sampled and not pngs:
            skip_notes.append("page render failed (pdf_oxide); pixel checks skipped")
        else:
            for page, png in sorted(pngs.items()):
                # pixel_findings decodes the PNG with PIL (CPU-bound) — keep it off the loop too.
                p_findings = await asyncio.to_thread(pixel_findings, png, page)
                if page in exempt_blank_pages:
                    p_findings = [f for f in p_findings if f.code != "blank_page"]
                findings.extend(p_findings)
        if vlm_judge is not None and pngs and max_vlm_pages > 0:
            flagged_set = set(flagged)
            chart_set = chart_pages_from_blocks(blocks)

            def _vlm_rank(page: int) -> tuple[int, int]:
                if page in flagged_set and page in chart_set:
                    return (0, page)
                if page in flagged_set:
                    return (1, page)
                if page in chart_set:
                    return (2, page)
                return (3, page)

            ordered = sorted(pngs, key=_vlm_rank)
            vlm_pages = ordered[:max_vlm_pages]
            images_b64: dict[int, str] = {}
            for page in vlm_pages:
                encoded = _encode_png_b64(pngs[page])
                if encoded is not None:
                    images_b64[page] = encoded
            if images_b64:
                findings.extend(await judge_pages_with_vlm(images_b64, vlm_judge))
        elif vlm_judge is None:
            skip_notes.append("vlm judge disabled")
    passed = not any(f.severity in ("major", "critical") for f in findings)
    stats: dict[str, str | int | float] = {
        "total_pages": total,
        "sampled": len(sampled),
        "vlm_checked": len(vlm_pages),
        "findings": len(findings),
        # Keep structural loss visible even when several pairs share a page;
        # ``block_overlap_findings`` reports every pair rather than stopping at
        # the first one. Out-of-bounds blocks are the deterministic truncation
        # proxy available before raster/VLM inspection.
        "overlap_pairs": sum(f.code == "block_overlap" for f in findings),
        "truncation_count": sum(f.code == "block_out_of_bounds" for f in findings),
    }
    return VisualGateResult(
        passed=passed,
        findings=tuple(findings),
        sampled_pages=tuple(sampled),
        vlm_pages=tuple(vlm_pages),
        skipped_reason="; ".join(skip_notes) if skip_notes else None,
        stats=stats,
    )

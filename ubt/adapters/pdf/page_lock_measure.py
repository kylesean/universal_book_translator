"""Plan-time measurement of a PDF's page-lock feasibility (the light G2 probe).

The delivery router promises a 1:1 page lock when ``page_lock_feasible`` is
true. That signal used to be a constant tied to font damage — a clean native
PDF was "feasible" without anything ever being measured — while the router's
docstring called it the measured veto. This module supplies the measurement:
it samples pages, builds the same paragraph records the facsimile renderer
builds, and runs the renderer's own solver (:func:`solve_page_2d`) at the body
floor.

Two of the three resulting signals are translation-independent, so they are
real measurements rather than estimates:

* **source overlap** — source boxes that collide cannot be reproduced 1:1 at
  any size, whatever the translation says;
* **geometric headroom** — the largest uniform size at which the page's *own*
  text still fits. That bounds how much expansion a translation can absorb
  before the page must spill, which is what ``expansion_risk`` reports.

The remaining question — does the *translated* text fit — cannot be answered
before a translation exists. It stays provisional here and is answered where
it genuinely can be: the renderer's per-page solve at export, plus the visual
gate. A page this probe cannot examine (unreadable, no text layer, solver
error) is reported as unmeasured, never as feasible.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ubt.render.layout_solver import FLOOR_PT, check_source_overlap, fits_all, solve_page_2d

logger = logging.getLogger(__name__)

#: Pages examined per document. Even sampling keeps the cost flat for a long
#: book while still touching every part of it (front matter, body, appendices).
DEFAULT_SAMPLE_PAGES = 8

#: Area headroom at which expansion risk becomes MEDIUM / HIGH. ``capacity`` is
#: the squared ratio of the largest fitting size to the floor: 1.44 means the
#: page holds 44 % more text area at the floor than it currently needs. Below
#: 1.35 a 20-40 % CJK/Latin area swing is plausible; below 1.10 the page is
#: already nearly full at the floor.
_MEDIUM_CAPACITY = 1.35
_HIGH_CAPACITY = 1.10

#: A page whose text covers less than this share of the page is front matter,
#: a plate, or a mostly-empty page; its headroom says nothing about body flow.
_MIN_TEXT_AREA_SHARE = 0.005


@dataclass(frozen=True)
class PageLockMeasurement:
    """What the probe actually measured, with the evidence that supports it."""

    measured: bool
    sampled_pages: tuple[int, ...] = ()
    source_overlap: bool = False
    feasible: bool = True
    #: Smallest page's area headroom across the sample (1.0 = exactly fits at
    #: the floor, 1.44 = 44 % spare area).
    expansion_capacity: float = 0.0
    expansion_risk: str = "low"
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _line_records(
    lines: list[Any],
) -> tuple[list[dict[str, Any]], float, float]:
    """Build solver records from extracted lines, plus the source size/pitch.

    Records mirror the ones the facsimile renderer feeds the solver: ``top`` /
    ``bot`` from the box, ``w`` from the box width, ``glyphs`` from the text
    length, and ``glyph_aspect`` measured from the line's own advance per
    glyph rather than assumed. ``x0``/``x1`` ride along so the overlap check
    can tell side-by-side boxes from a genuine collision.
    """
    records: list[dict[str, Any]] = []
    sizes: list[float] = []
    tops: list[float] = []
    for line in lines:
        text = (line.text or "").strip()
        if not text:
            continue
        x0, y0, x1, y1 = (float(v) for v in line.rect)
        glyphs = len(text)
        size = float(getattr(line, "font_size", 0.0) or 0.0)
        if size <= 0.0 and y1 > y0:
            # pdfium reports a line box, not a font size; the box height is the
            # best available proxy and is what the renderer's type audit uses.
            size = (y1 - y0) * 0.78
        advance = (x1 - x0) / glyphs if glyphs else 0.0
        aspect = advance / size if size > 0.0 else 0.5
        records.append(
            {
                "id": f"l{len(records)}",
                "top": y1,
                "bot": y0,
                "w": x1 - x0,
                "x0": x0,
                "x1": x1,
                "glyphs": glyphs,
                "glyph_aspect": min(1.6, max(0.2, aspect)),
            }
        )
        sizes.append(size)
        tops.append(y1)
    source_size = statistics.median(sizes) if sizes else 0.0
    gaps = [
        abs(tops[i] - tops[i + 1])
        for i in range(len(tops) - 1)
        if 0.0 < abs(tops[i] - tops[i + 1]) < source_size * 2.5
    ]
    leading = statistics.median(gaps) if gaps else source_size * 1.33
    return records, source_size, leading


def _capacity_at_floor(records: list[dict[str, Any]], em_ratio: float, floor_pt: float) -> float:
    """Largest uniform size that still fits, as area headroom over the floor."""
    if not records:
        return 0.0
    upper = floor_pt * 4.0
    if fits_all([records], upper, em_ratio):
        return 4.0
    if not fits_all([records], floor_pt, em_ratio):
        return 0.0
    lo, hi = floor_pt, upper
    for _ in range(18):
        mid = (lo + hi) / 2.0
        if fits_all([records], mid, em_ratio):
            lo = mid
        else:
            hi = mid
    return (lo / floor_pt) ** 2


def _sample_page_numbers(page_count: int, limit: int) -> list[int]:
    """Evenly spaced 1-based page numbers, always including the first and last."""
    if page_count <= 0:
        return []
    if page_count <= limit:
        return list(range(1, page_count + 1))
    step = (page_count - 1) / (limit - 1)
    return sorted({1 + round(i * step) for i in range(limit)})


def measure_page_lock(
    pdf_path: Path,
    *,
    max_pages: int = DEFAULT_SAMPLE_PAGES,
    floor_pt: float = FLOOR_PT,
) -> PageLockMeasurement:
    """Measure whether the sampled pages admit a floor-size page lock.

    Returns ``measured=False`` when nothing could be examined — a non-PDF, an
    unreadable file, an encrypted document, a scan with no text layer. The
    caller must treat that as "provisional", not as a pass: an unmeasurable
    document is exactly the one the veto exists for.
    """
    try:
        import pypdfium2 as pdfium

        from ubt.adapters.pdf.textgeom import extract_lines
    except Exception as exc:  # noqa: BLE001 - optional dependency, degrade honestly
        logger.debug("Page-lock probe unavailable: %s", exc)
        return PageLockMeasurement(measured=False, reasons=("pdfium unavailable",))

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Page-lock probe could not open %s: %s", pdf_path, exc)
        return PageLockMeasurement(measured=False, reasons=("document unreadable",))

    try:
        page_numbers = _sample_page_numbers(len(pdf), max_pages)
        if not page_numbers:
            return PageLockMeasurement(measured=False, reasons=("no pages",))

        sampled: list[int] = []
        overlapping: list[int] = []
        capacities: list[float] = []
        failing: list[int] = []
        examined = 0

        for page_no in page_numbers:
            try:
                lines, page_size = extract_lines(pdf_path, page_no)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Page-lock probe skipped page %d: %s", page_no, exc)
                continue
            records, source_size, leading = _line_records(lines)
            if not records or source_size <= 0.0:
                continue
            page_area = float(page_size[0]) * float(page_size[1])
            text_area = sum(float(r["w"]) * (float(r["top"]) - float(r["bot"])) for r in records)
            if page_area <= 0.0 or text_area / page_area < _MIN_TEXT_AREA_SHARE:
                continue
            examined += 1
            sampled.append(page_no)

            if check_source_overlap([records]):
                overlapping.append(page_no)
                continue

            em_ratio = max(1.15, min(1.65, leading / source_size if source_size else 1.33))
            solution = solve_page_2d(
                [records],
                source_size_pt=source_size,
                source_leading_pt=leading,
                floor_pt=floor_pt,
                resolve_overlaps=False,
            )
            if not solution.fits:
                failing.append(page_no)
                continue
            capacities.append(_capacity_at_floor(records, em_ratio, floor_pt))

        if examined == 0:
            return PageLockMeasurement(
                measured=False, reasons=("no page carried a measurable text layer",)
            )

        reasons: list[str] = []
        feasible = True
        if overlapping:
            feasible = False
            reasons.append(
                f"source boxes overlap on page(s) {', '.join(map(str, overlapping[:5]))}: "
                "a 1:1 page lock cannot reproduce them at any size"
            )
        if failing:
            feasible = False
            reasons.append(
                f"page(s) {', '.join(map(str, failing[:5]))} do not fit their own text at "
                f"the {floor_pt:g}pt body floor"
            )

        capacity = min(capacities) if capacities else 0.0
        if not feasible or capacity < _HIGH_CAPACITY:
            risk = "high"
        elif capacity < _MEDIUM_CAPACITY:
            risk = "medium"
        else:
            risk = "low"
        reasons.append(f"sampled {len(sampled)} page(s); area headroom at floor {capacity:.2f}x")

        return PageLockMeasurement(
            measured=True,
            sampled_pages=tuple(sampled),
            source_overlap=bool(overlapping),
            feasible=feasible,
            expansion_capacity=round(capacity, 4),
            expansion_risk=risk,
            reasons=tuple(reasons),
        )
    finally:
        with_close = getattr(pdf, "close", None)
        if callable(with_close):
            with_close()

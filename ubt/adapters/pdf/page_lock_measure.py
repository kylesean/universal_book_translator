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

* **source overlap** — body-text boxes that collide cannot be reproduced 1:1,
  whatever the translation says. The check runs on body lines only, and per
  column: pdfium reports punctuation, superscripts and table cells as their
  own "lines", and a flat per-page list would compare a left-column line with
  a right-column one. Both were false vetoes before they were filtered;
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
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK
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

#: A body line must be at least this wide and this tall. pdfium splits a stray
#: comma, a superscript or a table cell into its own "line": a 1.3pt-wide
#: fragment is ink, not a paragraph, and must not veto the page.
_MIN_BODY_LINE_WIDTH_PT = 20.0
_MIN_BODY_LINE_HEIGHT_PT = 4.0
_MAX_BODY_LINE_HEIGHT_PT = 22.0

#: Lines set well below the page's own median size are captions, figure axis
#: labels or table cells. They are placed as exhibits, never re-flowed as body
#: text, so they must not vote in the body-floor check either.
_MIN_BODY_SIZE_RATIO = 0.80

#: Body prose shares a left margin; figures, charts and centred headings do
#: not. When one margin carries at least this share of the page's glyphs it is
#: the body column and everything else on the page (axis labels, chart keys,
#: table cells) is an exhibit, not a paragraph.
_MIN_MARGIN_SHARE = 0.50
_MARGIN_TOLERANCE_PT = 8.0

#: Records whose left edge starts this far across the page belong to the right
#: column. Two-column pages must be checked per column; a flat list compares a
#: left-column line against a right-column one and calls them stacked.
_COLUMN_SPLIT_SHARE = 0.50


def _body_lines(lines: list[Any]) -> list[Any]:
    """The lines that can carry a body-text claim.

    ``_line_records`` used to convert every pdfium line, so a 1.3pt comma
    fragment sitting inside a body line's box read as a paragraph that
    overlapped it, and the router vetoed the whole document. Keep the lines
    that are wide enough and tall enough to be prose, then drop the ones set
    far below the page's own median size (table cells, axis labels, footnote
    marks). Mirrors ``probe_facsimile._body_lines`` and the ``0.8 × main`` rule
    already used by :mod:`ubt.adapters.pdf.textgeom`.
    """
    candidates: list[tuple[Any, float]] = []
    for line in lines:
        text = (line.text or "").strip()
        if not text:
            continue
        x0, y0, x1, y1 = (float(v) for v in line.rect)
        width = x1 - x0
        height = y1 - y0
        if width <= _MIN_BODY_LINE_WIDTH_PT:
            continue
        if not (_MIN_BODY_LINE_HEIGHT_PT < height < _MAX_BODY_LINE_HEIGHT_PT):
            continue
        size = float(getattr(line, "font_size", 0.0) or 0.0)
        if size <= 0.0:
            size = height * 0.78
        candidates.append((line, size))
    if not candidates:
        return []
    median = statistics.median(size for _, size in candidates)
    cutoff = median * _MIN_BODY_SIZE_RATIO
    kept = [(line, size) for line, size in candidates if size >= cutoff]
    if not kept:
        return []

    # A figure's labels are the same size as body text, so size alone cannot
    # separate them. They do not share the body's left margin, though: keep the
    # modal margin when it carries most of the page, and leave the records
    # untouched when no margin dominates (a genuine two-column page).
    margins: Counter[int] = Counter()
    for line, _ in kept:
        margins[round(float(line.rect[0]) / 4.0) * 4] += len((line.text or "").strip())
    total = sum(margins.values())
    ranked = margins.most_common(2)
    modal, weight = ranked[0]
    runner_up = ranked[1][1] / total if len(ranked) > 1 and total else 0.0
    # Two comparable margins are two columns, not one body and a figure; leave
    # them for ``_column_groups`` to split.
    if total and weight / total >= _MIN_MARGIN_SHARE and runner_up < _MIN_MARGIN_SHARE - 0.2:
        return [
            line for line, _ in kept if abs(float(line.rect[0]) - modal) <= _MARGIN_TOLERANCE_PT
        ]
    return [line for line, _ in kept]


def _column_groups(records: list[dict[str, Any]], page_width: float) -> list[list[dict[str, Any]]]:
    """Split records into left/right columns when the page really has two.

    A single flat list is returned for a one-column page (or for a page where
    one side holds too little text to be a column), which keeps the overlap
    check identical to before on ordinary prose.
    """
    if page_width <= 0.0 or not records:
        return [records]
    split = page_width * _COLUMN_SPLIT_SHARE
    left = [r for r in records if float(r["x0"]) < split]
    right = [r for r in records if float(r["x0"]) >= split]
    # Full-width prose crossing the split means the page is one column, however
    # much text happens to start on the right (a chart, a table). One or two
    # crossing lines are a spanning heading or caption and do not change that.
    crossing = [r for r in records if float(r["x0"]) < split - 5.0 and float(r["x1"]) > split + 5.0]
    if len(left) >= 3 and len(right) >= 3 and len(crossing) < 3:
        return [left, right]
    return [records]


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
        # Opening and querying the page count both walk libpdfium's global
        # state; they must hold the same lock as every other pdfium entry.
        with PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(str(pdf_path))
            page_count = len(pdf)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Page-lock probe could not open %s: %s", pdf_path, exc)
        return PageLockMeasurement(measured=False, reasons=("document unreadable",))

    try:
        page_numbers = _sample_page_numbers(page_count, max_pages)
        if not page_numbers:
            return PageLockMeasurement(measured=False, reasons=("no pages",))

        sampled: list[int] = []
        overlapping: list[int] = []
        capacities: list[float] = []
        failing: list[int] = []
        examined = 0
        ignored_fragments = 0

        for page_no in page_numbers:
            try:
                lines, page_size = extract_lines(pdf_path, page_no)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Page-lock probe skipped page %d: %s", page_no, exc)
                continue
            body_lines = _body_lines(lines)
            ignored_fragments += len(lines) - len(body_lines)
            records, source_size, leading = _line_records(body_lines)
            if not records or source_size <= 0.0:
                continue
            page_area = float(page_size[0]) * float(page_size[1])
            text_area = sum(float(r["w"]) * (float(r["top"]) - float(r["bot"])) for r in records)
            if page_area <= 0.0 or text_area / page_area < _MIN_TEXT_AREA_SHARE:
                continue
            examined += 1
            sampled.append(page_no)

            columns = _column_groups(records, float(page_size[0]))
            if check_source_overlap(columns):
                overlapping.append(page_no)
                continue

            em_ratio = max(1.15, min(1.65, leading / source_size if source_size else 1.33))
            solution = solve_page_2d(
                columns,
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
        if ignored_fragments:
            # Named so a reader can tell a veto from probe housekeeping: these
            # are punctuation splits, superscripts, table cells and axis
            # labels, none of which is a body paragraph.
            reasons.append(
                f"ignored {ignored_fragments} punctuation/subscript/table fragment(s) "
                "that pdfium reports as separate lines"
            )
        if overlapping:
            feasible = False
            reasons.append(
                f"source body boxes overlap on page(s) {', '.join(map(str, overlapping[:5]))}"
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
            with PDFIUM_LOCK:
                with_close()

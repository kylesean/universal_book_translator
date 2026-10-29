"""The source page's body metrics: its size and leading, measured from its own lines.

The facsimile channel's whole budget is expressed relative to the *source*
page's body size — the ceiling is ``source_size * 1.10`` and a paragraph's
height at size S is ``lines(S) * (leading/source_size) * S`` — so reading the
source size wrongly moves every page's delivered size with it.

The production renderer hardcoded ``source_size_pt=10.0, source_leading_pt=13.0``
for every document. The corpus measures 9.0–11.3pt, so a 11.3pt book was capped
at ``min(12.0, 10.0*1.10) = 11.0pt``: the delivered text was *smaller* than the
source it was meant to reproduce, on every page.

Two signals are available and neither alone is reliable:

* ``font_size`` from pdfium's char styles is the nominal size, and it is exactly
  right — except that some producers report ``1.0`` for every glyph (the
  reference textbook ``chapter-3.pdf`` among them).
* The ink box height is the height of the glyphs that actually landed on the
  page, which survives every producer, but it is the *ink* height (cap height
  for a line of prose), not the em.

So: prefer the reported size when it is plausible, and fall back to the ink
height scaled by the ratio measured on the reference page (9pt nominal over a
8.9pt ink box). The scale factor is a measured constant, not a guess, and it is
named here so the next measurement can replace it in one place.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: Nominal body size / measured ink-box height, from ``chapter-3.pdf``: a 9pt
#: Elsevier body renders an 8.9pt ink box over 38 body lines. Only used when the
#: producer does not report a usable ``font_size``.
INK_TO_EM_RATIO: float = 9.0 / 8.9

#: A ``font_size`` below this is a producer reporting a placeholder (``1.0``),
#: not a real size; no delivered book sets body text at 4.5pt.
MIN_PLAUSIBLE_SIZE_PT: float = 4.5
#: No body face in the corpus is outside this band (measured 9.0–11.3pt); a
#: value outside it is a display heading's size, not the body's.
MIN_BODY_SIZE_PT: float = 6.0
MAX_BODY_SIZE_PT: float = 14.0

#: Line pitches outside this band are not body leading: too small is two lines
#: sharing a baseline (a table row, a heading and its rule), too large is a
#: paragraph break.
MIN_LEADING_PT: float = 6.0
MAX_LEADING_PT: float = 40.0

#: Leading as a multiple of the size, used when the page has too few lines to
#: measure a pitch. 1.333 is the value the corpus clusters on.
DEFAULT_LEADING_RATIO: float = 1.333

#: A line box taller than this is not body prose (a rule, a sidebar glyph).
MAX_BODY_LINE_H_PT: float = 22.0
#: ...and one narrower than this is furniture (a folio, a review line number).
MIN_BODY_LINE_W_PT: float = 50.0

#: The bands a page's running head and folio occupy, as a share of page height.
#: Page furniture is not body text: it keeps its own box in the background layer
#: and must not constrain the body's size budget.
FURNITURE_BAND_TOP: float = 0.92
FURNITURE_BAND_BOTTOM: float = 0.08


def body_lines(lines: Sequence[object], page_h: float) -> list[object]:
    """The lines on a page that are body prose, not furniture or decoration."""
    return [
        line
        for line in lines
        if (line.rect[2] - line.rect[0]) > MIN_BODY_LINE_W_PT  # type: ignore[attr-defined]
        and MIN_PLAUSIBLE_SIZE_PT
        < (line.rect[3] - line.rect[1])  # type: ignore[attr-defined]
        < MAX_BODY_LINE_H_PT
        and page_h * FURNITURE_BAND_BOTTOM < line.rect[1]  # type: ignore[attr-defined]
        and line.rect[3] < page_h * FURNITURE_BAND_TOP  # type: ignore[attr-defined]
    ]


def _reported_sizes(lines: Sequence[object]) -> list[float]:
    return [
        float(line.font_size)  # type: ignore[attr-defined]
        for line in lines
        if float(getattr(line, "font_size", 0.0)) >= MIN_PLAUSIBLE_SIZE_PT
    ]


def source_body_size(lines: Sequence[object], default: float = 9.0) -> float:
    """The page's body size in points.

    ``lines`` should already be filtered by :func:`body_lines`; passing the
    whole page lets a running head's size dominate the mode.

    Prefers the size the producer reported (mode of the line sizes, which is the
    body rather than a heading); falls back to the mode of the ink heights
    scaled by :data:`INK_TO_EM_RATIO` when the producer reported none.
    """
    sizes = _reported_sizes(lines)
    if sizes:
        try:
            measured = float(statistics.mode([round(size, 1) for size in sizes]))
        except statistics.StatisticsError:  # pragma: no cover - tie between modes
            measured = float(statistics.median(sizes))
        if MIN_BODY_SIZE_PT <= measured <= MAX_BODY_SIZE_PT:
            return round(measured, 2)

    heights = [
        round(line.rect[3] - line.rect[1], 1)  # type: ignore[attr-defined]
        for line in lines
        if str(getattr(line, "text", "")).strip()
    ]
    if not heights:
        return default
    try:
        ink = float(statistics.mode(heights))
    except statistics.StatisticsError:  # pragma: no cover - tie between modes
        ink = float(statistics.median(heights))
    return round(max(MIN_BODY_SIZE_PT, min(MAX_BODY_SIZE_PT, ink * INK_TO_EM_RATIO)), 2)


def source_leading(lines: Sequence[object], source_size: float) -> float:
    """The page's body leading in points, from its most common line-to-line pitch.

    Falls back to ``source_size * 1.333`` when the page has too few lines to
    measure a pitch (a title page, a one-line caption page).
    """
    ordered = sorted(lines, key=lambda line: -line.rect[1])  # type: ignore[attr-defined]
    pitches = [
        round(
            ordered[i].rect[1] - ordered[i + 1].rect[1],  # type: ignore[attr-defined]
            1,
        )
        for i in range(len(ordered) - 1)
    ]
    plausible = [pitch for pitch in pitches if MIN_LEADING_PT < pitch < MAX_LEADING_PT]
    if not plausible:
        return round(source_size * DEFAULT_LEADING_RATIO, 2)
    try:
        return round(float(statistics.mode(plausible)), 2)
    except statistics.StatisticsError:  # pragma: no cover - tie between modes
        return round(float(statistics.median(plausible)), 2)


__all__ = [
    "DEFAULT_LEADING_RATIO",
    "FURNITURE_BAND_BOTTOM",
    "FURNITURE_BAND_TOP",
    "INK_TO_EM_RATIO",
    "MIN_PLAUSIBLE_SIZE_PT",
    "body_lines",
    "source_body_size",
    "source_leading",
]

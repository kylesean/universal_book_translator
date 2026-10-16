"""``column_right_share`` must not mistake page furniture for a right column.

Regression: a single-column paper with a rotated margin stamp (the arXiv/JMLR
watermark running vertically down the page edge) scored ``right_row_share``
≈ 0.44 on its body pages, so ``multicolumn_page_share`` reported 0.225 and the
renderer laid the delivery out in two columns -- a single-column source silently
rewritten as two columns. The stamp is emitted as one near-square glyph per
rect, which the old counter treated as a right-column segment.

Two guards are pinned here: narrow (furniture) fragments never form a segment,
and a right-started segment only counts when the same row also carries genuine
left-column text.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_text_pdf

from ubt.adapters.pdf.page_profiler import collect_page_facts, column_right_share
from ubt.core.policy.layout_policy import PROBE_COLUMN_SHARE

pytestmark = pytest.mark.fast

_WIDTH = 612.0


def _line(x: float, y: float, w: float, h: float = 10.0) -> tuple[float, float, float, float]:
    """A pdfium text rect ``(left, bottom, right, top)``."""
    return (x, y, x + w, y + h)


def test_a_single_column_page_scores_zero() -> None:
    # Full-width body lines: left margin at 72, running almost to the right edge.
    rects = [_line(72, 700 - i * 14, 468) for i in range(20)]
    assert column_right_share(rects, _WIDTH) == 0.0


def test_a_rotated_margin_stamp_is_not_a_right_column() -> None:
    """The bug: a vertical side stamp (one glyph per rect, down the right edge)."""
    rects = [_line(72, 700 - i * 14, 468) for i in range(12)]  # the real body
    # The stamp: single ~10pt glyphs at x≈510, stacked down the page.
    rects += [_line(510, 700 - i * 13, 10.0, 7.5) for i in range(20)]
    assert column_right_share(rects, _WIDTH) == 0.0, (
        "a rotated margin stamp must not read as a right column"
    )


def test_a_margin_stamp_is_not_counted_as_body_either() -> None:
    """A page of only a stamp (no left/right pairing) is not two-column."""
    rects = [_line(510, 700 - i * 13, 10.0, 7.5) for i in range(20)]
    assert column_right_share(rects, _WIDTH) == 0.0


def test_a_genuine_two_column_row_is_counted() -> None:
    """A left line and a right line at the same vertical band: the real shape."""
    rects: list[tuple[float, float, float, float]] = []
    for i in range(16):
        y = 700 - i * 14
        rects.append(_line(72, y, 200))  # left column, ends before the gutter
        rects.append(_line(330, y, 240))  # right column, starts past 0.45*width
    share = column_right_share(rects, _WIDTH)
    assert share > PROBE_COLUMN_SHARE, f"a real two-column page must score high, got {share}"


def test_a_right_only_line_without_a_left_pair_is_not_a_column() -> None:
    """Off-centre content alone (a header, a figure placed right) is not a column."""
    rects = [_line(330, 700 - i * 14, 240) for i in range(16)]
    assert column_right_share(rects, _WIDTH) == 0.0


def test_single_column_body_stays_below_the_multicolumn_bar(tmp_path: Path) -> None:
    """End to end through the real profiler: a single-column PDF is not flagged."""
    body = [f"Body line number {i} of a single-column page" for i in range(30)]
    pdf = write_text_pdf(tmp_path / "single.pdf", [body, body, body])
    facts = collect_page_facts(pdf)
    assert facts, "profiler returned no pages"
    flagged = [f for f in facts if f.right_row_share > PROBE_COLUMN_SHARE]
    assert flagged == [], f"single-column pages flagged as multi-column: {flagged}"

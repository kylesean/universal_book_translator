from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_text_pdf

from ubt.adapters.pdf.page_profiler import collect_page_facts, column_right_share
from ubt.core.policy.layout_policy import PROBE_COLUMN_SHARE

pytestmark = pytest.mark.fast

_WIDTH = 612.0


def _line(x: float, y: float, w: float, h: float = 10.0) -> tuple[float, float, float, float]:
    return (x, y, x + w, y + h)


def test_a_single_column_page_scores_zero() -> None:
    rects = [_line(72, 700 - i * 14, 468) for i in range(20)]
    assert column_right_share(rects, _WIDTH) == 0.0


def test_a_rotated_margin_stamp_is_not_a_right_column() -> None:
    rects = [_line(72, 700 - i * 14, 468) for i in range(12)]
    rects += [_line(510, 700 - i * 13, 10.0, 7.5) for i in range(20)]
    assert column_right_share(rects, _WIDTH) == 0.0, (
        "a rotated margin stamp must not read as a right column"
    )


def test_a_margin_stamp_is_not_counted_as_body_either() -> None:
    rects = [_line(510, 700 - i * 13, 10.0, 7.5) for i in range(20)]
    assert column_right_share(rects, _WIDTH) == 0.0


def test_a_genuine_two_column_row_is_counted() -> None:
    rects: list[tuple[float, float, float, float]] = []
    for i in range(16):
        y = 700 - i * 14
        rects.append(_line(72, y, 200))
        rects.append(_line(330, y, 240))
    share = column_right_share(rects, _WIDTH)
    assert share > PROBE_COLUMN_SHARE, f"a real two-column page must score high, got {share}"


def test_a_right_only_line_without_a_left_pair_is_not_a_column() -> None:
    rects = [_line(330, 700 - i * 14, 240) for i in range(16)]
    assert column_right_share(rects, _WIDTH) == 0.0


def test_single_column_body_stays_below_the_multicolumn_bar(tmp_path: Path) -> None:
    body = [f"Body line number {i} of a single-column page" for i in range(30)]
    pdf = write_text_pdf(tmp_path / "single.pdf", [body, body, body])
    facts = collect_page_facts(pdf)
    assert facts, "profiler returned no pages"
    flagged = [f for f in facts if f.right_row_share > PROBE_COLUMN_SHARE]
    assert flagged == [], f"single-column pages flagged as multi-column: {flagged}"

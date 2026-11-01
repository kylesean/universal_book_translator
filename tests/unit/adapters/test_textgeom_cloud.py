from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pdf_builders import write_marker_cloud_pdf, write_text_pdf

from ubt.adapters.pdf.textgeom import (
    _DEBRIS_MIN_COUNT,
    _drop_scatter_debris,
    extract_lines,
)

pytestmark = pytest.mark.fast

Rect = tuple[float, float, float, float]


def _line(x: float, y: float, width: float = 200.0, height: float = 10.0) -> Rect:
    return (x, y, x + width, y + height)


def _dot(x: float, y: float) -> Rect:
    return (x, y, x + 1.0, y + 1.0)


def _dots(n: int) -> list[Rect]:
    return [_dot(60.0 + i % 300, 700.0 - i // 300) for i in range(n)]


def test_a_dense_page_keeps_its_small_rects() -> None:
    rects = [_line(60.0, 700.0 - i * 12.0) for i in range(500)] + _dots(200)
    assert len(rects) > 600
    kept, dropped = _drop_scatter_debris(list(rects))
    assert dropped == 0
    assert kept == rects


def test_a_marker_cloud_drops_only_the_debris() -> None:
    debris = _dots(_DEBRIS_MIN_COUNT + 400)
    lines = [_line(60.0, 200.0 - i * 12.0) for i in range(50)]
    kept, dropped = _drop_scatter_debris(debris + lines)
    assert dropped == len(debris)
    assert kept == lines


def test_a_small_page_is_never_touched() -> None:
    rects = _dots(_DEBRIS_MIN_COUNT + 100)
    kept, dropped = _drop_scatter_debris(list(rects))
    assert (kept, dropped) == (rects, 0)


def test_a_tall_rect_is_only_debris_once_the_page_is_a_cloud() -> None:
    tall = (60.0, 40.0, 70.0, 140.0)
    dense = [tall] + [_line(60.0, 700.0 - i * 8.0) for i in range(650)]
    kept, dropped = _drop_scatter_debris(list(dense))
    assert tall in kept and dropped == 0

    cloud = [tall] + _dots(_DEBRIS_MIN_COUNT + 400)
    kept, dropped = _drop_scatter_debris(cloud)
    assert tall not in kept and dropped == len(cloud)


def test_extract_lines_reports_the_cloud_it_cleaned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_marker_cloud_pdf(
        tmp_path / "cloud.pdf", 700, body=["Real body line one.", "Real body line two."]
    )
    with caplog.at_level(logging.WARNING, logger="ubt.adapters.pdf.textgeom"):
        lines, _size = extract_lines(path, 1)

    texts = " ".join(line.text for line in lines)
    assert "Real body line one." in texts
    assert "Real body line two." in texts
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    message = record.getMessage()
    assert "dropped 700 debris rects" in message
    assert "not extracted" in message


def test_a_clean_page_says_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = write_text_pdf(tmp_path / "plain.pdf", [["One line only.", "And another."]])
    with caplog.at_level(logging.WARNING, logger="ubt.adapters.pdf.textgeom"):
        lines, _size = extract_lines(path, 1)
    assert [line.text for line in lines] == ["One line only.", "And another."]
    assert caplog.records == []


def test_dense_per_character_text_is_not_a_cloud() -> None:
    chars = [
        (
            60.0 + i % 50 * 4.0,
            700.0 - i // 50 * 8.0,
            60.0 + i % 50 * 4.0 + 3.5,
            700.0 - i // 50 * 8.0 + 4.9,
        )
        for i in range(700)
    ]
    kept, dropped = _drop_scatter_debris(list(chars))
    assert dropped == 0
    assert kept == chars


def test_a_marker_cloud_is_still_dropped() -> None:
    dots = [
        (60.0 + i % 300, 700.0 - i // 300, 60.0 + i % 300 + 1.0, 700.0 - i // 300 + 1.1)
        for i in range(_DEBRIS_MIN_COUNT + 400)
    ]
    kept, dropped = _drop_scatter_debris(list(dots))
    assert dropped == len(dots)
    assert kept == []

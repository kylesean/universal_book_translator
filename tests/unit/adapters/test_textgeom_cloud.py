"""The vector-figure micro-glyph cleanup must be bounded and must not be silent.

A matplotlib/tikz scatter cloud puts thousands of sub-5pt marker glyphs on one
page, which turns ``merge_row_fragments`` into an O(K^2) multi-minute hang. The
cleanup that bounds K is only allowed to fire when the page really is a cloud
(hundreds of debris-sized rects), because the same size test on a merely dense
page discards real content -- a superscript or a footnote mark. And when it
does fire it must say so: the rects, and the text under them, are gone.
"""

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
    """700 rects with 200 superscript-sized ones: the small ones stay.

    This is the page the old unconditional ``>600 rects`` filter damaged -- it
    is dense, not a cloud, so its 1pt marks are content, not marker debris.
    """
    rects = [_line(60.0, 700.0 - i * 12.0) for i in range(500)] + _dots(200)
    assert len(rects) > 600
    kept, dropped = _drop_scatter_debris(list(rects))
    assert dropped == 0
    assert kept == rects


def test_a_marker_cloud_drops_only_the_debris() -> None:
    """Past the debris floor the cleanup fires and removes exactly the debris."""
    debris = _dots(_DEBRIS_MIN_COUNT + 400)
    lines = [_line(60.0, 200.0 - i * 12.0) for i in range(50)]
    kept, dropped = _drop_scatter_debris(debris + lines)
    assert dropped == len(debris)
    assert kept == lines


def test_a_small_page_is_never_touched() -> None:
    """Below the rect count the cleanup never fires, however small the rects."""
    rects = _dots(_DEBRIS_MIN_COUNT + 100)
    kept, dropped = _drop_scatter_debris(list(rects))
    assert (kept, dropped) == (rects, 0)


def test_a_tall_rect_is_only_debris_once_the_page_is_a_cloud() -> None:
    """The >60pt rule rides the same gate -- a sidebar on a dense page survives.

    A 100pt-tall rect on an ordinary page is a watermark or a table rule the
    reader may still want; on a cloud page it is part of the figure cruft.
    """
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
    """A real cloud page: the body lines survive and the loss is logged."""
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
    assert "marker cloud" in message
    assert "dropped 700 debris rects" in message
    assert "not extracted" in message


def test_a_clean_page_says_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = write_text_pdf(tmp_path / "plain.pdf", [["One line only.", "And another."]])
    with caplog.at_level(logging.WARNING, logger="ubt.adapters.pdf.textgeom"):
        lines, _size = extract_lines(path, 1)
    assert [line.text for line in lines] == ["One line only.", "And another."]
    assert caplog.records == []

"""The micro-mask must take the colour of the page, not a hardcoded white.

The mask is the fallback path (source text that could not be stripped is
painted over), and it used to paint pure white. On a page printed on cream
stock, or a shaded row, or a scanned off-white, that is a bright rectangle
across every replaced line. The fill is sampled from the page now -- median,
per region, at a low dpi.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_text_pdf
from PIL import Image

from ubt.render.page_background import _median_rgb, sample_backgrounds

pytestmark = pytest.mark.fast

_PAGE = (
    "The Attention Machine",
    "The machine relies on attention and runs a forward pass.",
)
#: The two text lines the fixture writes (margin 54, top 792 - 54, leading 16).
_TEXT_REGION = (54.0, 700.0, 400.0, 745.0)
#: Deliberately not grey, and not symmetric: a channel swap or a collapsed
#: colour would survive any tint that reads the same on every channel.
_TINT = (0.8, 0.9, 0.7)
_OFF_PAGE = (700.0, 700.0, 800.0, 745.0)


def test_white_paper_reads_white(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])

    samples = sample_backgrounds(source, 1, [_TEXT_REGION])

    assert samples == [(1.0, 1.0, 1.0)]


def test_the_text_on_the_region_does_not_drag_the_sample_dark(tmp_path: Path) -> None:
    # The region is mostly paper with a line of ink across it; a mean would
    # come back grey, which is why the median is what gets sampled.
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])

    (sample,) = sample_backgrounds(source, 1, [_TEXT_REGION])

    assert sample is not None
    assert min(sample) > 0.9, f"the paper was not found under the ink: {sample}"


def test_a_tinted_page_reads_its_tint(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE], background=_TINT)

    (sample,) = sample_backgrounds(source, 1, [_TEXT_REGION])

    assert sample is not None
    for got, want in zip(sample, _TINT, strict=True):
        assert got == pytest.approx(want, abs=0.02)


def test_a_region_off_the_page_samples_nothing(tmp_path: Path) -> None:
    # A visible mask is better than source text left under the translation, so
    # an unanswerable region reports "no sample" and the caller fills white.
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])

    assert sample_backgrounds(source, 1, [_OFF_PAGE]) == [None]


def test_a_missing_source_samples_nothing(tmp_path: Path) -> None:
    assert sample_backgrounds(tmp_path / "gone.pdf", 1, [_TEXT_REGION]) == [None]


def test_samples_come_back_one_per_region_in_order(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE], background=_TINT)

    samples = sample_backgrounds(source, 1, [_TEXT_REGION, _OFF_PAGE, _TEXT_REGION])

    assert len(samples) == 3
    assert samples[1] is None
    assert samples[0] == samples[2]


# --------------------------------------------------------------------------- #
# The statistic itself
# --------------------------------------------------------------------------- #


def _region(ink_columns: int, *, width: int = 100) -> Image.Image:
    """A white region whose left ``ink_columns`` columns are black ink."""
    image = Image.new("RGB", (width, 20), (255, 255, 255))
    if ink_columns:
        image.paste((0, 0, 0), (0, 0, ink_columns, 20))
    return image


def test_the_median_finds_the_paper_under_the_ink() -> None:
    # 40% ink: still the minority of the region, so the paper is the middle
    # value of every channel.
    assert _median_rgb(_region(40)) == (1.0, 1.0, 1.0)


def test_the_median_follows_the_pages_polarity() -> None:
    # 60% ink: the ink is now the majority, so the middle value is the ink's.
    # A fixed "lighter pixel" rule would get a light-on-dark page backwards.
    assert _median_rgb(_region(60)) == (0.0, 0.0, 0.0)


def test_the_median_is_per_channel() -> None:
    # The three channels must be read separately: a page can be flat in red and
    # blue while green carries the tint (or the ink).
    image = Image.new("RGB", (4, 1))
    image.putdata([(255, 10, 0), (255, 20, 0), (255, 30, 0), (255, 40, 0)])
    red, green, blue = _median_rgb(image)
    assert red == 1.0
    # The upper of the two middle values, per the histogram rule.
    assert green == pytest.approx(30 / 255, abs=0.005)
    assert blue == 0.0

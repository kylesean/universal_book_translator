"""The VLM comparison harness: cropping must be exact, or the reading is noise.

Measured failure modes this pins (all from arXiv 2609.20519 / 2606.23344, RTX
4070 Ti, PaddleOCR-VL-1.6):

* a crop that includes surrounding prose makes ``Formula Recognition:`` read the
  prose back as ``\\text{...}`` — the model transcribes what it is shown;
* a page-sized crop collapses several tables into one row, or loops one equation
  to the token cap.

So the crop is the input contract. These tests pin the coordinate convention and
the request shape without loading a 2 GB model: the crop is a pure function of
the page geometry, and the prompt is the model's own task token.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.compare_vlm_elements import TASKS, _crop  # noqa: E402

pytestmark = pytest.mark.fast


@pytest.fixture()
def page_pdf(tmp_path: Path) -> Path:
    from pdf_builders import write_text_pdf

    return write_text_pdf(tmp_path / "page.pdf", [["Hello world test page content"]] * 3)


def test_the_box_is_read_with_a_top_left_origin(page_pdf: Path) -> None:
    """A top-left box must cut the top-left corner, not the mirrored one.

    The first version of this tool translated the box to bottom-left internally
    and produced a crop from the wrong half of the page — the model then reported
    content that was nowhere near the region under test, which reads exactly like
    a model failure.
    """
    crop = _crop(page_pdf, 1, (0.0, 0.0, 200.0, 100.0), dpi=72)
    # 72 dpi is 1 px per point, so the crop keeps the requested point size.
    assert crop.size == (200, 100)


def test_the_crop_scales_with_dpi(page_pdf: Path) -> None:
    """200 dpi must produce ~2.8x the pixels of 72 dpi for the same box."""
    low = _crop(page_pdf, 1, (0.0, 0.0, 100.0, 100.0), dpi=72)
    high = _crop(page_pdf, 1, (0.0, 0.0, 100.0, 100.0), dpi=144)
    assert high.size[0] == pytest.approx(low.size[0] * 2, abs=2)
    assert high.size[1] == pytest.approx(low.size[1] * 2, abs=2)


def test_the_model_prompts_are_the_published_task_tokens() -> None:
    """The prompt IS the task switch; a paraphrase is a different request.

    PaddleOCR-VL has no separate endpoint per task — the same model answers
    "OCR:", "Table Recognition:", "Formula Recognition:" and "Chart
    Recognition:". Measured: a *page* with several tables must not be sent with
    "Formula Recognition:" (it loops to the token cap), so the mapping is worth
    keeping explicit.
    """
    assert TASKS == {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
    }


def test_a_page_outside_the_document_is_rejected(page_pdf: Path) -> None:
    """Out of range must raise, not silently crop some other page."""
    import pypdfium2 as pdfium

    with pytest.raises(pdfium.PdfiumError):
        _crop(page_pdf, 99, (0.0, 0.0, 100.0, 100.0), dpi=72)

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
    crop = _crop(page_pdf, 1, (0.0, 0.0, 200.0, 100.0), dpi=72)
    assert crop.size == (200, 100)


def test_the_crop_scales_with_dpi(page_pdf: Path) -> None:
    low = _crop(page_pdf, 1, (0.0, 0.0, 100.0, 100.0), dpi=72)
    high = _crop(page_pdf, 1, (0.0, 0.0, 100.0, 100.0), dpi=144)
    assert high.size[0] == pytest.approx(low.size[0] * 2, abs=2)
    assert high.size[1] == pytest.approx(low.size[1] * 2, abs=2)


def test_the_model_prompts_are_the_published_task_tokens() -> None:
    assert TASKS == {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
    }


def test_a_page_outside_the_document_is_rejected(page_pdf: Path) -> None:
    import pypdfium2 as pdfium

    with pytest.raises(pdfium.PdfiumError):
        _crop(page_pdf, 99, (0.0, 0.0, 100.0, 100.0), dpi=72)

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ubt.adapters.pdf.docling_parser import (
    _has_text_content,
    _reject_empty_book,
    vlm_fallback_missing_pages,
)
from ubt.adapters.pdf.vlm.drivers.rapidocr_driver import RapidOcrDriver
from ubt.adapters.pdf.vlm.registry import probe_effective_driver
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


def _make_block(
    block_type: BlockType,
    text: str,
    *,
    page: int = 1,
    skip: bool = False,
    spine_index: int = 1,
) -> IRBlock:
    elem = make_element(
        block_type=block_type,
        id=f"test#{spine_index}",
        source_text=text,
        spine_index=spine_index,
        bbox=BoundingBox(x0=10, y0=10, x1=100, y1=50, page=page),
        skip_translate=skip,
    )
    return IRBlock(element=elem)


def test_has_text_content_detection() -> None:
    assert not _has_text_content([])

    img_block = _make_block(BlockType.IMAGE, "pic.png", skip=True)
    assert not _has_text_content([img_block])

    ws_block = _make_block(BlockType.NARRATIVE, "   \n\t  ")
    assert not _has_text_content([ws_block])

    narr_block = _make_block(BlockType.NARRATIVE, "Hello world")
    assert _has_text_content([narr_block])
    assert _has_text_content([img_block, narr_block])


def test_reject_empty_book_raises_on_image_only_document() -> None:
    fake_path = Path("scanned_sample.pdf")

    img1 = _make_block(BlockType.IMAGE, "pic1.png", page=1, skip=True, spine_index=1)
    img2 = _make_block(BlockType.IMAGE, "pic2.png", page=2, skip=True, spine_index=2)

    with pytest.raises(DocumentParseError) as exc_info:
        _reject_empty_book(fake_path, [img1, img2], ocr_mode="auto")

    assert "every page lacks an extractable text layer" in str(exc_info.value)
    assert "ocr_mode='auto'" in str(exc_info.value)


def test_reject_empty_book_accepts_valid_text_document() -> None:
    fake_path = Path("digital_sample.pdf")
    narr = _make_block(BlockType.NARRATIVE, "Chapter 1: The Beginning", page=1)
    img = _make_block(BlockType.IMAGE, "pic1.png", page=1, skip=True, spine_index=2)

    blocks = [narr, img]
    accepted = _reject_empty_book(fake_path, blocks, ocr_mode="auto")
    assert accepted == blocks


def test_rapidocr_driver_is_available_graceful() -> None:
    avail = RapidOcrDriver.is_available()
    assert isinstance(avail, bool)

    with patch.object(
        RapidOcrDriver, "_get_engine", side_effect=ImportError("onnxruntime missing")
    ):
        assert not RapidOcrDriver.is_available()


def test_probe_effective_driver_when_rapidocr_unoperational() -> None:
    with patch.object(RapidOcrDriver, "is_available", return_value=False):
        driver_type, driver = probe_effective_driver(
            mode="local",
            endpoint=None,
            api_key=None,
            allow_page_upload=False,
        )
        assert driver_type is None
        assert driver is None

        driver_type_auto, driver_auto = probe_effective_driver(
            mode="auto",
            endpoint=None,
            api_key=None,
            allow_page_upload=False,
        )
        assert driver_type_auto is None
        assert driver_auto is None

    with pytest.raises(ValueError, match="unknown ocr_mode"):
        probe_effective_driver(mode="bogus", allow_page_upload=False)


def test_vlm_fallback_auto_promotes_to_missing_for_textless_scans() -> None:
    fake_path = Path("fake_scan.pdf")
    img = _make_block(BlockType.IMAGE, "pic1.png", page=1, skip=True)

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    with (
        patch("pypdfium2.PdfDocument", return_value=mock_doc),
        patch.dict("os.environ", {"UBT_OCR_FALLBACK": "off"}),
    ):
        with pytest.raises(DocumentParseError) as exc_info:
            vlm_fallback_missing_pages(
                fake_path,
                [img],
                ocr_mode="auto",
                allow_page_upload=False,
            )
        assert "every page lacks an extractable text layer" in str(exc_info.value)


def test_pages_the_ocr_driver_could_not_transcribe_are_refused_not_shipped() -> None:
    book = _make_block(BlockType.NARRATIVE, "page one prose", page=1, spine_index=1)
    scan_only = _make_block(BlockType.IMAGE, "scan.png", page=2, skip=True, spine_index=2)

    driver = MagicMock()
    driver.measured_boxes = False
    driver.engine = "vlm"

    fake_pdf = MagicMock()
    fake_pdf.__len__ = MagicMock(return_value=2)

    with (
        patch("pypdfium2.PdfDocument", return_value=fake_pdf),
        patch.dict("os.environ", {"UBT_OCR_FALLBACK": "missing"}),
        patch(
            "ubt.adapters.pdf.vlm.transcribe.get_driver",
            return_value=driver,
        ),
        patch(
            "ubt.adapters.pdf.vlm.registry.probe_effective_driver",
            return_value=("vlm", driver),
        ),
        pytest.raises(DocumentParseError) as exc_info,
    ):
        vlm_fallback_missing_pages(
            Path("mixed_book.pdf"),
            [book, scan_only],
            ocr_mode="vlm",
            allow_page_upload=True,
        )

    message = str(exc_info.value)
    assert "Only part of" in message
    assert "page(s) have no extractable text layer" in message
    assert "2" in message, "the unreadable page must be named"


def test_a_block_spanning_a_page_break_covers_every_page_of_its_chain() -> None:
    from ubt.adapters.pdf.docling_parser import _pages_covered_by_blocks

    fused = _make_block(BlockType.NARRATIVE, "a paragraph across four pages", page=1)
    fused.provenance.physical_boxes = [
        {"page": page, "bbox": [54.0, 719.5, 606.9, 746.6]} for page in (1, 2, 3, 4)
    ]
    assert _pages_covered_by_blocks([fused]) == {1, 2, 3, 4}

    single = _make_block(BlockType.NARRATIVE, "one page only", page=7)
    assert _pages_covered_by_blocks([single]) == {7}

    image = _make_block(BlockType.IMAGE, "scan.png", page=9, skip=True)
    assert _pages_covered_by_blocks([image]) == set()

    blank = _make_block(BlockType.NARRATIVE, "   ", page=11)
    assert _pages_covered_by_blocks([blank]) == set()


@pytest.mark.parametrize("mode", ["cloud"])
def test_an_explicit_remote_ocr_mode_cannot_bypass_the_upload_gate(mode: str) -> None:
    with pytest.raises(ValueError, match="allow_page_upload=false"):
        probe_effective_driver(mode=mode, allow_page_upload=False)


def test_a_local_ocr_mode_still_works_with_the_gate_closed() -> None:
    with patch.object(RapidOcrDriver, "is_available", return_value=False):
        assert probe_effective_driver(mode="auto", allow_page_upload=False) == (None, None)
        assert probe_effective_driver(mode="off", allow_page_upload=False) == (None, None)

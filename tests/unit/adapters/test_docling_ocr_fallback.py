"""Tests for scanned PDF empty book rejection and OCR driver fallback resilience."""

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
    # Empty list
    assert not _has_text_content([])

    # Image-only list
    img_block = _make_block(BlockType.IMAGE, "pic.png", skip=True)
    assert not _has_text_content([img_block])

    # Whitespace-only block
    ws_block = _make_block(BlockType.NARRATIVE, "   \n\t  ")
    assert not _has_text_content([ws_block])

    # Legitimate textual block
    narr_block = _make_block(BlockType.NARRATIVE, "Hello world")
    assert _has_text_content([narr_block])
    assert _has_text_content([img_block, narr_block])


def test_reject_empty_book_raises_on_image_only_document() -> None:
    fake_path = Path("scanned_sample.pdf")

    # Image-only blocks from layout detector
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
    # In an environment without onnxruntime, is_available() must return False without throwing.
    # If onnxruntime happens to be installed, it returns True. Either way, it must be boolean.
    avail = RapidOcrDriver.is_available()
    assert isinstance(avail, bool)

    # When engine raises ImportError, is_available must return False
    with patch.object(
        RapidOcrDriver, "_get_engine", side_effect=ImportError("onnxruntime missing")
    ):
        assert not RapidOcrDriver.is_available()


def test_probe_effective_driver_when_rapidocr_unoperational() -> None:
    # When rapidocr is requested or probed in auto, but is_available() returns False
    with patch.object(RapidOcrDriver, "is_available", return_value=False):
        driver_type, driver = probe_effective_driver(
            mode="rapidocr",
            endpoint=None,
            api_key=None,
            allow_page_upload=False,
        )
        assert driver_type is None
        assert driver is None

        # auto mode with no upload allowed should return (None, None)
        driver_type_auto, driver_auto = probe_effective_driver(
            mode="auto",
            endpoint=None,
            api_key=None,
            allow_page_upload=False,
        )
        assert driver_type_auto is None
        assert driver_auto is None


def test_vlm_fallback_auto_promotes_to_missing_for_textless_scans() -> None:
    fake_path = Path("fake_scan.pdf")
    img = _make_block(BlockType.IMAGE, "pic1.png", page=1, skip=True)

    # When fallback mode is OFF, but PDF has NO text content and ocr_mode != 'off',
    # it promotes to MISSING and attempts driver probing rather than immediately returning image blocks.
    # With no driver available, it fails closed with DocumentParseError.
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    with (
        patch("pypdfium2.PdfDocument", return_value=mock_doc),
        patch.dict("os.environ", {"UBT_VLM_SCAN_FALLBACK": "off"}),
    ):
        with pytest.raises(DocumentParseError) as exc_info:
            vlm_fallback_missing_pages(
                fake_path,
                [img],
                ocr_mode="auto",
                allow_page_upload=False,
            )
        assert "every page lacks an extractable text layer" in str(exc_info.value)

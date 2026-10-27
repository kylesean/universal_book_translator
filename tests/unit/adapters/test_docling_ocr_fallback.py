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


def test_pages_the_ocr_driver_could_not_transcribe_are_refused_not_shipped() -> None:
    """A mixed text+scan book must not ship as a complete delivery.

    The OCR pass computed which pages had no text layer, used that set to decide
    what to bill, and then dropped it: the completion guard was called with no
    ``missing_pages``, so every page the driver failed on was silently absent
    while the run cleared all downstream gates. The no-OCR branch already passed
    this evidence; both paths must reach the guard with it.
    """
    book = _make_block(BlockType.NARRATIVE, "page one prose", page=1, spine_index=1)
    scan_only = _make_block(BlockType.IMAGE, "scan.png", page=2, skip=True, spine_index=2)

    # A vision-LLM driver reports no measured boxes, so it cannot supply geometry
    # for a page that has none -- exactly the "--ocr vlm on a mixed book" case.
    driver = MagicMock()
    driver.measured_boxes = False
    driver.engine = "vlm"

    fake_pdf = MagicMock()
    fake_pdf.__len__ = MagicMock(return_value=2)

    with (
        patch("pypdfium2.PdfDocument", return_value=fake_pdf),
        patch.dict("os.environ", {"UBT_VLM_SCAN_FALLBACK": "missing"}),
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
    """Page coverage must read the whole ``physical_boxes`` chain, not just ``bbox``.

    Docling emits ONE ``TextItem`` for a paragraph that runs across pages, with
    one ``prov`` entry per page. The IR keeps that chain in ``physical_boxes``,
    but ``bbox.page`` (and ``provenance.source_page``) name only the FIRST page.
    Reading just those reported pages 2-4 as having no text layer, so the
    partial-read guard refused a book it had in fact read completely — every
    crash/resume e2e failed on the default engine for that reason.
    """
    from ubt.adapters.pdf.docling_parser import _pages_covered_by_blocks

    fused = _make_block(BlockType.NARRATIVE, "a paragraph across four pages", page=1)
    # The chain Docling actually produced for the e2e fixture.
    fused.provenance.physical_boxes = [
        {"page": page, "bbox": [54.0, 719.5, 606.9, 746.6]} for page in (1, 2, 3, 4)
    ]
    assert _pages_covered_by_blocks([fused]) == {1, 2, 3, 4}

    # A single-box block still reports its own page.
    single = _make_block(BlockType.NARRATIVE, "one page only", page=7)
    assert _pages_covered_by_blocks([single]) == {7}

    # An IMAGE block covers nothing: an image-only page has no text layer.
    image = _make_block(BlockType.IMAGE, "scan.png", page=9, skip=True)
    assert _pages_covered_by_blocks([image]) == set()

    # A textless block covers nothing either.
    blank = _make_block(BlockType.NARRATIVE, "   ", page=11)
    assert _pages_covered_by_blocks([blank]) == set()


# --------------------------------------------------------------------------- #
# The egress gate: `allow_page_upload` is the only switch that permits page
# images to leave the machine. `ocr_mode` selects a route, it does not
# authorize one.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["vlm"])
def test_an_explicit_remote_ocr_mode_cannot_bypass_the_upload_gate(mode: str) -> None:
    """Naming a remote route is not consent to use it.

    Measured: with ``allow_page_upload=False`` (the default) both remote modes
    raise rather than shipping pages. This was verified because the opposite had
    been assumed — that an explicit ``ocr_mode=cloud`` was the user's consent
    action. It is not; ``allow_page_upload`` is, and it is operator-only.
    """
    with pytest.raises(ValueError, match="allow_page_upload=false"):
        probe_effective_driver(mode=mode, allow_page_upload=False)


def test_a_local_ocr_mode_still_works_with_the_gate_closed() -> None:
    """The gate blocks egress, not OCR: a local route keeps working."""
    with patch.object(RapidOcrDriver, "is_available", return_value=False):
        # No local engine available and egress closed -> no driver, no error.
        assert probe_effective_driver(mode="auto", allow_page_upload=False) == (None, None)
        assert probe_effective_driver(mode="off", allow_page_upload=False) == (None, None)

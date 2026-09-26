"""Regression tests for Batch 2 fixes: VLM Subsystem & Multi-engine OCR Drivers."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ubt.adapters.pdf.vlm.drivers.cloud_driver import CloudOcrDriver
from ubt.adapters.pdf.vlm.drivers.deepseek_driver import DeepSeekOcrDriver
from ubt.adapters.pdf.vlm.drivers.rapidocr_driver import RapidOcrDriver
from ubt.adapters.pdf.vlm.drivers.sidecar_driver import SidecarOcrDriver
from ubt.adapters.pdf.vlm.registry import probe_effective_driver
from ubt.adapters.pdf.vlm.transcribe import transcribe_page_to_blocks


@pytest.mark.fast
def test_sidecar_is_healthy_passes_api_key() -> None:
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.get.return_value = mock_resp
        mock_client.__enter__.return_value = mock_client
        mock_client_cls.return_value = mock_client

        healthy = SidecarOcrDriver.is_healthy("http://localhost:8765", api_key="secret-tok")
        assert healthy is True
        mock_client.get.assert_called_once()
        _, kwargs = mock_client.get.call_args
        assert kwargs.get("headers") == {"Authorization": "Bearer secret-tok"}


@pytest.mark.fast
def test_registry_auto_probe_passes_api_key_to_sidecar_health() -> None:
    with patch.object(SidecarOcrDriver, "is_healthy", return_value=True) as mock_health:
        probe_effective_driver(
            mode="auto", endpoint="http://localhost:8765", api_key="my-key", allow_page_upload=True
        )
        mock_health.assert_called_once()
        _, kwargs = mock_health.call_args
        assert kwargs.get("api_key") == "my-key" or mock_health.call_args[0][1] == "my-key"


@pytest.mark.fast
def test_sidecar_recognize_computes_measured_boxes_dynamically() -> None:
    driver = SidecarOcrDriver(endpoint="http://localhost:8765")
    img = Image.new("RGB", (100, 100), color="white")

    # 1. Sidecar returns lines without bounding boxes
    payload_no_boxes = {
        "lines": [{"text": "Hello world"}],
        "coord_system": "auto",
    }
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload_no_boxes
        mock_client.post.return_value = mock_resp
        mock_client.__enter__.return_value = mock_client
        mock_client_cls.return_value = mock_client

        transcript = driver.recognize(img, (100.0, 100.0), scale=1.0)
        assert transcript.measured_boxes is False, (
            "Sidecar without boxes must set measured_boxes=False"
        )


@pytest.mark.fast
def test_deepseek_driver_close_kills_worker() -> None:
    driver = DeepSeekOcrDriver()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = MagicMock()
    driver._proc = mock_proc

    driver.close()
    mock_proc.kill.assert_called_once()
    assert driver._proc is None


@pytest.mark.fast
def test_rapidocr_driver_empty_box_does_not_crash() -> None:
    # numpy ships in the optional `ocr` extra (see test_rapidocr_driver.py for
    # why the skip stays in-test rather than at module level).
    pytest.importorskip("numpy", reason="numpy ships in the optional `ocr` extra")
    driver = RapidOcrDriver()
    mock_engine = MagicMock()
    # Box is empty list []
    mock_engine.return_value = ([[[], "sample text", 0.95]], 0.05)
    driver._engine = mock_engine

    img = Image.new("RGB", (100, 100), color="white")
    # Must not raise ValueError: min() arg is an empty sequence
    transcript = driver.recognize(img, (100.0, 100.0), scale=1.0)
    assert len(transcript.lines) == 0


@pytest.mark.fast
def test_transcribe_page_bounds_check(tmp_path: Path) -> None:
    # Create a minimal 1-page PDF using pypdfium2
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(width=100, height=100)
    pdf_path = tmp_path / "test.pdf"
    pdf.save(str(pdf_path))
    pdf.close()

    # page_no = 0 must raise IndexError, not access pdf[-1]
    with pytest.raises(IndexError):
        transcribe_page_to_blocks(pdf_path, page_no=0)

    # page_no > total pages must raise IndexError
    with pytest.raises(IndexError):
        transcribe_page_to_blocks(pdf_path, page_no=5)


@pytest.mark.fast
def test_cloud_driver_measured_boxes_synchronized_in_init() -> None:
    # When endpoint is OpenAI or chat/completions, driver is vision LLM (measured_boxes=False)
    driver = CloudOcrDriver(endpoint="https://api.openai.com/v1")
    assert driver.measured_boxes is False, (
        "OpenAI endpoint must set measured_boxes=False in __init__"
    )


@pytest.mark.fast
def test_engine_selector_pikepdf_does_not_shadow_pdfium(tmp_path: Path) -> None:
    import pypdfium2 as pdfium

    from ubt.adapters.pdf import engine_selector

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(width=200, height=200)
    pdf_path = tmp_path / "probe_test.pdf"
    pdf.save(str(pdf_path))
    pdf.close()

    closed_docs: list[Any] = []
    real_pdfium_doc = pdfium.PdfDocument

    def monitored_pdf_doc(*args: Any, **kwargs: Any) -> Any:
        doc = real_pdfium_doc(*args, **kwargs)
        real_close = doc.close

        def _close() -> None:
            closed_docs.append(doc)
            real_close()

        doc.close = _close
        return doc

    with patch("pypdfium2.PdfDocument", side_effect=monitored_pdf_doc):
        plan = engine_selector.inspect_pdf_route_plan(pdf_path)
        assert plan is not None
        assert len(closed_docs) == 1, "Outer pdfium document must be closed in finally block"

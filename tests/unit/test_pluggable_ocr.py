"""Unit tests for pluggable OCR drivers (Sidecar, Cloud REST, Vision LLM) and registry probing."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.vlm.drivers.cloud_driver import DEFAULT_VISION_MODEL, CloudOcrDriver
from ubt.adapters.pdf.vlm.drivers.sidecar_driver import SidecarOcrDriver
from ubt.adapters.pdf.vlm.registry import (
    list_drivers,
    probe_effective_driver,
    register_driver,
)
from ubt.adapters.pdf.vlm.types import PageTranscript, VlmLine
from ubt.core.config import UBTConfig
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock


@pytest.fixture
def dummy_image() -> Image.Image:
    """Create a dummy 100x100 RGB image for driver testing."""
    return Image.new("RGB", (100, 100), color="white")


def test_registry_contains_pluggable_drivers() -> None:
    drivers = list_drivers()
    assert "sidecar" in drivers
    assert "http" in drivers
    assert "cloud" in drivers
    assert "vlm" in drivers
    assert "rapidocr" in drivers


def test_sidecar_is_healthy_probe() -> None:
    with patch("httpx.Client.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        assert SidecarOcrDriver.is_healthy("http://localhost:8765") is True

        mock_get.return_value = MagicMock(status_code=500)
        assert SidecarOcrDriver.is_healthy("http://localhost:8765") is False

        mock_get.side_effect = Exception("Connection refused")
        assert SidecarOcrDriver.is_healthy("http://localhost:8765") is False


def test_sidecar_recognize_paddle_grid(dummy_image: Image.Image) -> None:
    driver = SidecarOcrDriver(endpoint="http://mock-sidecar:8765", api_key="secret-key")

    mock_resp = {
        "lines": [
            {
                "text": "Chapter 1 Introduction",
                "box": [100, 100, 900, 200],  # [0, 1000] grid
                "confidence": 0.98,
            },
            {
                "text": "Body paragraph content",
                "box": [100, 250, 900, 400],
                "confidence": 0.95,
            },
        ],
        "coord_system": "norm_1000",
    }

    with patch("httpx.Client.post") as mock_post:
        mock_resp_obj = MagicMock(status_code=200)
        mock_resp_obj.json.return_value = mock_resp
        mock_resp_obj.raise_for_status.return_value = None
        mock_post.return_value = mock_resp_obj

        # Page size: 600 x 800 pt, scale 2.0
        transcript = driver.recognize(dummy_image, page_size_pt=(600.0, 800.0), scale=2.0)

        assert mock_post.called
        # Verify Bearer auth header
        called_headers = mock_post.call_args[1].get("headers", {})
        assert called_headers.get("Authorization") == "Bearer secret-key"

        assert len(transcript.lines) == 2
        assert transcript.lines[0].text == "Chapter 1 Introduction"
        assert transcript.lines[0].confidence == 0.98
        # Paddle grid box [100, 100, 900, 200] in 600x800 pt:
        # x0: 100/1000 * 600 = 60.0
        # x1: 900/1000 * 600 = 540.0
        # y0: 800 - (200/1000 * 800) = 800 - 160 = 640.0
        # y1: 800 - (100/1000 * 800) = 800 - 80 = 720.0
        box0 = transcript.lines[0].measured_box
        assert box0 is not None
        assert abs(box0[0] - 60.0) < 1.0
        assert abs(box0[2] - 540.0) < 1.0
        assert abs(box0[1] - 640.0) < 1.0
        assert abs(box0[3] - 720.0) < 1.0


def test_cloud_vision_llm(dummy_image: Image.Image) -> None:
    driver = CloudOcrDriver(
        endpoint="https://api.openai.com/v1",
        api_key="test-openai-key",
        provider="vlm",
        model="gpt-4o-mini",
    )

    mock_content = "Line 1 from Vision\nLine 2 from Vision\nLine 3"
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"choices": [{"message": {"content": mock_content}}]}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        transcript = driver.recognize(dummy_image, page_size_pt=(600.0, 800.0), scale=2.0)
        assert len(transcript.lines) == 3
        assert transcript.lines[0].text == "Line 1 from Vision"
        assert transcript.lines[1].text == "Line 2 from Vision"
        assert transcript.lines[2].text == "Line 3"
        assert transcript.measured_boxes is False


def test_cloud_rest_baidu_format(dummy_image: Image.Image) -> None:
    driver = CloudOcrDriver(
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic",
        api_key="baidu-token",
        provider="cloud",
    )

    baidu_resp = {
        "words_result_num": 2,
        "words_result": [
            {
                "words": "Baidu recognized line 1",
                "location": {"left": 10, "top": 20, "width": 80, "height": 30},
            },
            {
                "words": "Baidu recognized line 2",
                "location": {"left": 10, "top": 60, "width": 80, "height": 30},
            },
        ],
    }

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = baidu_resp
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        transcript = driver.recognize(dummy_image, page_size_pt=(100.0, 100.0), scale=1.0)
        assert len(transcript.lines) == 2
        assert transcript.lines[0].text == "Baidu recognized line 1"
        assert transcript.lines[0].measured_box is not None
        assert transcript.measured_boxes is True


def test_probe_effective_driver_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. off mode
    mode, drv = probe_effective_driver(mode="off")
    assert mode is None
    assert drv is None

    # 2. explicit sidecar mode. `http://custom:1234` is a non-local host, so the
    # egress gate (non-local sidecar + allow_page_upload=false now fails loudly)
    # must be opened for this dispatch/plumbing check.
    mode, drv = probe_effective_driver(
        mode="sidecar", endpoint="http://custom:1234", allow_page_upload=True
    )
    assert mode == "sidecar"
    assert isinstance(drv, SidecarOcrDriver)
    assert drv.endpoint == "http://custom:1234"

    # 2b. The sidecar egress gate itself: a non-local endpoint with uploads
    # disabled must fail loudly (explicit mode), not silently egress pages.
    with pytest.raises(ValueError, match="UBT_ALLOW_PAGE_UPLOAD"):
        probe_effective_driver(
            mode="sidecar", endpoint="http://remote:1234", allow_page_upload=False
        )

    # 3. explicit cloud mode
    mode, drv = probe_effective_driver(
        mode="cloud", endpoint="https://api.ocr.com", allow_page_upload=True
    )
    assert mode == "cloud"
    assert isinstance(drv, CloudOcrDriver)

    # 4. explicit vlm mode
    mode, drv = probe_effective_driver(mode="vlm", allow_page_upload=True)
    assert mode == "vlm"
    assert isinstance(drv, CloudOcrDriver)

    # 5. auto mode with healthy sidecar
    with patch.object(SidecarOcrDriver, "is_healthy", return_value=True):
        mode, drv = probe_effective_driver(mode="auto")
        assert mode == "sidecar"
        assert isinstance(drv, SidecarOcrDriver)

    # 6. auto mode with unhealthy sidecar but OPENAI_API_KEY: a working local
    # engine wins over cloud egress (privacy reordering).
    import importlib.util

    with patch.object(SidecarOcrDriver, "is_healthy", return_value=False):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test")
        mode, drv = probe_effective_driver(mode="auto", allow_page_upload=True)
        if importlib.util.find_spec("rapidocr") is not None:
            assert mode == "rapidocr"
        else:
            assert mode == "vlm"
            assert isinstance(drv, CloudOcrDriver)

    # 6b. allow_page_upload=false: cloud/vlm must never be auto-selected;
    # only a local engine (or nothing) may answer.
    with patch.object(SidecarOcrDriver, "is_healthy", return_value=False):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test")
        mode, drv = probe_effective_driver(mode="auto", allow_page_upload=False)
        assert mode in (None, "rapidocr")

    # 6c. explicit cloud/vlm against a page-upload ban fails loudly.
    with pytest.raises(ValueError, match="UBT_ALLOW_PAGE_UPLOAD"):
        probe_effective_driver(mode="cloud", allow_page_upload=False)
    with pytest.raises(ValueError, match="UBT_ALLOW_PAGE_UPLOAD"):
        probe_effective_driver(mode="vlm", allow_page_upload=False)

    # 7. auto mode with UBT_VLM_DRIVER override
    class _CustomOcr:
        name = "custom-ocr"
        measured_boxes = False

        def recognize(
            self, image: object, page_size_pt: tuple[float, float], scale: float
        ) -> PageTranscript:
            return PageTranscript(lines=(), engine=self.name, measured_boxes=False)

    register_driver("test-custom", _CustomOcr)
    monkeypatch.setenv("UBT_VLM_DRIVER", "test-custom")
    mode, drv = probe_effective_driver(mode="auto")
    assert mode == "test-custom"


def test_docling_fallback_missing_pages_with_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pypdfium2 as pdfium

    # Create a 2-page test PDF where page 2 has no text
    pdf_path = tmp_path / "test_missing.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 300)
    doc.new_page(200, 300)
    doc.save(str(pdf_path))
    doc.close()

    blk1 = IRBlock(
        id="b1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Page 1 text",
        bbox=BoundingBox(page=1, x0=10.0, y0=10.0, x1=100.0, y1=50.0),
        provenance={"parser": "docling"},
    )

    monkeypatch.setenv("UBT_VLM_SCAN_FALLBACK", "missing")

    # 1. If OCR driver probe returns None (no sidecar, no cloud key)
    with patch("ubt.adapters.pdf.vlm.registry.probe_effective_driver", return_value=(None, None)):
        res = DoclingPDFAdapter._vlm_fallback_missing_pages(pdf_path, [blk1], ocr_mode="off")
        assert len(res) == 1
        assert res[0].id == "b1"

    # 2. When mock Sidecar is healthy
    class _FakeSidecar:
        name = "sidecar:http://localhost:8765"
        measured_boxes = True

        def recognize(
            self, image: object, page_size_pt: tuple[float, float], scale: float
        ) -> PageTranscript:
            return PageTranscript(
                lines=(
                    VlmLine(
                        text="Sidecar transcribed page 2",
                        reading_index=0,
                        measured_box=(10.0, 10.0, 150.0, 40.0),
                    ),
                ),
                engine="sidecar",
                measured_boxes=True,
            )

    fake_sidecar = _FakeSidecar()
    with patch(
        "ubt.adapters.pdf.vlm.registry.probe_effective_driver",
        return_value=("sidecar", fake_sidecar),
    ):
        res = DoclingPDFAdapter._vlm_fallback_missing_pages(pdf_path, [blk1], ocr_mode="sidecar")
        assert len(res) == 2
        assert res[0].id == "b1"
        assert res[1].bbox is not None and res[1].bbox.page == 2
        assert "Sidecar transcribed page 2" in res[1].source_text
        assert res[1].provenance["parser"] == "vlm:sidecar"


def _multi_page_pdf(tmp_path: Path, pages: int) -> Path:
    import pypdfium2 as pdfium

    pdf_path = tmp_path / f"scan_{pages}p.pdf"
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(200, 300)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_vlm_fallback_fail_loud_on_empty_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scanned PDF with OCR off/unavailable must raise, not export an empty book.

    User-visible failure prevented: the old contract handed
    zero blocks downstream, every stage "succeeded" on nothing, and --strict
    counters all read zero while the job reported completion.
    """
    pdf_path = _multi_page_pdf(tmp_path, 2)
    # Fallback tier on but no driver reachable: loud failure, not silence.
    monkeypatch.setenv("UBT_VLM_SCAN_FALLBACK", "missing")
    with (
        patch("ubt.adapters.pdf.vlm.registry.probe_effective_driver", return_value=(None, None)),
        pytest.raises(DocumentParseError, match="No content could be parsed"),
    ):
        DoclingPDFAdapter._vlm_fallback_missing_pages(pdf_path, [], ocr_mode="off")
    # Tier off entirely (the default): same loud failure at the same seam.
    monkeypatch.delenv("UBT_VLM_SCAN_FALLBACK", raising=False)
    with pytest.raises(DocumentParseError, match="No content could be parsed"):
        DoclingPDFAdapter._vlm_fallback_missing_pages(pdf_path, [])


def test_vlm_proofread_keeps_table_image_formula_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A weak/all upgrade must not delete the originals it cannot re-create.

    User-visible failure prevented: proofread used to swap
    the whole page for VLM narrative blocks, vaporizing TABLE/IMAGE/FORMULA
    blocks together with their skip flags — tables evaporated and upgraded
    titles were re-typed as body text.
    """

    class _FakeProofread:
        name = "fake-proofread"
        measured_boxes = True

        def recognize(
            self, image: object, page_size_pt: tuple[float, float], scale: float
        ) -> PageTranscript:
            return PageTranscript(
                lines=(
                    VlmLine(
                        text="re-segmented body",
                        reading_index=0,
                        measured_box=(10.0, 10.0, 150.0, 40.0),
                    ),
                ),
                engine="fake-proofread",
                measured_boxes=True,
            )

    def _blk(bid: str, btype: BlockType, text: str, skip: bool = False) -> IRBlock:
        return IRBlock(
            id=bid,
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=btype,
            source_text=text,
            skip_translate=skip,
            bbox=BoundingBox(page=1, x0=10.0, y0=10.0, x1=100.0, y1=50.0),
            provenance={"parser": "docling"},
        )

    blocks = [
        _blk("t1", BlockType.NARRATIVE, "Body paragraph."),
        _blk("t2", BlockType.TABLE, "| a | b |"),
        _blk("t3", BlockType.IMAGE, "[figure]", skip=True),
        _blk("t4", BlockType.FORMULA, "E=mc^2"),
    ]
    # A real (empty) PDF: the function probes it with pdfium before the
    # upgrade loop; transcription itself is patched below.
    pdf_path = _multi_page_pdf(tmp_path, 1)
    monkeypatch.setenv("UBT_VLM_SCAN_FALLBACK", "weak")
    # Deterministic upgrade trigger (the real debris heuristic depends on
    # policy thresholds): force every page to look formula-heavy.
    monkeypatch.setattr("ubt.core.policy.layout_policy.formula_debris_share", lambda _t: 1.0)
    fresh_blocks = [
        IRBlock(
            id="vlm#v0010000",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="re-segmented body",
            bbox=BoundingBox(page=1, x0=10.0, y0=10.0, x1=150.0, y1=40.0),
            provenance={"parser": "vlm:fake-proofread"},
        )
    ]
    with (
        patch(
            "ubt.adapters.pdf.vlm.registry.probe_effective_driver",
            return_value=("fake-proofread", _FakeProofread()),
        ),
        patch(
            "ubt.adapters.pdf.vlm.transcribe.transcribe_page_to_blocks",
            return_value=(fresh_blocks, None),
        ),
    ):
        res = DoclingPDFAdapter._vlm_fallback_missing_pages(blocks=blocks, path=pdf_path)
    assert sum(b.block_type == BlockType.TABLE for b in res) == 1
    assert sum(b.block_type == BlockType.IMAGE for b in res) == 1
    assert sum(b.block_type == BlockType.FORMULA for b in res) == 1
    kept_image = next(b for b in res if b.block_type == BlockType.IMAGE)
    assert kept_image.id == "t3" and kept_image.skip_translate
    assert any(
        b.block_type == BlockType.NARRATIVE and b.provenance.get("parser", "").startswith("vlm:")
        for b in res
    )


def test_vlm_circuit_breaker_stops_billing_after_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Three failed paid calls must break the loop, not bill all N pages.

    User-visible failure prevented: a dead endpoint or
    revoked key burned its provider timeout once per page across a whole
    scanned book, with no document-level stop.
    """
    pdf_path = _multi_page_pdf(tmp_path, 6)
    monkeypatch.setenv("UBT_VLM_SCAN_FALLBACK", "missing")
    with (
        patch(
            "ubt.adapters.pdf.vlm.registry.probe_effective_driver", return_value=("boom", object())
        ),
        patch(
            "ubt.adapters.pdf.vlm.transcribe.transcribe_page_to_blocks",
            side_effect=RuntimeError("endpoint down"),
        ) as boom,
        caplog.at_level(logging.ERROR, logger="ubt.adapters.pdf.docling_parser"),
        pytest.raises(DocumentParseError, match="No content could be parsed"),
    ):
        DoclingPDFAdapter._vlm_fallback_missing_pages(pdf_path, [], ocr_mode="sidecar")
    assert boom.call_count == 3
    assert any("circuit breaker" in r.message for r in caplog.records)


def test_auto_mode_warns_before_selecting_paid_engine(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A silent auto pick of a paid vision model is a surprise bill."""
    import sys

    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test")
    monkeypatch.delenv("UBT_VLM_DRIVER", raising=False)
    monkeypatch.delenv("UBT_OCR_ENDPOINT", raising=False)
    monkeypatch.setattr(SidecarOcrDriver, "is_healthy", staticmethod(lambda *a, **k: False))
    # Force the local-rapidocr probe to miss (this venv has it installed);
    # an import guard sees a None module entry as ImportError.
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    with caplog.at_level(logging.WARNING, logger="ubt.adapters.pdf.vlm.registry"):
        # Uploads opted in: this case is about the surprise-bill warning, not
        # about the egress gate (that pairing has its own case above).
        mode, _drv = probe_effective_driver(mode="auto", allow_page_upload=True)
    assert mode == "vlm"
    assert any("PAID" in r.message for r in caplog.records)


def test_probe_default_forbids_page_egress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A caller with no opinion must not get "ship the pages".

    ``allow_page_upload`` used to default to True, so every new call site
    opted the book into cloud egress by omission — the opposite of the
    UBTConfig default and of the privacy review the ordering already follows.
    """
    import sys

    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test")
    monkeypatch.delenv("UBT_VLM_DRIVER", raising=False)
    monkeypatch.delenv("UBT_OCR_ENDPOINT", raising=False)
    monkeypatch.setattr(SidecarOcrDriver, "is_healthy", staticmethod(lambda *a, **k: False))
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    with pytest.raises(ValueError, match="UBT_ALLOW_PAGE_UPLOAD"):
        probe_effective_driver(mode="vlm")
    with pytest.raises(ValueError, match="UBT_ALLOW_PAGE_UPLOAD"):
        probe_effective_driver(mode="cloud")
    assert probe_effective_driver(mode="auto") == (None, None)


def test_probe_effective_driver_honours_the_configured_ocr_model() -> None:
    """The model the quote prices must be the model the channel bills.

    ``UBTConfig.ocr_model`` reaches the driver through ``AdapterRuntimeConfig``
    the same way ``ocr_endpoint`` does. A driver that read only ``os.environ``
    could quote one model and pay for another, because a ``.env``-only setting
    never lands in the environment (pydantic-settings reads the file, the
    process does not).
    """
    _, drv = probe_effective_driver(mode="vlm", model="org/vision-x", allow_page_upload=True)
    assert isinstance(drv, CloudOcrDriver)
    assert drv.model == "org/vision-x"

    # No pick given: the driver's own env/default fallback stays in charge.
    _, default_drv = probe_effective_driver(mode="vlm", allow_page_upload=True)
    assert isinstance(default_drv, CloudOcrDriver)
    assert default_drv.model == DEFAULT_VISION_MODEL


def test_config_ocr_model_default_matches_the_driver() -> None:
    """A default that drifts from the driver's would silently mis-quote OCR."""
    from ubt.core.config import UBTConfig

    assert UBTConfig.model_fields["ocr_model"].default == DEFAULT_VISION_MODEL


def test_ocr_usage_books_the_run_sink_from_a_worker_thread() -> None:
    """OCR tokens must reach the run's usage sink across the thread hop.

    The sink is a ``ContextVar`` and the OCR driver runs in a worker thread, so
    this books through the same hop the adapter uses (``asyncio.to_thread``,
    which copies the context — ``loop.run_in_executor`` does not, and the
    channel would then bill invisibly).
    """
    import asyncio

    from ubt.adapters.pdf.vlm.drivers.cloud_driver import _record_ocr_usage
    from ubt.core.router.provider import attach_usage_sink

    sink = attach_usage_sink()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        await asyncio.to_thread(
            _record_ocr_usage,
            DEFAULT_VISION_MODEL,
            {"prompt_tokens": 1200, "completion_tokens": 90},
        )
        # A 200 with no usage block is unmeasured, not free.
        await asyncio.to_thread(_record_ocr_usage, "unreported-vision-v9", None)
        # And the executor-shaped hop is the bug: prove it sees nothing.
        seen: list[dict[str, dict[str, int]] | None] = []
        await loop.run_in_executor(None, lambda: seen.append(_peek_sink()))
        assert seen == [None]

    asyncio.run(_run())

    assert sink[DEFAULT_VISION_MODEL]["prompt_tokens"] == 1200
    assert "unmeasured_calls" not in sink[DEFAULT_VISION_MODEL]
    assert sink["unreported-vision-v9"]["unmeasured_calls"] == 1


def _peek_sink() -> dict[str, dict[str, int]] | None:
    from ubt.core.router.provider import _usage_sink

    return _usage_sink.get()


def _runtime_cfg(**overrides: object) -> object:
    from ubt.core.ports import AdapterRuntimeConfig

    base: dict[str, object] = {
        "ocr_mode": "vlm",
        "ocr_endpoint": None,
        "ocr_api_key": "",
        "ocr_model": "org/vision-x",
        "formula_enrichment": "auto",
        "render_engine": "auto",
        "formula_render": "witness",
        "font_family": None,
        "math_backend": "typst",
        "allow_page_upload": True,
    }
    return AdapterRuntimeConfig(**{**base, **overrides})  # type: ignore[arg-type]


def test_adapter_forwards_the_runtime_ocr_model_to_the_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config → adapter → parser → probe, with no second environment read.

    The assessor and the spend pre-flight price ``UBTConfig.ocr_model``, so the
    driver that bills must be the one the runtime config named.
    """
    import pypdfium2 as pdfium

    pdf_path = tmp_path / "scan.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 300)
    doc.new_page(200, 300)
    doc.save(str(pdf_path))
    doc.close()

    adapter = DoclingPDFAdapter()
    adapter.apply_config(_runtime_cfg())  # type: ignore[arg-type]
    assert adapter.ocr_model == "org/vision-x"

    captured: dict[str, object] = {}

    def _record_probe(**kwargs: object) -> tuple[None, None]:
        captured.update(kwargs)
        return None, None

    monkeypatch.setenv("UBT_VLM_SCAN_FALLBACK", "missing")
    monkeypatch.setattr("ubt.adapters.pdf.vlm.registry.probe_effective_driver", _record_probe)
    blk = IRBlock(
        id="b1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Page 1 text",
        bbox=BoundingBox(page=1, x0=10.0, y0=10.0, x1=100.0, y1=50.0),
    )
    DoclingPDFAdapter._vlm_fallback_missing_pages(
        pdf_path,
        [blk],
        "vlm",
        None,
        None,
        adapter.ocr_model,
        None,
        True,
    )
    assert captured["model"] == "org/vision-x"


def test_cloud_ocr_driver_accepts_extra_headers() -> None:
    """Extra headers must be stored and merged into request headers."""
    driver = CloudOcrDriver(
        endpoint="https://ocr.example.com/v1",
        api_key="sk-test",
        extra_headers={"X-Custom-Auth": "secret", "X-Trace-Id": "123"},
    )
    assert driver.extra_headers == {"X-Custom-Auth": "secret", "X-Trace-Id": "123"}


def test_ocr_unavailable_hint_does_not_offer_vision_llm_for_scans() -> None:
    from ubt.adapters.pdf.docling_parser import _ocr_unavailable_hint

    hint = _ocr_unavailable_hint("auto")
    # The numbered remediation options are the "offers"; none may be the
    # vision-LLM mode, which yields text but no geometry and so cannot
    # transcribe a scan.
    offered = "\n".join(ln for ln in hint.splitlines() if ln.strip()[:1].isdigit())
    assert "--ocr vlm" not in offered
    # A measured-box engine must be named instead.
    assert "rapidocr" in offered
    assert "sidecar" in offered
    assert "--ocr cloud" in offered
    # And the hint must say why the vision route is not the answer.
    assert "cannot transcribe a scanned page" in hint


def test_unmeasured_vlm_driver_is_not_scan_capable() -> None:
    from ubt.adapters.pdf.docling_parser import _driver_can_transcribe_scans
    from ubt.adapters.pdf.vlm.drivers.cloud_driver import CloudOcrDriver

    assert _driver_can_transcribe_scans(CloudOcrDriver(provider="vlm")) is False
    assert _driver_can_transcribe_scans(CloudOcrDriver(provider="cloud")) is True


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
def test_cloud_driver_measured_boxes_synchronized_in_init() -> None:
    # When endpoint is OpenAI or chat/completions, driver is vision LLM (measured_boxes=False)
    driver = CloudOcrDriver(endpoint="https://api.openai.com/v1")
    assert driver.measured_boxes is False, (
        "OpenAI endpoint must set measured_boxes=False in __init__"
    )


_r0921_SRC = "The quick brown fox jumps over the lazy dog near the river bank."


def _r0921_capture_probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    from ubt.adapters.pdf.vlm import registry

    seen: dict[str, object] = {}

    def _probe(
        mode: str = "auto",
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        allow_page_upload: bool = False,
    ) -> tuple[None, None]:
        seen["allow_page_upload"] = allow_page_upload
        seen["model"] = model
        return (None, None)

    monkeypatch.setattr(registry, "probe_effective_driver", _probe)
    return seen


def _r0921_page_two_block() -> IRBlock:
    """One block on page 2, so page 1 is "missing" but the book isn't empty."""
    return IRBlock(
        id="b2",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text=_r0921_SRC,
        bbox=BoundingBox(page=2, x0=0.0, y0=0.0, x1=1.0, y1=1.0),
    )


def _r0921_two_page_pdf(tmp_path: Path) -> Path:
    import pypdf

    pdf = tmp_path / "two_pages.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with pdf.open("wb") as fh:
        writer.write(fh)
    return pdf


def test_dry_run_closes_every_page_egress_path() -> None:
    """A rehearsal must not make a real cloud call off the operator's config.

    ``ocr_mode``/``allow_page_upload``/``visual_judge_enabled`` are read from
    the adapter config, not the router, so a "zero-spend" dry run with OCR on
    still called the cloud endpoint with the operator's key and uploaded
    manuscript pages.
    """
    from ubt.core.engine.dry_run import create_dry_run_orchestrator

    config = UBTConfig(ocr_mode="vlm", allow_page_upload=True, visual_judge_enabled=True)
    orchestrator = create_dry_run_orchestrator(config)

    assert orchestrator.config.ocr_mode == "off"
    assert orchestrator.config.allow_page_upload is False
    assert orchestrator.config.visual_judge_enabled is False
    # The caller's config is untouched — the rehearsal works on a copy.
    assert config.ocr_mode == "vlm"
    assert config.allow_page_upload is True
    assert config.visual_judge_enabled is True


def test_page_upload_gate_prefers_the_resolved_config_over_the_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.adapters.pdf import docling_parser

    pdf = _r0921_two_page_pdf(tmp_path)
    seen = _r0921_capture_probe(monkeypatch)
    monkeypatch.setenv("UBT_VLM_SCAN_FALLBACK", "missing")
    monkeypatch.setenv("UBT_ALLOW_PAGE_UPLOAD", "true")

    docling_parser.vlm_fallback_missing_pages(
        pdf, [_r0921_page_two_block()], ocr_mode="vlm", allow_page_upload=False
    )

    assert seen["allow_page_upload"] is False, "env must not override the resolved config"
    assert seen["model"] is None, "no model configured must not invent one"


def test_page_upload_gate_falls_back_to_env_without_a_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``None`` is the direct-caller path (tests, library use): keep the env read."""
    from ubt.adapters.pdf import docling_parser

    pdf = _r0921_two_page_pdf(tmp_path)
    seen = _r0921_capture_probe(monkeypatch)
    monkeypatch.setenv("UBT_VLM_SCAN_FALLBACK", "missing")
    monkeypatch.setenv("UBT_ALLOW_PAGE_UPLOAD", "true")

    docling_parser.vlm_fallback_missing_pages(pdf, [_r0921_page_two_block()], ocr_mode="vlm")

    assert seen["allow_page_upload"] is True

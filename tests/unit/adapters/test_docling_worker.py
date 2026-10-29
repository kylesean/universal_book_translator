"""Unit tests for the isolated Docling worker process pool and warm converter lifecycle."""

from __future__ import annotations

import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.docling_parser import (
    cleanup_in_process_docling_converter,
    extract_with_docling,
)
from ubt.adapters.pdf.docling_worker import (
    DoclingWorkerPool,
    _worker_convert_task,
    close_docling_worker_pool,
    get_docling_worker_pool,
    is_docling_process_isolation_enabled,
)

# ``Paragraph``/``Span`` are defined in ``ubt.model`` and only re-imported by
# ``ubt.core.ir.models``; importing them from the defining module keeps the
# dependency explicit instead of relying on an incidental re-export.
from ubt.core.ir.models import IRBlock
from ubt.core.ports import AdapterRuntimeConfig
from ubt.model.ast import Paragraph
from ubt.model.span import Span

pytestmark = pytest.mark.fast


def test_is_docling_process_isolation_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UBT_DOCLING_ISOLATE_PROCESS", raising=False)
    assert is_docling_process_isolation_enabled() is True

    monkeypatch.setenv("UBT_DOCLING_ISOLATE_PROCESS", "0")
    assert is_docling_process_isolation_enabled() is False

    monkeypatch.setenv("UBT_DOCLING_ISOLATE_PROCESS", "false")
    assert is_docling_process_isolation_enabled() is False

    monkeypatch.setenv("UBT_DOCLING_ISOLATE_PROCESS", "1")
    assert is_docling_process_isolation_enabled() is True


def test_docling_worker_pool_lifecycle() -> None:
    pool = DoclingWorkerPool()
    assert pool._closed is False
    assert pool._executor is None

    # Getting executor creates it
    executor = pool._ensure_executor()
    assert isinstance(executor, concurrent.futures.ProcessPoolExecutor)
    assert pool._ensure_executor() is executor

    # Closing shuts down executor
    pool.close()
    assert pool._closed is True
    assert pool._executor is None

    # Re-using closed pool raises RuntimeError
    with pytest.raises(RuntimeError, match="DoclingWorkerPool is closed"):
        pool._ensure_executor()


def test_docling_worker_pool_singleton() -> None:
    close_docling_worker_pool()
    pool1 = get_docling_worker_pool()
    pool2 = get_docling_worker_pool()
    assert pool1 is pool2

    close_docling_worker_pool()
    assert pool1._closed is True

    pool3 = get_docling_worker_pool()
    assert pool3 is not pool1
    close_docling_worker_pool()


def test_docling_worker_task_invalidation(tmp_path: Path) -> None:
    import ubt.adapters.pdf.docling_worker as dw

    dw._WORKER_CONVERTER = MagicMock()
    dw._WORKER_CONVERTER_KEY = (False, "cpu", 0.08)

    mock_doc = MagicMock()
    mock_converter = MagicMock()

    with patch(
        "ubt.adapters.pdf.docling_parser.convert_docling_document",
        return_value=(mock_doc, mock_converter, "fake-cache-key-123"),
    ) as mock_convert:
        # Run task with different key (enrich=True) -> should invalidate old converter
        key = _worker_convert_task(
            path_str=str(tmp_path / "doc.pdf"),
            page_range=(1, 10),
            enrich=True,
            device="cuda",
            formula_crop_margin=0.08,
            file_sha256="abc1234",
        )
        assert key == "fake-cache-key-123"
        # Notice existing_converter passed to convert_docling_document was None because it was invalidated
        assert mock_convert.call_args.kwargs["existing_converter"] is None
        assert dw._WORKER_CONVERTER is mock_converter
        assert dw._WORKER_CONVERTER_KEY == (True, "cuda", 0.08)

        # Second call with same key -> reuses warm converter
        key2 = _worker_convert_task(
            path_str=str(tmp_path / "doc.pdf"),
            page_range=(11, 20),
            enrich=True,
            device="cuda",
            formula_crop_margin=0.08,
            file_sha256="abc1234",
        )
        assert key2 == "fake-cache-key-123"
        assert mock_convert.call_args.kwargs["existing_converter"] is mock_converter

    # Clean up worker globals
    dw._WORKER_CONVERTER = None
    dw._WORKER_CONVERTER_KEY = None


def test_docling_worker_pool_crash_handling(tmp_path: Path) -> None:
    pool = DoclingWorkerPool()
    mock_executor = MagicMock()
    mock_future = MagicMock()
    mock_future.result.side_effect = BrokenProcessPool("Process died with SIGSEGV")
    mock_executor.submit.return_value = mock_future
    pool._executor = mock_executor

    with pytest.raises(BrokenProcessPool):
        pool.convert(
            path=tmp_path / "test.pdf",
            page_range=(1, 5),
            enrich=False,
            device="cpu",
            formula_crop_margin=0.08,
            file_sha256="sha-xyz",
        )
    # Pool marked itself closed on BrokenProcessPool
    assert pool._closed is True


def test_extract_with_docling_worker_fallback(tmp_path: Path) -> None:
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    fake_block = IRBlock(
        element=Paragraph(id="p1", spine_index=1, span=Span(page=1), text="Fallback text")
    )

    # Mock pool to simulate crash, and mock in-process convert_docling_document to succeed
    mock_pool = MagicMock()
    mock_pool.convert.side_effect = RuntimeError("Worker process failed")

    with (
        patch("ubt.adapters.pdf.docling_worker.get_docling_worker_pool", return_value=mock_pool),
        patch(
            "ubt.adapters.pdf.docling_parser.convert_docling_document",
            return_value=(MagicMock(), MagicMock(), "in-process-key"),
        ),
        patch(
            "ubt.adapters.pdf.docling_parser.map_docling_document_to_blocks",
            return_value=[fake_block],
        ),
    ):
        blocks = extract_with_docling(
            path=pdf,
            page_range=None,
            enrich=False,
            isolate_process=True,
        )
        assert len(blocks) == 1
        # ``IRBlock.source_text`` dispatches by element class; ``element.text``
        # exists only on prose and mypy cannot narrow the union here.
        assert blocks[0].source_text == "Fallback text"


def test_docling_adapter_lifecycle_and_teardown() -> None:
    adapter = DoclingPDFAdapter()
    assert adapter.docling_isolate_process is True

    # apply_config updates docling_isolate_process
    runtime_config = AdapterRuntimeConfig(
        ocr_mode="off",
        ocr_endpoint=None,
        ocr_api_key="",
        ocr_model="",
        formula_enrichment="off",
        formula_crop_margin=0.08,
        toc_mode="off",
        docling_device="cpu",
        font_family=None,
        allow_page_upload=False,
        docling_isolate_process=False,
    )
    adapter.apply_config(runtime_config)
    assert adapter.docling_isolate_process is False

    with (
        patch("ubt.adapters.pdf.docling_adapter.close_docling_worker_pool") as mock_close_pool,
        patch(
            "ubt.adapters.pdf.docling_adapter.cleanup_in_process_docling_converter"
        ) as mock_cleanup,
    ):
        adapter.close()
        mock_close_pool.assert_called_once()
        mock_cleanup.assert_called_once()


def test_in_process_converter_cleanup() -> None:
    import ubt.adapters.pdf.docling_parser as dp

    mock_conv = MagicMock()
    dp._WARM_IN_PROCESS_CONVERTER = mock_conv
    dp._WARM_IN_PROCESS_KEY = (True, "cpu", 0.08)

    with patch("ubt.adapters.pdf.docling_parser._cleanup_docling_converter") as mock_cleanup:
        cleanup_in_process_docling_converter()
        mock_cleanup.assert_called_once_with(mock_conv)
        assert dp._WARM_IN_PROCESS_CONVERTER is None
        assert dp._WARM_IN_PROCESS_KEY is None

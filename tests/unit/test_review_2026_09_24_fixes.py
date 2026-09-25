"""Failure-first (RED -> GREEN) regression tests for the 2026-09-24 comprehensive review fixes."""

from __future__ import annotations

import asyncio
import io
import sqlite3
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.adapters.epub.adapter import EPUBAdapter
from ubt.adapters.pdf.diagram_localizer import DiagramLocalizer
from ubt.adapters.pdf.docling_parser import vlm_fallback_missing_pages
from ubt.api.app import create_app
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.ledger_flusher import CheckpointBatchFlusher
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.exceptions import UBTError
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    BoundingBox,
    ChapterMeta,
    DocumentIR,
    FlowID,
    IRBlock,
)
from ubt.core.job_options import overrides_from_request
from ubt.core.memory.hierarchical_memory import EpochSnapshot, HierarchicalMemoryManager
from ubt.core.router.capabilities import ModelProfile, PromptStrategy
from ubt.core.router.rate_limiter import SqliteTokenBucket
from ubt.core.router.registry import ModelCapabilityRegistry
from ubt.core.validators.consistency import canonicalize_numeric_token


@pytest.mark.fast
@pytest.mark.asyncio
async def test_flusher_applies_backoff_on_transient_failure_and_drains_on_task_error(
    tmp_path: Path,
) -> None:
    """[CRITICAL-T1-1] Flusher must backoff between consecutive failures and
    close() must drain remaining items even if the background task died."""
    ledger = SQLiteJobLedger(tmp_path / "ledger.sqlite")
    doc = DocumentIR(
        doc_id="doc_backoff",
        source_path="/tmp/test.epub",
        format_type="epub",
        metadata={},
        blocks=[
            IRBlock(id="b_001", flow_id=FlowID.MAIN_STORY, spine_index=1, source_text="One"),
            IRBlock(id="b_002", flow_id=FlowID.MAIN_STORY, spine_index=2, source_text="Two"),
        ],
    )
    ledger.init_job("job_backoff", doc, target_lang="zh")
    original_save = ledger.save_checkpoints_batch

    # Part 1: Verify retry backoff delay is > 0 when _save fails transiently
    flusher = CheckpointBatchFlusher(ledger, flush_interval=0.01, max_batch_size=10)
    assert getattr(flusher, "_retry_base_delay", 0.0) > 0.0, (
        "CheckpointBatchFlusher must define a positive _retry_base_delay for transient SQLite failures"
    )

    # Part 2: Simulate background task dying with RuntimeError, then ledger recovering
    # before close() is called with additional pending items in the queue.
    fail_now = True

    def controlled_save(updates: list[dict[str, Any]], **kwargs: Any) -> int:
        if fail_now:
            raise sqlite3.OperationalError("database is locked")
        return original_save(updates, **kwargs)

    ledger.__dict__["save_checkpoints_batch"] = controlled_save
    await flusher.enqueue(
        {"block_id": "b_001", "target_text": "译文1", "status": BlockStatus.DRAFTED}
    )

    # Wait for background task to hit _MAX_CONSECUTIVE_FAILURES and terminate
    assert flusher._flusher_task is not None
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(asyncio.shield(flusher._flusher_task), timeout=2.0)

    # Now the ledger recovers before close() is called
    fail_now = False
    with pytest.raises(RuntimeError):
        await flusher.close()

    # Even though close() surfaced the task RuntimeError, pending items must ALREADY be drained and saved!
    blocks = {b.id: b for b in ledger.get_all_blocks("job_backoff")}
    assert blocks["b_001"].target_text == "译文1"
    ledger.close()


@pytest.mark.fast
@pytest.mark.asyncio
async def test_sqlite_token_bucket_has_async_thread_offloaded_reporters(tmp_path: Path) -> None:
    """[CRITICAL-T3-1] SqliteTokenBucket must provide report_success_async and report_429_async
    that offload synchronous SQLite transactions via asyncio.to_thread."""
    bucket = SqliteTokenBucket(tmp_path / "rate.sqlite", initial_rpm=60, initial_tpm=60000)
    assert hasattr(bucket, "report_success_async"), (
        "Missing report_success_async on SqliteTokenBucket"
    )
    assert hasattr(bucket, "report_429_async"), "Missing report_429_async on SqliteTokenBucket"

    with patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy_to_thread:
        await bucket.report_429_async()
        await bucket.report_success_async()
        assert spy_to_thread.call_count == 2
    bucket.close()


@pytest.mark.fast
@pytest.mark.asyncio
async def test_sse_stream_releases_global_slot_when_response_not_iterated(tmp_path: Path) -> None:
    """[CRITICAL-T4-1] StreamingResponse returned by /jobs/{job_id}/stream must attach a
    BackgroundTask / slot guard so slots are released even if the client disconnects
    before iterating body_iterator."""
    cfg = UBTConfig(db_dir=tmp_path / "ledgers")
    app = create_app(config=cfg)
    # Locate the stream_progress route handler
    stream_route: Any = next(
        r for r in app.routes if getattr(r, "path", None) == "/jobs/{job_id}/stream"
    )
    endpoint: Any = stream_route.endpoint

    # Register a dummy active job in the app's JobManager
    closure_vars = {
        name: cell.cell_contents
        for name, cell in zip(
            endpoint.__code__.co_freevars, endpoint.__closure__ or (), strict=False
        )
    }
    manager = closure_vars["manager"]
    global_subscribers = closure_vars["global_subscribers"]

    record = manager.create_job("job_slot_leak_test")
    assert record is not None

    mock_req = MagicMock()
    mock_req.is_disconnected = AsyncMock(return_value=True)

    resp = await endpoint(job_id=record.job_id, request=mock_req, x_ubt_tenant=None)
    assert resp.background is not None, (
        "StreamingResponse must attach a BackgroundTask to release subscriber slots on abort"
    )
    # Execute background cleanup without ever iterating resp.body_iterator
    await resp.background()
    assert sum(getattr(global_subscribers, "_counts", {}).values()) == 0
    assert len(record.subscribers) == 0


@pytest.mark.fast
def test_service_api_key_is_secret_str_and_masked_in_repr() -> None:
    """[HIGH-T4-3] UBTConfig.service_api_key must be SecretStr so repr() and str() never leak it."""
    cfg = UBTConfig(service_api_key=SecretStr("top-secret-inbound-gate-key"))
    assert isinstance(cfg.service_api_key, SecretStr)
    assert cfg.service_api_key.get_secret_value() == "top-secret-inbound-gate-key"
    assert "top-secret-inbound-gate-key" not in repr(cfg)
    assert "top-secret-inbound-gate-key" not in str(cfg)


@pytest.mark.fast
def test_model_profiles_post_forbids_overriding_builtin_profiles() -> None:
    """[HIGH-T4-5] POST /api/v1/model-profiles must forbid overriding existing/built-in profiles
    (override=False -> 409 Conflict) and require verify_api_key when service_api_key is configured."""
    reg = ModelCapabilityRegistry()
    with pytest.raises(ValueError, match="already"):
        reg.register(
            ModelProfile(
                model_pattern="deepseek",
                prompt_strategy=PromptStrategy.MINIMAL,
            ),
            override=False,
        )

    app = create_app(config=UBTConfig(service_api_key=SecretStr("gate-secret-123")))
    client = TestClient(app)

    # Unauthenticated request must be rejected with 401
    res_unauth = client.post(
        "/api/v1/model-profiles",
        json={"model_pattern": "custom-new-model", "prompt_strategy": "minimal"},
    )
    assert res_unauth.status_code == 401

    # Authenticated attempt to override built-in 'deepseek' must be rejected with 409 Conflict
    res_conflict = client.post(
        "/api/v1/model-profiles",
        headers={"X-API-Key": "gate-secret-123"},
        json={"model_pattern": "deepseek", "prompt_strategy": "minimal"},
    )
    assert res_conflict.status_code == 409


@pytest.mark.fast
def test_vlm_fallback_missing_pages_closes_driver_and_logs_correct_remaining_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """[HIGH-T2-1 & MEDIUM-T2-4] vlm_fallback_missing_pages must call driver.close() in finally
    and log len(missing) - idx (not len(missing) - page_no + 1) when circuit breaker trips."""
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    mock_driver = MagicMock()
    mock_driver.close = MagicMock()

    existing_block = IRBlock(
        id="b1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Existing page 1 text",
        bbox=BoundingBox(page=1, x0=10, y0=10, x1=100, y1=50),
    )

    import pypdfium2 as pdfium

    from ubt.adapters.pdf.vlm.transcribe import VlmFallbackMode

    pdf = pdfium.PdfDocument.new()
    for _ in range(53):
        pdf.new_page(width=200, height=200)
    pdf.save(str(pdf_path))
    pdf.close()

    with (
        patch(
            "ubt.adapters.pdf.vlm.transcribe.get_fallback_mode",
            return_value=VlmFallbackMode.MISSING,
        ),
        patch(
            "ubt.adapters.pdf.vlm.registry.probe_effective_driver",
            return_value=("deepseek", mock_driver),
        ),
        patch(
            "ubt.adapters.pdf.vlm.transcribe.transcribe_page_to_blocks",
            side_effect=RuntimeError("VLM worker crashed"),
        ),
    ):
        result = vlm_fallback_missing_pages(
            pdf_path,
            [existing_block],
            ocr_mode="deepseek",
            page_range=(50, 53),
        )

    assert len(result) == 1
    mock_driver.close.assert_called_once()
    # Circuit breaker trips at idx=3 (page_no=53), so remaining pages is 4 - 3 = 1 (NOT 4 - 53 + 1 = -48!)
    assert "remaining 1 page(s)" in caplog.text


@pytest.mark.fast
def test_diagram_localizer_unescapes_xml_entities_in_pdftotext_bbox(tmp_path: Path) -> None:
    """[MEDIUM-T2-3] DiagramLocalizer.extract_text_spans must html.unescape XML entities."""
    localizer = DiagramLocalizer.__new__(DiagramLocalizer)
    localizer.glossary = {"r&d <5v>": "研发 <5V>"}
    sample_xml = (
        '<doc><page width="200" height="200">'
        '<word xMin="20.0" yMin="20.0" xMax="80.0" yMax="40.0">R&amp;D &lt;5V&gt;</word>'
        "</page></doc>"
    )
    with patch("subprocess.check_output", return_value=sample_xml):
        spans = localizer.extract_text_spans(
            tmp_path / "fig.pdf",
            page_no=1,
            bbox=BoundingBox(page=1, x0=10, y0=10, x1=190, y1=190),
            page_height=200.0,
        )
    assert len(spans) == 1
    assert spans[0].text == "R&D <5V>"


@pytest.mark.fast
def test_canonicalize_numeric_token_preserves_zero_leading_three_digit_decimals() -> None:
    """[HIGH-T3-2] 0.125 and 0.500 are decimals, never thousands-separated integers."""
    assert canonicalize_numeric_token("0.125") == "0.125"
    assert canonicalize_numeric_token("0.500") == "0.5"
    assert canonicalize_numeric_token("1.500") == "1500"
    assert canonicalize_numeric_token("1,500") == "1500"


@pytest.mark.fast
def test_hierarchical_memory_get_l3_summary_retains_multi_epoch_history() -> None:
    """[HIGH-T3-3] get_l3_summary must include earlier epochs (clamped to _MAX_L3_CHARS),
    not overwrite/discard earlier epochs when a new epoch closes."""
    mem = HierarchicalMemoryManager()
    mem._epochs = [
        EpochSnapshot(
            epoch_index=0,
            start_spine=0,
            end_spine=9,
            summary_text="Epoch 0: Alice enters the rabbit hole.",
        ),
        EpochSnapshot(
            epoch_index=1, start_spine=10, end_spine=19, summary_text="Epoch 1: The Mad Tea-Party."
        ),
    ]
    l3 = mem.get_l3_summary()
    assert "Epoch 0: Alice enters the rabbit hole." in l3
    assert "Epoch 1: The Mad Tea-Party." in l3


@pytest.mark.fast
@pytest.mark.asyncio
async def test_pipeline_orchestrator_marks_cancelled_on_keyboard_interrupt(tmp_path: Path) -> None:
    """[HIGH-T4-4] KeyboardInterrupt during pipeline execution must mark job status as 'cancelled'."""
    cfg = UBTConfig(db_dir=tmp_path / "ledgers")
    cfg.db_dir.mkdir(parents=True, exist_ok=True)
    input_file = tmp_path / "book.txt"
    input_file.write_text("Hello world\n", encoding="utf-8")

    orchestrator = PipelineOrchestrator(config=cfg)
    ledger = SQLiteJobLedger(cfg.db_dir / "job_kb_int.sqlite")
    manifest = BookManifest(
        doc_id="doc_kb",
        title="KB Test",
        source_path=str(input_file),
        chapters=[ChapterMeta(chapter_id="ch1", title="Ch1", spine_index=0)],
    )
    ledger.init_job_from_manifest("job_kb_int", manifest)
    ledger.close()

    async def raise_kb_interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt("user pressed Ctrl+C")
        yield  # make it an async generator

    with (
        patch("ubt.core.engine.pipeline.run_ingest_stage", side_effect=raise_kb_interrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        async for _ in orchestrator.run(
            input_path=input_file,
            output_path=tmp_path / "out.txt",
            job_id="job_kb_int",
        ):
            pass

    check_ledger = SQLiteJobLedger(cfg.db_dir / "job_kb_int.sqlite", read_only=True)
    assert check_ledger.get_job_status("job_kb_int") == "cancelled"
    check_ledger.close()


@pytest.mark.fast
def test_overrides_from_request_blocks_ocr_endpoint_and_epub_locates_single_quoted_opf() -> None:
    """[MEDIUM-T4-2 & MEDIUM-T2-5] overrides_from_request must block ocr_endpoint when
    allow_provider_keys=False, and EPUBAdapter._locate_opf must parse single-quoted full-path."""
    with pytest.raises(UBTError, match="ocr_endpoint"):
        overrides_from_request(
            {"ocr_endpoint": "http://169.254.169.254/latest"}, allow_provider_keys=False
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "META-INF/container.xml",
            "<?xml version='1.0'?><container><rootfiles>"
            "<rootfile full-path='OEBPS/content.opf' media-type='application/oebps-package+xml'/>"
            "</rootfiles></container>",
        )
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf:
        adapter = EPUBAdapter()
        assert adapter._locate_opf(zf) == "OEBPS/content.opf"


@pytest.mark.fast
def test_ledger_get_conn_initializes_inside_lock(tmp_path: Path) -> None:
    """[HIGH-T1-3] _get_conn must check and call _init_connection while holding self._lock."""
    ledger = SQLiteJobLedger(tmp_path / "lock_test.sqlite")
    ledger.close()
    assert ledger._conn is None

    lock_held_during_init: list[bool] = []
    orig_init = ledger._init_connection

    def checked_init() -> None:
        # RLock._is_owned() is True iff the current thread holds self._lock
        is_owned = getattr(ledger._lock, "_is_owned", lambda: False)()
        lock_held_during_init.append(bool(is_owned))
        orig_init()

    ledger.__dict__["_init_connection"] = checked_init
    with ledger._get_conn() as conn:
        assert conn is not None

    assert lock_held_during_init == [True], (
        "_init_connection must be called inside `with self._lock:` in _get_conn()"
    )
    ledger.close()


@pytest.mark.fast
def test_noise_aggregator_intercepts_child_logger_orphan_pdf_cell() -> None:
    """Child logger 'docling_ibm_models.tableformer.data_management.matching_post_processor'
    must not bypass NoiseAggregator during propagation."""
    import logging

    from ubt.core.log_aggregate import install_noise_aggregators, noise_aggregators
    from ubt.core.log_config import setup_logging

    stream = io.StringIO()
    setup_logging(level="INFO", stream=stream)
    install_noise_aggregators()
    agg = noise_aggregators()["table_structure_guess"]
    before_seen = agg.seen

    child_logger = logging.getLogger(
        "docling_ibm_models.tableformer.data_management.matching_post_processor"
    )
    for cell_id in range(2277, 2282):
        child_logger.warning(
            "Orphan pdf_cell %d recovered to col=1 by nearest-column fallback (row=34, x=877.6, dist=156.1)",
            cell_id,
        )

    assert agg.seen == before_seen + 5
    assert "Orphan pdf_cell 2278" not in stream.getvalue()


@pytest.mark.fast
def test_translate_cli_and_main_share_same_rich_console() -> None:
    """translate.py and main.py must share one Rich Console instance so RichHandler
    does not tear the Progress live bar."""
    import ubt.cli.commands.translate as translate_mod
    import ubt.cli.main as main_mod

    assert translate_mod.console is main_mod.console

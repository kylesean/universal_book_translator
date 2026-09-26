"""Unit and integration tests for FastAPI microservice layer."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from ubt.api.app import JobManager, JobRecord, JobSubmitRequest, create_app
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.progress import ProgressSnapshot
from ubt.core.ir.models import BookManifest, ChapterMeta
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


@pytest.fixture
def sample_api_doc(tmp_path: Path) -> Path:
    p = tmp_path / "api_doc.md"
    p.write_text("# Chapter 1\n\nShort paragraph for API testing.\n", encoding="utf-8")
    return p


@pytest.fixture
def api_client(tmp_path: Path) -> TestClient:
    db_dir = tmp_path / "api_ledgers"
    config = UBTConfig(db_dir=db_dir, rate_limit_rpm=600, allowed_dirs=str(tmp_path))
    app = create_app(config=config)
    return TestClient(app)


def test_health_check(api_client: TestClient) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "universal-book-translator"


def test_submit_job_missing_file_fails(api_client: TestClient) -> None:
    payload = {
        "input_path": "/nonexistent/path/nowhere.epub",
        "target_lang": "zh",
    }
    resp = api_client.post("/jobs/submit", json=payload)
    # Missing file outside the sandbox is denied (403) before existence (400).
    assert resp.status_code in (400, 403)
    assert "Source file not found" in resp.text or "Access denied" in resp.text


def test_submit_job_success_and_query_status(
    api_client: TestClient, sample_api_doc: Path, tmp_path: Path
) -> None:
    payload = {
        "input_path": str(sample_api_doc),
        "output_path": str(tmp_path / "api_out.md"),
        "target_lang": "zh",
    }
    resp = api_client.post("/jobs/submit", json=payload)
    assert resp.status_code == 202
    data = resp.json()
    job_id = data["job_id"]
    assert job_id.startswith("job_")
    assert "/stream" in data["stream_url"]
    assert "/status" in data["status_url"]

    # Query status
    status_resp = api_client.get(f"/jobs/{job_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    # A synchronous TestClient tears the job's asyncio task down between the
    # submit and the status query, so the task is legitimately cancelled.
    # P1-1/P1-10 record that as a terminal "cancelled" instead of leaving a
    # zombie "running". Anything but a failure is valid here; completion is
    # covered by the end-to-end pipeline tests, not this submit+query smoke.
    assert status_data["status"] in ("submitted", "running", "completed", "cancelled")


def test_status_nonexistent_job_returns_404(api_client: TestClient) -> None:
    resp = api_client.get("/jobs/ghost_job/status")
    assert resp.status_code == 404
    assert "Job not found" in resp.text


def test_report_and_download_nonexistent_job_returns_404(api_client: TestClient) -> None:
    resp1 = api_client.get("/jobs/ghost_job/report")
    assert resp1.status_code == 404

    resp2 = api_client.get("/jobs/ghost_job/download")
    assert resp2.status_code == 404


def test_visual_report_unknown_job_returns_404(api_client: TestClient) -> None:
    resp = api_client.get("/jobs/ghost_job/visual-report")
    assert resp.status_code == 404


def test_visual_report_no_report_returns_400(
    api_client: TestClient, sample_api_doc: Path, tmp_path: Path
) -> None:
    payload = {
        "input_path": str(sample_api_doc),
        "output_path": str(tmp_path / "api_out_vis.md"),
        "target_lang": "zh",
    }
    resp = api_client.post("/jobs/submit", json=payload)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    vis_resp = api_client.get(f"/jobs/{job_id}/visual-report")
    assert vis_resp.status_code == 400


def test_visual_report_ledger_fallback(tmp_path: Path) -> None:
    """ROI-5: restarted server (no memory record) serves the persisted ledger report."""
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import DocumentIR, FlowID, IRBlock

    db_dir = tmp_path / "api_ledgers_vis"
    db_dir.mkdir(parents=True, exist_ok=True)
    job_id = "job_vis_ledger"
    ledger = SQLiteJobLedger(db_dir / f"{job_id}.sqlite")
    doc_ir = DocumentIR(
        doc_id="doc_vis_ledger",
        source_path="test.pdf",
        format_type="pdf",
        blocks=[IRBlock(id="b1", flow_id=FlowID.MAIN_STORY, spine_index=1, source_text="hello")],
    )
    ledger.init_job(job_id, doc_ir, target_lang="zh")
    ledger.record_visual_report(job_id, {"passed": True, "findings": []})
    ledger.close()

    config = UBTConfig(db_dir=db_dir, rate_limit_rpm=600)
    client = TestClient(create_app(config=config))
    resp = client.get(f"/jobs/{job_id}/visual-report")
    assert resp.status_code == 200
    assert resp.json()["passed"] is True


def test_visual_report_rejected_path_falls_back_to_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sandbox-rejected stored path must not mask the ledger's report (X21).

    ``resolve_secure_path`` signals a rejected or missing path with
    ``HTTPException``, but the endpoint caught ``ValueError``/
    ``FileNotFoundError``. The exception therefore escaped, so the ledger
    fallback — which holds the report actually persisted — was unreachable and
    the job answered 400/403 instead of returning its report.
    """
    import importlib

    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import DocumentIR, FlowID, IRBlock

    app_module = importlib.import_module("ubt.api.app")

    class _Progress:
        # A path that cannot be resolved: resolve_secure_path raises 400 for it.
        visual_report_file = str(tmp_path / "gone.json")

    class _Record:
        progress = _Progress()

    class _StubManager(JobManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

        def get_job(self, job_id: str) -> Any:
            return _Record()

    db_dir = tmp_path / "vis_rejected_ledgers"
    db_dir.mkdir(parents=True, exist_ok=True)
    job_id = "job_vis_rejected"
    ledger = SQLiteJobLedger(db_dir / f"{job_id}.sqlite")
    doc_ir = DocumentIR(
        doc_id="doc_vis_rejected",
        source_path="test.pdf",
        format_type="pdf",
        blocks=[IRBlock(id="b1", flow_id=FlowID.MAIN_STORY, spine_index=1, source_text="hello")],
    )
    ledger.init_job(job_id, doc_ir, target_lang="zh")
    ledger.record_visual_report(job_id, {"passed": True, "findings": []})
    ledger.close()

    monkeypatch.setattr(app_module, "JobManager", _StubManager)
    config = UBTConfig(db_dir=db_dir, rate_limit_rpm=600)
    client = TestClient(create_app(config=config))
    resp = client.get(f"/jobs/{job_id}/visual-report")
    assert resp.status_code == 200
    assert resp.json()["passed"] is True


def test_only_the_exact_health_route_skips_auth(tmp_path: Path) -> None:
    """A path merely *ending* in ``/health`` must not bypass the API key (X33).

    The check matched with ``endswith("/health")``, so any future route named
    e.g. ``/jobs/{id}/health`` would have been unauthenticated by default. The
    probe route below is registered after ``create_app`` so it inherits the same
    app-level auth dependency every real route gets.
    """
    config = UBTConfig(db_dir=tmp_path / "health_ledgers", service_api_key=SecretStr("secret"))
    app = create_app(config=config)

    @app.get("/probe/health")
    async def _probe() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/probe/health").status_code in (401, 403)


@pytest.mark.asyncio
async def test_sse_stream_terminates_for_cancelled_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§10.3-#2: reconnecting /stream to a cancelled job must not hang forever.

    The event generator's terminal-state checks used to list only
    ("completed", "failed"); a job cancelled while still "submitted" (its task
    never ran, so execute_job's finally never pushed a sentinel) left a fresh
    subscriber spinning in the wait_for/continue loop until the client gave up.
    The preloaded manager gives a deterministic cancelled record; the stream must
    emit one terminal event and close.
    """
    import importlib

    app_module = importlib.import_module("ubt.api.app")

    class _CancelledJobManager(JobManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            record = JobRecord(
                job_id="job_cancelled_stream",
                request=JobSubmitRequest(input_path="/books/x.pdf", target_lang="zh"),
            )
            record.status = "cancelled"
            self.jobs[record.job_id] = record

    monkeypatch.setattr(app_module, "JobManager", _CancelledJobManager)
    config = UBTConfig(db_dir=tmp_path / "ledgers", allowed_dirs=str(tmp_path))
    app = create_app(config=config)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def consume() -> list[str]:
            events: list[str] = []
            async with client.stream("GET", "/jobs/job_cancelled_stream/stream") as resp:
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        events.append(line)
            return events

        # wait_for is the assertion: the pre-fix generator never yields a
        # terminal event and spins until this times out.
        events = await asyncio.wait_for(consume(), timeout=5)
    # Unified frame contract (queue mode emits the same): one current snapshot,
    # then exactly one terminal frame carrying the job's status.
    assert events and events[0] == "event: progress", events
    assert "event: cancelled" in events, f"stream did not terminate (got {events!r})"


@pytest.mark.asyncio
async def test_sse_stream_endpoint_for_job(sample_api_doc: Path, tmp_path: Path) -> None:
    db_dir = tmp_path / "api_ledgers_async"
    config = UBTConfig(db_dir=db_dir, rate_limit_rpm=600, allowed_dirs=str(tmp_path))
    # Mock the provider: without a router the pipeline builds a real
    # OpenAI-compatible provider, and a slow/hung network keeps the job
    # "running" forever, which buffers this SSE response indefinitely.
    mock_provider = MockModelProvider(default_response="这是SSE流测试的翻译段落。")
    router = ModelRouter(
        provider=mock_provider,
        draft_model="mock-draft",
        repair_model="mock-repair",
    )
    app = create_app(config=config, router=router, qe_runner=MockQERunner(default_score=0.88))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "input_path": str(sample_api_doc),
            "output_path": str(tmp_path / "api_out_stream.md"),
            "target_lang": "zh",
        }
        resp = await client.post("/jobs/submit", json=payload)
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Stream SSE events
        async with client.stream("GET", f"/jobs/{job_id}/stream") as stream_resp:
            assert stream_resp.status_code == 200
            assert "text/event-stream" in stream_resp.headers["content-type"]
            lines: list[str] = []
            async for line in stream_resp.aiter_lines():
                if line:
                    lines.append(line)
                if len(lines) >= 4:
                    break
            assert any("event:" in line or "data:" in line for line in lines)


@pytest.mark.asyncio
async def test_full_job_lifecycle_via_api(sample_api_doc: Path, tmp_path: Path) -> None:
    db_dir = tmp_path / "api_ledgers_lifecycle"
    config = UBTConfig(
        db_dir=db_dir,
        rate_limit_rpm=600,
        draft_model="mock-draft",
        repair_model="mock-repair",
        allowed_dirs=str(tmp_path),
    )
    mock_provider = MockModelProvider(
        default_response="这是用于API验证的端到端测试翻译文本段落。",
        custom_responses={
            "Translate\n# Chapter 1": "# 第1章\n",
            "Translate\nShort paragraph": "这是用于API验证的端到端测试翻译文本段落。",
        },
    )
    router = ModelRouter(
        provider=mock_provider,
        draft_model="mock-draft",
        repair_model="mock-repair",
    )
    qe = MockQERunner(default_score=0.88)
    app = create_app(config=config, router=router, qe_runner=qe)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        out_path = tmp_path / "api_out_lifecycle.md"
        payload = {
            "input_path": str(sample_api_doc),
            "output_path": str(out_path),
            "target_lang": "zh",
        }
        resp = await client.post("/jobs/submit", json=payload)
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Stream until job finishes
        async with client.stream("GET", f"/jobs/{job_id}/stream") as stream_resp:
            async for line in stream_resp.aiter_lines():
                if "event: completed" in line or "event: failed" in line:
                    break

        # Check final status
        status_resp = await client.get(f"/jobs/{job_id}/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["status"] == "completed"
        assert status_data["total_blocks"] >= 2
        assert status_data["completed_blocks"] >= 2
        assert status_data["output_file"] is not None

        # Check quality report endpoint
        report_resp = await client.get(f"/jobs/{job_id}/report")
        assert report_resp.status_code == 200
        rep_json = report_resp.json()
        assert rep_json["job_id"] == job_id
        assert "score_metrics" in rep_json

        # Check download endpoint
        download_resp = await client.get(f"/jobs/{job_id}/download")
        assert download_resp.status_code == 200
        assert len(download_resp.content) > 0


def test_sandbox_denial_body_has_no_server_paths(api_client: TestClient, tmp_path: Path) -> None:
    """Risk: a sandbox 403 used to interpolate the server's absolute allowed
    bases into ``detail``, handing the host's directory layout to any caller of
    a service that is unauthenticated by default."""
    resp = api_client.post(
        "/jobs/submit",
        json={"input_path": str(tmp_path.parent / "elsewhere.md"), "target_lang": "zh"},
    )
    assert resp.status_code == 403
    assert str(tmp_path) not in resp.text
    assert "allowed directories" in resp.json()["detail"]


def test_system_deny_list_still_applies_without_a_whitelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Risk: the deny list must stay the fallback sandbox when the operator
    configures no whitelist (a system path is still refused, fail closed)."""
    for name in ("UBT_ALLOWED_DIRS", "UBT_ALLOWED_DIR"):
        monkeypatch.delenv(name, raising=False)
    config = UBTConfig(db_dir=tmp_path / "ledgers")
    client = TestClient(create_app(config=config))
    resp = client.post("/jobs/submit", json={"input_path": "/etc/passwd", "target_lang": "zh"})
    assert resp.status_code == 403
    assert "restricted system directory" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_submit_without_output_path_yields_downloadable_artifact(
    sample_api_doc: Path, tmp_path: Path
) -> None:
    """Risk: with ``output_path`` omitted the pipeline wrote to the cwd-relative
    ``tmp/output/...`` that never passed the sandbox, so ``/download`` then
    rejected the very artifact the job had just produced (403 when absolute,
    400 when the relative path was re-anchored to the whitelist base)."""
    config = UBTConfig(
        db_dir=tmp_path / "api_ledgers_default_out",
        rate_limit_rpm=600,
        draft_model="mock-draft",
        repair_model="mock-repair",
        allowed_dirs=str(tmp_path),
    )
    router = ModelRouter(
        provider=MockModelProvider(default_response="这是省略输出路径时的默认产物测试。"),
        draft_model="mock-draft",
        repair_model="mock-repair",
    )
    app = create_app(config=config, router=router, qe_runner=MockQERunner(default_score=0.88))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs/submit",
            json={"input_path": str(sample_api_doc), "target_lang": "zh"},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        async with client.stream("GET", f"/jobs/{job_id}/stream") as stream_resp:
            async for line in stream_resp.aiter_lines():
                if "event: completed" in line or "event: failed" in line:
                    break

        status_data = (await client.get(f"/jobs/{job_id}/status")).json()
        assert status_data["status"] == "completed", status_data
        output_file = status_data["output_file"]
        assert output_file is not None
        expected = tmp_path / "tmp" / "output" / f"{sample_api_doc.stem}_bilingual.md"
        # L8: /status echoes the artifact's basename, never the host's absolute
        # path — /download below still resolves the stored path server-side.
        assert output_file == expected.name
        assert not Path(output_file).is_absolute()

        download_resp = await client.get(f"/jobs/{job_id}/download")
        assert download_resp.status_code == 200
        assert len(download_resp.content) > 0

        # The derived artifact's quality report stays fetchable too.
        report_resp = await client.get(f"/jobs/{job_id}/report")
        assert report_resp.status_code == 200


def test_unsatisfiable_default_output_fails_with_4xx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Risk: when the derived default output location itself is outside the
    sandbox (here: the anchor sits under a denied system prefix, as a real
    ``/var/lib/ubt`` deployment does) the submit must fail with a clear 4xx —
    the old cwd-relative default escaped the sandbox and only surfaced later as
    a 403/400 on ``/download``, i.e. after the whole job had run."""
    import importlib

    app_module = importlib.import_module("ubt.api.app")
    monkeypatch.chdir(tmp_path)
    # Simulate the default output anchor resolving under a system prefix.
    monkeypatch.setenv("UBT_OUTPUT_DIR", str(tmp_path / "tmp"))
    monkeypatch.setattr(app_module, "SYSTEM_DISALLOWED_PREFIXES", ((tmp_path / "tmp").resolve(),))

    input_file = tmp_path / "book.md"
    input_file.write_text("# Chapter 1\n\nHello world\n", encoding="utf-8")
    config = UBTConfig(db_dir=tmp_path / "ledgers")
    client = TestClient(create_app(config=config))

    resp = client.post("/jobs/submit", json={"input_path": str(input_file), "target_lang": "zh"})
    assert resp.status_code == 403
    assert "restricted system directory" in resp.json()["detail"]


def test_report_outside_sandbox_is_not_masked_as_a_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Risk: ``/report`` re-validates the recorded report path through the
    sandbox, but its broad ``except Exception`` turned the resulting 403 into a
    500 — masking a client error as a server fault (e.g. after UBT_ALLOWED_DIRS
    was narrowed so the artifact was no longer readable)."""
    import importlib

    app_module = importlib.import_module("ubt.api.app")
    outside_report = tmp_path.parent / "ubt_outside_quality_report.json"
    outside_report.write_text('{"job_id": "job_outside_report"}', encoding="utf-8")

    class _PreloadedJobManager(JobManager):
        """Manager seeded with one completed record whose report escaped the sandbox."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            record = JobRecord(
                job_id="job_outside_report",
                request=JobSubmitRequest(input_path=str(outside_report)),
            )
            record.status = "completed"
            record.progress = ProgressSnapshot(report_file=str(outside_report))
            self.jobs[record.job_id] = record

    monkeypatch.setattr(app_module, "JobManager", _PreloadedJobManager)
    config = UBTConfig(db_dir=tmp_path / "ledgers", allowed_dirs=str(tmp_path / "books"))
    client = TestClient(create_app(config=config))

    try:
        resp = client.get("/jobs/job_outside_report/report")
        assert resp.status_code == 403
    finally:
        outside_report.unlink(missing_ok=True)


def test_path_traversal_rejection(api_client: TestClient) -> None:
    """Ensure directory traversal patterns ('..') are blocked with 403 Forbidden."""
    payload = {
        "input_path": "../../../../etc/passwd",
        "target_lang": "zh",
    }
    resp = api_client.post("/jobs/submit", json=payload)
    assert resp.status_code == 403
    assert "Directory traversal" in resp.json()["detail"]


def test_sensitive_system_path_rejection(api_client: TestClient, system_probe_path: str) -> None:
    """Ensure direct absolute paths to system directories are blocked."""
    payload = {
        "input_path": system_probe_path,
        "target_lang": "zh",
    }
    resp = api_client.post("/jobs/submit", json=payload)
    assert resp.status_code == 403
    assert "restricted system directory" in resp.json()["detail"]


def test_sensitive_dotdir_rejection(api_client: TestClient) -> None:
    """Ensure access to credentials/ssh dirs is blocked."""
    payload = {
        "input_path": "/home/user/.ssh/id_rsa",
        "target_lang": "zh",
    }
    resp = api_client.post("/jobs/submit", json=payload)
    assert resp.status_code == 403
    assert "Accessing sensitive configuration directory" in resp.json()["detail"]


def test_job_manager_prunes_old_jobs() -> None:
    """Ensure JobManager evicts oldest completed jobs when exceeding max_retained_jobs."""
    from ubt.api.app import JobManager, JobSubmitRequest

    manager = JobManager(max_retained_jobs=5)
    for i in range(10):
        req = JobSubmitRequest(input_path=f"/fake/path_{i}.epub", target_lang="zh")
        record = manager.create_job(req)
        record.status = "completed"  # Mark as finished so it is eligible for pruning

    # Only 5 jobs should be retained
    assert len(manager.jobs) <= 5


# ---------------------------------------------------------------------------
# Status endpoint must not block the event loop, and the
# persisted-ledger fallback must not fabricate a per-block cost constant.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_endpoint_offloads_ledger_scan_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disk-fallback path opens the SQLite ledger and runs a
    full ORDER BY mtqe_score scan. The endpoint must be a sync ``def`` (which
    FastAPI dispatches to its threadpool) so one status poll cannot freeze the
    event loop and every SSE stream attached to it."""
    import asyncio
    import importlib
    import inspect
    import time

    from fastapi.routing import APIRoute

    # NOTE: ubt/api/__init__.py re-exports a FastAPI instance named `app`,
    # which shadows the `ubt.api.app` submodule in attribute lookups —
    # importlib gives us the real module for monkeypatching.
    app_module = importlib.import_module("ubt.api.app")

    db_dir = tmp_path / "api_ledgers_block"
    config = UBTConfig(db_dir=db_dir, rate_limit_rpm=600)
    app = create_app(config=config)

    # Signature guard: the status route must not be a coroutine function.
    status_route = next(
        r for r in app.routes if isinstance(r, APIRoute) and r.path == "/jobs/{job_id}/status"
    )
    assert not inspect.iscoroutinefunction(status_route.endpoint)

    # Simulate a slow ledger scan on the disk-fallback path (blocking sleep,
    # NOT asyncio.sleep — exactly the pattern a real SQLite scan produces).
    class _SlowLedger:
        def __init__(self, path: Path) -> None:
            self._path = path

        def __enter__(self) -> "_SlowLedger":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def get_job_snapshot(self, job_id: str) -> dict[str, object]:
            time.sleep(0.4)
            return {
                "job_id": job_id,
                "status": "completed",
                "total": 5,
                "completed": 5,
                "repaired": 0,
                "failed": 0,
                "avg_qe_score": 0.9,
                "bottom_15_avg_qe": 0.8,
                "created_at": "2026-09-09T00:00:00+00:00",
            }

        def get_job_metadata_value(self, job_id: str, key: str) -> None:
            return None

    monkeypatch.setattr(app_module, "SQLiteJobLedger", _SlowLedger)

    # The fallback only runs when the ledger file exists on disk.
    job_id = "job_ubt001"
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / f"{job_id}.sqlite").touch()

    ticks: list[float] = []
    loop = asyncio.get_running_loop()

    async def _ticker(deadline: float) -> None:
        last = loop.time()
        while loop.time() < deadline:
            await asyncio.sleep(0.02)
            now = loop.time()
            ticks.append(now - last)
            last = now

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ticker_task = asyncio.create_task(_ticker(loop.time() + 1.5))
        resp = await client.get(f"/jobs/{job_id}/status")
        await ticker_task

    assert resp.status_code == 200
    # The persisted-ledger fallback reports "unknown" cost instead of
    # fabricating a per-block constant.
    assert resp.json()["estimated_cost_usd"] is None
    # The event loop kept ticking while the blocking scan ran in the pool.
    assert len(ticks) >= 20
    # A frozen loop would stall for the full 0.4s sleep in a single tick gap.
    assert max(ticks) < 0.3


def test_status_corrupted_ledger_returns_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib
    from typing import Any

    app_module = importlib.import_module("ubt.api.app")

    class _CorruptLedger:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "_CorruptLedger":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get_job_snapshot(self, *args: Any) -> Any:
            raise RuntimeError("Disk corruption error")

        def get_visual_report(self, *args: Any) -> Any:
            raise RuntimeError("Corrupt visual report")

    monkeypatch.setattr(app_module, "SQLiteJobLedger", _CorruptLedger)
    db_dir = tmp_path / "corrupt_ledgers"
    db_dir.mkdir()
    job_id = "job_corrupt_001"
    (db_dir / f"{job_id}.sqlite").touch()

    config = UBTConfig(db_dir=db_dir)
    app = create_app(config=config)
    client = TestClient(app)

    # 1. get_job_status returns 500 instead of masking as 404
    resp = client.get(f"/jobs/{job_id}/status")
    assert resp.status_code == 500
    assert "Database error inspecting job ledger" in resp.json()["detail"]

    # 2. get_visual_report returns 500 instead of masking as 400
    resp_vis = client.get(f"/jobs/{job_id}/visual-report")
    assert resp_vis.status_code == 500
    assert "Database error reading visual report" in resp_vis.json()["detail"]


def test_submit_job_id_is_idempotent(
    api_client: TestClient, sample_api_doc: Path, tmp_path: Path
) -> None:
    """Resubmitting a stable job_id returns the existing job, not a duplicate."""
    payload = {
        "input_path": str(sample_api_doc),
        "output_path": str(tmp_path / "idem_out.md"),
        "target_lang": "zh",
        "job_id": "job_idem_fixed_1",
    }
    first = api_client.post("/jobs/submit", json=payload)
    assert first.status_code == 202
    assert first.json()["job_id"] == "job_idem_fixed_1"

    second = api_client.post("/jobs/submit", json=payload)
    assert second.status_code == 202
    assert second.json()["job_id"] == "job_idem_fixed_1"


def test_cancel_unknown_job_returns_404(api_client: TestClient) -> None:
    resp = api_client.post("/jobs/ghost_job/cancel")
    assert resp.status_code == 404


def test_cancel_known_job_reaches_terminal_state(
    api_client: TestClient, sample_api_doc: Path, tmp_path: Path
) -> None:
    payload = {
        "input_path": str(sample_api_doc),
        "output_path": str(tmp_path / "cancel_out.md"),
        "target_lang": "zh",
        "job_id": "job_cancel_fixed_1",
    }
    assert api_client.post("/jobs/submit", json=payload).status_code == 202
    resp = api_client.post("/jobs/job_cancel_fixed_1/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("cancelled", "completed", "failed")


def test_api_get_status_sqlite_disk_fallback(tmp_path: Path) -> None:
    """Fix 6: Verify API get_status recovers job stats from on-disk SQLite when in-memory record is absent."""
    config = UBTConfig(db_dir=tmp_path)
    app = create_app(config=config)
    client = TestClient(app)

    job_id = "job_restarted_001"
    db_file = tmp_path / f"{job_id}.sqlite"
    ledger = SQLiteJobLedger(db_file)

    manifest = BookManifest(
        doc_id="sha256_restart_test",
        title="Restarted Book",
        source_path="book.md",
        chapters=[ChapterMeta(chapter_id="c1", title="Chapter 1", spine_index=1)],
    )
    ledger.init_job_from_manifest(job_id, manifest)
    ledger.finalize_job(job_id, status="completed")

    response = client.get(f"/jobs/{job_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "completed"


def test_api_assess_job(tmp_path: Path) -> None:
    """Verify POST /jobs/assess returns document quote without spend."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Hello world! This is a test book for assessment.", encoding="utf-8")

    config = UBTConfig(db_dir=tmp_path, allowed_dirs=str(tmp_path))
    app = create_app(config=config)
    client = TestClient(app)

    resp = client.post(
        "/jobs/assess",
        json={"input_path": str(sample_file), "target_lang": "zh"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "document" in data
    assert "route" in data
    assert "cost" in data
    assert data["document"]["format_ext"] == "txt"


@pytest.mark.asyncio
async def test_deep_assess_is_capped_at_the_job_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deep=True`` loads the full adapter stack, so it must share the job cap.

    ``/jobs/submit`` refuses beyond ``max_running_jobs`` but ``/jobs/assess``
    with ``deep=True`` had no cap: it was the one default-open endpoint that
    could pile unbounded heavy ingest onto the process.
    """
    import importlib

    import ubt.core.assess as assess_mod

    app_module = importlib.import_module("ubt.api.app")

    class _SingleSlotJobManager(JobManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.max_running_jobs = 1

    class _Report:
        def to_dict(self) -> dict[str, Any]:
            return {"status": "ok"}

    started = asyncio.Event()
    gate = asyncio.Event()

    async def _slow_assess(*_args: Any, **_kwargs: Any) -> Any:
        started.set()
        await gate.wait()
        return _Report()

    monkeypatch.setattr(app_module, "JobManager", _SingleSlotJobManager)
    monkeypatch.setattr(assess_mod, "assess_document_async", _slow_assess)

    sample_file = tmp_path / "deep.txt"
    sample_file.write_text("Hello world! " * 50, encoding="utf-8")
    config = UBTConfig(db_dir=tmp_path / "ledgers", allowed_dirs=str(tmp_path))
    app = create_app(config=config)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"input_path": str(sample_file), "target_lang": "zh", "deep": True}
        first = asyncio.create_task(client.post("/jobs/assess", json=payload))
        await asyncio.wait_for(started.wait(), timeout=5)

        second = await client.post("/jobs/assess", json=payload)
        assert second.status_code == 503
        assert "at capacity" in second.text.lower()

        gate.set()
        first_resp = await asyncio.wait_for(first, timeout=5)
        assert first_resp.status_code == 200


def test_nonloopback_bind_without_isolation_is_refused(tmp_path: Path) -> None:
    """A bind guard that only runs in ``run_server`` is bypassed by uvicorn.

    ``uvicorn ubt.api.app:app --host 0.0.0.0`` imports the app object directly,
    so the "loopback unless fully isolated" contract must be enforced per
    request from the socket's local address.
    """
    config = UBTConfig(db_dir=tmp_path / "ledgers")
    app = create_app(config=config)
    client = TestClient(app, base_url="http://0.0.0.0:8000")

    resp = client.get("/health")
    assert resp.status_code == 403
    assert "non-loopback" in resp.text


def test_nonloopback_bind_with_full_isolation_is_allowed(tmp_path: Path) -> None:
    config = UBTConfig(
        db_dir=tmp_path / "ledgers",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr("secret"),
    )
    app = create_app(config=config)
    client = TestClient(app, base_url="http://192.0.2.10:8000")

    # The isolation guards are present, so the request is served (auth is a
    # separate layer; /health stays public).
    assert client.get("/health").status_code == 200


def test_nonloopback_bind_override_is_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UBT_ALLOW_INSECURE_BIND", "1")
    config = UBTConfig(db_dir=tmp_path / "ledgers")
    app = create_app(config=config)
    client = TestClient(app, base_url="http://0.0.0.0:8000")

    assert client.get("/health").status_code == 200


def test_cancel_cannot_rewrite_a_completed_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Export persists "completed" before the worker flips the in-memory record.

    Several awaits sit between them, so a cancel can observe ``running`` while
    the durable row already says ``completed`` — and ``finalize_job`` is an
    unconditional UPDATE, which used to erase a finished job's status even
    though its artifact was on disk.
    """
    import importlib

    api_module = importlib.import_module("ubt.api.app")

    captured: dict[str, JobManager] = {}

    class _Capturing(JobManager):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)  # type: ignore[arg-type]
            captured["manager"] = self

    monkeypatch.setattr(api_module, "JobManager", _Capturing)
    config = UBTConfig(db_dir=tmp_path)
    app = create_app(config=config)
    client = TestClient(app)

    job_id = "job_completed_then_cancel"
    ledger = SQLiteJobLedger(tmp_path / f"{job_id}.sqlite")
    manifest = BookManifest(
        doc_id="sha256_cancel_test",
        title="Finished Book",
        source_path="book.md",
        chapters=[ChapterMeta(chapter_id="c1", title="Chapter 1", spine_index=1)],
    )
    ledger.init_job_from_manifest(job_id, manifest)
    ledger.finalize_job(job_id, status="completed")
    ledger.close()

    record = captured["manager"].create_job(
        JobSubmitRequest(
            input_path=str(tmp_path / "book.md"),
            output_path=str(tmp_path / "o.md"),
            target_lang="zh",
        ),
        job_id=job_id,
    )
    record.status = "running"

    resp = client.post(f"/jobs/{job_id}/cancel")
    assert resp.status_code == 200

    with SQLiteJobLedger(tmp_path / f"{job_id}.sqlite") as check:
        assert check.get_job_status(job_id) == "completed"


def test_submit_job_id_is_stored_as_validated(
    api_client: TestClient, sample_api_doc: Path, tmp_path: Path
) -> None:
    """L3: ``validate_job_id`` strips, so ``create_job`` must get its return value.

    The handler validated a copy and then stored the raw ``req.job_id``: the
    manager was keyed on ``"  job_padded_1  "`` while the idempotency lookup
    used ``"job_padded_1"`` — a resubmit missed its own record and started
    (and billed) a duplicate run.
    """
    payload = {
        "input_path": str(sample_api_doc),
        "output_path": str(tmp_path / "padded_out.md"),
        "target_lang": "zh",
        "job_id": "  job_padded_1  ",
    }
    first = api_client.post("/jobs/submit", json=payload)
    assert first.status_code == 202
    assert first.json()["job_id"] == "job_padded_1"

    # The record lives under the validated id, so lookups resolve...
    assert api_client.get("/jobs/job_padded_1/status").status_code == 200
    # ...and the resubmit hits the idempotency branch instead of forking a
    # second job under a whitespace-variant key.
    second = api_client.post("/jobs/submit", json=payload)
    assert second.status_code == 202
    assert second.json()["job_id"] == "job_padded_1"


def test_global_stream_subscriber_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """L2: per-job SSE caps do not bound the process — the global one must.

    Each open stream polls from a worker thread once a second, so
    ``_MAX_SUBSCRIBERS_PER_JOB`` per job across N jobs still exhausts the
    shared default executor. One counter, one shared key: 64 streams total,
    then 429 until one closes.
    """
    import importlib

    app_module = importlib.import_module("ubt.api.app")

    class _PreloadedJobManager(JobManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            record = JobRecord(
                job_id="job_global_cap",
                request=JobSubmitRequest(input_path="/books/x.pdf", target_lang="zh"),
            )
            record.status = "cancelled"
            self.jobs[record.job_id] = record

    monkeypatch.setattr(app_module, "JobManager", _PreloadedJobManager)
    # The production default is what the review fixed; shrink it to one slot so
    # the accounting assertions below are exact (any leak is visible).
    assert app_module._MAX_GLOBAL_STREAM_SUBSCRIBERS == 64
    monkeypatch.setattr(app_module, "_MAX_GLOBAL_STREAM_SUBSCRIBERS", 1)
    config = UBTConfig(db_dir=tmp_path / "ledgers", allowed_dirs=str(tmp_path))
    app = create_app(config=config)
    counter = app.state.global_stream_subscribers
    key = app_module._GLOBAL_SUBSCRIBER_KEY

    assert counter.acquire(key)
    client = TestClient(app)
    resp = client.get("/jobs/job_global_cap/stream")
    assert resp.status_code == 429
    assert "global limit" in resp.json()["detail"]
    counter.release(key)

    # The slot is free again, so the stream is served...
    ok = client.get("/jobs/job_global_cap/stream")
    assert ok.status_code == 200
    # ...and the closed stream gave its slot back (a leak here would make this
    # acquire fail against the single-slot ceiling).
    assert counter.acquire(key)
    counter.release(key)


def test_status_never_echoes_host_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L8: ``/status`` returned the pipeline's absolute artifact paths.

    The host directory layout (home directory, project path) went out over a
    port that is unauthenticated by default. The echo now carries basenames;
    ``/download`` and ``/report`` still resolve the stored absolute path
    server-side, so nothing legitimate is lost.
    """
    import importlib

    from ubt.core.engine.ledger import SQLiteJobLedger as _Ledger
    from ubt.core.ir.models import BookManifest as _Manifest
    from ubt.core.ir.models import ChapterMeta as _ChapterMeta

    app_module = importlib.import_module("ubt.api.app")
    artifacts = tmp_path / "published"
    artifacts.mkdir()

    class _PreloadedJobManager(JobManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            record = JobRecord(
                job_id="job_paths",
                request=JobSubmitRequest(input_path="/books/x.pdf", target_lang="zh"),
            )
            record.status = "completed"
            record.progress = ProgressSnapshot(
                total_blocks=1,
                completed_blocks=1,
                output_file=str(artifacts / "book_bilingual.md"),
                report_file=str(artifacts / "book_quality_report.json"),
                visual_report_file=str(artifacts / "book_visual_report.json"),
            )
            self.jobs[record.job_id] = record

    monkeypatch.setattr(app_module, "JobManager", _PreloadedJobManager)
    config = UBTConfig(db_dir=tmp_path / "ledgers", allowed_dirs=str(tmp_path))
    client = TestClient(create_app(config=config))
    resp = client.get("/jobs/job_paths/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_file"] == "book_bilingual.md"
    assert data["report_file"] == "book_quality_report.json"
    assert data["visual_report_file"] == "book_visual_report.json"
    assert str(tmp_path) not in resp.text, "host path leaked through /status"

    # Same rule on the persisted-ledger fallback (fresh server, no record):
    # the artifact paths live in job_meta and used to be echoed verbatim.
    job_id = "job_ledger_paths"
    db_dir = tmp_path / "ledgers"
    db_dir.mkdir(parents=True, exist_ok=True)
    with _Ledger(db_dir / f"{job_id}.sqlite") as ledger:
        ledger.init_job_from_manifest(
            job_id,
            _Manifest(
                doc_id="doc_ledger_paths",
                title="t",
                source_path="book.md",
                chapters=[_ChapterMeta(chapter_id="c1", title="One", spine_index=1)],
            ),
        )
        ledger.set_job_metadata_value(
            job_id, "output_file", str(artifacts / "ledger_bilingual.pdf")
        )
        ledger.finalize_job(job_id, status="completed")
    fallback = TestClient(create_app(config=config)).get(f"/jobs/{job_id}/status")
    assert fallback.status_code == 200
    assert fallback.json()["output_file"] == "ledger_bilingual.pdf"
    assert str(tmp_path) not in fallback.text, "host path leaked via the ledger fallback"


def test_bootstrap_converges_env_file_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1: the ASGI entry point must chmod ``.env`` to 0600 before config reads it.

    dotenv files are created ``0644`` by editors and ``cp`` while holding the
    API credentials (review 2026-09 M1). Unit behaviour lives in
    ``tests/unit/test_fs_perms.py``; this pins the *wiring*.
    """
    import importlib

    app_module = importlib.import_module("ubt.api.app")
    calls: list[object] = []
    monkeypatch.setattr(app_module, "restrict_env_file", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(app_module, "setup_logging", lambda *a, **k: None)
    app_module._bootstrap_asgi_app()
    assert calls, "_bootstrap_asgi_app() must converge .env before create_app()"


def test_api_status_clamps_an_orphaned_embedded_job(tmp_path: Path) -> None:
    """A crash mid-run leaves the row "initialized"; nothing will move it again.

    Embedded mode has no durable worker and no startup reconcile, so a polling
    client would wait forever on a job that no longer exists.
    """
    config = UBTConfig(db_dir=tmp_path)
    client = TestClient(create_app(config=config))

    job_id = "job_orphaned_midrun"
    ledger = SQLiteJobLedger(tmp_path / f"{job_id}.sqlite")
    manifest = BookManifest(
        doc_id="sha256_orphan_test",
        title="Crashed Book",
        source_path="book.md",
        chapters=[ChapterMeta(chapter_id="c1", title="Chapter 1", spine_index=1)],
    )
    ledger.init_job_from_manifest(job_id, manifest)
    ledger.close()

    data = client.get(f"/jobs/{job_id}/status").json()
    assert data["status"] == "failed"
    assert "interrupted" in str(data["error"])

    # A completed row is still reported as completed.
    with SQLiteJobLedger(tmp_path / f"{job_id}.sqlite") as done:
        done.finalize_job(job_id, status="completed")
    assert client.get(f"/jobs/{job_id}/status").json()["status"] == "completed"


@pytest.mark.fast
def test_api_download_job_exists_not_ready_returns_400(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from ubt.api.app import create_app
    from ubt.core.config import UBTConfig
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir import BookManifest

    db_dir = tmp_path / "ledgers"
    db_dir.mkdir()
    config = UBTConfig(db_dir=db_dir)
    app = create_app(config)
    client = TestClient(app)

    job_id = "job_test_not_ready"
    # Create DB file so job exists on disk
    db_file = db_dir / f"{job_id}.sqlite"
    with SQLiteJobLedger(db_file) as ldg:
        ldg.init_job_from_manifest(
            job_id,
            BookManifest(
                doc_id="test_doc", source_path=str(tmp_path / "book.md"), title="Test", chapters=[]
            ),
        )

    resp = client.get(f"/jobs/{job_id}/download")
    assert resp.status_code == 400
    assert "not ready" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Engine-knob parity: the job payload surface must not be narrower than the CLI
# ---------------------------------------------------------------------------


def test_submit_request_carries_every_engine_knob_into_the_config() -> None:
    """A knob missing from ``JobSubmitRequest`` is a 422, not a silent default.

    The model is ``extra="forbid"``, so a REST/queue/MCP client cannot merely
    *fail to set* a knob the CLI can set — the submit is rejected outright
    before a job exists. The CLI built 40+ request keys while this model carried
    19, which meant a web client could not cap spend, raise the concurrency
    ceiling, choose the OCR engine or pin the formula policy at all.

    The test walks the exact production path — ``request.model_dump()`` ->
    ``overrides_from_request(..., allow_provider_keys=False)`` ->
    ``apply_config_overrides`` (``ubt/api/manager.py::execute_job``) — so a knob
    that parses but never reaches ``UBTConfig`` fails here too.
    """
    from ubt.core.job_options import apply_config_overrides, overrides_from_request

    payload = {
        "input_path": "book.md",
        # Cost and throughput ceilings.
        "budget_usd": 5.0,
        "max_concurrency": 8,
        "batch_limit": 3,
        "macro_chunk_size": 12,
        "short_max_pages": 12,
        # Long-chain behaviour.
        "enable_rolling_summary": True,
        "chapter_streaming_enabled": True,
        "offline_batch_enabled": True,
        "qe_engine": "tiered",
        # Quality gates.
        "visual_judge_enabled": True,
        "visual_judge_model": "gpt-4o-mini",
        "prompt_strategy": "rich",
        # Output shape.
        "translate_chrome": True,
        "facing_spread": True,
        "emit_both": True,
        "cover_mode": "never",
        # Formulas and OCR.
        "formula_enrichment": "on",
        "formula_render": "image",
        "math_backend": "mathjax",
        "ocr_mode": "rapidocr",
        # Content.
        "domain": "semiconductor",
    }
    request = JobSubmitRequest.model_validate(payload)
    overrides = overrides_from_request(request.model_dump(), allow_provider_keys=False)
    config = apply_config_overrides(UBTConfig.from_env(), overrides)

    for key, expected in payload.items():
        if key == "input_path":
            continue
        assert getattr(config, key) == expected, (
            f"{key!r}: payload value {expected!r} did not reach UBTConfig "
            f"(got {getattr(config, key)!r}; override={overrides.get(key)!r})"
        )


def test_submit_request_still_refuses_credentials_and_server_owned_keys() -> None:
    """Widening the engine surface must not widen the credential surface.

    Two independent defences guard provider credentials: this model's
    ``extra="forbid"`` (the key is not a field at all) and
    ``overrides_from_request(allow_provider_keys=False)`` (defence in depth for
    a payload written by an older/other surface). Both must hold, or a job
    payload could redirect the provider endpoint or spend a key the operator
    never handed the service.
    """
    from ubt.core.exceptions import UBTError
    from ubt.core.job_options import overrides_from_request

    for key in (
        "api_key",
        "base_url",
        "api_mode",
        "ocr_api_key",
        "ocr_endpoint",
        "service_api_key",
        # Server-owned state: the service picks its own storage and profile.
        "db_dir",
        "provider_profile",
        # A filesystem path: it needs the same sandbox as ``input_path`` before
        # it can be accepted, so it stays out until that is wired (see
        # ``resolve_secure_path`` in the submit handler).
        "glossary",
    ):
        with pytest.raises(ValidationError):
            JobSubmitRequest.model_validate({"input_path": "book.md", key: "probe"})

    # Defence in depth: the shared mapping rejects the credential family even
    # when a caller bypasses the model (e.g. a queue row from an older release).
    with pytest.raises(UBTError):
        overrides_from_request({"api_key": "sk-leaked"}, allow_provider_keys=False)


def test_submit_accepts_the_engine_knobs_over_http(
    api_client: TestClient, sample_api_doc: Path, tmp_path: Path
) -> None:
    """Shares the knob set with the mapping test, over the real HTTP contract.

    ``extra="forbid"`` made every knob the model omitted a *rejection*, so a web
    client's first submit carrying a spend cap failed before a job existed. The
    422-versus-202 boundary is the entire user-visible defect, and only a real
    request through the app exercises it (routing, sandbox, intake, response).
    """
    payload = {
        "input_path": str(sample_api_doc),
        "output_path": str(tmp_path / "knobs_out.md"),
        "target_lang": "zh",
        "budget_usd": 5.0,
        "max_concurrency": 4,
        "batch_limit": 2,
        "macro_chunk_size": 3,
        "short_max_pages": 10,
        "enable_rolling_summary": True,
        "chapter_streaming_enabled": True,
        "offline_batch_enabled": True,
        "qe_engine": "heuristic",
        "visual_judge_enabled": True,
        "visual_judge_model": "gpt-4o-mini",
        "prompt_strategy": "rich",
        "translate_chrome": True,
        "facing_spread": True,
        "emit_both": True,
        "cover_mode": "never",
        "formula_enrichment": "on",
        "formula_render": "image",
        "math_backend": "mathjax",
        "ocr_mode": "rapidocr",
        "domain": "semiconductor",
        "dry_run": True,
    }
    resp = api_client.post("/jobs/submit", json=payload)
    assert resp.status_code == 202, resp.text
    assert resp.json()["job_id"].startswith("job_")


def test_api_rejects_unsupported_target_before_ingest() -> None:
    from pydantic import ValidationError

    from ubt.api.models import JobSubmitRequest

    with pytest.raises(ValidationError):
        JobSubmitRequest(input_path="book.md", target_lang="pt-BR")
    # A supported region tag is accepted and preserved verbatim for font selection.
    req = JobSubmitRequest(input_path="book.md", target_lang="zh-CN")
    assert req.target_lang == "zh-CN"


def test_submit_without_output_path_works_without_an_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from ubt.api.app import create_app
    from ubt.core.config import UBTConfig

    monkeypatch.delenv("UBT_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("XDG_DOCUMENTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "book.md"
    input_file.write_text("# Title\n\nSome source prose.\n", encoding="utf-8")

    config = UBTConfig(
        db_dir=tmp_path / "ledgers",
        rate_limit_rpm=600,
        draft_model="mock-draft",
        repair_model="mock-repair",
    )
    assert config.allowed_base_dirs() == []  # the bug's precondition

    client = TestClient(create_app(config=config))
    resp = client.post("/jobs/submit", json={"input_path": str(input_file), "target_lang": "zh"})
    assert resp.status_code == 202, resp.text


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

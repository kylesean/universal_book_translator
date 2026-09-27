"""REST API in queue mode: submit enqueues; status/cancel/stream read the queue."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ubt.api.app import QueueSubscriberCounter, create_app
from ubt.core.config import UBTConfig
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.job_queue import JobQueue, QueuedJob
from ubt.core.engine.job_worker import JobWorker
from ubt.core.job_options import sidecar_path


@pytest.fixture
def sample_doc(tmp_path: Path) -> Path:
    path = tmp_path / "doc.md"
    path.write_text("# Chapter 1\n\nHello world.\n", encoding="utf-8")
    return path


@pytest.fixture
def queue(tmp_path: Path) -> Iterator[JobQueue]:
    q = JobQueue(tmp_path / "q.sqlite", global_max_running=4, default_tenant_max_running=4)
    yield q
    q.close()


def _config(tmp_path: Path) -> UBTConfig:
    return UBTConfig(db_dir=tmp_path / "ledgers", allowed_dirs=str(tmp_path), job_mode="queue")


def _client(tmp_path: Path, queue: JobQueue) -> TestClient:
    return TestClient(create_app(config=_config(tmp_path), queue=queue))


def test_submit_enqueues_and_status_reports_position(
    tmp_path: Path, queue: JobQueue, sample_doc: Path
) -> None:
    client = _client(tmp_path, queue)
    payload = {"input_path": str(sample_doc), "target_lang": "zh"}
    first = client.post("/jobs/submit", json=payload, headers={"X-UBT-Tenant": "acme"})
    assert first.status_code == 202
    body = first.json()
    assert body["status"] == "queued"  # not "submitted": nothing runs in the API
    job_id = body["job_id"]

    # A higher-priority job jumps ahead of it.
    urgent = client.post("/jobs/submit", json={**payload, "job_id": "job_urgent", "priority": 10})
    assert urgent.status_code == 202

    # Reads are tenant-scoped: poll the job as the tenant that submitted it.
    status = client.get(f"/jobs/{job_id}/status", headers={"X-UBT-Tenant": "acme"}).json()
    assert status["status"] == "queued"
    assert status["queue_position"] == 2

    landed = queue.get(job_id)
    assert landed is not None and landed.tenant_id == "acme"
    urgent_job = queue.get("job_urgent")
    assert urgent_job is not None and urgent_job.tenant_id == "default"


def test_submit_is_idempotent_in_queue_mode(
    tmp_path: Path, queue: JobQueue, sample_doc: Path
) -> None:
    client = _client(tmp_path, queue)
    payload = {"input_path": str(sample_doc), "job_id": "job_fixed"}
    assert client.post("/jobs/submit", json=payload).json()["job_id"] == "job_fixed"
    assert client.post("/jobs/submit", json=payload).status_code == 202
    assert len(queue.list_jobs()) == 1


def test_invalid_tenant_header_rejected(tmp_path: Path, queue: JobQueue, sample_doc: Path) -> None:
    client = _client(tmp_path, queue)
    resp = client.post(
        "/jobs/submit",
        json={"input_path": str(sample_doc)},
        headers={"X-UBT-Tenant": "bad tenant!"},
    )
    assert resp.status_code == 400


def test_cancel_queued_job(tmp_path: Path, queue: JobQueue, sample_doc: Path) -> None:
    client = _client(tmp_path, queue)
    job_id = client.post("/jobs/submit", json={"input_path": str(sample_doc)}).json()["job_id"]
    resp = client.post(f"/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    assert client.get(f"/jobs/{job_id}/status").json()["status"] == "cancelled"
    assert client.post("/jobs/job_absent/cancel").status_code == 404


def test_worker_completion_is_visible_through_status(
    tmp_path: Path, queue: JobQueue, sample_doc: Path
) -> None:
    client = _client(tmp_path, queue)
    job_id = client.post("/jobs/submit", json={"input_path": str(sample_doc)}).json()["job_id"]

    async def _gen(
        job: QueuedJob, config: UBTConfig
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        yield TranslationProgressEvent(
            event_type=EventType.DRAFT_BATCH_COMPLETED,
            job_id=job.job_id,
            total_blocks=6,
            completed_blocks=6,
        )

    worker = JobWorker(queue, UBTConfig(), worker_id="w", event_source=_gen)
    assert asyncio.run(worker.run_until_idle()) == 1

    status = client.get(f"/jobs/{job_id}/status").json()
    assert status["status"] == "completed"
    assert status["completed_blocks"] == 6
    assert status["queue_position"] is None


def test_mid_run_row_keeps_every_counter_key_even_when_unknown(
    tmp_path: Path, queue: JobQueue, sample_doc: Path
) -> None:
    """A row written before any usage report still carries all seven counters.

    Queue-mode SSE forwards the row dict verbatim, so a client that indexes
    ``estimated_cost_usd`` must not watch the key vanish mid-run; only the
    artifact paths may be absent, until an export names them.
    """
    client = _client(tmp_path, queue)
    job_id = client.post("/jobs/submit", json={"input_path": str(sample_doc)}).json()["job_id"]

    async def _gen(
        job: QueuedJob, config: UBTConfig
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        yield TranslationProgressEvent(
            event_type=EventType.DRAFT_BATCH_COMPLETED,
            job_id=job.job_id,
            total_blocks=4,
            completed_blocks=2,
        )

    worker = JobWorker(queue, UBTConfig(), worker_id="w", event_source=_gen)
    assert asyncio.run(worker.run_until_idle()) == 1

    row = queue.get(job_id)
    assert row is not None
    assert row.progress["estimated_cost_usd"] is None
    assert "output_file" not in row.progress

    status = client.get(f"/jobs/{job_id}/status").json()
    assert status["completed_blocks"] == 2
    assert status["estimated_cost_usd"] is None
    assert status["output_file"] is None


@pytest.mark.asyncio
async def test_sse_stream_emits_terminal_event_in_queue_mode(
    tmp_path: Path, queue: JobQueue, sample_doc: Path
) -> None:
    client = _client(tmp_path, queue)
    job_id = client.post("/jobs/submit", json={"input_path": str(sample_doc)}).json()["job_id"]

    async def _gen(
        job: QueuedJob, config: UBTConfig
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        yield TranslationProgressEvent(
            event_type=EventType.EXPORT_COMPLETED,
            job_id=job.job_id,
            total_blocks=1,
            completed_blocks=1,
            artifact_path=str(tmp_path / "out.md"),
        )

    worker = JobWorker(queue, UBTConfig(), worker_id="w", event_source=_gen)
    assert await worker.run_until_idle() == 1

    app = create_app(config=_config(tmp_path), queue=queue)
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as ac,
        ac.stream("GET", f"/jobs/{job_id}/stream") as resp,
    ):
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = [line async for line in resp.aiter_lines() if line.startswith("event:")]
    assert any("completed" in event for event in events)


def test_queue_subscriber_counter_caps_per_job() -> None:
    """Queue mode needs the same subscriber ceiling as embedded.

    Every open stream polls the queue from a worker thread once a second, so N
    subscribers occupy N threads of the shared default executor and stall every
    other endpoint. The embedded branch already refused past
    ``_MAX_SUBSCRIBERS_PER_JOB``; the queue branch returned before that check.
    """
    counter = QueueSubscriberCounter(1)
    assert counter.acquire("job_a") is True
    assert counter.acquire("job_a") is False
    # Per job, not global: a second job still gets its own slot.
    assert counter.acquire("job_b") is True
    # Releasing frees the slot for the next subscriber.
    counter.release("job_a")
    assert counter.acquire("job_a") is True
    # An unbalanced release must not drive the count negative.
    counter.release("job_never_seen")
    assert counter.acquire("job_never_seen") is True


def test_completed_queue_job_stays_downloadable(tmp_path: Path, sample_doc: Path) -> None:
    """``/download`` and ``/report`` must find a queued job's artifacts.

    The worker stores them in the queue row's ``progress`` snapshot and never
    writes ledger metadata (only the embedded API path does), so both routes
    404'd for every job that ran through the queue -- including after the
    in-memory record was evicted, which is the case the ledger fallback exists
    for.
    """
    artifact = tmp_path / "out.md"
    artifact.write_text("# Kapitel 1\n\nHallo Welt.\n", encoding="utf-8")
    report = sidecar_path(artifact, "quality_report.json")
    report.write_text('{"summary": {"total_blocks": 1}}', encoding="utf-8")

    queue = JobQueue(tmp_path / "q.sqlite", global_max_running=4, default_tenant_max_running=4)
    config = _config(tmp_path)
    client = TestClient(create_app(config=config, queue=queue))
    job_id = client.post(
        "/jobs/submit",
        json={"input_path": str(sample_doc), "job_id": "job_dl", "target_lang": "de"},
    ).json()["job_id"]

    async def _one(
        job: QueuedJob, _cfg: UBTConfig
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        yield TranslationProgressEvent(
            event_type=EventType.EXPORT_COMPLETED,
            job_id=job.job_id,
            total_blocks=1,
            completed_blocks=1,
            artifact_path=str(artifact),
        )

    assert asyncio.run(JobWorker(queue, config, worker_id="w", event_source=_one).run_until_idle())

    status = client.get(f"/jobs/{job_id}/status").json()
    assert status["output_file"] == artifact.name
    assert client.get(f"/jobs/{job_id}/download").status_code == 200
    assert client.get(f"/jobs/{job_id}/report").status_code == 200
    queue.close()


def test_api_queue_progress_frame_satisfies_schema() -> None:
    import json
    from typing import Any

    from ubt.api.app import _progress_frame
    from ubt.core.engine.progress import ProgressSnapshot

    # Empty raw progress from initial queue row
    raw_queue_progress: dict[str, Any] = {}
    normalized_progress = {**ProgressSnapshot().to_payload(), **raw_queue_progress}
    frame = _progress_frame(normalized_progress, "queued")

    # Verify SSE format and presence of standard fields
    assert frame.startswith("event: progress\ndata: ")
    data = json.loads(frame.split("data: ", 1)[1])
    assert data["total_blocks"] == 0
    assert data["completed_blocks"] == 0
    assert data["status"] == "queued"
    assert "progress_percent" in data


def test_queued_job_download_and_report_returns_400_not_404(
    tmp_path: Path, queue: JobQueue, sample_doc: Path
) -> None:
    """When a job is still queued or running, /download and /report must return HTTP 400 (not ready),
    never HTTP 404 (not found)."""
    client = _client(tmp_path, queue)
    resp = client.post(
        "/jobs/submit",
        json={"input_path": str(sample_doc), "job_id": "job_pending", "target_lang": "zh"},
    )
    assert resp.status_code == 202

    dl_resp = client.get("/jobs/job_pending/download")
    assert dl_resp.status_code == 400
    assert "not ready" in dl_resp.json()["detail"].lower()

    rpt_resp = client.get("/jobs/job_pending/report")
    assert rpt_resp.status_code == 400
    assert "not yet generated" in rpt_resp.json()["detail"].lower()


def test_cross_tenant_cannot_probe_or_reclaim_failed_job(
    tmp_path: Path, queue: JobQueue, sample_doc: Path
) -> None:
    """A cross-tenant submit must get HTTP 404 even if the target job is in FAILED or CANCELLED status."""
    client = _client(tmp_path, queue)
    resp = client.post(
        "/jobs/submit",
        json={"input_path": str(sample_doc), "job_id": "job_failed_a", "target_lang": "zh"},
        headers={"X-UBT-Tenant": "tenant_a"},
    )
    assert resp.status_code == 202

    # Cancel the job in the queue as tenant_a
    cancel_resp = client.post(
        "/jobs/job_failed_a/cancel",
        headers={"X-UBT-Tenant": "tenant_a"},
    )
    assert cancel_resp.status_code == 200

    # Tenant B attempts to submit with the same job_id
    cross_resp = client.post(
        "/jobs/submit",
        json={"input_path": str(sample_doc), "job_id": "job_failed_a", "target_lang": "zh"},
        headers={"X-UBT-Tenant": "tenant_b"},
    )
    assert cross_resp.status_code == 404
    assert "Job not found" in cross_resp.json()["detail"]

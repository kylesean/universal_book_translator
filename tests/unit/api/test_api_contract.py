from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.api.manager import JobManager, JobRecord
from ubt.api.models import JobSubmitRequest
from ubt.api.security import _require_api_key_gate, resolve_secure_path, session_cookie_value
from ubt.core.config import UBTConfig
from ubt.core.engine.job_queue import JobStatus
from ubt.core.exceptions import BudgetExceededError
from ubt.core.job_options import resolve_target_output

_API_KEY = "test-key-contract"
_AUTH = {"X-API-Key": _API_KEY}
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

pytestmark = pytest.mark.fast


def _config(tmp_path: Path, **overrides: Any) -> UBTConfig:
    return UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
        **overrides,
    )


def _write_doc(tmp_path: Path, name: str = "input.md") -> Path:
    doc = tmp_path / name
    doc.write_text("# Chapter 1\n\nHello contract world.\n", encoding="utf-8")
    return doc


@pytest.fixture()
def authed(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(config=_config(tmp_path))) as client:
        yield client


@pytest.fixture()
def open_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("UBT_ALLOW_NO_AUTH", "1")
    config = UBTConfig(db_dir=tmp_path / "db", allowed_dirs=str(tmp_path))
    with TestClient(create_app(config=config)) as client:
        yield client


def test_sandbox_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as err:
        resolve_secure_path(tmp_path / "sub" / ".." / "evil.md", config=_config(tmp_path))
    assert err.value.status_code == 403
    assert "traversal" in str(err.value.detail)


def test_sandbox_rejects_system_dirs_by_default(tmp_path: Path) -> None:
    config = UBTConfig(db_dir=tmp_path / "db")
    with pytest.raises(HTTPException) as err:
        resolve_secure_path("/etc/passwd", config=config)
    assert err.value.status_code == 403
    assert "restricted system directory" in str(err.value.detail)


def test_sandbox_sensitive_names_beat_the_allowlist(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as err:
        resolve_secure_path(tmp_path / "project" / ".env", config=_config(tmp_path))
    assert err.value.status_code == 403
    assert "sensitive" in str(err.value.detail)


def test_sandbox_accepts_file_inside_base(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path)
    resolved = resolve_secure_path(doc, config=_config(tmp_path))
    assert resolved == doc.resolve()


def test_sandbox_missing_file_is_400(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as err:
        resolve_secure_path(tmp_path / "ghost.md", must_exist=True, config=_config(tmp_path))
    assert err.value.status_code == 400
    assert "Source file not found" in str(err.value.detail)


def test_health_is_exempt_from_auth(authed: TestClient) -> None:
    response = authed.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "universal-book-translator"
    assert "version" in body


def test_missing_and_wrong_keys_are_401(authed: TestClient) -> None:
    for headers in ({}, {"X-API-Key": "not-the-key"}):
        response = authed.get("/jobs/whatever/status", headers=headers)
        assert response.status_code == 401, headers
        assert response.json()["detail"] == "Invalid or missing API key"


def test_correct_key_reaches_the_route(authed: TestClient) -> None:
    response = authed.get("/health", headers=_AUTH)
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    missing = authed.get("/jobs/does-not-exist/status", headers=_AUTH)
    assert missing.status_code == 404
    assert "Job not found" in missing.json()["detail"]


def test_docs_surface_closes_when_keyed(authed: TestClient) -> None:
    assert authed.get("/openapi.json").status_code == 404
    assert authed.get("/docs").status_code == 404


def test_open_server_boots_and_serves_docs(open_client: TestClient) -> None:
    response = open_client.get("/openapi.json")
    assert response.status_code == 200


def test_blank_key_is_a_500_not_silent_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UBT_ALLOW_NO_AUTH", "1")
    config = UBTConfig(
        db_dir=tmp_path / "db", allowed_dirs=str(tmp_path), service_api_key=SecretStr("   ")
    )
    with TestClient(create_app(config=config)) as client:
        response = client.get("/jobs/whatever/status")
    assert response.status_code == 500
    assert "blank/whitespace-only" in response.json()["detail"]


def test_boot_refuses_without_key_or_override(tmp_path: Path) -> None:
    config = UBTConfig(db_dir=tmp_path / "db")
    with pytest.raises(SystemExit):
        _require_api_key_gate(config)


def test_boot_strict_mode_with_no_key_fails_fast(tmp_path: Path) -> None:
    config = UBTConfig(db_dir=tmp_path / "db", strict_auth=True)
    with pytest.raises(SystemExit):
        _require_api_key_gate(config)


def test_boot_strict_mode_wins_over_the_no_auth_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UBT_ALLOW_NO_AUTH", "1")
    config = UBTConfig(db_dir=tmp_path / "db", strict_auth=True)
    with pytest.raises(SystemExit):
        _require_api_key_gate(config)


def test_the_session_cookie_is_a_digest_never_the_key(tmp_path: Path) -> None:
    key = "the-service-key"
    value = session_cookie_value(key)
    assert value != key and key not in value
    assert len(value) == 64


def test_session_round_trip_authenticates_headerless_requests(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path)))
    assert client.get("/jobs", headers=_AUTH).status_code == 200

    signed = client.post("/system/session", headers=_AUTH)
    assert signed.status_code == 200
    assert signed.json() == {"authenticated": True}
    cookie = client.cookies.get("ubt_session")
    assert cookie and cookie != _API_KEY

    client.cookies.clear()
    client.cookies.set("ubt_session", cookie)
    assert client.get("/jobs").status_code == 200
    assert client.get("/jobs/nosuchjob00/stream").status_code == 404

    dropped = client.delete("/system/session")
    assert dropped.status_code == 200
    expired = dropped.headers["set-cookie"]
    assert "ubt_session=" in expired and "Max-Age=0" in expired


def test_session_sign_in_requires_a_valid_key(authed: TestClient) -> None:
    assert authed.post("/system/session").status_code == 401
    assert authed.post("/system/session", headers={"X-API-Key": "wrong"}).status_code == 401
    assert "ubt_session" not in authed.cookies


def test_a_forged_session_cookie_is_rejected(authed: TestClient) -> None:
    authed.cookies.set("ubt_session", "0" * 64)
    assert authed.get("/jobs").status_code == 401


def test_submit_rejects_traversal_before_creating_a_job(authed: TestClient) -> None:
    response = authed.post("/jobs/submit", json={"input_path": "../evil.md"}, headers=_AUTH)
    assert response.status_code == 403
    assert "traversal" in response.json()["detail"]


def test_submit_missing_input_is_400(authed: TestClient, tmp_path: Path) -> None:
    response = authed.post(
        "/jobs/submit", json={"input_path": str(tmp_path / "ghost.md")}, headers=_AUTH
    )
    assert response.status_code == 400
    assert "Source file not found" in response.json()["detail"]


def test_invalid_payload_is_422(authed: TestClient) -> None:
    response = authed.post("/jobs/submit", json={}, headers=_AUTH)
    assert response.status_code == 422
    body = response.json()
    assert isinstance(body["detail"], list)


def test_submit_response_shape_and_rehearsal_labelling(authed: TestClient, tmp_path: Path) -> None:
    doc = _write_doc(tmp_path)
    response = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "submitted"
    assert body["rehearsal"] is True
    assert body["stream_url"] == f"/jobs/{body['job_id']}/stream"
    assert body["status_url"] == f"/jobs/{body['job_id']}/status"


def test_status_reaches_a_terminal_state(authed: TestClient, tmp_path: Path) -> None:
    doc = _write_doc(tmp_path)
    created = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH)
    job_id = created.json()["job_id"]
    deadline = time.monotonic() + 30
    status = ""
    while time.monotonic() < deadline:
        response = authed.get(f"/jobs/{job_id}/status", headers=_AUTH)
        assert response.status_code == 200
        status = response.json()["status"]
        if status in _TERMINAL_STATUSES:
            break
        time.sleep(0.2)
    assert status == "completed", f"rehearsal job ended as {status!r}"


def _await_terminal(authed: TestClient, job_id: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = authed.get(f"/jobs/{job_id}/status", headers=_AUTH).json()["status"]
        if status in _TERMINAL_STATUSES:
            return
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} never reached a terminal state")


def _ids(body: str) -> list[int]:
    return [int(line.split(": ", 1)[1]) for line in body.splitlines() if line.startswith("id: ")]


def test_stream_frames_carry_ids_and_replay_from_a_cursor(
    authed: TestClient, tmp_path: Path
) -> None:
    doc = _write_doc(tmp_path)
    job_id = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH).json()[
        "job_id"
    ]
    _await_terminal(authed, job_id)

    first = authed.get(f"/jobs/{job_id}/stream", headers=_AUTH)
    assert first.status_code == 200
    assert first.headers["content-type"].startswith("text/event-stream")
    assert "event: progress" in first.text
    ids = _ids(first.text)
    assert ids, "the stream must stamp each frame with an id: cursor"

    replayed = authed.get(f"/jobs/{job_id}/stream?last_event_id=0", headers=_AUTH)
    replay_ids = _ids(replayed.text)
    assert replay_ids[: len(ids)] == ids
    assert len(replay_ids) > len(ids)


def test_stream_rejects_an_unknown_job(authed: TestClient) -> None:
    assert authed.get("/jobs/nosuchjob00/stream", headers=_AUTH).status_code == 404


def test_resume_rejects_a_completed_job(authed: TestClient, tmp_path: Path) -> None:
    doc = _write_doc(tmp_path)
    job_id = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH).json()[
        "job_id"
    ]
    _await_terminal(authed, job_id)
    assert authed.post(f"/jobs/{job_id}/resume", headers=_AUTH).status_code == 409


def test_resume_keeps_the_output_path_claim_that_submit_guards(
    authed: TestClient, tmp_path: Path
) -> None:
    doc = _write_doc(tmp_path)
    out = tmp_path / "shared_deliverable.md"
    submitted = authed.post(
        "/jobs/submit",
        json={"input_path": str(doc), "output_path": str(out), "job_id": "sharedpath01"},
        headers=_AUTH,
    )
    assert submitted.status_code == 202
    manager: JobManager = authed.app.state.job_manager  # type: ignore[attr-defined]
    stale = manager.create_job(
        JobSubmitRequest(input_path=str(doc), output_path=str(out)), job_id="sharedpath02"
    )
    stale.status = JobStatus.FAILED

    response = authed.post("/jobs/sharedpath02/resume", headers=_AUTH)

    assert response.status_code == 409
    assert "output_path already claimed by live job sharedpath01" in response.json()["detail"]
    refused = manager.get_job("sharedpath02")
    assert refused is not None and refused.status == JobStatus.FAILED


def test_resume_ignores_terminal_holders_of_the_output_path(
    authed: TestClient, tmp_path: Path
) -> None:
    doc = _write_doc(tmp_path)
    out = tmp_path / "serial_deliverable.md"
    manager: JobManager = authed.app.state.job_manager  # type: ignore[attr-defined]
    first = manager.create_job(
        JobSubmitRequest(input_path=str(doc), output_path=str(out), dry_run=True),
        job_id="serialpath1",
    )
    first.status = JobStatus.FAILED
    second = manager.create_job(
        JobSubmitRequest(input_path=str(doc), output_path=str(out), dry_run=True),
        job_id="serialpath2",
    )
    second.status = JobStatus.CANCELLED

    response = authed.post("/jobs/serialpath2/resume", headers=_AUTH)

    assert response.status_code == 200
    _await_terminal(authed, "serialpath2")


def test_resume_unknown_job_is_404(authed: TestClient) -> None:
    assert authed.post("/jobs/nosuchjob00/resume", headers=_AUTH).status_code == 404


def test_resume_after_restart_restores_the_request_from_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.engine.ledger import SQLiteJobLedger

    resumed_requests: list[JobSubmitRequest] = []

    async def _capture(self: JobManager, record: JobRecord, config: Any) -> None:
        resumed_requests.append(record.request)
        record.status = JobStatus.RUNNING

    monkeypatch.setattr(JobManager, "execute_job", _capture)
    doc = _write_doc(tmp_path)
    out = tmp_path / "restart_output.md"
    job_id = "restart-resume-1"
    config = _config(tmp_path)
    with TestClient(create_app(config=config)) as client:
        submitted = client.post(
            "/jobs/submit",
            json={
                "input_path": str(doc),
                "output_path": str(out),
                "job_id": job_id,
                "profile": "textbook",
                "draft_model": "draft-after-restart",
                "repair_model": "repair-after-restart",
                "pages": "1",
            },
            headers=_AUTH,
        )
        assert submitted.status_code == 202, submitted.text

    with SQLiteJobLedger(config.db_dir / f"{job_id}.sqlite") as ledger:
        ledger.finalize_job(job_id, status=JobStatus.FAILED)
        stored = ledger.get_job_metadata_value(job_id, "submitted_request")
    assert isinstance(stored, dict)
    assert stored["job_id"] == job_id
    assert stored["draft_model"] == "draft-after-restart"
    assert stored["profile"] == "textbook"

    with TestClient(create_app(config=_config(tmp_path))) as client:
        resumed = client.post(f"/jobs/{job_id}/resume", headers=_AUTH)

    assert resumed.status_code == 200, resumed.text
    assert len(resumed_requests) == 2
    assert resumed_requests[-1].job_id == job_id
    assert resumed_requests[-1].fresh is False
    assert resumed_requests[-1].output_path == str(out.resolve())
    assert resumed_requests[-1].draft_model == "draft-after-restart"


def test_concurrent_resumes_spawn_exactly_one_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two resumes racing through the claimant await must yield one pipeline.

    The output-path claimant re-check is an ``await`` sitting between the
    FAILED/CANCELLED check and the status flip. Without the resume gate both
    requests pass the check, both flip the shared record, and both spawn a
    pipeline; the loser then dies on the writer lock and its handler flips the
    shared record to FAILED while the winner still runs — a state the cancel
    route refuses to touch, so the winner burns its budget uncancellable.
    """
    import ubt.api.routes.job as job_routes

    captured: list[str] = []

    async def _capture(self: JobManager, record: JobRecord, config: Any) -> None:
        captured.append(record.job_id)
        record.status = JobStatus.RUNNING

    def _slow_claimant(manager: JobManager, queue: Any, job_id: str, claimed: Path) -> None:
        # Hold the await window open long enough that the second request is
        # already inside the handler when the first one reaches the flip.
        time.sleep(0.05)
        return None

    monkeypatch.setattr(JobManager, "execute_job", _capture)
    monkeypatch.setattr(job_routes, "output_path_claimant", _slow_claimant)

    doc = _write_doc(tmp_path)
    app = create_app(config=_config(tmp_path))
    manager: JobManager = app.state.job_manager
    record = manager.create_job(
        JobSubmitRequest(
            input_path=str(doc),
            output_path=str(tmp_path / "race_out.md"),
            dry_run=True,
        ),
        job_id="racejob01",
    )
    record.status = JobStatus.FAILED

    async def _double_resume() -> list[int]:
        # A loopback Host: the trusted-host middleware rejects arbitrary names.
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            responses = await asyncio.gather(
                client.post("/jobs/racejob01/resume", headers=_AUTH),
                client.post("/jobs/racejob01/resume", headers=_AUTH),
            )
        return [response.status_code for response in responses]

    codes = asyncio.run(_double_resume())

    assert sorted(codes) == [200, 409], f"expected one win and one 409, got {codes}"
    assert captured == ["racejob01"]


def test_resume_reconstructs_an_old_ledger_without_a_request_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.engine.ledger import SQLiteJobLedger

    started: list[JobSubmitRequest] = []

    async def _capture(self: JobManager, record: JobRecord, config: Any) -> None:
        started.append(record.request)
        record.status = JobStatus.RUNNING

    monkeypatch.setattr(JobManager, "execute_job", _capture)
    doc = _write_doc(tmp_path)
    job_id = "old-ledger-resume"
    config = _config(tmp_path)
    with SQLiteJobLedger(config.db_dir / f"{job_id}.sqlite") as ledger:
        ledger.init_job_placeholder(job_id, str(doc), "zh", title="Old job")
        ledger.set_job_metadata_value(job_id, "source_lang", "en")
        ledger.set_job_metadata_value(job_id, "genre_profile", "textbook")
        ledger.set_job_metadata_value(
            job_id,
            "config_snapshot",
            {
                "draft_model": "old-draft",
                "repair_model": "old-repair",
                "prompt_strategy": "rich",
            },
        )
        ledger.set_job_metadata_value(
            job_id,
            "run_identity",
            {
                "profile_name": "textbook",
                "engine_signature": "old-engine",
                "mock_run": False,
            },
        )
        ledger.set_job_metadata_value(job_id, "chapter_window", [1, None])
        ledger.set_job_metadata_value(job_id, "selected_pages", [])
        ledger.finalize_job(job_id, status=JobStatus.FAILED)

    with TestClient(create_app(config=_config(tmp_path))) as client:
        resumed = client.post(f"/jobs/{job_id}/resume", headers=_AUTH)

    assert resumed.status_code == 200, resumed.text
    assert len(started) == 1
    request = started[0]
    assert request.job_id == job_id
    assert request.profile == "textbook"
    assert request.draft_model == "old-draft"
    assert request.repair_model == "old-repair"
    assert request.preset == "high_fidelity"
    assert request.fresh is False
    assert request.output_path == str((tmp_path / "db" / "outputs" / job_id).resolve())


def test_resume_refuses_a_live_ledger_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.writer_lock import LedgerWriterLock

    doc = _write_doc(tmp_path)
    job_id = "busy-ledger-resume"
    config = _config(tmp_path)
    db_path = config.db_dir / f"{job_id}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_placeholder(
            job_id,
            str(doc),
            "zh",
            submitted_request={
                "input_path": str(doc),
                "job_id": job_id,
                "dry_run": True,
            },
        )
        ledger.finalize_job(job_id, status=JobStatus.FAILED)

    lock = LedgerWriterLock(db_path, job_id)
    lock.acquire()
    try:
        with TestClient(create_app(config=_config(tmp_path))) as client:
            resumed = client.post(f"/jobs/{job_id}/resume", headers=_AUTH)
    finally:
        lock.release()

    assert resumed.status_code == 409
    assert "live writer" in resumed.json()["detail"]


def test_status_of_unknown_job_is_404(authed: TestClient) -> None:
    response = authed.get("/jobs/does-not-exist/status", headers=_AUTH)
    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]


def test_submit_conflicting_output_is_409(authed: TestClient, tmp_path: Path) -> None:
    doc = _write_doc(tmp_path)
    out = tmp_path / "existing.md"
    out.write_text("already delivered", encoding="utf-8")
    response = authed.post(
        "/jobs/submit",
        json={"input_path": str(doc), "output_path": str(out)},
        headers=_AUTH,
    )
    assert response.status_code == 409
    assert "output_path already exists" in response.json()["detail"]


def test_submit_fresh_does_not_overwrite_an_unrelated_existing_file(
    authed: TestClient, tmp_path: Path
) -> None:
    doc = _write_doc(tmp_path)
    secret = tmp_path / "ubt.toml"
    secret.write_text("[provider]\napi_key = 'do-not-lose-me'\n", encoding="utf-8")
    response = authed.post(
        "/jobs/submit",
        json={"input_path": str(doc), "output_path": str(secret), "fresh": True},
        headers=_AUTH,
    )
    assert response.status_code == 409
    assert "output_path already exists" in response.json()["detail"]
    assert "do-not-lose-me" in secret.read_text(encoding="utf-8")


def test_is_own_prior_output_requires_the_same_job(tmp_path: Path) -> None:
    from ubt.api.app import _is_own_prior_output

    class _Prior:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload

    class _Queue:
        def __init__(self, prior: Any) -> None:
            self._prior = prior

        def get(self, job_id: str) -> Any:
            return self._prior

    class _Manager:
        def get_job(self, job_id: str) -> Any:
            return None

    doc = tmp_path / "book.md"
    doc.write_text("x", encoding="utf-8")
    out = tmp_path / "book.pdf"
    prior = _Prior({"input_path": str(doc), "output_path": str(out)})
    target = resolve_target_output(out, doc)
    assert _is_own_prior_output("job-1", target, _Queue(prior), _Manager()) is True
    assert _is_own_prior_output("job-2", target, _Queue(None), _Manager()) is False
    assert _is_own_prior_output(None, target, _Queue(prior), _Manager()) is False


def test_resubmit_with_same_id_is_idempotent(authed: TestClient, tmp_path: Path) -> None:
    doc = _write_doc(tmp_path)
    payload = {"input_path": str(doc), "job_id": "contract-idempotent-1"}
    first = authed.post("/jobs/submit", json=payload, headers=_AUTH)
    assert first.status_code == 202
    second = authed.post("/jobs/submit", json=payload, headers=_AUTH)
    assert second.status_code == 202
    assert second.json()["job_id"] == first.json()["job_id"]


def test_a_repeated_submit_reports_that_it_reused_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    async def _stay_running(self: JobManager, record: JobRecord, config: Any) -> None:
        record.status = JobStatus.RUNNING

    monkeypatch.setattr(JobManager, "execute_job", _stay_running)
    with TestClient(create_app(config=_config(tmp_path))) as client:
        doc = _write_doc(tmp_path)
        payload = {"input_path": str(doc), "job_id": "contract-reused-1"}

        first = client.post("/jobs/submit", json=payload, headers=_AUTH)
        second = client.post("/jobs/submit", json=payload, headers=_AUTH)

    assert first.status_code == second.status_code == 202
    assert first.json()["reused"] is False, "the first submit must start the run"
    assert second.json()["reused"] is True, "the repeat must not claim to have started it"
    assert second.json()["job_id"] == first.json()["job_id"]


def test_resubmit_with_different_arguments_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    async def _stay_running(self: JobManager, record: JobRecord, config: Any) -> None:
        record.status = JobStatus.RUNNING

    monkeypatch.setattr(JobManager, "execute_job", _stay_running)
    with TestClient(create_app(config=_config(tmp_path))) as client:
        doc = _write_doc(tmp_path)
        first = client.post(
            "/jobs/submit",
            json={
                "input_path": str(doc),
                "job_id": "contract-mismatch-1",
                "target_lang": "zh",
            },
            headers=_AUTH,
        )
        second = client.post(
            "/jobs/submit",
            json={
                "input_path": str(doc),
                "job_id": "contract-mismatch-1",
                "target_lang": "ja",
            },
            headers=_AUTH,
        )

    assert first.status_code == 202
    assert second.status_code == 409
    assert "different arguments" in second.json()["detail"]
    assert "target_lang" in second.json()["detail"]


def test_an_accepted_job_survives_a_restart_before_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    async def _stay_running(self: JobManager, record: JobRecord, config: Any) -> None:
        record.status = JobStatus.RUNNING

    monkeypatch.setattr(JobManager, "execute_job", _stay_running)
    doc = _write_doc(tmp_path)
    job_id = "accepted-restart-1"
    with TestClient(create_app(config=_config(tmp_path))) as client:
        response = client.post(
            "/jobs/submit",
            json={"input_path": str(doc), "job_id": job_id},
            headers=_AUTH,
        )
        assert response.status_code == 202, response.text

    with TestClient(create_app(config=_config(tmp_path))) as client:
        status = client.get(f"/jobs/{job_id}/status", headers=_AUTH)

    assert status.status_code == 200, status.text
    assert status.json()["status"] == "failed"
    assert "server restarted" in (status.json()["error"] or "")


def test_submit_capacity_exceeded_is_429(
    authed: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.manager import JobManager
    from ubt.core.exceptions import ServerCapacityError

    doc = _write_doc(tmp_path)
    monkeypatch.setattr(
        JobManager,
        "create_job",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(
            ServerCapacityError("Server at capacity")
        ),
    )
    response = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH)
    assert response.status_code == 429
    assert "Server at capacity" in response.json()["detail"]


def test_a_cancelled_job_holds_its_slot_until_its_task_settles(tmp_path: Path) -> None:
    from ubt.core.exceptions import ServerCapacityError

    class _Unwinding:
        def done(self) -> bool:
            return False

    class _Settled:
        def done(self) -> bool:
            return True

    manager = JobManager(max_running_jobs=1)
    record = manager.create_job(JobSubmitRequest(input_path=str(_write_doc(tmp_path))))
    record.status = JobStatus.CANCELLED
    record.task = _Unwinding()  # type: ignore[assignment]
    with pytest.raises(ServerCapacityError):
        manager.ensure_capacity()

    record.task = _Settled()  # type: ignore[assignment]
    manager.ensure_capacity()


def test_default_output_lands_in_per_job_outputs_dir(
    authed: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.manager import JobManager
    from ubt.core.exceptions import DocumentParseError

    doc = _write_doc(tmp_path)
    captured: dict[str, Any] = {}

    def _capture(self: JobManager, request: Any, job_id: str | None = None) -> Any:
        captured["request"] = request
        captured["job_id"] = job_id
        raise DocumentParseError("stop before the run starts")

    monkeypatch.setattr(JobManager, "create_job", _capture)
    response = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH)
    assert response.status_code == 400

    job_id = captured["job_id"]
    assert isinstance(job_id, str) and job_id.startswith("job_")
    out = Path(captured["request"].output_path)
    assert out == (tmp_path / "db" / "outputs" / job_id).resolve()
    assert not out.is_dir()
    assert "uploads" not in out.parts


def test_default_output_dir_honors_requested_job_id(
    authed: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.manager import JobManager
    from ubt.core.exceptions import DocumentParseError

    doc = _write_doc(tmp_path)
    captured: dict[str, Any] = {}

    def _capture(self: JobManager, request: Any, job_id: str | None = None) -> Any:
        captured["request"] = request
        captured["job_id"] = job_id
        raise DocumentParseError("stop before the run starts")

    monkeypatch.setattr(JobManager, "create_job", _capture)
    response = authed.post(
        "/jobs/submit", json={"input_path": str(doc), "job_id": "myjob123"}, headers=_AUTH
    )
    assert response.status_code == 400
    assert captured["job_id"] == "myjob123"
    assert (
        Path(captured["request"].output_path)
        == (tmp_path / "db" / "outputs" / "myjob123").resolve()
    )


def test_submit_unsupported_format_is_415(
    authed: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.manager import JobManager
    from ubt.core.exceptions import UnsupportedDocumentFormatError

    doc = _write_doc(tmp_path)
    monkeypatch.setattr(
        JobManager,
        "create_job",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(
            UnsupportedDocumentFormatError("Unsupported format")
        ),
    )
    response = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH)
    assert response.status_code == 415


def test_submit_budget_exceeded_is_402(
    authed: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.manager import JobManager
    from ubt.core.exceptions import BudgetExceededError

    doc = _write_doc(tmp_path)
    monkeypatch.setattr(
        JobManager,
        "create_job",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(BudgetExceededError("Budget exceeded")),
    )
    response = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH)
    assert response.status_code == 402


def test_job_submit_request_carries_emit_both() -> None:
    from ubt.api.models import JobSubmitRequest

    req = JobSubmitRequest(input_path="/tmp/x.pdf", emit_both=True)
    assert req.emit_both is True


def test_output_path_claimant_sees_a_job_still_sitting_in_the_queue(tmp_path: Path) -> None:
    from ubt.api.app import _output_path_claimant
    from ubt.core.engine.job_queue import JobStatus as QueueStatus

    class _Record:
        def __init__(self, job_id: str, output_path: str | None) -> None:
            self.job_id = job_id
            self.status = QueueStatus.RUNNING
            self.request = JobSubmitRequest(input_path="x.md", output_path=output_path)

    class _Manager:
        def __init__(self, records: list[_Record]) -> None:
            self.jobs = {record.job_id: record for record in records}

    class _Queued:
        def __init__(self, job_id: str, status: QueueStatus, output_path: str | None) -> None:
            self.job_id = job_id
            self.status = status
            self.payload = {"output_path": output_path}

    class _Queue:
        def __init__(self, rows: list[_Queued]) -> None:
            self._rows = rows

        def list_jobs(self, *, limit: int = 100) -> list[_Queued]:
            return self._rows

    target = tmp_path / "book_mono.pdf"
    queued = _Queue([_Queued("queued0001", QueueStatus.QUEUED, str(target))])
    assert _output_path_claimant(_Manager([]), queued, "newjob0001", target) == "queued0001"

    done = _Queue([_Queued("donejob001", QueueStatus.COMPLETED, str(target))])
    assert _output_path_claimant(_Manager([]), done, "newjob0001", target) is None
    assert _output_path_claimant(_Manager([]), queued, "queued0001", target) is None

    live = _Manager([_Record("livejob001", str(target))])
    assert _output_path_claimant(live, _Queue([]), "newjob0001", target) == "livejob001"


def test_submit_in_queue_mode_refuses_a_path_a_queued_job_already_claims(tmp_path: Path) -> None:
    from ubt.core.engine.job_queue import JobQueue

    queue = JobQueue(tmp_path / "db" / "job_queue.sqlite")
    try:
        with TestClient(create_app(config=_config(tmp_path), queue=queue)) as client:
            doc = _write_doc(tmp_path)
            out = tmp_path / "shared_mono.md"
            first = client.post(
                "/jobs/submit",
                json={"input_path": str(doc), "output_path": str(out), "job_id": "queuejob001"},
                headers=_AUTH,
            )
            assert first.status_code == 202
            assert first.json()["status"] == "queued"

            second = client.post(
                "/jobs/submit",
                json={"input_path": str(doc), "output_path": str(out), "job_id": "queuejob002"},
                headers=_AUTH,
            )
            assert second.status_code == 409
            assert "already claimed by live job queuejob001" in second.json()["detail"]
    finally:
        queue.close()


def test_queue_mode_stream_serves_a_terminal_job_without_polling_threads(tmp_path: Path) -> None:
    from ubt.core.engine.job_queue import JobQueue, JobStatus

    queue = JobQueue(tmp_path / "db" / "job_queue.sqlite")
    try:
        app = create_app(config=_config(tmp_path), queue=queue)
        with TestClient(app) as client:
            queue.enqueue("donejob0001", {"input_path": str(_write_doc(tmp_path))}, now=1.0)
            claimed = queue.claim("worker01", now=2.0)
            assert claimed is not None
            queue.complete("donejob0001", "worker01", status=JobStatus.COMPLETED, now=3.0)

            assert app.state.sse_broadcaster is not None
            response = client.get("/jobs/donejob0001/stream", headers=_AUTH)
            assert response.status_code == 200
            assert "event: completed" in response.text
    finally:
        queue.close()


def test_is_own_prior_output_directory_licenses_only_the_derived_deliverable(
    tmp_path: Path,
) -> None:
    from ubt.api.app import _is_own_prior_output

    class _Prior:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload

    class _Queue:
        def __init__(self, prior: Any) -> None:
            self._prior = prior

        def get(self, job_id: str) -> Any:
            return self._prior

    class _Manager:
        def get_job(self, job_id: str) -> Any:
            return None

    doc = tmp_path / "book.md"
    doc.write_text("x", encoding="utf-8")
    books = tmp_path / "books"
    books.mkdir()
    prior = _Prior({"input_path": str(doc), "output_path": f"{books}/"})
    queue, manager = _Queue(prior), _Manager()

    own = resolve_target_output(books, doc)
    assert _is_own_prior_output("job-1", own, queue, manager) is True
    victim = books / "someone_elses_manuscript.md"
    assert _is_own_prior_output("job-1", victim, queue, manager) is False


@pytest.mark.asyncio
async def test_job_manager_safe_error_formatting(tmp_path: Path) -> None:
    manager = JobManager()
    dummy_input = tmp_path / "dummy.md"
    dummy_input.write_text("hello", encoding="utf-8")
    req = JobSubmitRequest(input_path=str(dummy_input))
    record = JobRecord("test-err-fmt", req)
    base_config = UBTConfig(
        provider="openai",
        api_key=SecretStr("test-provider-key"),
        service_api_key=SecretStr(_API_KEY),
    )

    with patch("ubt.core.engine.job_runner.PipelineOrchestrator") as mock_orch_cls:

        async def mock_run_budget(*args: object, **kwargs: object) -> AsyncGenerator[Any, None]:
            raise BudgetExceededError("Budget cap of $5.00 exceeded")
            if False:
                yield None

        mock_orch_cls.return_value.run = mock_run_budget
        await manager.execute_job(record, base_config)

    assert record.status == "failed"
    assert record.error == "BudgetExceededError: Budget cap of $5.00 exceeded"

    with patch("ubt.core.engine.job_runner.PipelineOrchestrator") as mock_orch_cls:

        async def mock_run_secret(*args: object, **kwargs: object) -> AsyncGenerator[Any, None]:
            raise RuntimeError("Internal DB host 10.0.0.5 connection refused password=secret")
            if False:
                yield None

        mock_orch_cls.return_value.run = mock_run_secret
        await manager.execute_job(record, base_config)

    assert "password=secret" not in (record.error or "")
    assert record.error == f"RuntimeError (see server logs; job_id={record.job_id})"


def test_job_submit_request_accepts_pdf_engine() -> None:
    req = JobSubmitRequest(input_path="book.pdf", pdf_engine="pdfium")
    assert req.pdf_engine == "pdfium"

    req_auto = JobSubmitRequest(input_path="book.pdf", pdf_engine="auto")
    assert req_auto.pdf_engine == "auto"


def test_reparse_is_the_one_way_a_finished_job_is_parsed_again(
    authed: TestClient, tmp_path: Path
) -> None:
    doc = _write_doc(tmp_path)
    job_id = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH).json()[
        "job_id"
    ]
    _await_terminal(authed, job_id)

    res = authed.post(f"/jobs/{job_id}/reparse", headers=_AUTH)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["job_id"] == job_id
    assert body["status"] in ("submitted", "queued", "running")

    _await_terminal(authed, job_id)


def test_reparse_refuses_a_job_that_is_still_running(authed: TestClient, tmp_path: Path) -> None:
    doc = _write_doc(tmp_path)
    job_id = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH).json()[
        "job_id"
    ]
    res = authed.post(f"/jobs/{job_id}/reparse", headers=_AUTH)
    assert res.status_code == 409
    assert "only a finished job" in res.json()["detail"].lower()
    _await_terminal(authed, job_id)


def test_reparse_refuses_an_unknown_job(authed: TestClient) -> None:
    res = authed.post("/jobs/unknownjob99/reparse", headers=_AUTH)
    assert res.status_code == 404


def test_concurrent_submits_of_one_output_path_spawn_one_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two submits racing the claimant await must yield one accepted job.

    The claimant scan is a to_thread snapshot; without the submit side of the
    claim gate, two concurrent submits of one output_path could both observe
    "unclaimed" and both spawn a pipeline onto the same deliverable.
    """
    import ubt.api.routes.job as job_routes

    spawned: list[str] = []

    async def _capture(self: JobManager, record: JobRecord, config: Any) -> None:
        spawned.append(record.job_id)
        record.status = JobStatus.RUNNING

    # Delay, then delegate to the REAL scan: the rejection under test comes
    # from the claimant scan itself, so stubbing it to None would stub away
    # the assertion.
    real_claimant = job_routes.output_path_claimant

    def _slow_claimant(manager: JobManager, queue: Any, job_id: str, claimed: Path) -> Any:
        # Hold the scan window open so the second request is already waiting
        # on the claim gate when the first one commits.
        time.sleep(0.05)
        return real_claimant(manager, queue, job_id, claimed)

    monkeypatch.setattr(JobManager, "execute_job", _capture)
    monkeypatch.setattr(job_routes, "output_path_claimant", _slow_claimant)

    doc = _write_doc(tmp_path)
    app = create_app(config=_config(tmp_path))

    async def _double_submit() -> list[int]:
        # A loopback Host: the trusted-host middleware rejects arbitrary names.
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            responses = await asyncio.gather(
                client.post(
                    "/jobs/submit",
                    json={
                        "input_path": str(doc),
                        "output_path": str(tmp_path / "clash.md"),
                        "job_id": "clashjob01",
                        "dry_run": True,
                    },
                    headers=_AUTH,
                ),
                client.post(
                    "/jobs/submit",
                    json={
                        "input_path": str(doc),
                        "output_path": str(tmp_path / "clash.md"),
                        "job_id": "clashjob02",
                        "dry_run": True,
                    },
                    headers=_AUTH,
                ),
            )
        return [r.status_code for r in responses]

    codes = asyncio.run(_double_submit())

    assert sorted(codes) == [202, 409], f"expected one accept and one 409, got {codes}"
    assert spawned == ["clashjob01"]

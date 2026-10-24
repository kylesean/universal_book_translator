"""REST contract tripwire: the API surface's guarantees, pinned as fast tests.

The REST surface was the one delivery path with zero coverage; these tests are
the minimal defense — the *contract*, not the implementation:

- the auth gate (``X-API-Key``, opt-in, ``/health`` exempt, docs closed when keyed);
- the path sandbox (traversal, system dirs, sensitive names, allowlist containment);
- the job intake contract (202 shape, rehearsal auto-labelling, sandbox rejection
  before creation, output-conflict 409, idempotent resubmit, terminal status).

Everything runs in-process over the ASGI transport with injected configuration —
no network, no provider key, no model spend (keyless submission auto-runs as a
zero-token rehearsal, which is itself one of the pinned contracts).
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
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
    """The keyed app: auth enforced, docs surface closed."""
    with TestClient(create_app(config=_config(tmp_path))) as client:
        yield client


@pytest.fixture()
def open_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The named escape hatch: no key + ``UBT_ALLOW_NO_AUTH`` boots open."""
    monkeypatch.setenv("UBT_ALLOW_NO_AUTH", "1")
    config = UBTConfig(db_dir=tmp_path / "db", allowed_dirs=str(tmp_path))
    with TestClient(create_app(config=config)) as client:
        yield client


# --------------------------------------------------------------------------- #
# Path sandbox (pure guard — no HTTP).
# --------------------------------------------------------------------------- #


def test_sandbox_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as err:
        resolve_secure_path(tmp_path / "sub" / ".." / "evil.md", config=_config(tmp_path))
    assert err.value.status_code == 403
    assert "traversal" in str(err.value.detail)


def test_sandbox_rejects_system_dirs_by_default(tmp_path: Path) -> None:
    # No operator allowlist: the fallback sandbox (cwd + db_dir) plus the
    # hard-coded system deny list protect /etc even though the containment
    # rule alone would also refuse it.
    config = UBTConfig(db_dir=tmp_path / "db")
    with pytest.raises(HTTPException) as err:
        resolve_secure_path("/etc/passwd", config=config)
    assert err.value.status_code == 403
    assert "restricted system directory" in str(err.value.detail)


def test_sandbox_sensitive_names_beat_the_allowlist(tmp_path: Path) -> None:
    # An allowlist is a scope decision, not a licence to serve secrets that
    # live beside the books (.ssh/.env/credentials.json, casefolded).
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


# --------------------------------------------------------------------------- #
# Auth gate.
# --------------------------------------------------------------------------- #


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
    # Auth passes and the request reaches the handler. Assert on a route that
    # must succeed for a valid key: an unknown-id 404 also holds when the whole
    # authenticated surface is broken (e.g. a router that 404s everything), so
    # it proves nothing about the gate letting a request through.
    response = authed.get("/health", headers=_AUTH)
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    # And an authenticated route answers as its handler intends (404 for an
    # unknown job, with the handler's detail rather than the auth gate's).
    missing = authed.get("/jobs/does-not-exist/status", headers=_AUTH)
    assert missing.status_code == 404
    assert "Job not found" in missing.json()["detail"]


def test_docs_surface_closes_when_keyed(authed: TestClient) -> None:
    # A keyed server must not publish its schema to unauthenticated readers.
    assert authed.get("/openapi.json").status_code == 404
    assert authed.get("/docs").status_code == 404


def test_open_server_boots_and_serves_docs(open_client: TestClient) -> None:
    response = open_client.get("/openapi.json")
    assert response.status_code == 200


def test_blank_key_is_a_500_not_silent_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A set-but-whitespace key is a misconfiguration: refuse requests instead
    # of silently dropping the gate the operator believes is on. (Non-health
    # route: /health is exempt from the gate entirely.)
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
    # Strict mode is a deployment promise; UBT_ALLOW_NO_AUTH is a process
    # convenience that tools set implicitly (``ubt console`` exports it). The
    # override winning let a strict production server boot open whenever
    # anything set the variable.
    monkeypatch.setenv("UBT_ALLOW_NO_AUTH", "1")
    config = UBTConfig(db_dir=tmp_path / "db", strict_auth=True)
    with pytest.raises(SystemExit):
        _require_api_key_gate(config)


def test_the_session_cookie_is_a_digest_never_the_key(tmp_path: Path) -> None:
    # The cookie rides requests a browser cannot attach headers to (SSE,
    # previews, downloads). It must not be a replayable copy of the key.
    key = "the-service-key"
    value = session_cookie_value(key)
    assert value != key and key not in value
    assert len(value) == 64  # sha256 hexdigest


def test_session_round_trip_authenticates_headerless_requests(tmp_path: Path) -> None:
    # Sign in with the header, then drop it: the cookie alone must carry an
    # SSE-stream request (the exact case EventSource cannot header-authenticate)
    # and a normal API call, and clearing it must close the gate again.
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
    assert client.get("/jobs/nosuchjob00/stream").status_code == 404  # past the gate

    # Logout is a client-side clear: the response must expire the cookie on the
    # browser. (The value is a deterministic digest of the key, so the server
    # cannot invalidate a copied cookie -- rotating the key is the revocation
    # path, exactly as it is for the header itself.)
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


# --------------------------------------------------------------------------- #
# Job intake contract.
# --------------------------------------------------------------------------- #


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
    # No injected router and no provider key: the intake must auto-label the
    # run as a rehearsal so a keyless server can never report a mock as real.
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
    # A zero-token rehearsal over a two-line markdown must finish cleanly.
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
    # A finished rehearsal job gives a bounded, deterministic stream.
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

    # Reconnecting with a cursor replays the missed frames before the snapshot.
    replayed = authed.get(f"/jobs/{job_id}/stream?last_event_id=0", headers=_AUTH)
    replay_ids = _ids(replayed.text)
    assert replay_ids[: len(ids)] == ids
    assert len(replay_ids) > len(ids)  # ...then a fresh snapshot follows


def test_stream_rejects_an_unknown_job(authed: TestClient) -> None:
    assert authed.get("/jobs/nosuchjob00/stream", headers=_AUTH).status_code == 404


def test_resume_rejects_a_completed_job(authed: TestClient, tmp_path: Path) -> None:
    doc = _write_doc(tmp_path)
    job_id = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH).json()[
        "job_id"
    ]
    _await_terminal(authed, job_id)
    # Only a failed/cancelled job resumes; a completed one is a 409.
    assert authed.post(f"/jobs/{job_id}/resume", headers=_AUTH).status_code == 409


def test_resume_keeps_the_output_path_claim_that_submit_guards(
    authed: TestClient, tmp_path: Path
) -> None:
    # While a job sits failed/cancelled its output path is up for grabs, so a
    # fresh submit may already own it. Resuming anyway would run two pipelines
    # into one deliverable (last writer wins) — the same collision submit
    # refuses. The claim is re-checked at resume.
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
    # The refusal left the record resumable, like the capacity check.
    refused = manager.get_job("sharedpath02")
    assert refused is not None and refused.status == JobStatus.FAILED


def test_resume_ignores_terminal_holders_of_the_output_path(
    authed: TestClient, tmp_path: Path
) -> None:
    # The mirror of the guard: a failed/cancelled holder does not block, else
    # re-running a failed job would be impossible once any earlier job failed.
    doc = _write_doc(tmp_path)
    out = tmp_path / "serial_deliverable.md"
    manager: JobManager = authed.app.state.job_manager  # type: ignore[attr-defined]
    # dry_run mirrors what the keyed app itself would set for a keyless submit
    # (zero-token rehearsal), so the resumed run completes without a provider.
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
    """A restart must not turn a resumable ledger into a 404.

    The manager is process memory; the ledger is the durable record. The
    accepted request is persisted with the placeholder, so the restarted app can
    reconstruct the exact model/profile/output choices and continue rather than
    asking the operator to resubmit and potentially mix a different run into the
    same checkpoints.
    """
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


def test_resume_reconstructs_an_old_ledger_without_a_request_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-snapshot ledgers remain resumable when their run identity is complete."""
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
    """A restart must not become a second writer of a live job."""
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
    # ``fresh`` resumes the ledger; it must not license clobbering an arbitrary
    # file inside the sandbox (ubt.toml, job_queue.sqlite, another job's output).
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
    # The colliding file must be untouched.
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
    # A different job id may not overwrite another job's output.
    assert _is_own_prior_output("job-2", target, _Queue(None), _Manager()) is False
    # No job id at all is never authorized.
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
    """``reused`` is how a caller tells "I started this" from "it already existed".

    The status code and the job id are identical either way, so without this flag
    a double-clicked submit is indistinguishable from a fresh run — and the
    console navigated to a progress page for a job whose bar was already at 60%.

    The record is held non-terminal by stubbing ``execute_job``: a real
    ``TestClient`` cancels the background task when the request scope ends, and a
    ``cancelled`` id is deliberately *re-runnable* (resubmitting a dead id is a
    request to run it again), so it would take the fresh-run branch and prove
    nothing about reuse.
    """

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


def test_an_accepted_job_survives_a_restart_before_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Submit must leave a durable row before the background task can be lost.

    The SOCKS-proxy failure happened after intake but before ingest created a
    ledger. Restarting the console erased the only in-memory record, so the id
    the client had just received answered 404 forever. The placeholder must
    survive and report a terminal failure instead.
    """

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
    """The cancel window must not free a capacity slot.

    ``cancel_job`` flips the status to CANCELLED and cancels the task, but the
    coroutine unwinds asynchronously (releasing the writer lock, closing
    provider connections). ``ensure_capacity`` counted by status alone, so a
    submit landing in that window saw a free slot and could start a pipeline
    while the cancelled one was still tearing down — breaching the cap.
    """
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

    # Once the cancelled task has actually settled, the slot is genuinely free.
    record.task = _Settled()  # type: ignore[assignment]
    manager.ensure_capacity()


def test_default_output_lands_in_per_job_outputs_dir(
    authed: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``output_path`` → the API derives ``db_dir/outputs/<job_id>/<stem>_bilingual<ext>``.

    Deliverables must not sit beside the staged upload (sources and
    translations mixed in one directory) nor silently in the caller's cwd —
    the per-job directory is the contract the console's deliverable links and
    the operator's cleanup scripts both rely on.
    """
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
    # The directory exists before the worker starts, and the source staging
    # area stays free of deliverables.
    assert out.is_dir()
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
    # ``manager.jobs`` only knows what this process started; in queue mode the
    # rival claimant is usually still QUEUED, and the in-memory scan alone let
    # both submissions through.
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

    # A retired row's claim is up for grabs, and the submitter's own id is not a
    # rival to itself.
    done = _Queue([_Queued("donejob001", QueueStatus.COMPLETED, str(target))])
    assert _output_path_claimant(_Manager([]), done, "newjob0001", target) is None
    assert _output_path_claimant(_Manager([]), queued, "queued0001", target) is None

    # The in-memory map still wins its own case.
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
    """The queue-mode stream routes through the shared broadcaster, not a poll pool.

    A terminal job must answer immediately with its terminal frame: the
    broadcaster fans the row out from its own poll loop and the subscriber's
    first frame is the snapshot the route already read. This pins the route's
    wiring to ``QueueSseBroadcaster`` (the per-stream ``job_queue.get`` polling
    and the 4-thread pool are gone).
    """
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
    """A prior run that named a *directory* licenses only its derived deliverable.

    The reported defect: ``_is_own_prior_output``'s directory branch compared
    only the parent directory, so a job id that once ran with
    ``output_path='/books/'`` could then overwrite ``/books/anything.md`` — a
    sibling job's output or a human's manuscript. The licence is now the exact
    derived deliverable name(s), never the whole tree.
    """
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

    # The derived deliverable inside the directory IS the job's own output.
    own = resolve_target_output(books, doc)
    assert _is_own_prior_output("job-1", own, queue, manager) is True
    # A different file in that directory is NOT.
    victim = books / "someone_elses_manuscript.md"
    assert _is_own_prior_output("job-1", victim, queue, manager) is False


@pytest.mark.asyncio
async def test_job_manager_safe_error_formatting(tmp_path: Path) -> None:
    """Verify safe domain errors (e.g. BudgetExceededError) are exposed directly while generic ones are masked."""
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

    # 1. BudgetExceededError should be directly readable in record.error
    with patch("ubt.core.engine.job_runner.PipelineOrchestrator") as mock_orch_cls:

        async def mock_run_budget(*args: object, **kwargs: object) -> AsyncGenerator[Any, None]:
            raise BudgetExceededError("Budget cap of $5.00 exceeded")
            if False:
                yield None

        mock_orch_cls.return_value.run = mock_run_budget
        await manager.execute_job(record, base_config)

    assert record.status == "failed"
    assert record.error == "BudgetExceededError: Budget cap of $5.00 exceeded"

    # 2. Sensitive internal exceptions must be masked to prevent data/credential leaks
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
    """Verify JobSubmitRequest validates and accepts pdf_engine."""
    req = JobSubmitRequest(input_path="book.pdf", pdf_engine="pdfium")
    assert req.pdf_engine == "pdfium"

    req_auto = JobSubmitRequest(input_path="book.pdf", pdf_engine="auto")
    assert req_auto.pdf_engine == "auto"


def test_reparse_is_the_one_way_a_finished_job_is_parsed_again(
    authed: TestClient, tmp_path: Path
) -> None:
    """Re-rendering and resuming both keep the old parse; only this discards it.

    A parse-stage fix (block splitting, table reconstruction, figure and formula
    crops) cannot reach a finished job any other way: ``resume`` explicitly
    refuses to re-fresh because that would throw away the checkpoints it exists
    to keep, and a re-export re-runs only the export stage over the text the
    ledger already holds — so the delivered file keeps the artefacts the old
    code cut, however many times it is rendered.
    """
    doc = _write_doc(tmp_path)
    job_id = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH).json()[
        "job_id"
    ]
    _await_terminal(authed, job_id)

    res = authed.post(f"/jobs/{job_id}/reparse", headers=_AUTH)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["job_id"] == job_id
    # It starts a new run rather than returning the finished one.
    assert body["status"] in ("submitted", "queued", "running")

    _await_terminal(authed, job_id)


def test_reparse_refuses_a_job_that_is_still_running(authed: TestClient, tmp_path: Path) -> None:
    """Discarding a live run's ledger would corrupt it from underneath."""
    doc = _write_doc(tmp_path)
    job_id = authed.post("/jobs/submit", json={"input_path": str(doc)}, headers=_AUTH).json()[
        "job_id"
    ]
    # Submit and re-parse immediately: the run is certainly not finished yet.
    res = authed.post(f"/jobs/{job_id}/reparse", headers=_AUTH)
    assert res.status_code == 409
    assert "only a finished job" in res.json()["detail"].lower()
    _await_terminal(authed, job_id)


def test_reparse_refuses_an_unknown_job(authed: TestClient) -> None:
    """A job with no in-session record must be resubmitted, not silently
    reconstructed: the ledger does not carry the model/preset/output choices,
    so a rebuilt request would run a different job than the one asked for."""
    res = authed.post("/jobs/unknownjob99/reparse", headers=_AUTH)
    assert res.status_code == 404

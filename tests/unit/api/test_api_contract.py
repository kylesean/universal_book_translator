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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.api.security import _require_api_key_gate, resolve_secure_path
from ubt.core.config import UBTConfig

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
    # Auth passes and the request hits the route itself: unknown id is a 404
    # from the handler, not a 401 from the gate.
    response = authed.get("/jobs/does-not-exist/status", headers=_AUTH)
    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]


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


def test_resubmit_with_same_id_is_idempotent(authed: TestClient, tmp_path: Path) -> None:
    doc = _write_doc(tmp_path)
    payload = {"input_path": str(doc), "job_id": "contract-idempotent-1"}
    first = authed.post("/jobs/submit", json=payload, headers=_AUTH)
    assert first.status_code == 202
    second = authed.post("/jobs/submit", json=payload, headers=_AUTH)
    assert second.status_code == 202
    assert second.json()["job_id"] == first.json()["job_id"]

"""Contract for POST /jobs/upload — the web console's real upload channel.

Browsers cannot hand the server a filesystem path (``File.path`` is an
Electron-only property), so the wizard stages document bytes here and submits
the returned ``file_path``. These tests pin that the staged path is real,
sandbox-valid, format-gated, and size-capped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import ubt.api.app as app_module
from ubt.api.app import UPLOAD_MAX_BYTES, create_app
from ubt.api.security import resolve_secure_path
from ubt.core.config import UBTConfig

pytestmark = pytest.mark.fast

_API_KEY = "upload-test-key"
_AUTH = {"X-API-Key": _API_KEY}


def _config(tmp_path: Path) -> UBTConfig:
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    config.db_dir.mkdir(parents=True, exist_ok=True)
    return config


def test_upload_stages_file_inside_sandbox(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path)))
    res = client.post(
        "/jobs/upload",
        files={"file": ("My Book.md", b"# Hello", "text/markdown")},
        headers=_AUTH,
    )
    assert res.status_code == 201
    body = res.json()
    assert body["file_name"] == "My Book.md"
    assert body["size_bytes"] == len(b"# Hello")
    staged = Path(body["file_path"])
    # The stored bytes are the uploaded bytes…
    assert staged.read_bytes() == b"# Hello"
    # …under db_dir/uploads with the original suffix preserved…
    assert staged.parent == (tmp_path / "db" / "uploads").resolve()
    assert staged.suffix == ".md"
    # …and the returned path passes the sandbox validator used by
    # /jobs/assess and /jobs/submit (the whole point of the channel).
    assert (
        resolve_secure_path(body["file_path"], must_exist=True, config=_config(tmp_path)) == staged
    )


def test_upload_rejects_unsupported_format(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path)))
    res = client.post(
        "/jobs/upload",
        files={"file": ("payload.exe", b"MZ", "application/octet-stream")},
        headers=_AUTH,
    )
    assert res.status_code == 415
    assert "Unsupported document format" in res.json()["detail"]


def test_upload_rejects_oversized_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "UPLOAD_MAX_BYTES", 4)
    client = TestClient(create_app(_config(tmp_path)))
    res = client.post(
        "/jobs/upload",
        files={"file": ("big.md", b"x" * 10, "text/markdown")},
        headers=_AUTH,
    )
    assert res.status_code == 413
    assert "exceeds" in res.json()["detail"]
    # The partial file is not left behind.
    uploads = (tmp_path / "db" / "uploads").resolve()
    assert not any(uploads.glob("*.md")) if uploads.exists() else True


def test_upload_names_do_not_collide_or_traverse(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path)))
    paths = set()
    for _ in range(2):
        res = client.post(
            "/jobs/upload",
            files={"file": ("../weird name!.md", b"data", "text/markdown")},
            headers=_AUTH,
        )
        assert res.status_code == 201
        paths.add(res.json()["file_path"])
    assert len(paths) == 2
    for raw in paths:
        staged = Path(raw)
        assert ".." not in staged.parts
        assert staged.exists()


def test_upload_requires_auth(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path)))
    res = client.post(
        "/jobs/upload",
        files={"file": ("book.md", b"data", "text/markdown")},
    )
    assert res.status_code in (401, 403)


def test_upload_max_constant_is_sane() -> None:
    assert UPLOAD_MAX_BYTES == 512 * 1024 * 1024

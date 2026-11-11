"""The console mount serves a public-asset allowlist, nothing else.

F7 regression: the SPA mount sat outside the API-key dependency by
construction (FastAPI dependencies do not apply to ``app.mount``), so anything
later dropped into ``ubt/api/static`` — a ``.env``, a secret ``.txt``, a
database — was served unauthenticated. The mount now refuses dotfiles, any
non-public extension, and planted extension-less files.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.fast


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    """A console-like bundle with planted non-public files beside the assets."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app-abc123.js").write_text("console.log('bundle')", encoding="utf-8")
    (tmp_path / "index.html").write_text("<html><body>shell</body></html>", encoding="utf-8")
    (tmp_path / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "robots.txt").write_text("User-agent: *", encoding="utf-8")
    # Planted secrets of every refusal class.
    (tmp_path / ".env").write_text("UBT_API_KEY=real-key", encoding="utf-8")
    (tmp_path / ".secrets").mkdir()
    (tmp_path / ".secrets" / "secret.txt").write_text("hidden", encoding="utf-8")
    (tmp_path / "secret.txt").write_text(
        "extension is public, name is not a route", encoding="utf-8"
    )
    (tmp_path / "config.yaml").write_text("key: value", encoding="utf-8")
    (tmp_path / "db.sqlite").write_bytes(b"SQLite format 3")
    (tmp_path / "DROPHERE").write_text("extension-less planted file", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def client(bundle: Path) -> Iterator[TestClient]:
    from ubt.api.app import _SPAStaticFiles

    app = FastAPI()
    app.mount("/", _SPAStaticFiles(directory=str(bundle), html=True), name="console")
    with TestClient(app) as c:
        yield c


def test_bundle_assets_serve(client: TestClient) -> None:
    response = client.get("/assets/app-abc123.js")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"


def test_the_shell_serves_and_is_never_cached(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "shell" in response.text
    assert response.headers["Cache-Control"] == "no-cache"


def test_public_text_and_icons_serve(client: TestClient) -> None:
    assert client.get("/robots.txt").status_code == 200
    assert client.get("/favicon.svg").status_code == 200


def test_a_planted_dotenv_returns_404(client: TestClient) -> None:
    assert client.get("/.env").status_code == 404


def test_secrets_inside_a_dot_directory_return_404(client: TestClient) -> None:
    assert client.get("/.secrets/secret.txt").status_code == 404


def test_a_planted_non_public_extension_returns_404(client: TestClient) -> None:
    assert client.get("/config.yaml").status_code == 404
    assert client.get("/db.sqlite").status_code == 404


def test_a_planted_extension_less_file_returns_404(client: TestClient) -> None:
    assert client.get("/DROPHERE").status_code == 404


def test_a_client_route_falls_back_to_the_shell_for_a_browser(client: TestClient) -> None:
    response = client.get("/wizard", headers={"Accept": "text/html"})
    assert response.status_code == 200
    assert "shell" in response.text
    assert response.headers["Cache-Control"] == "no-cache"


def test_an_api_client_still_gets_a_json_404(client: TestClient) -> None:
    """A non-HTML client probing a missing surface must not receive the shell.

    The probe path must not collide with a route the host app registers itself:
    a bare ``FastAPI()`` serves ``/docs`` from its own table before the mount
    is consulted, which would make this test pass for the wrong reason.
    """
    response = client.get("/api/typo", headers={"Accept": "application/json"})
    assert response.status_code == 404
    assert "shell" not in response.text


def test_a_traversal_attempt_returns_404(client: TestClient) -> None:
    assert client.get("/..%2f..%2fetc%2fpasswd").status_code == 404


def test_the_real_console_mount_still_serves_the_shell(tmp_path: Path) -> None:
    """The shipped static dir, when present, keeps its index.html reachable."""
    static_dir = Path(__file__).resolve().parents[3] / "ubt" / "api" / "static"
    if not (static_dir / "index.html").exists():
        pytest.skip("console bundle not built in this checkout")
    from ubt.api.app import _SPAStaticFiles

    app = FastAPI()
    app.mount("/", _SPAStaticFiles(directory=str(static_dir), html=True), name="console")
    with TestClient(app) as c:
        assert c.get("/").status_code == 200

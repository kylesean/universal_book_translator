"""The session cookie: per-install secret, expiry, Secure flag, rotation.

F4 regression: the cookie was a static HMAC over the API key alone, set with
``secure=False`` and no expiry — it never rotated and a cookie leaked over
plain http stayed valid forever. It is now keyed over a random per-install
secret (0600, rotatable), carries a one-week ``Max-Age``, and is ``Secure``
whenever the request arrived over https (or the deployment says production).
"""

from __future__ import annotations

import stat
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

pytestmark = pytest.mark.fast

from ubt.core.config import UBTConfig  # noqa: E402

_API_KEY = "test-key-cookie"
_AUTH = {"X-API-Key": _API_KEY}


def _config(tmp_path: Path) -> UBTConfig:
    return UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )


@pytest.fixture()
def keyed(tmp_path: Path) -> Iterator[TestClient]:
    from ubt.api.app import create_app

    with TestClient(create_app(config=_config(tmp_path))) as client:
        yield client


def _secret_path(tmp_path: Path) -> Path:
    return tmp_path / "db" / "session_secret.key"


def _mint(client: TestClient) -> str:
    response = client.post("/system/session", headers=_AUTH)
    assert response.status_code == 200
    cookie = client.cookies.get("ubt_session")
    assert cookie
    return cookie


def test_the_cookie_carries_expiry_httponly_and_samesite(keyed: TestClient) -> None:
    cookie = _mint(keyed)
    assert cookie != _API_KEY and _API_KEY not in cookie
    raw = keyed.post("/system/session", headers=_AUTH).headers["set-cookie"]
    assert "HttpOnly" in raw
    assert "SameSite=lax" in raw
    assert "Max-Age=604800" in raw
    # A plain-http console must still receive the cookie: a Secure cookie over
    # http is silently dropped by the browser.
    assert "Secure" not in raw


def test_the_cookie_is_secure_over_https(tmp_path: Path) -> None:
    from ubt.api.app import create_app

    with TestClient(create_app(config=_config(tmp_path)), base_url="https://testserver") as client:
        raw = client.post("/system/session", headers=_AUTH).headers["set-cookie"]
    assert "Secure" in raw


def test_production_env_forces_the_secure_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.app import create_app

    monkeypatch.setenv("UBT_ENV", "production")
    with TestClient(create_app(config=_config(tmp_path))) as client:
        raw = client.post("/system/session", headers=_AUTH).headers["set-cookie"]
    assert "Secure" in raw


def test_the_cookie_secure_env_overrides_the_scheme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.app import create_app

    monkeypatch.setenv("UBT_COOKIE_SECURE", "1")
    with TestClient(create_app(config=_config(tmp_path))) as client:
        assert "Secure" in client.post("/system/session", headers=_AUTH).headers["set-cookie"]

    monkeypatch.setenv("UBT_COOKIE_SECURE", "0")
    with TestClient(create_app(config=_config(tmp_path)), base_url="https://testserver") as client:
        raw = client.post("/system/session", headers=_AUTH).headers["set-cookie"]
    assert "Secure" not in raw


def test_the_secret_file_is_owner_only_and_created_once(tmp_path: Path) -> None:
    from ubt.api.security import load_or_create_session_secret

    first = load_or_create_session_secret(_config(tmp_path))
    second = load_or_create_session_secret(_config(tmp_path))
    assert first == second
    mode = stat.S_IMODE(_secret_path(tmp_path).stat().st_mode)
    assert mode == 0o600


def test_the_value_changes_after_secret_rotation(tmp_path: Path) -> None:
    from ubt.api.security import load_or_create_session_secret, session_cookie_value

    config = _config(tmp_path)
    secret_a = load_or_create_session_secret(config)
    value_a = session_cookie_value(_API_KEY, session_secret=secret_a)

    # The documented rotation: delete the file; the next boot mints a fresh one.
    _secret_path(tmp_path).unlink()
    secret_b = load_or_create_session_secret(config)
    value_b = session_cookie_value(_API_KEY, session_secret=secret_b)

    assert secret_a != secret_b
    assert value_a != value_b


def test_an_old_cookie_is_rejected_after_secret_rotation(tmp_path: Path, keyed: TestClient) -> None:
    old_cookie = _mint(keyed)
    client = keyed
    # Send each cookie explicitly rather than through the shared jar: the
    # session endpoint sets the cookie on every call, so the jar accumulates a
    # second entry under the same name and httpx raises CookieConflict on
    # lookup. An explicit cookie overrides the jar for that request.
    client.cookies.clear()

    # A minted cookie authenticates headerless requests...
    assert client.get("/jobs", cookies={"ubt_session": old_cookie}).status_code == 200

    # ...until the per-install secret is rotated out from under it.
    _secret_path(tmp_path).unlink()
    assert client.get("/jobs", cookies={"ubt_session": old_cookie}).status_code == 401

    # A fresh mint against the new secret authenticates again.
    fresh = _mint(client)
    client.cookies.clear()
    assert fresh != old_cookie
    assert client.get("/jobs", cookies={"ubt_session": fresh}).status_code == 200


def test_changing_the_api_key_rotates_the_cookie(tmp_path: Path) -> None:
    from ubt.api.security import session_cookie_value

    secret = b"x" * 32
    assert session_cookie_value("key-a", session_secret=secret) != session_cookie_value(
        "key-b", session_secret=secret
    )

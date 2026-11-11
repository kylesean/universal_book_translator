"""The proxy-aware loopback decision for credential/base_url writes.

F2 regression: the credential gate answered from the socket *bind*, so behind
the reverse-proxy posture (a proxy bound to the same loopback host) every
remote caller looked local and a caller holding the API key could re-point a
provider at an attacker host and harvest the stored key. Forwarded headers now
mark such requests non-local unless ``UBT_TRUST_PROXY`` declares the front
proxy trusted — and a spoofed forwarded header without that declaration is
denied exactly like any other forged claim.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

pytestmark = pytest.mark.fast

_API_KEY = "test-key-proxy"
_AUTH = {"X-API-Key": _API_KEY}

_CREDENTIAL_BLOCK = """\
[providers.demo]
base_url = "https://api.example/v1"
api_mode = "openai-chat"
api_key = "sk-REAL-SECRET"
draft_model = "demo-model"
"""


@pytest.fixture()
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from ubt.api.app import create_app
    from ubt.core.config import UBTConfig

    (tmp_path / "ubt.toml").write_text(_CREDENTIAL_BLOCK, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UBT_TRUST_PROXY", raising=False)
    config = UBTConfig(
        db_dir=tmp_path / "db", allowed_dirs=str(tmp_path), service_api_key=SecretStr(_API_KEY)
    )
    with TestClient(create_app(config=config)) as client:
        yield client


def test_forwarded_headers_without_trust_deny_the_credential_write(
    app_client: TestClient, tmp_path: Path
) -> None:
    response = app_client.put(
        "/system/providers/demo",
        json={"api_key": "sk-via-proxy"},
        headers={**_AUTH, "X-Forwarded-For": "203.0.113.5"},
    )
    assert response.status_code == 400
    assert "credential cannot be written" in response.json()["detail"]
    assert "sk-via-proxy" not in (tmp_path / "ubt.toml").read_text(encoding="utf-8")


def test_a_spoofed_loopback_forward_is_denied_while_trust_is_off(
    app_client: TestClient,
) -> None:
    """The forgeable claim must never widen the decision: presence means proxied."""
    for headers in (
        {"X-Forwarded-For": "127.0.0.1"},
        {"X-Forwarded-For": "127.0.0.1, 10.0.0.1"},
        {"X-Real-IP": "127.0.0.1"},
    ):
        response = app_client.put(
            "/system/providers/demo", json={"api_key": "sk-spoofed"}, headers={**_AUTH, **headers}
        )
        assert response.status_code == 400, headers


def test_an_absent_forwarded_header_is_not_proxied(app_client: TestClient) -> None:
    """An empty header value is absent, not a proxy claim."""
    response = app_client.put(
        "/system/providers/demo",
        json={"api_key": "sk-empty-header"},
        headers={**_AUTH, "X-Forwarded-For": ""},
    )
    assert response.status_code == 200


def test_forwarded_headers_without_trust_deny_repointing_a_credentialed_block(
    app_client: TestClient, tmp_path: Path
) -> None:
    """The base_url gate is the harvesting path; it must close with the same rule."""
    response = app_client.put(
        "/system/providers/demo",
        json={"base_url": "https://attacker.example/v1"},
        headers={**_AUTH, "X-Forwarded-For": "203.0.113.5"},
    )
    assert response.status_code == 400
    assert "credential" in response.json()["detail"]
    assert "attacker.example" not in (tmp_path / "ubt.toml").read_text(encoding="utf-8")


def test_trusted_proxy_forwarded_loopback_client_allows_the_write(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("UBT_TRUST_PROXY", "1")
    response = app_client.put(
        "/system/providers/demo",
        json={"api_key": "sk-local-operator"},
        headers={**_AUTH, "X-Forwarded-For": "127.0.0.1"},
    )
    assert response.status_code == 200, response.json()
    assert 'api_key = "sk-local-operator"' in (tmp_path / "ubt.toml").read_text(encoding="utf-8")


def test_trusted_proxy_x_real_ip_is_honoured(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UBT_TRUST_PROXY", "1")
    response = app_client.put(
        "/system/providers/demo",
        json={"api_key": "sk-nope"},
        headers={**_AUTH, "X-Real-IP": "127.0.0.1"},
    )
    assert response.status_code == 200


def test_trusted_proxy_forwarded_remote_client_is_denied(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UBT_TRUST_PROXY", "1")
    response = app_client.put(
        "/system/providers/demo",
        json={"api_key": "sk-via-trusted-proxy"},
        headers={**_AUTH, "X-Forwarded-For": "203.0.113.5"},
    )
    assert response.status_code == 400


def test_trusted_proxy_non_ip_forwarded_value_fails_closed(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UBT_TRUST_PROXY", "1")
    response = app_client.put(
        "/system/providers/demo",
        json={"api_key": "sk-hostname"},
        headers={**_AUTH, "X-Forwarded-For": "caller.attacker.example"},
    )
    assert response.status_code == 400


def test_a_direct_unproxied_request_keeps_the_historical_answer(
    app_client: TestClient, tmp_path: Path
) -> None:
    """No forwarded headers: the testserver bind decides, as the console tests assume."""
    response = app_client.put(
        "/system/providers/demo",
        json={"api_key": "sk-direct"},
        headers=_AUTH,
    )
    assert response.status_code == 200
    assert 'api_key = "sk-direct"' in (tmp_path / "ubt.toml").read_text(encoding="utf-8")


def test_the_decision_helper_table(monkeypatch: pytest.MonkeyPatch) -> None:
    from ubt.api.security import credential_route_is_local

    # No forwarded headers: bind-only, exactly the historical contract.
    assert credential_route_is_local(server_bind="127.0.0.1") is True
    assert credential_route_is_local(server_bind="0.0.0.0") is False
    assert credential_route_is_local(server_bind=None) is True

    monkeypatch.delenv("UBT_TRUST_PROXY", raising=False)
    # Forwarded headers present, trust off: non-local, even claiming loopback.
    assert credential_route_is_local(server_bind="127.0.0.1", x_forwarded_for="127.0.0.1") is False
    assert credential_route_is_local(server_bind="127.0.0.1", x_real_ip="10.1.2.3") is False

    monkeypatch.setenv("UBT_TRUST_PROXY", "1")
    # Trust on: the forwarded client IP decides; a non-IP value fails closed.
    assert credential_route_is_local(server_bind="127.0.0.1", x_forwarded_for="127.0.0.1") is True
    assert (
        credential_route_is_local(server_bind="127.0.0.1", x_forwarded_for="203.0.113.9") is False
    )
    assert (
        credential_route_is_local(server_bind="127.0.0.1", x_forwarded_for="attacker.example")
        is False
    )

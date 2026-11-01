from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.core.config import UBTConfig

pytestmark = pytest.mark.fast

_API_KEY = "assess-parity-key"
_AUTH = {"X-API-Key": _API_KEY}

_PROVIDERS_TOML = """\
[defaults]
provider = "alpha"

[providers.alpha]
base_url = "http://127.0.0.1:9090/v1"
api_mode = "openai-chat"
draft_model = "alpha-model"
repair_model = "alpha-model"

[providers.beta]
base_url = "http://127.0.0.1:9091/v1"
api_mode = "openai-chat"
draft_model = "beta-model"
repair_model = "beta-model"
"""


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    (tmp_path / "ubt.toml").write_text(_PROVIDERS_TOML, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    config.db_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(config=config)) as test_client:
        yield test_client


def _document(tmp_path: Path) -> Path:
    doc = tmp_path / "book.md"
    doc.write_text(
        "# Chapter One\n\n"
        + ("The quick brown fox. " * 200)
        + "\n\n# Chapter Two\n\n"
        + ("More prose here. " * 200),
        encoding="utf-8",
    )
    return doc


def test_the_quote_prices_the_selected_provider(client: TestClient, tmp_path: Path) -> None:
    doc = _document(tmp_path)

    default_quote = client.post("/jobs/assess", json={"input_path": str(doc)}, headers=_AUTH)
    beta_quote = client.post(
        "/jobs/assess", json={"input_path": str(doc), "provider": "beta"}, headers=_AUTH
    )

    assert default_quote.status_code == 200, default_quote.text
    assert beta_quote.status_code == 200, beta_quote.text

    assert default_quote.json()["meta"]["provider"] == "alpha"
    assert beta_quote.json()["meta"]["provider"] == "beta"
    assert beta_quote.json()["cost"]["draft_model"] == "beta-model"


def test_an_unknown_provider_is_a_named_400(client: TestClient, tmp_path: Path) -> None:
    doc = _document(tmp_path)
    response = client.post(
        "/jobs/assess", json={"input_path": str(doc), "provider": "no-such-block"}, headers=_AUTH
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "no-such-block" in detail
    assert "alpha" in detail


def test_ocr_mode_reaches_the_quote(client: TestClient, tmp_path: Path) -> None:
    doc = _document(tmp_path)
    response = client.post(
        "/jobs/assess", json={"input_path": str(doc), "ocr_mode": "local"}, headers=_AUTH
    )
    assert response.status_code == 200, response.text
    assert response.json()["meta"]["ocr_mode"] == "local"


def test_concurrency_drives_the_runtime_estimate(client: TestClient, tmp_path: Path) -> None:
    doc = _document(tmp_path)
    serial = client.post(
        "/jobs/assess", json={"input_path": str(doc), "max_concurrency": 1}, headers=_AUTH
    )
    parallel = client.post(
        "/jobs/assess", json={"input_path": str(doc), "max_concurrency": 8}, headers=_AUTH
    )
    assert serial.status_code == parallel.status_code == 200
    assert (
        parallel.json()["runtime"]["est_seconds_low"] < serial.json()["runtime"]["est_seconds_low"]
    )


def test_the_assess_model_forbids_a_credential(tmp_path: Path) -> None:
    doc = _document(tmp_path)
    from ubt.api.models import JobAssessRequest

    for leaked in ("api_key", "base_url", "api_mode", "ocr_api_key"):
        assert leaked not in JobAssessRequest.model_fields
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        JobAssessRequest(input_path=str(doc), api_key="sk-should-not-be-here")  # type: ignore[call-arg]

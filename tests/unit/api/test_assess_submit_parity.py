"""The quote must price the job the caller is about to submit.

``POST /jobs/assess`` and ``POST /jobs/submit`` are two calls about the same
intent, and they disagreed. Assess applied only ``preset``; every other knob it
had was ignored, so the dollar figure came from ``[defaults] provider`` while the
console's dropdown selected a different block. A user comparing two providers saw
the same total twice, and choosing cloud OCR added a vision line to the run that
the quote had never priced.

These tests pin the parity rather than the implementation: the request model
accepts the selection, the route applies it through the *same* mapping the submit
path uses, and the quote's own provenance (``meta.provider``) names what was
priced.
"""

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

#: Two declared providers with different models, so a quote priced against one is
#: distinguishable from a quote priced against the other.
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
    """A keyed app rooted in a throwaway directory holding a two-provider file.

    ``ubt.core.providers`` resolves ``ubt.toml`` relative to the process CWD, so
    the test chdirs — the same way a real server finds the operator's file.
    """
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
    """``provider`` must reach the pricing config, not just the submit path."""
    doc = _document(tmp_path)

    default_quote = client.post("/jobs/assess", json={"input_path": str(doc)}, headers=_AUTH)
    beta_quote = client.post(
        "/jobs/assess", json={"input_path": str(doc), "provider": "beta"}, headers=_AUTH
    )

    assert default_quote.status_code == 200, default_quote.text
    assert beta_quote.status_code == 200, beta_quote.text

    assert default_quote.json()["meta"]["provider"] == "alpha"
    # The provenance the console renders. Without it the two panels are
    # indistinguishable and the user cannot tell which block was priced.
    assert beta_quote.json()["meta"]["provider"] == "beta"
    assert beta_quote.json()["cost"]["draft_model"] == "beta-model"


def test_an_unknown_provider_is_a_named_400(client: TestClient, tmp_path: Path) -> None:
    """A bad name must be the caller's error, naming the block — never a 500."""
    doc = _document(tmp_path)
    response = client.post(
        "/jobs/assess", json={"input_path": str(doc), "provider": "no-such-block"}, headers=_AUTH
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "no-such-block" in detail
    # The message must list what the operator *did* declare, or it is unactionable.
    assert "alpha" in detail


def test_ocr_mode_reaches_the_quote(client: TestClient, tmp_path: Path) -> None:
    """Cloud OCR adds a vision line; a quote that never learns the mode prices 0.

    ``ocr_page_calls`` is only non-zero when ``config.ocr_mode`` is cloud/vlm, so
    it is the direct witness that the mode survived the trip into the pricing
    config. A text-layer markdown file has no scanned pages, so the *count* stays
    at zero either way — what this pins is that a mode the request stated is
    accepted and applied rather than silently dropped.
    """
    doc = _document(tmp_path)
    response = client.post(
        "/jobs/assess", json={"input_path": str(doc), "ocr_mode": "cloud"}, headers=_AUTH
    )
    assert response.status_code == 200, response.text
    assert response.json()["meta"]["ocr_mode"] == "cloud"


def test_concurrency_drives_the_runtime_estimate(client: TestClient, tmp_path: Path) -> None:
    """``max_concurrency`` is one of the three inputs to the wall-clock figure.

    Asserting the direction (more parallelism ⇒ shorter estimate) rather than an
    exact number keeps this from re-deriving the heuristic in the test.
    """
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
    """The quote prices a provider; it must never *become* a credential path.

    ``extra="forbid"`` is the boundary: a payload naming ``api_key`` or
    ``base_url`` is a 422 here exactly as it is on submit.
    """
    doc = _document(tmp_path)
    from ubt.api.models import JobAssessRequest

    for leaked in ("api_key", "base_url", "api_mode", "ocr_api_key"):
        assert leaked not in JobAssessRequest.model_fields
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        JobAssessRequest(input_path=str(doc), api_key="sk-should-not-be-here")  # type: ignore[call-arg]

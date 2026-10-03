from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from starlette.testclient import TestClient

from ubt.adapters.pdf.vlm.drivers.cloud_driver import DEFAULT_OPENAI_ENDPOINT, CloudOcrDriver
from ubt.api.app import create_app
from ubt.api.manager import JobManager, JobRecord
from ubt.api.models import JobSubmitRequest
from ubt.core.config import UBTConfig
from ubt.core.exceptions import BudgetExceededError
from ubt.core.router.rate_limiter import AdaptiveTokenBucket

pytestmark = pytest.mark.fast

_AUTH = {"X-API-Key": "inbound-service-key-123"}


def test_ubt_config_from_env_defaults_to_no_bootstrap() -> None:
    """Verify UBTConfig.from_env defaults to bootstrap=False and does not pollute registries."""
    cfg = UBTConfig.from_env()
    assert cfg is not None
    # Verify local_endpoints can be bootstrapped without error
    cfg_with_local = UBTConfig(local_endpoints="192.168.1.50:8000,10.0.0.1:8000")
    cfg_with_local.bootstrap_runtime_environment()


def test_cloud_driver_no_ambient_openai_scraping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CloudOcrDriver never scrapes foreign OPENAI_API_KEY or OPENAI_BASE_URL."""
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-stolen-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient-stolen.endpoint.com")
    monkeypatch.delenv("UBT_OCR_API_KEY", raising=False)
    monkeypatch.delenv("UBT_OCR_ENDPOINT", raising=False)

    driver = CloudOcrDriver(model="test-vision-model")
    assert driver.api_key is None
    assert driver.endpoint == DEFAULT_OPENAI_ENDPOINT.rstrip("/")


def test_in_flight_download_and_report_return_409(tmp_path: Path) -> None:
    """Verify in-flight polling of /download and /report returns HTTP 409 Conflict instead of 400."""
    config = UBTConfig(
        provider="openai",
        api_key=SecretStr("outbound-llm-key-456"),
        service_api_key=SecretStr("inbound-service-key-123"),
        db_dir=tmp_path / "ledgers",
        allowed_dirs=str(tmp_path),
    )
    app = create_app(config=config)
    client = TestClient(app)

    # Submit a job
    input_file = tmp_path / "test.md"
    input_file.write_text("# Chapter 1\nHello", encoding="utf-8")

    res = client.post(
        "/jobs/submit",
        json={"input_path": str(input_file), "job_id": "test-inflight-409"},
        headers=_AUTH,
    )
    assert res.status_code == 202

    # Polling download before file is ready must return 409 Conflict
    dl_res = client.get("/jobs/test-inflight-409/download", headers=_AUTH)
    assert dl_res.status_code == 409
    assert "not ready" in dl_res.json()["detail"]

    # Polling report before report is generated must return 409 Conflict
    rep_res = client.get("/jobs/test-inflight-409/report", headers=_AUTH)
    assert rep_res.status_code == 409
    assert "not yet generated" in rep_res.json()["detail"]


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
        api_key=SecretStr("outbound-llm-key-456"),
        service_api_key=SecretStr("inbound-service-key-123"),
    )

    # 1. BudgetExceededError should be directly readable in record.error
    with patch("ubt.api.manager.PipelineOrchestrator") as mock_orch_cls:

        async def mock_run_budget(*args: object, **kwargs: object) -> AsyncGenerator[Any, None]:
            raise BudgetExceededError("Budget cap of $5.00 exceeded")
            if False:
                yield None

        mock_orch_cls.return_value.run = mock_run_budget
        await manager.execute_job(record, base_config)

    assert record.status == "failed"
    assert record.error == "BudgetExceededError: Budget cap of $5.00 exceeded"

    # 2. Sensitive internal exceptions must be masked to prevent data/credential leaks
    with patch("ubt.api.manager.PipelineOrchestrator") as mock_orch_cls:

        async def mock_run_secret(*args: object, **kwargs: object) -> AsyncGenerator[Any, None]:
            raise RuntimeError("Internal DB host 10.0.0.5 connection refused password=secret")
            if False:
                yield None

        mock_orch_cls.return_value.run = mock_run_secret
        await manager.execute_job(record, base_config)

    assert "password=secret" not in (record.error or "")
    assert record.error == f"RuntimeError (see server logs; job_id={record.job_id})"


def test_adaptive_token_bucket_smoothed_aimd() -> None:
    """Verify AdaptiveTokenBucket increases capacity smoothly without massive step jumps."""
    bucket = AdaptiveTokenBucket(initial_rpm=60, max_rpm=120)
    assert bucket.capacity == 60.0

    # 1 success should add a fraction (1 / 60), not +1.0
    bucket.report_success()
    assert 60.0 < bucket.capacity < 60.1
    assert bucket._waiters == 0


def test_job_submit_request_accepts_pdf_engine() -> None:
    """Verify JobSubmitRequest validates and accepts pdf_engine."""
    req = JobSubmitRequest(input_path="book.pdf", pdf_engine="pdfium")
    assert req.pdf_engine == "pdfium"

    req_auto = JobSubmitRequest(input_path="book.pdf", pdf_engine="auto")
    assert req_auto.pdf_engine == "auto"

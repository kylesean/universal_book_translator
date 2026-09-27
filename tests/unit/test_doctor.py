"""Unit tests for the ubt doctor diagnostic command and --probe functionality."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_doctor_offline_json_run() -> None:
    """Offline doctor run outputs valid JSON with summary and checks."""
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code in (0, 1)  # 0 or 1 depending on environment credentials
    payload = json.loads(result.stdout)
    assert "status" in payload
    assert "summary" in payload
    assert "checks" in payload
    check_names = {c["name"] for c in payload["checks"]}
    assert "API key" in check_names
    assert "Base URL" in check_names


def test_doctor_probe_live_endpoint_success() -> None:
    """--probe against reachable endpoint confirming draft model."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "gemini-3.8-flash"},
                    {"id": "gemini-3.1-pro"},
                ]
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    with patch("httpx.Client", return_value=mock_client):
        result = runner.invoke(
            app,
            [
                "doctor",
                "--json",
                "--probe",
            ],
            env={
                "UBT_PROVIDER": "gemini",
                "GEMINI_API_KEY": "sk-dummy-key",
                "UBT_DRAFT_MODEL": "gemini-3.8-flash",
            },
        )
        assert result.exit_code in (0, 1)
        payload = json.loads(result.stdout)
        probe_checks = [c for c in payload["checks"] if c["name"] == "Live Endpoint Probe"]
        assert len(probe_checks) == 1
        assert probe_checks[0]["status"] == "OK"
        assert "gemini-3.8-flash" in probe_checks[0]["detail"]


def test_doctor_probe_model_missing_warn() -> None:
    """--probe against reachable endpoint where draft model is not deployed."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "other-model-a"},
                    {"id": "other-model-b"},
                ]
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    with patch("httpx.Client", return_value=mock_client):
        result = runner.invoke(
            app,
            [
                "doctor",
                "--json",
                "--probe",
            ],
            env={
                "UBT_PROVIDER": "gemini",
                "GEMINI_API_KEY": "sk-dummy-key",
                "UBT_DRAFT_MODEL": "gemini-3.8-flash",
            },
        )
        payload = json.loads(result.stdout)
        probe_checks = [c for c in payload["checks"] if c["name"] == "Live Endpoint Probe"]
        assert len(probe_checks) == 1
        assert probe_checks[0]["status"] == "WARN"
        assert "not found" in probe_checks[0]["detail"]
        assert "other-model-a" in probe_checks[0]["detail"]


def test_doctor_probe_auth_failure() -> None:
    """--probe with invalid credentials reports FAIL."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Invalid API key"})

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    with patch("httpx.Client", return_value=mock_client):
        result = runner.invoke(
            app,
            [
                "doctor",
                "--json",
                "--probe",
            ],
            env={
                "UBT_PROVIDER": "gemini",
                "GEMINI_API_KEY": "sk-dummy-key",
            },
        )
        payload = json.loads(result.stdout)
        probe_checks = [c for c in payload["checks"] if c["name"] == "Live Endpoint Probe"]
        assert len(probe_checks) == 1
        assert probe_checks[0]["status"] == "FAIL"
        assert "401" in probe_checks[0]["detail"]


def test_doctor_probe_connection_error() -> None:
    """--probe with unreachable endpoint reports FAIL."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    with patch("httpx.Client", return_value=mock_client):
        result = runner.invoke(
            app,
            [
                "doctor",
                "--json",
                "--probe",
            ],
            env={
                "UBT_PROVIDER": "gemini",
                "GEMINI_API_KEY": "sk-dummy-key",
            },
        )
        payload = json.loads(result.stdout)
        probe_checks = [c for c in payload["checks"] if c["name"] == "Live Endpoint Probe"]
        assert len(probe_checks) == 1
        assert probe_checks[0]["status"] == "FAIL"
        assert "Cannot connect" in probe_checks[0]["detail"]

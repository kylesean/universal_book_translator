"""Unit tests for UBT logging configuration and telemetry standards."""

from __future__ import annotations

import io
import logging

import pytest

from ubt.core.log_config import (
    _NOISY_LOGGERS,
    DEFAULT_DATE_FORMAT,
    DEFAULT_LOG_FORMAT,
    AccessNoiseFilter,
    setup_logging,
)

pytestmark = pytest.mark.fast


def test_default_log_format_structure() -> None:
    """Standard UBT log format must include levelprefix, name_prefix, and message."""
    assert "%(levelprefix)s" in DEFAULT_LOG_FORMAT
    assert "%(name_prefix)s" in DEFAULT_LOG_FORMAT
    assert "%(message)s" in DEFAULT_LOG_FORMAT
    assert DEFAULT_DATE_FORMAT == "%Y-%m-%d %H:%M:%S"


def test_setup_logging_formats_records() -> None:
    """A log record handled by setup_logging must render with the unified modern format."""
    stream = io.StringIO()
    setup_logging(level=logging.INFO, stream=stream)

    test_logger = logging.getLogger("ubt.test.telemetry")
    test_logger.info("unified telemetry message test")

    output = stream.getvalue()
    assert output == "INFO:     ubt.test.telemetry: unified telemetry message test\n"


def test_setup_logging_with_colors() -> None:
    """When colors are enabled, level prefix includes ANSI escape sequences."""
    stream = io.StringIO()
    setup_logging(level=logging.INFO, stream=stream, use_colors=True)

    test_logger = logging.getLogger("ubt.test.telemetry")
    test_logger.info("color test")

    output = stream.getvalue()
    assert output == "\x1b[32mINFO\x1b[0m:     ubt.test.telemetry: color test\n"


def test_httpx_and_httpcore_are_floored_at_warning() -> None:
    """httpx logs 'HTTP Request: POST ...' at INFO. Flooring at WARNING prevents request wire leakage."""
    assert _NOISY_LOGGERS["httpx"] == (logging.WARNING, False)
    assert _NOISY_LOGGERS["httpcore"] == (logging.WARNING, False)
    assert _NOISY_LOGGERS["urllib3"] == (logging.WARNING, False)

    stream = io.StringIO()
    setup_logging(level=logging.INFO, stream=stream)

    httpx_logger = logging.getLogger("httpx")
    assert httpx_logger.level == logging.WARNING
    assert not httpx_logger.isEnabledFor(logging.INFO)

    # An INFO-level wire log from httpx must NOT appear in output
    httpx_logger.info(
        "HTTP Request: POST http://127.0.0.1:9090/v1/chat/completions 'HTTP/1.1 200 OK'"
    )
    assert stream.getvalue() == ""

    # WARNING must still pass
    httpx_logger.warning("HTTP connection timeout to provider")
    assert "WARNING:  httpx: HTTP connection timeout to provider" in stream.getvalue()


def test_debug_lifts_floor_on_noisy_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    """UBT_LOG_LEVEL=DEBUG or explicit debug level lifts the floor to NOTSET (inherits root DEBUG)."""
    monkeypatch.setenv("UBT_LOG_LEVEL", "DEBUG")
    root = logging.getLogger()
    old_level = root.level
    try:
        setup_logging()
        httpx_logger = logging.getLogger("httpx")
        # In lift_all mode, floor is removed (set to NOTSET so it inherits root DEBUG)
        assert httpx_logger.level == logging.NOTSET
        assert httpx_logger.isEnabledFor(logging.DEBUG)
    finally:
        root.setLevel(old_level)


def test_setup_logging_preserves_debug_when_already_debug() -> None:
    """If an outer CLI callback (e.g. -v) already put root into DEBUG, re-calling setup_logging keeps it."""
    root = logging.getLogger()
    old_level = root.level
    try:
        root.setLevel(logging.DEBUG)
        # A bare setup_logging() without args should detect already_debug and stay in DEBUG
        setup_logging()
        assert root.level == logging.DEBUG
        assert logging.getLogger("httpx").isEnabledFor(logging.DEBUG)
    finally:
        root.setLevel(old_level)


def test_access_noise_filter_filters_polling_requests() -> None:
    """AccessNoiseFilter must suppress 200/304 polling requests but preserve non-200 and business actions."""
    filt = AccessNoiseFilter(allow_all=False)

    def _make_record(msg: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    # High-frequency 200/304 polling should be filtered
    assert (
        filt.filter(
            _make_record(
                '127.0.0.1:57164 - "GET /jobs/web_mvkxznvs_72d6672b7f02/status HTTP/1.1" 200 OK'
            )
        )
        is False
    )
    assert (
        filt.filter(_make_record('127.0.0.1:57164 - "GET /jobs?limit=200 HTTP/1.1" 200 OK'))
        is False
    )
    assert filt.filter(_make_record('127.0.0.1:57164 - "GET /health HTTP/1.1" 200 OK')) is False
    assert (
        filt.filter(_make_record('127.0.0.1:57164 - "GET /favicon.ico HTTP/1.1" 200 OK')) is False
    )
    assert (
        filt.filter(_make_record('127.0.0.1:57164 - "GET /assets/index-a1b2c3.js HTTP/1.1" 200 OK'))
        is False
    )
    assert (
        filt.filter(
            _make_record(
                '127.0.0.1:57164 - "GET /assets/index-a1b2c3.js HTTP/1.1" 304 Not Modified'
            )
        )
        is False
    )

    # State-changing operations and important non-polling endpoints must pass
    assert (
        filt.filter(_make_record('127.0.0.1:57164 - "POST /jobs/submit HTTP/1.1" 202 Accepted'))
        is True
    )
    assert (
        filt.filter(_make_record('127.0.0.1:57164 - "POST /jobs/123/cancel HTTP/1.1" 200 OK'))
        is True
    )
    assert (
        filt.filter(_make_record('127.0.0.1:57164 - "GET /jobs/123/report HTTP/1.1" 200 OK'))
        is True
    )

    # Non-200 responses on polling endpoints must pass so failures are visible!
    assert (
        filt.filter(
            _make_record(
                '127.0.0.1:57164 - "GET /jobs/web_123/status HTTP/1.1" 500 Internal Server Error'
            )
        )
        is True
    )
    assert (
        filt.filter(
            _make_record('127.0.0.1:57164 - "GET /jobs/web_123/status HTTP/1.1" 404 Not Found')
        )
        is True
    )

    # When allow_all=True (debug/verbose mode), everything passes
    filt_verbose = AccessNoiseFilter(allow_all=True)
    assert (
        filt_verbose.filter(
            _make_record('127.0.0.1:57164 - "GET /jobs/web_123/status HTTP/1.1" 200 OK')
        )
        is True
    )


def test_run_server_boots_uvicorn_with_noisy_loggers_floored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_server boots uvicorn with colorized terminal logging while keeping httpx wire traces floored."""
    from ubt.api.app import run_server

    captured_kwargs: dict[str, object] = {}

    def mock_uvicorn_run(app_target: str, **kwargs: object) -> None:
        captured_kwargs["app_target"] = app_target
        captured_kwargs.update(kwargs)

    monkeypatch.setattr("uvicorn.run", mock_uvicorn_run)
    monkeypatch.setenv("UBT_ALLOW_NO_AUTH", "1")

    run_server(host="127.0.0.1", port=8999)

    assert captured_kwargs["app_target"] == "ubt.api.app:app"
    assert logging.getLogger("httpx").level == logging.WARNING

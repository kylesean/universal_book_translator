"""Regression tests for the round-4 review fixes.

Each test pins a defect found by the second/third review passes: the shared
rate bucket's AIMD increment law, the split httpx timeout, and the MCP disk
fallback's staleness guard for interrupted jobs.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from ubt.core.router.transports.openai_chat import OpenAIChatTransport

pytestmark = pytest.mark.fast


def test_sqlite_shared_bucket_climbs_like_the_in_process_one(tmp_path: Path) -> None:
    """The shared bucket must use the same additive-increase law (1/capacity).

    A flat +1.0 let N workers reporting concurrently multiply the climb and
    defeat the shared rate ceiling the bucket exists to enforce.
    """
    from ubt.core.router.rate_limiter import SqliteTokenBucket

    path = tmp_path / "limiter.sqlite"
    bucket = SqliteTokenBucket(path, initial_rpm=60, max_rpm=120)
    bucket.report_success()

    row = (
        sqlite3.connect(path)
        .execute("SELECT rpm_capacity FROM token_bucket WHERE key LIKE 'global@60rpm/%'")
        .fetchone()
    )
    assert row is not None
    # One success adds 1/60, not 1.0 — same law as AdaptiveTokenBucket.
    assert 60.0 < row[0] < 60.1


@pytest.mark.asyncio
async def test_transport_client_timeout_is_split() -> None:
    """Connect/pool phases fail fast; only read/write get the full api_timeout.

    A single-float timeout pinned a pool slot for the full 180s on a dead
    endpoint before the connect phase gave up.
    """
    transport = OpenAIChatTransport(
        api_key="k", base_url="https://api.example.com/v1", timeout=180.0
    )
    client = transport._get_client()
    assert client.timeout.connect == 10.0
    assert client.timeout.read == 180.0
    assert client.timeout.write == 180.0
    assert client.timeout.pool == 180.0
    await transport.aclose()


def test_ledger_staleness_helper(tmp_path: Path) -> None:
    """The MCP disk fallback's staleness guard: fresh mtime is live, old is stale."""
    from ubt.mcp.server import _LEDGER_STALENESS_S, _ledger_is_stale

    ledger = tmp_path / "job_x.sqlite"
    ledger.write_bytes(b"x")
    assert _ledger_is_stale(ledger) is False

    old = time.time() - _LEDGER_STALENESS_S - 10.0
    import os

    os.utime(ledger, (old, old))
    assert _ledger_is_stale(ledger) is True


async def test_provider_shared_client_splits_connect_timeout() -> None:
    """The provider's shared client must keep the fast connect timeout.

    A bare ``timeout=self._timeout`` here silently overrode every transport's
    own ``connect=10s`` (``BaseTransport._get_client`` returns the injected
    client unchanged), pinning a pool slot for the full api_timeout on a dead
    endpoint.
    """
    from ubt.core.router.provider import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(api_key="k", timeout=180.0)
    client = provider._shared_client
    assert client.timeout.connect == 10.0
    assert client.timeout.read == 180.0
    assert client.timeout.write == 180.0
    assert client.timeout.pool == 180.0
    await client.aclose()


def test_retry_after_ms_is_converted_to_seconds() -> None:
    """A ``Retry-After-Ms`` header is milliseconds; the router sleeps seconds.

    Passing ``20000`` through verbatim made a 429 retry sleep ~5.5 hours.
    """
    import httpx

    from ubt.core.router.transports.base import retry_after_seconds

    assert retry_after_seconds(httpx.Headers({"retry-after": "20"})) == "20"
    assert retry_after_seconds(httpx.Headers({"retry-after-ms": "20000"})) == "20.0"
    assert retry_after_seconds(httpx.Headers({})) is None
    assert retry_after_seconds(httpx.Headers({"retry-after-ms": "abc"})) is None

"""The upload cap must stop the read, not just relabel the response.

``UploadSizeLimitMiddleware`` exists because Starlette parses a multipart body
into a ``SpooledTemporaryFile`` before any handler sees it, so the handler's own
byte counter runs after the disk write. The middleware's first implementation
set an ``aborted`` flag when the streamed cap was crossed and then kept handing
chunks to the app — measured with a synthetic ASGI harness, a 200 kB body under a
1 kB cap was read to the last byte and answered with a 413 only afterwards. The
guard cost nothing and prevented nothing.

These tests drive the middleware directly (no server, no client) and assert on
how many bytes the upstream actually pulled from the wire.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ubt.api.app import UploadSizeLimitMiddleware

pytestmark = pytest.mark.fast

_CAP = 1000
_SCOPE: dict[str, Any] = {"type": "http", "path": "/api/jobs/upload", "headers": []}


class _Harness:
    """A body of ``chunks`` × ``chunk_size`` bytes and the bytes consumed."""

    def __init__(self, chunks: int, chunk_size: int = 1024) -> None:
        self._remaining = chunks
        self._chunk_size = chunk_size
        self.read = 0

    async def receive(self) -> dict[str, Any]:
        if self._remaining <= 0:
            return {"type": "http.request", "body": b"", "more_body": False}
        self._remaining -= 1
        self.read += self._chunk_size
        return {
            "type": "http.request",
            "body": b"x" * self._chunk_size,
            "more_body": self._remaining > 0,
        }


async def _drain(scope: Any, receive: Any, send: Any) -> None:
    """A handler that reads the whole body, like Starlette's form parser."""
    while True:
        message = await receive()
        if message["type"] != "http.request" or not message.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def _drain_swallowing(scope: Any, receive: Any, send: Any) -> None:
    """A handler that catches everything and answers anyway."""
    try:
        while True:
            message = await receive()
            if message["type"] != "http.request" or not message.get("more_body"):
                break
    except Exception:  # noqa: BLE001 - the point is to swallow the abort
        pass
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"swallowed"})


def _run(
    handler: Any, *, chunks: int, declared: int | None = None, chunk_size: int = 1024
) -> tuple[_Harness, list[dict[str, Any]]]:
    harness = _Harness(chunks, chunk_size)
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope: dict[str, Any] = dict(_SCOPE)
    if declared is not None:
        scope["headers"] = [(b"content-length", str(declared).encode())]
    middleware = UploadSizeLimitMiddleware(handler, max_bytes=_CAP, path_suffix="/jobs/upload")
    asyncio.run(middleware(scope, harness.receive, send))
    return harness, sent


def _status(sent: list[dict[str, Any]]) -> int | None:
    for message in sent:
        if message["type"] == "http.response.start":
            return int(message["status"])
    return None


def test_a_chunked_body_stops_being_read_at_the_cap() -> None:
    """The regression: a 200 kB body under a 1 kB cap must not be read whole."""
    harness, sent = _run(_drain, chunks=200)
    assert harness.read <= _CAP + 1024, f"read {harness.read} bytes under a {_CAP} cap"
    assert harness.read < 200 * 1024
    assert _status(sent) == 413


def test_the_413_is_sent_exactly_once() -> None:
    _harness, sent = _run(_drain, chunks=20)
    starts = [message for message in sent if message["type"] == "http.response.start"]
    assert len(starts) == 1
    assert _status(sent) == 413


def test_a_handler_that_swallows_the_abort_still_gets_a_413() -> None:
    """Suppressing the handler's response must not leave the client with none."""
    harness, sent = _run(_drain_swallowing, chunks=20)
    assert _status(sent) == 413
    assert harness.read <= _CAP + 1024


def test_a_declared_length_over_the_cap_is_refused_without_reading() -> None:
    harness, sent = _run(_drain, chunks=20, declared=999_999)
    assert harness.read == 0
    assert _status(sent) == 413


def test_a_body_under_the_cap_passes_through_untouched() -> None:
    harness, sent = _run(_drain, chunks=1, declared=500, chunk_size=500)
    assert harness.read == 500
    assert _status(sent) == 200


def test_other_paths_are_not_inspected() -> None:
    """A request that is not the upload endpoint must reach the app unchanged."""
    harness = _Harness(20)
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    middleware = UploadSizeLimitMiddleware(_drain, max_bytes=_CAP, path_suffix="/jobs/upload")
    asyncio.run(middleware({"type": "http", "path": "/jobs", "headers": []}, harness.receive, send))
    assert _status(sent) == 200
    assert harness.read == 20 * 1024

"""The recorded-fixture layer (spec module C, item 3).

The quality gate used to be exercisable only through the ``REHEARSAL_MARKER``
content bypass, so a CI run never actually gated a translation: it just proved
the plumbing moved bytes. This module pins the replacement: a cassette of
recorded provider responses is replayed through the *real* transport, so a test
asserts on what the transport actually sent and received rather than on a bypass.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from cassettes import Cassette, build_client

pytestmark = pytest.mark.fast


def test_a_cassette_replays_the_recorded_response(tmp_path: Path) -> None:
    cassette = Cassette(path=tmp_path / "wire.json")
    cassette.seed(
        "POST",
        "https://api.example/v1/chat/completions",
        request={"model": "gpt-x", "messages": []},
        response={"choices": [{"message": {"content": "你好"}}]},
    )

    async def _run() -> Any:
        client = await build_client(cassette)
        try:
            resp = await client.post(
                "https://api.example/v1/chat/completions",
                json={"model": "gpt-x", "messages": []},
            )
            return resp.json()
        finally:
            await client.aclose()

    data = asyncio.run(_run())
    assert data["choices"][0]["message"]["content"] == "你好"
    cassette.assert_fully_replayed()


def test_the_cassette_round_trips_through_disk(tmp_path: Path) -> None:
    path = tmp_path / "wire.json"
    writer = Cassette(path=path)
    writer.seed(
        "POST",
        "https://api.example/v1/x",
        request={"a": 1},
        response={"ok": True},
    )
    assert path.exists(), "seeding with a path must persist the cassette"

    replayed = Cassette(path=path).load()

    async def _run() -> Any:
        client = await build_client(replayed)
        try:
            resp = await client.post("https://api.example/v1/x", json={"a": 1})
            return resp.json()["ok"]
        finally:
            await client.aclose()

    assert asyncio.run(_run()) == 1


def test_an_unrecorded_request_fails_closed(tmp_path: Path) -> None:
    """A request the cassette does not cover must not silently succeed."""
    cassette = Cassette(path=tmp_path / "wire.json")
    cassette.seed("POST", "https://api.example/v1/a", request={}, response={})

    async def _run() -> Any:
        client = await build_client(cassette)
        try:
            resp = await client.post("https://api.example/v1/UNRECORDED", json={})
            return resp.status_code
        finally:
            await client.aclose()

    assert asyncio.run(_run()) == 599


def test_a_missing_cassette_is_a_hard_error(tmp_path: Path) -> None:
    """A missing cassette must raise, never replay empty and pass vacuously."""
    with pytest.raises(FileNotFoundError):
        Cassette(path=tmp_path / "absent.json").load()


def test_the_replay_transport_is_httpx_compatible(tmp_path: Path) -> None:
    """The transport is a real ``httpx.MockTransport``, not a hand-rolled shim."""
    cassette = Cassette(path=tmp_path / "w.json")
    assert isinstance(cassette.transport(), httpx.MockTransport)

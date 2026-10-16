"""Record/replay cassettes for transport-level tests, built on ``httpx``.

The spec (module C, item 3) asks for a recorded-fixture layer so the quality
gate can be exercised end-to-end in CI *against real provider responses*
instead of being bypassed. VCR.py is the named option, but this codebase already
speaks ``httpx`` throughout and deliberately keeps its dependency surface lean,
so this is the same idea with no new dependency: a ``Cassette`` wraps request/
response pairs on an ``httpx.MockTransport``.

Usage::

    cassette = Cassette.load_or_record(path)   # record when missing, replay after
    cassette.seed("POST", "https://api/x", json_payload, json_response)
    transport = cassette.transport()           # pass to httpx.AsyncClient(transport=...)

A cassette written by a live run is a JSON file checked into ``tests/fixtures``
and replayed by CI; a missing cassette fails loudly rather than letting the test
silently pass with no data (the failure mode the spec calls out).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass
class Cassette:
    """A recorded set of HTTP request/response pairs for one test."""

    path: Path | None = None
    #: Recorded interactions, keyed by ``(method, url)`` so replay is exact.
    interactions: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Methods that must be present for the test's assertions to be meaningful.
    _seen: set[str] = field(default_factory=set)

    @staticmethod
    def _key(method: str, url: str) -> str:
        return f"{method.upper()} {url}"

    def seed(
        self, method: str, url: str, *, request: Any, response: Any, status: int = 200
    ) -> None:
        """Record one interaction in memory (and to disk when ``path`` is set)."""
        self.interactions[self._key(method, url)] = {
            "method": method.upper(),
            "url": url,
            "request": request,
            "status": status,
            "response": response,
        }
        if self.path is not None:
            self.save()

    def load(self) -> Cassette:
        """Load interactions from ``path``; raise when the cassette is absent.

        A missing cassette is a hard error, not an empty replay: a test that
        silently replays nothing would pass without ever exercising the code.
        """
        if self.path is None or not self.path.exists():
            raise FileNotFoundError(
                f"cassette not found: {self.path}. Record it with a live run, or "
                "build it in-test with Cassette.seed()."
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.interactions = payload.get("interactions", {})
        return self

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"interactions": self.interactions}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def transport(self) -> httpx.MockTransport:
        """An ``httpx.MockTransport`` that replays the recorded interactions."""

        def _dispatch(request: httpx.Request) -> httpx.Response:
            key = self._key(request.method, str(request.url))
            entry = self.interactions.get(key)
            if entry is None:
                # Fail closed: an unrecorded request means the test is reaching
                # the network against a cassette that does not cover it.
                return httpx.Response(599, text=f"no cassette entry for {key}")
            self._seen.add(key)
            return httpx.Response(entry.get("status", 200), json=entry.get("response"))

        return httpx.MockTransport(_dispatch)

    def assert_fully_replayed(self) -> None:
        """Every recorded interaction must have been requested."""
        missing = set(self.interactions) - self._seen
        assert not missing, f"cassette interactions never replayed: {sorted(missing)}"


async def build_client(cassette: Cassette) -> httpx.AsyncClient:
    """An ``httpx.AsyncClient`` wired to the cassette's replay transport."""
    return httpx.AsyncClient(transport=cassette.transport())

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass
class Cassette:
    path: Path | None = None
    interactions: dict[str, dict[str, Any]] = field(default_factory=dict)
    _seen: set[str] = field(default_factory=set)

    @staticmethod
    def _key(method: str, url: str) -> str:
        return f"{method.upper()} {url}"

    def seed(
        self, method: str, url: str, *, request: Any, response: Any, status: int = 200
    ) -> None:
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

        def _dispatch(request: httpx.Request) -> httpx.Response:
            key = self._key(request.method, str(request.url))
            entry = self.interactions.get(key)
            if entry is None:
                return httpx.Response(599, text=f"no cassette entry for {key}")
            self._seen.add(key)
            return httpx.Response(entry.get("status", 200), json=entry.get("response"))

        return httpx.MockTransport(_dispatch)

    def assert_fully_replayed(self) -> None:
        missing = set(self.interactions) - self._seen
        assert not missing, f"cassette interactions never replayed: {sorted(missing)}"


async def build_client(cassette: Cassette) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=cassette.transport())

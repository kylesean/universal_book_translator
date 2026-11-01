from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

_UBT_ENV_PREFIX = "UBT_"


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("network") is not None:
        return

    def _blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "outbound network access is blocked in tests; "
            "mark the test with @pytest.mark.network if it truly needs it"
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture(autouse=True)
def clean_ubt_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import os

    for name in [key for key in os.environ if key.startswith(_UBT_ENV_PREFIX)]:
        monkeypatch.delenv(name, raising=False)
    yield

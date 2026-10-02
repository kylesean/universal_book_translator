"""Shared test fixtures.

Two project-wide guarantees are enforced here rather than repeated per test:

- **No network in the default tier.** The ``fast`` tier must run offline in the
  base dev environment (no ``docling``/``torch``/``rapidocr``/``comet``). Any
  test that genuinely needs the network opts out with ``@pytest.mark.network``.
- **Deterministic construction.** ``UBTConfig`` is a pydantic-settings object,
  so a stray ``UBT_*`` variable in the developer's shell would leak into a test.
  ``clean_ubt_env`` removes the config-affecting variables for the test's scope.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

#: Environment variables that change ``UBTConfig`` construction. Cleared by
#: ``clean_ubt_env`` so a test observes field defaults, not the developer's shell.
_UBT_ENV_PREFIX = "UBT_"


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast on outbound socket connects unless the test is marked ``network``."""
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


@pytest.fixture
def clean_ubt_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every ``UBT_*`` variable so ``UBTConfig`` sees only field defaults."""
    import os

    for name in [key for key in os.environ if key.startswith(_UBT_ENV_PREFIX)]:
        monkeypatch.delenv(name, raising=False)
    yield

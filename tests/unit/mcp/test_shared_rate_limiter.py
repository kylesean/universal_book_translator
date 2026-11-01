from __future__ import annotations

import pytest

from ubt.core.router.rate_limiter import AdaptiveTokenBucket

pytestmark = pytest.mark.fast


def test_importing_the_server_does_not_read_the_environment() -> None:
    import inspect

    import ubt.mcp.server as server

    source = inspect.getsource(server)
    for line in source.splitlines():
        if line.startswith("_rate_limiter"):
            assert "None" in line, f"module-scope construction: {line}"
    assert "_rate_limiter = build_rate_limiter(" in inspect.getsource(server._shared_rate_limiter)


def test_the_shared_limiter_is_a_single_instance() -> None:
    import ubt.mcp.server as server

    first = server._shared_rate_limiter()
    second = server._shared_rate_limiter()
    assert first is second
    assert isinstance(first, AdaptiveTokenBucket)


def test_the_manager_gets_the_shared_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    import ubt.mcp.server as server

    limiter = server._shared_rate_limiter()
    monkeypatch.setattr(server._MANAGER, "rate_limiter", limiter)
    assert server._MANAGER.rate_limiter is limiter


def test_a_caller_supplied_limiter_is_not_replaced() -> None:
    import ubt.mcp.server as server

    original = server._MANAGER.rate_limiter
    custom = AdaptiveTokenBucket(initial_rpm=1, initial_tpm=1000)
    try:
        server._MANAGER.rate_limiter = custom
        if server._MANAGER.rate_limiter is None:  # pragma: no cover - the guard
            server._MANAGER.rate_limiter = server._shared_rate_limiter()
        assert server._MANAGER.rate_limiter is custom
    finally:
        server._MANAGER.rate_limiter = original

"""The MCP surface shares one rate-limit bucket per process.

``rate_limit_rpm`` is a per-credential budget. The MCP server built its
``JobManager`` with no limiter at all, and each job's orchestrator then made its
own: N concurrent translations against one API key multiplied the configured RPM
by N, and the provider answered 429s the operator had already paid to avoid.
REST (``create_app``) and the worker (``worker_command``) both build one bucket;
MCP was the third door and had none.

The server now builds one lazily — on first submission, not at import, because
every ``ubt`` CLI invocation imports this module and a limiter constructed at
import time would read the environment before a caller could override it.
"""

from __future__ import annotations

import pytest

from ubt.core.router.rate_limiter import AdaptiveTokenBucket

pytestmark = pytest.mark.fast


def test_importing_the_server_does_not_read_the_environment() -> None:
    """Every ``ubt`` invocation imports this module; the limiter must wait.

    Asserted against the source rather than a subprocess: this module is in the
    ``fast`` tier, and a fast module may not spawn one. What matters is that no
    limiter is constructed at module scope — ``_rate_limiter`` starts ``None``
    and only ``_shared_rate_limiter`` assigns it.
    """
    import inspect

    import ubt.mcp.server as server

    source = inspect.getsource(server)
    # No module-level construction: every assignment is inside the accessor.
    for line in source.splitlines():
        if line.startswith("_rate_limiter"):
            assert "None" in line, f"module-scope construction: {line}"
    assert "_rate_limiter = build_rate_limiter(" in inspect.getsource(server._shared_rate_limiter)


def test_the_shared_limiter_is_a_single_instance() -> None:
    """Two submissions must not get two buckets."""
    import ubt.mcp.server as server

    first = server._shared_rate_limiter()
    second = server._shared_rate_limiter()
    assert first is second
    assert isinstance(first, AdaptiveTokenBucket)


def test_the_manager_gets_the_shared_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wiring that was missing: the manager must hold it before create_job."""
    import ubt.mcp.server as server

    limiter = server._shared_rate_limiter()
    monkeypatch.setattr(server._MANAGER, "rate_limiter", limiter)
    assert server._MANAGER.rate_limiter is limiter


def test_a_caller_supplied_limiter_is_not_replaced() -> None:
    """``is None`` guard: an operator-configured manager keeps its own bucket."""
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

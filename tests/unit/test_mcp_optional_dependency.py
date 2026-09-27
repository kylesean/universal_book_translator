"""A missing ``mcp`` extra is a catchable error, not ``SystemExit``.

``ubt.mcp.server`` imports the optional ``mcp`` package at module scope. That
guard used to ``raise SystemExit``, which kills the whole process: a library
caller (an agent framework probing which tools exist, a test harness) cannot
catch it. Import must raise a normal exception instead, while the console-script
shim still turns it into a clean exit for ``ubt-mcp``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from ubt.core.exceptions import OptionalDependencyError


def _block_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``mcp`` unimportable and drop the cached server/package modules.

    Every ``mcp`` submodule must go, not just the top-level entry: the import
    system serves ``from mcp.server.mcpserver import MCPServer`` straight from
    ``sys.modules['mcp.server.mcpserver']`` when it is cached, never consulting
    the poisoned ``sys.modules['mcp']`` — so a partial block silently lets the
    real server boot.
    """
    for name in [n for n in sys.modules if n == "mcp" or n.startswith("mcp.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "mcp", None)
    for name in ("ubt.mcp.server", "ubt.mcp._entrypoint", "ubt.mcp"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_importing_server_without_mcp_is_catchable(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_mcp(monkeypatch)

    with pytest.raises(OptionalDependencyError, match=r"optional 'mcp' extra"):
        importlib.import_module("ubt.mcp.server")


def test_importing_package_without_mcp_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import ubt.mcp`` must not require the extra — only the server does."""
    _block_mcp(monkeypatch)

    package = importlib.import_module("ubt.mcp")
    assert package.__all__ == ["main", "mcp"]
    with pytest.raises(OptionalDependencyError):
        _ = package.main


def test_console_entrypoint_converts_missing_mcp_to_clean_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_mcp(monkeypatch)

    entrypoint = importlib.import_module("ubt.mcp._entrypoint")
    with pytest.raises(SystemExit, match=r"optional 'mcp' extra"):
        entrypoint.main()


def test_transitive_missing_dependency_keeps_its_own_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dep missing *inside* an installed ``mcp`` must keep its own error.

    The guard converted any ``ModuleNotFoundError`` into the "install [mcp]"
    hint, so a missing transitive dep (e.g. jsonschema) printed the wrong fix
    and permanently hid the real cause.
    """
    pkg = tmp_path / "mcp" / "server"
    pkg.mkdir(parents=True)
    (tmp_path / "mcp" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "mcpserver.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'jsonschema'\", name='jsonschema')\n"
    )
    for name in [n for n in sys.modules if n == "mcp" or n.startswith("mcp.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "ubt.mcp.server", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as exc:
        importlib.import_module("ubt.mcp.server")
    assert exc.value.name == "jsonschema"


def test_lazy_reexports_are_visible_to_dir() -> None:
    """``__getattr__`` without ``__dir__`` hid the lazy names from tooling."""
    import ubt.core.engine
    import ubt.mcp

    assert "main" in dir(ubt.mcp)
    assert "mcp" in dir(ubt.mcp)
    assert "PipelineOrchestrator" in dir(ubt.core.engine)
    assert "RepairLoop" in dir(ubt.core.engine)

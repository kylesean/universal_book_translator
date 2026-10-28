"""Tests for core layering and the ports boundary invariant.

Ensures that ubt.core does not have forbidden reverse edges into adapters or
compiler packages (ubt.pipeline, ubt.segment, ubt.translate), and that only
ubt/core/ports.py is permitted to hold lazy bridge forwarders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.dump_compiler_edges as dce

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_DIR = _REPO_ROOT / "ubt" / "core"


def test_core_has_no_module_level_compiler_or_adapter_edges() -> None:
    """Zero module-level imports of pipeline/segment/translate or adapters in core."""
    forbidden_compiler: list[tuple[str, int, str]] = []
    forbidden_adapters: list[tuple[str, int, str]] = []

    for f in sorted(_CORE_DIR.rglob("*.py")):
        top, _ = dce.scan(f)
        for ln, mod in top:
            if any(mod == c or mod.startswith(c + ".") for c in dce.COMPILER) and not (
                mod == dce._EXEMPT_LEAF or mod.startswith(dce._EXEMPT_LEAF + ".")
            ):
                forbidden_compiler.append((str(f.relative_to(_REPO_ROOT)), ln, mod))
            if mod == "ubt.adapters" or mod.startswith("ubt.adapters."):
                forbidden_adapters.append((str(f.relative_to(_REPO_ROOT)), ln, mod))

    assert not forbidden_compiler, f"Forbidden module-level compiler edges: {forbidden_compiler}"
    assert not forbidden_adapters, f"Forbidden module-level adapter edges: {forbidden_adapters}"


def test_only_ports_may_hold_lazy_bridge_imports_in_core() -> None:
    """Non-ports.py modules in core must not lazy-import adapters or compiler packages."""
    leaks: list[tuple[str, int, str]] = []
    forbidden_targets = ("ubt.adapters", "ubt.pipeline", "ubt.segment", "ubt.translate")

    for f in sorted(_CORE_DIR.rglob("*.py")):
        if f.name == "ports.py":
            continue
        _, lazy = dce.scan(f)
        for ln, mod in lazy:
            for bad in forbidden_targets:
                if mod == bad or mod.startswith(bad + "."):
                    leaks.append((str(f.relative_to(_REPO_ROOT)), ln, mod))

    assert not leaks, (
        "Detected lazy import leaks in core bypassing ubt/core/ports.py:\n"
        + "\n".join(f"  {f}:{ln} -> {mod}" for f, ln, mod in leaks)
    )


def test_dump_compiler_edges_script_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    """scripts/dump_compiler_edges.py executes without error and reports 0 leaks."""
    exit_code = dce.main()
    captured = capsys.readouterr()
    assert exit_code == 0, f"dump_compiler_edges failed:\n{captured.out}"
    assert "函数级 core→adapters/compiler（仅限 ports.py，其他模块验收：0）: 0" in captured.out

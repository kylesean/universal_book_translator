"""Regression: importing the core engine must not pull in adapters.

Adapters depend on core (IR models), so a reverse module-scope
edge would be a package cycle and would drag heavy dependencies into every core
import. ``ubt/core/ports.py`` is the single sanctioned bridge (function-level
lazy imports); ``test_license_guard.py`` enforces the AST-level ban. This
test enforces the runtime-level guarantee end to end: a fresh interpreter
that imports the whole engine carries zero ``ubt.adapters`` modules.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]

# Runtime half of the architecture guard (the AST half is test_license_guard):
# it must run in the per-edit `pytest -m fast` gate, or a core->adapters leak
# could land silently.
pytestmark = pytest.mark.fast

PROBE = (
    "import sys\n"
    "import ubt.core.engine.pipeline\n"
    "import ubt.core.engine.stages.ingest\n"
    "import ubt.core.engine.stages.export\n"
    "leaked = sorted(m for m in sys.modules if m.startswith('ubt.adapters'))\n"
    "print('LEAKED:' + ','.join(leaked))\n"
)


def test_core_import_does_not_pull_adapters() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = [line for line in proc.stdout.splitlines() if line.startswith("LEAKED:")]
    assert out, f"probe produced no marker line: {proc.stdout[-2000:]}"
    leaked = out[-1][len("LEAKED:") :]
    assert leaked == "", f"Core imports pulled in adapter modules: {leaked}"


def test_ports_lazy_default_still_resolves(tmp_path: Path) -> None:
    """The ports bridge must keep working with production defaults: the lazy
    import inside ``resolve_adapter`` is the sanctioned edge, so resolving an
    adapter through it succeeds without any prior injection."""
    from ubt.core.ports import is_pdf_engine_adapter, reset_ports, resolve_adapter

    reset_ports()  # ensure production defaults, no test leakage
    try:
        doc = tmp_path / "book.md"
        doc.write_text("# heading\n", encoding="utf-8")
        adapter = resolve_adapter(doc, pdf_engine="auto")
        assert adapter is not None
        # Non-PDF adapters lack the engine_name duck-typed marker.
        assert is_pdf_engine_adapter(adapter) is False
    finally:
        reset_ports()


def test_ports_reset_restores_default_visual_gate_runner() -> None:
    """set_visual_gate_runner override is cleared when reset_ports is invoked."""
    from ubt.core.ports import get_visual_gate_runner, reset_ports, set_visual_gate_runner

    dummy_called = False

    def dummy_runner(*args: object, **kwargs: object) -> object:
        nonlocal dummy_called
        dummy_called = True
        return None

    set_visual_gate_runner(dummy_runner)
    assert get_visual_gate_runner() is dummy_runner

    reset_ports()
    assert get_visual_gate_runner() is not dummy_runner

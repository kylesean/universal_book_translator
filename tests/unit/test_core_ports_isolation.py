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

REPO_ROOT = Path(__file__).parents[2]

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

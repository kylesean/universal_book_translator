"""Guards for the PdfiumGateway contract (docs/PDFIUM_THREAD_SAFETY_2026-09-20.md).

pdfium's lazy global font-table build and page parsing are not thread
safe; every in-process entry to libpdfium must hold
``ubt.adapters.pdf.pdfium_gate.PDFIUM_LOCK`` and the gate must be
imported before pypdfium2 so the pinned-font policy installs.
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

UBT_PKG = Path(__file__).resolve().parents[2] / "ubt"

GATE_MODULE = "ubt.adapters.pdf.pdfium_gate"


def _module_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    "py_file",
    sorted(UBT_PKG.rglob("*.py")),
    ids=lambda p: str(p.relative_to(UBT_PKG.parent)),
)
def test_pdfium_importers_hold_the_gate(py_file: Path) -> None:
    """Any module importing pypdfium2 must import the gate first.

    Import order is load-bearing twice over: the gate's decorator is
    needed by locked entries, and the font policy must install before
    pypdfium2 initializes the library at import time.
    """
    src = py_file.read_text(encoding="utf-8")
    if "pypdfium2" not in src:
        pytest.skip("no pdfium usage")
    if py_file.name == "pdfium_gate.py":
        pytest.skip("the gate itself")
    tree = ast.parse(src, filename=str(py_file))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    uses_pdfium = any(
        name == "pypdfium2" or name.startswith("pypdfium2.") or name == "pypdfium2_raw"
        for name in imported
    )
    if not uses_pdfium:
        pytest.skip("pdfium mentioned only in strings/comments")
    gate_imported = any(
        name == GATE_MODULE or name.startswith(GATE_MODULE + ".") for name in imported
    )
    assert gate_imported, (
        f"{py_file}: imports pypdfium2 without ubt.adapters.pdf.pdfium_gate — "
        "all pdfium entry points must hold PDFIUM_LOCK and install the "
        "font policy (see docs/PDFIUM_THREAD_SAFETY_2026-09-20.md)"
    )
    # Gate import must precede pypdfium2 imports at the source level so the
    # font policy installs before the library initializes.
    first_pdfium = min(
        (
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            and any(a.name.startswith("pypdfium2") for a in node.names)
        ),
        default=10**9,
    )
    first_gate = min(
        (
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and (
                (
                    isinstance(node, ast.Import)
                    and any(a.name.startswith(GATE_MODULE) for a in node.names)
                )
                or (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.startswith(GATE_MODULE)
                )
            )
        ),
        default=10**9,
    )
    assert first_gate < first_pdfium, (
        f"{py_file}: pypdfium2 is imported before pdfium_gate — the pinned "
        "font policy would not install for this process"
    )


def test_lock_is_reentrant() -> None:
    """Nested entry must not self-deadlock, and the failure must be fast.

    The bare `with PDFIUM_LOCK, PDFIUM_LOCK: pass` form this guards could only
    fail as a pytest-timeout hang — a lost reentrancy parks the thread inside the
    second acquire — which is the worst possible signature for the PDFium gate and
    burns the whole 600 s budget to report one regression. Running the nested
    entry on a daemon thread with a short join turns that hang into a red line;
    daemon, so a wedged thread cannot outlive the session either.
    """
    from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK

    completed = threading.Event()

    def nested() -> None:
        with PDFIUM_LOCK, PDFIUM_LOCK:
            completed.set()

    worker = threading.Thread(target=nested, daemon=True)
    worker.start()
    worker.join(timeout=2.0)
    assert completed.is_set(), "nested PDFIUM_LOCK entry did not complete within 2 s"


def test_serialized_decorator_serializes_threads() -> None:
    from ubt.adapters.pdf.pdfium_gate import pdfium_serialized

    inflight = 0
    max_inflight = 0
    guard = threading.Lock()

    @pdfium_serialized
    def critical() -> None:
        nonlocal inflight, max_inflight
        with guard:
            inflight += 1
            max_inflight = max(max_inflight, inflight)
        threading.Event().wait(0.02)
        with guard:
            inflight -= 1

    ts = [threading.Thread(target=critical) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert max_inflight == 1, "pdfium_serialized allowed overlapping critical sections"


def test_open_document_roundtrip(tmp_path: Path) -> None:
    """open_document yields a usable locked document (needs a real PDF)."""
    sample = UBT_PKG.parent / "docs" / "synthetic-duo.pdf"
    if not sample.exists():
        pytest.skip("synthetic sample PDF unavailable")
    from ubt.adapters.pdf.pdfium_gate import open_document

    with open_document(sample) as doc:
        assert len(doc) >= 1
        page = doc[0]
        assert float(page.get_width()) > 0
        page.close()

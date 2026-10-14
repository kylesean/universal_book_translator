"""``ubt.analyze`` must not reach up into ``ubt.adapters`` (layering invariant).

The adapters build on this layer -- ``ubt.adapters.epub``/``html``/``docx``/
``markdown`` import ``ubt.analyze``'s walkers, numbering and classifier rules --
so a call-time ``from ubt.adapters.base import decode_markup`` *inside a
reader* is a reverse edge: the lower layer reaches up into the higher one. Such
an edge compiles and passes a smoke test, then breaks the moment the adapter is
loaded for a different format, and it made the "parallel representation"
criticism true (the reader depended on the adapter rather than the reverse).

The decode helpers live in the leaf :mod:`ubt.core.text_decode` now. The PDF
geometry seam this test used to whitelist is gone too: the reader moved to
``ubt.adapters.pdf.reader_pdf``. This test pins the unconditional invariant:
NO ``ubt.adapters`` import anywhere under ``ubt/analyze``.

Pure-AST: imports are parsed, never executed, so it cannot be fooled by an
import side effect and costs milliseconds.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANALYZE = _REPO_ROOT / "ubt" / "analyze"


def _known_modules() -> set[str]:
    """Every importable ``ubt.*`` module name, so a ``from x import y`` alias is
    only treated as a submodule when ``x.y`` really is one."""
    known: set[str] = set()
    for path in (_REPO_ROOT / "ubt").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.with_suffix("").relative_to(_REPO_ROOT).parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        known.add(".".join(parts))
    return known


_KNOWN_MODULES = _known_modules()

#: There are NO allowed ``ubt.analyze -> ubt.adapters`` edges. The one
#: documented seam this set used to cover (the PDF reader importing
#: ``textgeom``/``pdfium_gate``) is gone: the reader moved to its only
#: production consumer's layer (``ubt.adapters.pdf.reader_pdf``), so the
#: invariant is now unconditional — any new edge here is an inversion, not
#: an exception to negotiate.
_ALLOWED_ADAPTER_EDGES: frozenset[str] = frozenset()


def _adapter_imports(path: Path) -> list[tuple[int, str]]:
    """Every ``ubt.adapters`` module this file imports, at any nesting depth.

    A ``from ubt.adapters.pdf import textgeom`` records the package name and,
    only when ``ubt.adapters.pdf.textgeom`` is a real module, that submodule too
    -- so an unlisted submodule aliased through an allowed package is caught
    without mistaking ``from ... import PDFIUM_LOCK`` for a module.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("ubt.adapters"):
                found.append((node.lineno, node.module))
                for alias in node.names:
                    candidate = f"{node.module}.{alias.name}"
                    if candidate in _KNOWN_MODULES:
                        found.append((node.lineno, candidate))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ubt.adapters"):
                    found.append((node.lineno, alias.name))
    return found


def test_analyze_does_not_reach_up_into_adapters() -> None:
    """No ``ubt/analyze`` module imports ``ubt.adapters`` off the allowlist."""
    offenders: list[str] = []
    for path in sorted(_ANALYZE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for lineno, module in _adapter_imports(path):
            if module not in _ALLOWED_ADAPTER_EDGES:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno} -> {module}")

    assert not offenders, (
        "ubt.analyze imports ubt.adapters outside the documented geometry seam "
        "(layering inversion -- the lower layer reaches up into the higher one). "
        "Move the shared helper to a leaf module (e.g. ubt.core.text_decode) "
        f"instead: {offenders}"
    )


def test_the_decode_helpers_live_in_the_leaf_module() -> None:
    """The markup decoder every reader shares has exactly one leaf owner.

    Guards the specific regression: the decoder drifted into ``ubt.adapters.base``
    and the readers imported it from there. If it moves again, it must move to a
    module *below* the readers, not into the adapters.
    """
    from ubt.core import text_decode

    assert callable(text_decode.decode_markup)
    # The reader-facing wrappers must resolve through the leaf, not the adapter.
    from ubt.analyze.reader_html import decode_bytes

    assert decode_bytes(b"hi") == "hi"

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANALYZE = _REPO_ROOT / "ubt" / "analyze"


def _known_modules() -> set[str]:
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

_ALLOWED_ADAPTER_EDGES: frozenset[str] = frozenset()


def _adapter_imports(path: Path) -> list[tuple[int, str]]:
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
    from ubt.core import text_decode

    assert callable(text_decode.decode_markup)
    from ubt.analyze.reader_html import decode_bytes

    assert decode_bytes(b"hi") == "hi"

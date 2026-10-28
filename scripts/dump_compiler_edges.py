"""Dump the core→compiler module-level import edges (read-only; exits non-zero on violation).

The acceptance tool for the layering invariant: a module-level ``ubt.core``
import of ``ubt.pipeline`` / ``ubt.segment`` / ``ubt.translate`` is a forbidden
reverse edge, and a ``ubt.core.engine <-> ubt.pipeline`` package cycle fails
this check. TYPE_CHECKING imports never enter the runtime import graph and are
excluded, as is the exempted inward edge to the shared leaf ``ubt.model``.

Run from the repository root: ``.venv/bin/python scripts/dump_compiler_edges.py``.
"""

import ast
from pathlib import Path

COMPILER = ("ubt.pipeline", "ubt.segment", "ubt.translate", "ubt.model")
#: The zero-out-edge shared leaf: core→model is an inward edge, exempted by design.
_EXEMPT_LEAF = "ubt.model"


def pkg_of(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    return f"ubt.core.{parts[2]}" if parts[1] == "core" and len(parts) > 3 else f"ubt.{parts[1]}"


def _is_type_checking(node: ast.AST) -> bool:
    """Whether this ``if`` is a ``TYPE_CHECKING`` guard (any spelling)."""
    test = getattr(node, "test", None)
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _resolve_relative(path: Path, level: int, module: str | None) -> str | None:
    """Absolute module name for a ``from ..x import y`` (``level`` dots)."""
    if level == 0:
        return module
    package = list(path.with_suffix("").parts)[:-1]
    if level - 1 > len(package):
        return module
    base = package[: len(package) - (level - 1)] if level > 1 else package
    parts = [*base, *([module] if module else [])]
    return ".".join(parts) if parts else module


def scan(path: Path):
    """Module-level and function-level ``ubt`` imports, with TYPE_CHECKING excluded.

    Module level means *any* import that runs at import time — including one
    nested in a module-level ``try:`` or ``class`` body, which the previous
    ``tree.body``-only walk could not see. A relative import is resolved to an
    absolute name so it is judged by the same rule.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top: list[tuple[int, str]] = []
    lazy: list[tuple[int, str]] = []

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.fn_depth = 0
            self.tc_depth = 0

        def visit_If(self, n: ast.If) -> None:
            guarded = _is_type_checking(n)
            if guarded:
                self.tc_depth += 1
            self.generic_visit(n)
            if guarded:
                self.tc_depth -= 1

        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
            self.fn_depth += 1
            self.generic_visit(n)
            self.fn_depth -= 1

        def visit_AsyncFunctionDef(self, n: ast.AsyncFunctionDef) -> None:
            self.fn_depth += 1
            self.generic_visit(n)
            self.fn_depth -= 1

        def visit_ImportFrom(self, n: ast.ImportFrom) -> None:
            if not self.tc_depth:
                resolved = _resolve_relative(path, n.level, n.module)
                if resolved:
                    (lazy if self.fn_depth else top).append((n.lineno, resolved))

        def visit_Import(self, n: ast.Import) -> None:
            if not self.tc_depth:
                target = lazy if self.fn_depth else top
                target.extend((n.lineno, a.name) for a in n.names)

    V().visit(tree)
    return top, lazy


def main() -> int:
    print("== 模块级 core→编译器包 ==")
    n = 0
    forbidden = 0
    for f in sorted(Path("ubt/core").rglob("*.py")):
        for ln, mod in scan(f)[0]:
            if any(mod == c or mod.startswith(c + ".") for c in COMPILER):
                print(f"  {f}:{ln} -> {mod}")
                n += 1
                if not (mod == _EXEMPT_LEAF or mod.startswith(_EXEMPT_LEAF + ".")):
                    forbidden += 1
    print(f"  合计: {n}（验收：pipeline/segment/translate 三类 = 0）")

    # ``core`` is the domain layer: it must not import an adapter. Without this the
    # gate only watched the compiler packages, so any core module that reached for
    # ``ubt.adapters`` (docling, typst, playwright) passed CI green while dragging
    # those trees into the domain's import graph — the exact edge
    # ARCHITECTURE.md says is forbidden and claims is enforced.
    #
    # ``sys.modules`` lookups from a function body and ``TYPE_CHECKING`` imports do
    # not enter the runtime graph and are excluded by ``scan``.
    adapters_forbidden: list[tuple[str, int, str]] = []
    for f in sorted(Path("ubt/core").rglob("*.py")):
        for ln, mod in scan(f)[0]:
            if mod == "ubt.adapters" or mod.startswith("ubt.adapters."):
                adapters_forbidden.append((str(f), ln, mod))
    if adapters_forbidden:
        print("== 模块级 core→adapters（验收：0）==")
        for f_path, ln, mod in adapters_forbidden:
            print(f"  {f_path}:{ln} -> {mod}")
    else:
        print("== 模块级 core→adapters（验收：0）: 0 ==")

    edges: dict[str, set[str]] = {}
    for f in Path("ubt").rglob("*.py"):
        src = pkg_of(f)
        for _, mod in scan(f)[0]:
            if mod.startswith("ubt."):
                parts = mod.split(".")
                cand = Path(*parts, "__init__.py")
                tgt = pkg_of(cand if cand.exists() else Path(*parts, "x.py"))
                if tgt != src:
                    edges.setdefault(src, set()).add(tgt)

    a, b = "ubt.core.engine", "ubt.pipeline"
    ab, ba = b in edges.get(a, set()), a in edges.get(b, set())
    cycle = ab and ba
    print(f"== {a} <-> {b}: cycle={cycle}（验收：False）==")

    # Render toolbox is a DOWNSTREAM-free layer: ubt/render may not reach back
    # into ubt/adapters at module level. The one edge that made this a real
    # package-level cycle (render/typst/math_transpiler importing the
    # adapters/pdf/typst_symbols shim) is gone; keep it gone.
    render_violations: list[str] = []
    for src, tgts in sorted(edges.items()):
        if src == "ubt.render" or src.startswith("ubt.render."):
            for tgt in sorted(tgts):
                if tgt == "ubt.adapters" or tgt.startswith("ubt.adapters."):
                    render_violations.append(f"{src} -> {tgt}")
    if render_violations:
        print("== render→adapters 模块级边（验收：0）==")
        for edge in render_violations:
            print(f"  {edge}")
    else:
        print("== render→adapters 模块级边（验收：0）: 0 ==")

    # --------------------------------------------------------------------------- #
    # Port-boundary invariant: within ubt.core, only ubt/core/ports.py is
    # permitted to contain function-level (lazy) bridges into ubt.adapters or the
    # compiler packages (ubt.pipeline, ubt.segment, ubt.translate).
    # Any lazy import elsewhere in ubt.core is an architectural leak bypassing the
    # designated port bridge.
    # --------------------------------------------------------------------------- #
    lazy_leaks: list[tuple[str, int, str]] = []
    for f in sorted(Path("ubt/core").rglob("*.py")):
        if f.name == "ports.py":
            continue
        for ln, mod in scan(f)[1]:
            for bad in ("ubt.adapters", "ubt.pipeline", "ubt.segment", "ubt.translate"):
                if mod == bad or mod.startswith(bad + "."):
                    lazy_leaks.append((str(f), ln, mod))

    if lazy_leaks:
        print("== 形式主义 lazy import 违规（核心域绕过 ports.py 验收：0）==")
        for f_str, ln, mod in lazy_leaks:
            print(f"  {f_str}:{ln} -> {mod}")
    else:
        print("== 函数级 core→adapters/compiler（仅限 ports.py，其他模块验收：0）: 0 ==")

    return (
        0
        if forbidden == 0
        and not cycle
        and not render_violations
        and not adapters_forbidden
        and not lazy_leaks
        else 1
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())

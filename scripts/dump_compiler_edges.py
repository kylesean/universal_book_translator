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


def scan(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = []

    for node in tree.body:
        if isinstance(node, ast.If) and getattr(node.test, "id", "") == "TYPE_CHECKING":
            continue
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            top += [(node.lineno, a.name) for a in node.names]

    class V(ast.NodeVisitor):
        depth = 0
        lazy: list[tuple[int, str]] = []

        def visit_FunctionDef(self, n):
            self.depth += 1
            self.generic_visit(n)
            self.depth -= 1

        def _imp(self, node):
            if self.depth and isinstance(node, ast.ImportFrom) and node.module:
                self.lazy.append((node.lineno, node.module))

        visit_ImportFrom = _imp

        def visit_Import(self, node):
            if self.depth:
                self.lazy += [(node.lineno, a.name) for a in node.names]

    v = V()
    v.visit(tree)
    return top, v.lazy


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
raise SystemExit(0 if forbidden == 0 and not cycle else 1)

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
    for f, ln, mod in adapters_forbidden:
        print(f"  {f}:{ln} -> {mod}")
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

raise SystemExit(
    0 if forbidden == 0 and not cycle and not render_violations and not adapters_forbidden else 1
)

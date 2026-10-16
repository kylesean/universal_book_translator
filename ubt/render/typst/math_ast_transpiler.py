"""AST-based LaTeX math transpiler for Typst using pylatexenc.

This module provides structural transpilation of complex LaTeX math environments
(matrices, cases, aligned equations, arrays) into native Typst math constructs
(``mat(...)``, ``cases(...)``, multiline alignments).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence

from pylatexenc.latexwalker import (  # type: ignore[import-untyped]
    LatexCharsNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexNode,
    LatexSpecialsNode,
    LatexWalker,
)

logger = logging.getLogger(__name__)

#: Supported matrix environments and their Typst delimiters
_MATRIX_DELIMS: dict[str, str] = {
    "pmatrix": "(",
    "pmatrix*": "(",
    "bmatrix": "[",
    "bmatrix*": "[",
    "vmatrix": "|",
    "vmatrix*": "|",
    "Vmatrix": "‖",
    "Vmatrix*": "‖",
    "matrix": "#none",
    "matrix*": "#none",
    "smallmatrix": "#none",
    "smallmatrix*": "#none",
    "array": "#none",
}

#: Environments that represent piecewise/conditional cases
_CASES_ENVS: frozenset[str] = frozenset({"cases", "dcases", "cases*", "dcases*"})

#: Environments that represent multiline alignments or equations. ``multline``
#: and its starred form break lines with ``\\`` just like ``align`` but must be
#: listed explicitly: an unlisted environment is unwrapped node-by-node, which
#: leaves the ``\\`` as literal LaTeX (``a \\ b``) instead of a real line break.
_ALIGN_ENVS: frozenset[str] = frozenset(
    {
        "aligned",
        "align",
        "align*",
        "alignedat",
        "split",
        "gather",
        "gather*",
        "multline",
        "multline*",
        "equation",
        "equation*",
        "eqnarray",
        "eqnarray*",
    }
)

#: Environment names that take a mandatory ``{n}`` argument (an alignment
#: column count). ``pylatexenc`` keeps that argument as a sibling node, so it
#: would otherwise leak into the emitted cells as a stray ``{2}``.
_ENVS_WITH_MANDATORY_ARG: frozenset[str] = frozenset({"alignedat", "alignedat*"})

#: Environment names that may carry an optional ``[pos]`` argument (the
#: ``*``-variant alignment of starred matrix/smallmatrix). The argument is
#: dropped: Typst does not take a positional alignment for ``mat``.
_ENVS_DROP_OPTIONAL_ARG: frozenset[str] = frozenset(
    {
        "matrix",
        "matrix*",
        "pmatrix",
        "pmatrix*",
        "bmatrix",
        "bmatrix*",
        "vmatrix",
        "vmatrix*",
        "Vmatrix",
        "Vmatrix*",
        "smallmatrix",
        "smallmatrix*",
        "array",
    }
)


def has_latex_environments(latex: str) -> bool:
    """Fast check whether the string contains a LaTeX ``\\begin{...}`` environment.

    The extractors emit a *spaced* dialect — ``\\begin { pmatrix }`` — so a
    literal ``\\begin{`` test misses the production spelling. Accept whitespace
    between ``\\begin`` and its opening brace, which is valid LaTeX and the
    exact shape Docling emits. (``typstify_math`` also normalizes before this
    gate, but the check stays permissive so it cannot silently skip an
    environment if the call order ever changes.)
    """
    return re.search(r"\\begin\s*\{", latex) is not None


def _drop_leading_group(nodes: list[LatexNode]) -> list[LatexNode]:
    """Drop a leading brace group (an environment's mandatory ``{...}`` arg)."""
    out = list(nodes)
    while out:
        head = out[0]
        if isinstance(head, LatexGroupNode):
            out.pop(0)
            continue
        if isinstance(head, LatexCharsNode) and not head.chars.strip():
            out.pop(0)
            continue
        break
    return out


def _drop_leading_optional(nodes: list[LatexNode]) -> list[LatexNode]:
    """Drop a leading ``[...]`` optional argument, wherever pylatexenc put it.

    For starred matrices the argument is not a distinct node: it is the prefix
    of the first ``LatexCharsNode`` (``\\begin{pmatrix*}[r] 1`` -> ``'[r] 1 '``),
    so the bracket run is stripped from that node's ``chars``.
    """
    out = list(nodes)
    while out:
        head = out[0]
        if _is_optional_arg_node(head):
            out.pop(0)
            continue
        if isinstance(head, LatexCharsNode):
            if not head.chars.strip():
                out.pop(0)
                continue
            match = re.match(r"^\s*\[[^\]]*\]\s*", head.chars)
            if match:
                # ``latex_verbatim()`` recomputes from ``pos``/``len`` against
                # the original source, so shifting those (not ``chars``) is what
                # actually drops the ``[...]`` prefix from the emitted cell.
                offset = match.end()
                head.pos += offset
                head.len -= offset
                head.chars = head.chars[offset:]
                if not head.chars.strip():
                    out.pop(0)
            break
        break
    return out


def _is_optional_arg_node(node: LatexNode) -> bool:
    """Whether ``node`` is a ``[...]`` optional argument rather than content."""
    macro = getattr(node, "macroname", None)
    if isinstance(node, LatexMacroNode) and macro in ("[", "]"):
        return True
    # ``pylatexenc`` may attach an optional-argument node as ``nodeargd``; the
    # ArgumentParserNode is not a content node, so treat any non-content node
    # whose verbatim form is bracketed as the optional argument.
    verbatim = getattr(node, "latex_verbatim", None)
    if callable(verbatim):
        text = verbatim().strip()
        if text.startswith("[") and text.endswith("]"):
            return True
    return False


class _LeafTranspileFailure(Exception):
    """Internal signal: a leaf chunk could not be transpiled (fail closed)."""


def _is_empty_cell(cell: Sequence[LatexNode]) -> bool:
    """Whether an extracted grid cell contains only whitespace."""
    for n in cell:
        if isinstance(n, LatexCharsNode):
            if n.chars.strip():
                return False
        else:
            return False
    return True


def _parse_grid(nodelist: Sequence[LatexNode]) -> list[list[list[LatexNode]]]:
    """Parse a list of LaTeX AST nodes into a 2D grid: rows of cells."""
    rows: list[list[list[LatexNode]]] = []
    curr_row: list[list[LatexNode]] = []
    curr_cell: list[LatexNode] = []

    for n in nodelist:
        if isinstance(n, LatexMacroNode) and n.macroname in ("\\", "cr", "tabularnewline"):
            curr_row.append(curr_cell)
            rows.append(curr_row)
            curr_row = []
            curr_cell = []
        elif (isinstance(n, LatexSpecialsNode) and n.specials_chars == "&") or (
            isinstance(n, LatexCharsNode) and n.chars.strip() == "&"
        ):
            curr_row.append(curr_cell)
            curr_cell = []
        else:
            curr_cell.append(n)

    if curr_cell or curr_row:
        curr_row.append(curr_cell)
        rows.append(curr_row)

    # Prune trailing empty rows (e.g. from a trailing \\ before \end{...})
    while rows and all(_is_empty_cell(c) for c in rows[-1]):
        rows.pop()

    # NOTE: no leading-empty-cell pruning. pylatexenc already consumes an
    # ``array``/``matrix`` column spec (``{cc}``/``{c|c}``) — it never appears in
    # ``nodelist`` — so the only thing a leading-empty prune could drop is a
    # legitimately empty first cell (a blank-corner matrix ``& b \\ c & d``),
    # silently collapsing a column.
    return rows


class TypstMathASTVisitor:
    """Visitor that transforms a LaTeX AST into Typst math syntax."""

    def __init__(self, leaf_transpiler: Callable[[str], str | None] | None = None) -> None:
        self._leaf_transpiler = leaf_transpiler

    def transpile_string(self, latex: str) -> str | None:
        """Parse LaTeX string and transpile its AST to Typst math."""
        if not latex or not latex.strip():
            return None
        try:
            w = LatexWalker(latex)
            nodes, _, _ = w.get_latex_nodes()
            return self.transpile_nodes(nodes)
        except _LeafTranspileFailure:
            # A leaf chunk the caller's transpiler could not represent. This is
            # the caller's documented fail-closed signal, so it must propagate
            # as ``None`` (degrade the whole formula to readable Unicode)
            # rather than let raw LaTeX reach the Typst compiler, where one
            # bad cell would fail the entire document.
            return None
        except Exception as exc:
            logger.debug("Failed to transpile LaTeX AST for %r: %s", latex[:60], exc)
            return None

    def transpile_nodes(self, nodes: Sequence[LatexNode]) -> str:
        """Transpile a sequence of nodes, transforming environments and leaf chunks."""
        parts: list[str] = []
        chunk: list[LatexNode] = []

        for n in nodes:
            if isinstance(n, LatexEnvironmentNode):
                if chunk:
                    chunk_str = self._transpile_leaf_chunk(chunk)
                    if chunk_str:
                        parts.append(chunk_str)
                    chunk = []
                env_str = self._transpile_environment(n)
                if env_str:
                    parts.append(env_str)
            else:
                chunk.append(n)

        if chunk:
            chunk_str = self._transpile_leaf_chunk(chunk)
            if chunk_str:
                parts.append(chunk_str)

        return " ".join(p for p in parts if p)

    def _transpile_leaf_chunk(self, chunk: Sequence[LatexNode]) -> str:
        """Transpile a contiguous run of non-environment nodes.

        Raises :class:`_LeafTranspileFailure` when the leaf transpiler returns
        ``None``: that is its "I cannot represent this" signal, and swallowing
        it here emits raw LaTeX that Typst rejects, failing the whole document.
        Propagating it lets ``transpile_string`` answer ``None`` and the caller
        degrade one formula to Unicode.
        """
        chunk_latex = "".join(n.latex_verbatim() for n in chunk).strip()
        if not chunk_latex:
            return ""
        if self._leaf_transpiler is not None:
            res = self._leaf_transpiler(chunk_latex)
            if res is None:
                raise _LeafTranspileFailure(chunk_latex)
            return res
        return chunk_latex

    def _transpile_cell(self, cell_nodes: Sequence[LatexNode]) -> str:
        """Transpile an individual cell in a matrix, case, or alignment."""
        return self.transpile_nodes(cell_nodes)

    def _transpile_environment(self, env_node: LatexEnvironmentNode) -> str | None:
        """Transpile a LaTeXEnvironmentNode into Typst syntax."""
        env_name = env_node.environmentname
        nodelist = list(env_node.nodelist or [])

        # Drop the environment's own specification arguments before building the
        # grid. ``pylatexenc`` leaves them as leading group/optional nodes, and
        # they are not content: ``alignedat{2}`` leaked ``{2}`` into the first
        # cell, and ``pmatrix*[r]`` leaked ``[r]`` into a matrix cell.
        if env_name in _ENVS_WITH_MANDATORY_ARG:
            nodelist = _drop_leading_group(nodelist)
        if env_name in _ENVS_DROP_OPTIONAL_ARG:
            nodelist = _drop_leading_optional(nodelist)

        grid = _parse_grid(nodelist)

        if env_name in _MATRIX_DELIMS:
            delim = _MATRIX_DELIMS[env_name]
            rendered_rows: list[str] = []
            for row in grid:
                rendered_cells = [self._transpile_cell(c) for c in row]
                rendered_rows.append(", ".join(rendered_cells))
            body = "; ".join(rendered_rows)
            delim_str = "#none" if delim == "#none" else f'"{delim}"'
            if delim == "(":
                return f"mat({body})"
            return f"mat(delim: {delim_str}, {body})"

        if env_name in _CASES_ENVS:
            rendered_cases: list[str] = []
            for row in grid:
                rendered_cells = [self._transpile_cell(c) for c in row]
                rendered_cases.append(" & ".join(rendered_cells))
            cases_str = ", ".join(rendered_cases)
            return f"cases({cases_str})"

        if env_name in _ALIGN_ENVS:
            rendered_lines: list[str] = []
            for row in grid:
                rendered_cells = [self._transpile_cell(c) for c in row]
                rendered_lines.append(" & ".join(rendered_cells))
            return " \\\n ".join(rendered_lines)

        # Fallback for unrecognized environments: unwrap inner nodes
        return self.transpile_nodes(nodelist)


def transpile_latex_ast(
    latex: str,
    leaf_transpiler: Callable[[str], str | None] | None = None,
) -> str | None:
    """Convenience helper to transpile a LaTeX math expression using AST visitor."""
    visitor = TypstMathASTVisitor(leaf_transpiler=leaf_transpiler)
    return visitor.transpile_string(latex)

"""Adaptive mathematical formula breaker with AST operator precedence and cost optimization."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.math.cost import (
    CATEGORY_BASE_COSTS,
    get_operator_tables,
)
from ubt.core.math.models import (
    BreakPoint,
    MathAuditReport,
    MathEquationAudit,
    MathSyntax,
)

_INLINE_MATH_RE: Final[re.Pattern[str]] = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^$\n]+\$)")


class AdaptiveMathBreaker:
    """Adaptive line breaker for long mathematical equations in Typst and LaTeX."""

    def __init__(
        self,
        default_target_width: int = 50,
        default_syntax: MathSyntax = MathSyntax.TYPST,
        align_operators: bool = True,
    ) -> None:
        self.default_target_width = default_target_width
        self.default_syntax = default_syntax
        self.align_operators = align_operators

    def break_equation(
        self,
        equation: str,
        target_width: int | None = None,
        syntax: MathSyntax | str | None = None,
        align: bool | None = None,
    ) -> str:
        """Break a long equation across lines using operator precedence and cost optimization."""
        eq = equation.strip()
        width = target_width if target_width is not None else self.default_target_width
        syn = MathSyntax(syntax) if syntax is not None else self.default_syntax
        do_align = align if align is not None else self.align_operators

        # Short equation or already formatted multiline
        if len(eq) <= width or "\n" in eq or "\\\n" in eq or "\\\\" in eq:
            return eq

        # Step 1: Discover all candidate breakpoints at depth 0
        candidates = self._find_candidate_breakpoints(eq, syn)
        if not candidates:
            return eq

        # Step 2: Select optimal breakpoints using min-penalty path
        selected_bps = self._select_optimal_breakpoints(eq, candidates, width)
        if not selected_bps:
            return eq

        # Step 3: Reconstruct expression with line breaks and optional alignment
        return self._format_multiline(eq, selected_bps, syn, do_align)

    def _find_candidate_breakpoints(self, equation: str, syntax: MathSyntax) -> list[BreakPoint]:
        """Scan equation tracking parentheses/bracket depth and collect valid operator break points."""
        tables = get_operator_tables(syntax)
        candidates: list[BreakPoint] = []
        depth = 0
        i = 0
        n = len(equation)

        while i < n:
            c = equation[i]

            # Track grouping depth
            if c in "({[":
                depth += 1
                i += 1
                continue
            if c in ")}]":
                if depth > 0:
                    depth -= 1
                i += 1
                continue

            # In LaTeX, handle \left and \right
            if syntax == MathSyntax.LATEX:
                if equation.startswith("\\left", i):
                    depth += 1
                    i += 5
                    continue
                if equation.startswith("\\right", i):
                    if depth > 0:
                        depth -= 1
                    i += 6
                    continue

            # Only collect candidate breaks at depth 0
            if depth == 0:
                for cat, ops in tables:
                    matched_op: str | None = None
                    for op in ops:
                        if equation.startswith(op, i):
                            matched_op = op
                            break
                    if matched_op:
                        base_cost = CATEGORY_BASE_COSTS[cat]
                        candidates.append(
                            BreakPoint(
                                index=i,
                                operator=matched_op,
                                category=cat,
                                penalty_cost=base_cost,
                                depth=depth,
                            )
                        )
                        i += len(matched_op)
                        break
                else:
                    i += 1
            else:
                i += 1

        return candidates

    def _select_optimal_breakpoints(
        self,
        equation: str,
        candidates: list[BreakPoint],
        target_width: int,
    ) -> list[BreakPoint]:
        """Select a subset of breakpoints minimizing global line penalty."""
        selected: list[BreakPoint] = []
        line_start = 0

        # Greedy lookahead: when current line exceeds target_width or a strong relation is reached
        for idx, bp in enumerate(candidates):
            line_len = bp.index - line_start

            # If line is approaching target_width (>= 0.7 * target_width)
            # or if the current line has exceeded target_width
            if line_len >= 0.7 * target_width:
                # Lookahead to see if a relation operator exists within the next 20%
                better_lookahead = None
                for future in candidates[idx + 1 : idx + 3]:
                    future_len = future.index - line_start
                    if future_len <= target_width * 1.2 and future.penalty_cost < bp.penalty_cost:
                        better_lookahead = future
                        break

                if better_lookahead and line_len < target_width:
                    # Defer to better lookahead operator
                    continue

                selected.append(bp)
                line_start = bp.index

        return selected

    def _format_multiline(
        self,
        equation: str,
        breakpoints: list[BreakPoint],
        syntax: MathSyntax,
        align: bool,
    ) -> str:
        """Splice equation at selected breakpoints inserting line break and alignment markers."""
        if not breakpoints:
            return equation

        chunks: list[str] = []

        # Break lines: insert break before the operator token
        first_bp = breakpoints[0]

        # First line
        first_line = equation[: first_bp.index].strip()
        if align and " = " in first_line:
            first_line = first_line.replace(" = ", " & = ", 1)
        chunks.append(first_line)

        for i, bp in enumerate(breakpoints):
            next_idx = breakpoints[i + 1].index if i + 1 < len(breakpoints) else len(equation)
            raw_segment = equation[bp.index : next_idx].strip()

            # Align operator
            op_str = bp.operator.strip()
            rest_of_segment = raw_segment[len(op_str) :].strip()

            aligned_op = f"& {op_str}" if align else op_str

            if syntax == MathSyntax.TYPST:
                chunks.append(f" \\\n {aligned_op} {rest_of_segment}")
            else:
                chunks.append(f" \\\\\n {aligned_op} {rest_of_segment}")

        result = "".join(chunks)

        if syntax == MathSyntax.LATEX and align and not result.startswith("\\begin{aligned}"):
            result = f"\\begin{{aligned}}\n  {result}\n\\end{{aligned}}"

        return result


def break_equation(
    equation: str,
    target_width: int = 50,
    syntax: MathSyntax | str = MathSyntax.TYPST,
    align: bool = True,
) -> str:
    """Convenience helper to break a single equation."""
    breaker = AdaptiveMathBreaker(
        default_target_width=target_width,
        default_syntax=MathSyntax(syntax),
        align_operators=align,
    )
    return breaker.break_equation(equation)


def strip_formula_delimiters(formula: str) -> str:
    """Unwrap outer $$...$$, $...$, or \\[...\\] math block delimiters."""
    f = formula.strip()
    if f.startswith("$$") and f.endswith("$$") and len(f) >= 4:
        return f[2:-2].strip()
    if f.startswith("$") and f.endswith("$") and len(f) >= 2:
        return f[1:-1].strip()
    if f.startswith(r"\[") and f.endswith(r"\]") and len(f) >= 4:
        return f[2:-2].strip()
    return f


def audit_math_blocks(
    blocks: Sequence[IRBlock],
    target_width: int = 50,
) -> MathAuditReport:
    """Audit all mathematical equations in document IR blocks for width compliance."""
    report = MathAuditReport()
    breaker = AdaptiveMathBreaker(default_target_width=target_width)

    eq_counter = 0
    for block in blocks:
        block_id = getattr(block, "id", None) or (
            block.element.id if getattr(block, "element", None) else ""
        )
        page = (
            getattr(block, "page", None)
            or getattr(block.element, "page", None)
            or (
                getattr(block.element.span, "page", None)
                if getattr(block.element, "span", None)
                else None
            )
        )
        b_type = str(getattr(block.block_type, "value", block.block_type) or "")
        text = (
            block.target_text
            or block.source_text
            or (getattr(block.element, "source", None) if block.element else None)
            or (getattr(block.element, "text", None) if block.element else None)
            or ""
        )

        # 1. Formula blocks
        if b_type == "formula" or getattr(block, "block_type", None) == BlockType.FORMULA:
            eq_counter += 1
            clean_body = strip_formula_delimiters(text)
            lines = [line.strip() for line in clean_body.splitlines() if line.strip()]
            char_len = max((len(line) for line in lines), default=len(clean_body))
            is_multi = len(lines) > 1 or r"\\" in clean_body
            exceeds = (
                any(len(line) > target_width for line in lines)
                if lines
                else len(clean_body) > target_width
            )

            bps = breaker._find_candidate_breakpoints(clean_body, MathSyntax.TYPST)
            rec_breaks = len(breaker._select_optimal_breakpoints(clean_body, bps, target_width))

            audit = MathEquationAudit(
                equation_id=f"eq-{eq_counter}",
                raw_text=clean_body,
                char_length=char_len,
                is_multiline=is_multi,
                exceeds_threshold=exceeds,
                threshold=target_width,
                recommended_breaks_count=rec_breaks,
                syntax=MathSyntax.TYPST,
                block_id=block_id,
                page=page,
            )
            report.total_equations += 1
            if is_multi:
                report.multiline_equations += 1
            if exceeds:
                report.oversized_equations += 1
            report.audits.append(audit)

        # 2. Inline formulas in prose exceeding threshold
        elif "$" in text:
            for m in _INLINE_MATH_RE.finditer(text):
                inline_content = strip_formula_delimiters(m.group(0))
                if len(inline_content) > target_width:
                    eq_counter += 1
                    bps = breaker._find_candidate_breakpoints(inline_content, MathSyntax.TYPST)
                    rec_breaks = len(
                        breaker._select_optimal_breakpoints(inline_content, bps, target_width)
                    )

                    audit = MathEquationAudit(
                        equation_id=f"inline-eq-{eq_counter}",
                        raw_text=inline_content,
                        char_length=len(inline_content),
                        is_multiline=False,
                        exceeds_threshold=True,
                        threshold=target_width,
                        recommended_breaks_count=rec_breaks,
                        syntax=MathSyntax.TYPST,
                        block_id=block_id,
                        page=page,
                    )
                    report.total_equations += 1
                    report.oversized_equations += 1
                    report.audits.append(audit)

    return report

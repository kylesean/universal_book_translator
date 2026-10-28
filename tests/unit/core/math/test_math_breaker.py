"""Unit tests for adaptive mathematical formula breaking, operator precedence, and audits."""

from __future__ import annotations

import pytest

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.core.math.breaker import (
    AdaptiveMathBreaker,
    audit_math_blocks,
    break_equation,
)
from ubt.core.math.cost import calculate_line_penalty
from ubt.core.math.models import MathSyntax

pytestmark = pytest.mark.fast


def test_calculate_line_penalty_behavior() -> None:
    # Depth > 0 penalty
    nested_penalty = calculate_line_penalty(40, 50, 10.0, depth=1)
    assert nested_penalty >= 2000.0

    # Overrun penalty
    normal_penalty = calculate_line_penalty(45, 50, 10.0, depth=0)
    overrun_penalty = calculate_line_penalty(60, 50, 10.0, depth=0)
    assert overrun_penalty > normal_penalty


def test_break_equation_typst_nested_protection() -> None:
    breaker = AdaptiveMathBreaker(
        default_target_width=20, default_syntax=MathSyntax.TYPST, align_operators=False
    )
    # The inside of frac(a + b, c + d) should NOT be split
    eq = "f(x) = frac(a + b + c, d + e + f) + g + h"
    broken = breaker.break_equation(eq)
    assert "frac(a + b + c, d + e + f)" in broken
    assert "\\\n" in broken
    assert " + g" in broken or " + h" in broken


def test_break_equation_typst_relation_and_alignment() -> None:
    breaker = AdaptiveMathBreaker(
        default_target_width=25, default_syntax=MathSyntax.TYPST, align_operators=True
    )
    eq = "L(theta) = E[log p(x | z)] - D_KL(q(z | x) || p(z)) + lambda * R(theta)"
    broken = breaker.break_equation(eq)
    assert "\\\n" in broken
    # Alignment '&' injected
    assert "& =" in broken or "& -" in broken


def test_break_equation_latex_alignment() -> None:
    eq = r"E = m c^2 + \frac{1}{2} m v^2 + \int_0^T F(t) dt + C_1 + C_2"
    broken = break_equation(eq, target_width=30, syntax=MathSyntax.LATEX, align=True)
    assert r"\begin{aligned}" in broken
    assert r"\end{aligned}" in broken
    assert r"\\" in broken
    assert "&" in broken


def test_audit_math_blocks_scoring() -> None:
    el_h1 = make_element(
        id="h1", spine_index=0, block_type=BlockType.HEADING, source_text="Physics", level=1
    )
    # Safe short formula
    el_f1 = make_element(
        id="f1", spine_index=1, block_type=BlockType.FORMULA, source_text="E = mc^2"
    )
    # Oversized formula
    el_f2 = make_element(
        id="f2",
        spine_index=2,
        block_type=BlockType.FORMULA,
        source_text="H(psi) = - frac(hbar^2, 2m) nabla^2 psi + V(x, y, z) psi + sum_(i=1)^N lambda_i phi_i(x)",
    )
    # Narrative with long inline math
    el_p1 = make_element(
        id="p1",
        spine_index=3,
        block_type=BlockType.NARRATIVE,
        source_text="Equation $A = B + C + D + E + F + G + H + I + J + K + L + M$ is oversized.",
    )

    blocks = [
        IRBlock(element=el_h1, target_text="物理学"),
        IRBlock(element=el_f1, target_text="E = mc^2"),
        IRBlock(
            element=el_f2,
            target_text="H(psi) = - frac(hbar^2, 2m) nabla^2 psi + V(x, y, z) psi + sum_(i=1)^N lambda_i phi_i(x)",
        ),
        IRBlock(
            element=el_p1,
            target_text="公式 $A = B + C + D + E + F + G + H + I + J + K + L + M$ 超长了。",
        ),
    ]

    report = audit_math_blocks(blocks, target_width=35)
    assert report.total_equations == 3
    assert report.oversized_equations == 2  # f2 and inline equation
    assert 0.0 < report.compliance_score < 100.0
    assert len(report.audits) == 3

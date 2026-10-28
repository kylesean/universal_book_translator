"""Penalty cost calculation and operator precedence tables for mathematical line breaking."""

from __future__ import annotations

from typing import Final

from ubt.core.math.models import MathSyntax, OperatorCategory

# Base costs per operator category (lower is more favorable for line breaking)
CATEGORY_BASE_COSTS: Final[dict[OperatorCategory, float]] = {
    OperatorCategory.RELATION: 10.0,
    OperatorCategory.ADDITIVE: 30.0,
    OperatorCategory.MULTIPLICATIVE: 65.0,
    OperatorCategory.PUNCTUATION: 100.0,
}

#: Typst operator tables
TYPST_RELATIONS: Final[tuple[str, ...]] = (
    " = ",
    " <= ",
    " >= ",
    " approx ",
    " equiv ",
    " != ",
    " arrow.r ",
    " ==> ",
    " --> ",
    " propto ",
    " subset ",
    " in ",
    " < ",
    " > ",
)

TYPST_ADDITIVES: Final[tuple[str, ...]] = (
    " + ",
    " - ",
    " plus.minus ",
    " minus.plus ",
)

TYPST_MULTIPLICATIVES: Final[tuple[str, ...]] = (
    " times ",
    " dot ",
    " * ",
)

#: LaTeX operator tables
LATEX_RELATIONS: Final[tuple[str, ...]] = (
    " = ",
    " \\le ",
    " \\ge ",
    " \\approx ",
    " \\equiv ",
    " \\neq ",
    " \\to ",
    " \\implies ",
    " \\iff ",
    " \\propto ",
    " \\subset ",
    " \\in ",
    " < ",
    " > ",
)

LATEX_ADDITIVES: Final[tuple[str, ...]] = (
    " + ",
    " - ",
    " \\pm ",
    " \\mp ",
    " \\oplus ",
)

LATEX_MULTIPLICATIVES: Final[tuple[str, ...]] = (
    " \\times ",
    " \\cdot ",
    " \\otimes ",
    " * ",
)


def get_operator_tables(
    syntax: MathSyntax,
) -> list[tuple[OperatorCategory, tuple[str, ...]]]:
    """Return categorized operators ordered by breaking precedence."""
    if syntax == MathSyntax.LATEX:
        return [
            (OperatorCategory.RELATION, LATEX_RELATIONS),
            (OperatorCategory.ADDITIVE, LATEX_ADDITIVES),
            (OperatorCategory.MULTIPLICATIVE, LATEX_MULTIPLICATIVES),
            (OperatorCategory.PUNCTUATION, ("; ", ", ")),
        ]
    return [
        (OperatorCategory.RELATION, TYPST_RELATIONS),
        (OperatorCategory.ADDITIVE, TYPST_ADDITIVES),
        (OperatorCategory.MULTIPLICATIVE, TYPST_MULTIPLICATIVES),
        (OperatorCategory.PUNCTUATION, ("; ", ", ")),
    ]


def calculate_line_penalty(
    current_line_length: int,
    target_width: int,
    operator_base_cost: float,
    depth: int = 0,
) -> float:
    """Calculate the comprehensive penalty for breaking at a candidate location.

    Penalizes:
    - Operating inside nested expressions (depth > 0 pays severe penalty: 1000 * depth);
    - Exceeding the target column width (quadratic penalty for overruns);
    - Line underfill if line is broken prematurely (< 0.5 * target_width);
    - Operator base cost (relations << additives << multiplicatives).
    """
    if depth > 0:
        return 2000.0 * depth + operator_base_cost

    diff = current_line_length - target_width
    if diff > 0:
        # Overrun penalty: quadratic growth
        width_penalty = 1.8 * (diff**2)
    else:
        # Underfill penalty: mild linear penalty if line is under 40% of target width
        underfill = (0.4 * target_width) - current_line_length
        width_penalty = 0.5 * max(0.0, underfill)

    return operator_base_cost + width_penalty

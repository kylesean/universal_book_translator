"""Operator break tables and base costs for mathematical line breaking."""

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

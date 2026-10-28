"""Adaptive mathematical formula breaking, operator cost models, and layout audits."""

from __future__ import annotations

from ubt.core.math.breaker import (
    AdaptiveMathBreaker,
    audit_math_blocks,
    break_equation,
)
from ubt.core.math.cost import (
    CATEGORY_BASE_COSTS,
    calculate_line_penalty,
    get_operator_tables,
)
from ubt.core.math.models import (
    BreakPoint,
    MathAuditReport,
    MathEquationAudit,
    MathSyntax,
    OperatorCategory,
)

__all__ = [
    "CATEGORY_BASE_COSTS",
    "AdaptiveMathBreaker",
    "BreakPoint",
    "MathAuditReport",
    "MathEquationAudit",
    "MathSyntax",
    "OperatorCategory",
    "audit_math_blocks",
    "break_equation",
    "calculate_line_penalty",
    "get_operator_tables",
]

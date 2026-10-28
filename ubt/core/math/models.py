"""Data models for adaptive mathematical formula breaking and layout audits."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MathSyntax(StrEnum):
    """Supported mathematical notation syntax."""

    TYPST = "typst"
    LATEX = "latex"


class OperatorCategory(StrEnum):
    """Categorization of mathematical operators by break suitability and precedence."""

    RELATION = "relation"  # '=', '<=', 'approx', etc. (Lowest penalty, best break)
    ADDITIVE = "additive"  # '+', '-', '\pm' (Moderate penalty)
    MULTIPLICATIVE = "multiplicative"  # '\times', '\cdot' (Higher penalty)
    PUNCTUATION = "punctuation"  # ';', ',' (Conditional break)


@dataclass(frozen=True, slots=True)
class BreakPoint:
    """A candidate mathematical line-break location."""

    index: int
    operator: str
    category: OperatorCategory
    penalty_cost: float
    depth: int = 0


@dataclass(slots=True)
class MathEquationAudit:
    """Audit analysis for a single mathematical equation in a document."""

    equation_id: str
    raw_text: str
    char_length: int
    is_multiline: bool
    exceeds_threshold: bool
    threshold: int
    recommended_breaks_count: int = 0
    syntax: MathSyntax = MathSyntax.TYPST
    block_id: str = ""
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "equation_id": self.equation_id,
            "char_length": self.char_length,
            "is_multiline": self.is_multiline,
            "exceeds_threshold": self.exceeds_threshold,
            "threshold": self.threshold,
            "recommended_breaks_count": self.recommended_breaks_count,
            "syntax": str(self.syntax),
            "block_id": self.block_id,
            "page": self.page,
            "preview": self.raw_text[:50] + ("..." if len(self.raw_text) > 50 else ""),
        }


@dataclass(slots=True)
class MathAuditReport:
    """Comprehensive mathematical formula layout health report for a document."""

    total_equations: int = 0
    oversized_equations: int = 0
    multiline_equations: int = 0
    audits: list[MathEquationAudit] = field(default_factory=list)

    @property
    def compliance_score(self) -> float:
        """Percentage of equations within safe layout boundaries (0.0 to 100.0)."""
        if self.total_equations == 0:
            return 100.0
        safe = self.total_equations - self.oversized_equations
        return max(0.0, round((safe / self.total_equations) * 100.0, 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_equations": self.total_equations,
            "oversized_equations": self.oversized_equations,
            "multiline_equations": self.multiline_equations,
            "compliance_score": self.compliance_score,
            "audits": [a.to_dict() for a in self.audits],
        }

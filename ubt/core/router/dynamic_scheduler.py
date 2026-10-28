"""Dynamic Pareto-optimal multi-model routing scheduler and complexity classifier.

Solves the monolithic LLM cost vs. quality dilemma in long-form book translation:
- Assigning flagship models (e.g. Claude-3.7-Sonnet / GPT-4o) to simple narrative/headings
  burns excessive token budget with negligible quality gains.
- Assigning budget/small models to complex mathematical proofs, dense code listings,
  or multi-column tables triggers high rejection rates and costly repair loops.

Architecture:
1. BlockComplexityClassifier:
   Extracts multi-dimensional features (inline/display math, LaTeX macros, code syntax,
   table grids, terminology density, syntactic depth) and computes a normalized
   complexity score in [0.0, 1.0], categorized into TRIVIAL, STANDARD, COMPLEX, EXTREME.
2. DynamicParetoScheduler:
   Maps complexity tiers to model configurations based on Pareto frontier trade-offs.
   Escalates models dynamically on validation failure or repair retries.
3. Simulation and Diagnostics:
   Projects entire book translation token distribution, estimated cost, and savings
   percentage against monolithic flagship runs without invoking live model APIs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.core.ir.models import BlockType
from ubt.core.router.pricing import resolve_model_prices

if TYPE_CHECKING:
    from ubt.core.ir.models import IRBlock


class ComplexityTier(StrEnum):
    """Categorical complexity grade for text blocks."""

    TRIVIAL = "trivial"  # Short headings, simple boilerplate, transitions
    STANDARD = "standard"  # Normal expository narrative paragraphs
    COMPLEX = "complex"  # High terminology density, inline math, tables
    EXTREME = "extreme"  # Deep LaTeX proofs, multi-line code listings, nested math


class DynamicRoutingPolicy(StrEnum):
    """Routing optimization strategy along the Pareto frontier."""

    PARETO = "pareto"  # Sweet-spot trade-off: budget for trivial/std, standard/flagship for complex
    QUALITY_FIRST = "quality_first"  # Prioritize maximum quality: flagship for complex & extreme
    COST_FIRST = "cost_first"  # Prioritize lowest expenditure: budget for all but extreme


class BlockComplexityMetrics(BaseModel):
    """Quantitative feature analysis and complexity rating for a single IR block."""

    model_config = ConfigDict(extra="ignore")

    block_id: str
    block_type: str
    char_count: int
    word_count: int
    math_inline_count: int = 0
    math_display_count: int = 0
    math_macro_count: int = 0
    code_line_count: int = 0
    code_syntax_score: float = 0.0
    term_matches_count: int = 0
    syntactic_depth_score: float = 0.0
    score: float = Field(ge=0.0, le=1.0)
    tier: ComplexityTier
    reasons: list[str] = Field(default_factory=list)


class DynamicModelAssignment(BaseModel):
    """Routing decision assigning a target model to a specific block."""

    model_config = ConfigDict(extra="ignore")

    block_id: str
    assigned_model: str
    tier: ComplexityTier
    metrics: BlockComplexityMetrics
    reason: str
    is_escalated: bool = False
    estimated_cost_usd: float = 0.0
    monolithic_flagship_cost_usd: float = 0.0
    savings_ratio: float = 0.0


class DynamicRoutingSimulationReport(BaseModel):
    """Document-wide simulation report comparing dynamic routing against monolithic baselines."""

    model_config = ConfigDict(extra="ignore")

    total_blocks: int
    tier_counts: dict[str, int]
    model_counts: dict[str, int]
    model_shares: dict[str, float]
    total_estimated_cost_usd: float
    total_monolithic_flagship_cost_usd: float
    total_savings_usd: float
    savings_percentage: float
    average_complexity_score: float
    assignments: list[DynamicModelAssignment] = Field(default_factory=list)


# Precompiled regex patterns for fast feature extraction
_MATH_INLINE_PATTERN = re.compile(r"(?<!\\)\$(?!\$)[^$\n]+(?<!\\)\$")
_MATH_DISPLAY_PATTERN = re.compile(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]")
_MATH_MACROS_PATTERN = re.compile(
    r"\\(?:frac|sum|int|prod|partial|sqrt|bmatrix|pmatrix|alpha|beta|gamma|theta|lambda|sigma|nabla|infty|approx|equiv|leq|geq|times|cdot|in|subset|forall|exists|mathbb|mathcal)\b"
)
_CODE_KEYWORDS_PATTERN = re.compile(
    r"\b(?:class|def|fn|function|return|import|export|struct|impl|async|await|var|let|const|int|float|char|bool|try|catch|except|finally|public|private|interface|for|while|if|else|match|switch)\b"
)
_CODE_OPERATORS_PATTERN = re.compile(r"->|=>|::|[{}\[\]()]|&&|\|\||!=|==")


class BlockComplexityClassifier:
    """Evaluates intrinsic textual, mathematical, and programmatic complexity of IR blocks."""

    def __init__(
        self,
        trivial_threshold: float = 0.15,
        complex_threshold: float = 0.55,
        extreme_threshold: float = 0.80,
    ) -> None:
        self.trivial_threshold = trivial_threshold
        self.complex_threshold = complex_threshold
        self.extreme_threshold = extreme_threshold

    def classify_block(
        self,
        block: IRBlock | dict[str, Any],
        glossary_terms: Sequence[str] | None = None,
    ) -> BlockComplexityMetrics:
        """Analyze a block and compute its normalized complexity metrics."""
        b_id = str(
            getattr(block, "id", None)
            or (block.get("id") if isinstance(block, dict) else "unknown")
        )
        b_type_val = getattr(block, "block_type", None) or (
            block.get("block_type") if isinstance(block, dict) else "narrative"
        )
        b_type = str(b_type_val).lower()
        source_text = str(
            getattr(block, "source_text", None)
            or (block.get("source_text") if isinstance(block, dict) else "")
        ).strip()

        char_cnt = len(source_text)
        words = source_text.split()
        word_cnt = len(words)

        reasons: list[str] = []

        # 1. Base score from block type
        if b_type == BlockType.HEADING.value or b_type == "heading":
            score = 0.05
            reasons.append("Heading structure")
        elif b_type == BlockType.TABLE.value or b_type == "table":
            score = 0.50
            reasons.append("Tabular structured data")
        elif b_type == BlockType.CODE.value or b_type == "code":
            score = 0.65
            reasons.append("Code listing block")
        elif b_type == BlockType.FORMULA.value or b_type == "formula":
            score = 0.85
            reasons.append("Standalone mathematical formula")
        else:
            score = 0.20

        # 2. Mathematical features
        inline_matches = _MATH_INLINE_PATTERN.findall(source_text)
        display_matches = _MATH_DISPLAY_PATTERN.findall(source_text)
        macro_matches = _MATH_MACROS_PATTERN.findall(source_text)

        math_inline_cnt = len(inline_matches)
        math_display_cnt = len(display_matches)
        math_macro_cnt = len(macro_matches)

        if math_display_cnt > 0:
            score += min(0.35, math_display_cnt * 0.20)
            reasons.append(f"{math_display_cnt} display equation(s)")
        if math_inline_cnt > 0:
            score += min(0.25, math_inline_cnt * 0.08)
            reasons.append(f"{math_inline_cnt} inline equation(s)")
        if math_macro_cnt > 0:
            score += min(0.20, math_macro_cnt * 0.03)
            reasons.append(f"{math_macro_cnt} LaTeX math macro(s)")

        # 3. Code features
        lines = [line for line in source_text.splitlines() if line.strip()]
        code_line_cnt = len(lines)
        code_kw_cnt = len(_CODE_KEYWORDS_PATTERN.findall(source_text))
        code_op_cnt = len(_CODE_OPERATORS_PATTERN.findall(source_text))
        code_syntax_score = min(1.0, (code_kw_cnt * 0.08) + (code_op_cnt * 0.03))

        if b_type == "code" or "```" in source_text or code_syntax_score >= 0.35:
            score += min(0.30, code_syntax_score * 0.35)
            if code_kw_cnt > 0:
                reasons.append(f"Code syntax keywords ({code_kw_cnt})")

        # 4. Terminology density
        term_cnt = 0
        if glossary_terms:
            text_lower = source_text.lower()
            for term in glossary_terms:
                t_clean = term.strip().lower()
                if t_clean and t_clean in text_lower:
                    term_cnt += 1

        if term_cnt > 0:
            density_bonus = min(0.25, term_cnt * 0.05)
            score += density_bonus
            reasons.append(f"{term_cnt} domain glossary term(s)")

        # 5. Syntactic depth & sentence structure
        # High sentence length and rich punctuation (semicolons, parentheses)
        semicolon_cnt = source_text.count(";")
        paren_cnt = source_text.count("(")
        depth_score = min(0.20, (semicolon_cnt * 0.04) + (paren_cnt * 0.02))
        if word_cnt > 35:
            depth_score += 0.05
        score += depth_score

        # Short paragraph discount
        if (
            char_cnt < 45
            and math_inline_cnt == 0
            and math_display_cnt == 0
            and b_type in ("narrative", "dialogue")
        ):
            score = max(0.05, score * 0.40)
            reasons.append("Short brief paragraph")

        # Normalize score
        final_score = round(max(0.0, min(1.0, score)), 4)

        # Categorize tier
        if final_score < self.trivial_threshold:
            tier = ComplexityTier.TRIVIAL
        elif final_score < self.complex_threshold:
            tier = ComplexityTier.STANDARD
        elif final_score < self.extreme_threshold:
            tier = ComplexityTier.COMPLEX
        else:
            tier = ComplexityTier.EXTREME

        if not reasons:
            reasons.append("Standard expository prose")

        return BlockComplexityMetrics(
            block_id=b_id,
            block_type=b_type,
            char_count=char_cnt,
            word_count=word_cnt,
            math_inline_count=math_inline_cnt,
            math_display_count=math_display_cnt,
            math_macro_count=math_macro_cnt,
            code_line_count=code_line_cnt,
            code_syntax_score=round(code_syntax_score, 3),
            term_matches_count=term_cnt,
            syntactic_depth_score=round(depth_score, 3),
            score=final_score,
            tier=tier,
            reasons=reasons,
        )


class DynamicParetoScheduler:
    """Dynamic multi-model router optimizing translation quality and token expenditure."""

    def __init__(
        self,
        policy: DynamicRoutingPolicy = DynamicRoutingPolicy.PARETO,
        budget_model: str = "deepseek-chat",
        standard_model: str = "deepseek-chat",
        flagship_model: str = "claude-3-7-sonnet",
        classifier: BlockComplexityClassifier | None = None,
    ) -> None:
        self.policy = policy
        self.budget_model = budget_model
        self.standard_model = standard_model
        self.flagship_model = flagship_model
        self.classifier = classifier or BlockComplexityClassifier()

    def _estimate_block_cost(self, model: str, char_count: int) -> float:
        """Estimate single block token cost in USD using prices table."""
        try:
            in_rate, out_rate = resolve_model_prices(model)
        except Exception:
            in_rate, out_rate = (0.27, 1.10)  # Default fallback rate

        prompt_tokens = max(1, (char_count // 4) + 250)
        completion_tokens = max(1, char_count // 4)
        return round((prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000, 6)

    def resolve_model_for_block(
        self,
        block: IRBlock | dict[str, Any],
        glossary_terms: Sequence[str] | None = None,
        is_retry_or_repair: bool = False,
    ) -> DynamicModelAssignment:
        """Assign the optimal model tier to an IR block based on complexity and retry state."""
        metrics = self.classifier.classify_block(block, glossary_terms=glossary_terms)
        b_id = metrics.block_id

        # Determine effective tier with escalation on retry
        effective_tier = metrics.tier
        is_escalated = False

        if is_retry_or_repair:
            is_escalated = True
            if effective_tier == ComplexityTier.TRIVIAL:
                effective_tier = ComplexityTier.STANDARD
            elif effective_tier == ComplexityTier.STANDARD:
                effective_tier = ComplexityTier.COMPLEX
            elif effective_tier == ComplexityTier.COMPLEX:
                effective_tier = ComplexityTier.EXTREME

        # Apply routing policy
        if self.policy == DynamicRoutingPolicy.QUALITY_FIRST:
            if effective_tier in (ComplexityTier.COMPLEX, ComplexityTier.EXTREME):
                assigned_model = self.flagship_model
                dispatch_reason = (
                    "Quality-First policy: escalated to Flagship model for complex block"
                )
            else:
                assigned_model = self.standard_model
                dispatch_reason = "Quality-First policy: standard tier for non-complex block"

        elif self.policy == DynamicRoutingPolicy.COST_FIRST:
            if effective_tier == ComplexityTier.EXTREME:
                assigned_model = self.standard_model
                dispatch_reason = "Cost-First policy: capped at Standard model for extreme block"
            else:
                assigned_model = self.budget_model
                dispatch_reason = "Cost-First policy: Budget model assigned"

        else:  # PARETO (default sweet-spot)
            if effective_tier == ComplexityTier.TRIVIAL:
                assigned_model = self.budget_model
                dispatch_reason = "Pareto sweet-spot: Budget model for trivial block"
            elif effective_tier == ComplexityTier.STANDARD:
                assigned_model = self.budget_model
                dispatch_reason = "Pareto sweet-spot: Budget model for standard narrative"
            elif effective_tier == ComplexityTier.COMPLEX:
                assigned_model = self.standard_model
                dispatch_reason = "Pareto sweet-spot: Standard model for complex block"
            else:  # EXTREME
                assigned_model = self.flagship_model
                dispatch_reason = (
                    "Pareto sweet-spot: Flagship model for extreme mathematical/code proof"
                )

        if is_escalated:
            dispatch_reason += f" (Escalated from {metrics.tier.value} on retry)"

        # Financial cost estimation
        est_cost = self._estimate_block_cost(assigned_model, metrics.char_count)
        flagship_cost = self._estimate_block_cost(self.flagship_model, metrics.char_count)
        savings_ratio = (
            round((flagship_cost - est_cost) / flagship_cost, 4) if flagship_cost > 0 else 0.0
        )

        return DynamicModelAssignment(
            block_id=b_id,
            assigned_model=assigned_model,
            tier=effective_tier,
            metrics=metrics,
            reason=dispatch_reason,
            is_escalated=is_escalated,
            estimated_cost_usd=est_cost,
            monolithic_flagship_cost_usd=flagship_cost,
            savings_ratio=max(0.0, savings_ratio),
        )

    def simulate_document_routing(
        self,
        blocks: Sequence[IRBlock | dict[str, Any]],
        glossary_terms: Sequence[str] | None = None,
    ) -> DynamicRoutingSimulationReport:
        """Simulate routing across all blocks in a document, computing total savings."""
        if not blocks:
            return DynamicRoutingSimulationReport(
                total_blocks=0,
                tier_counts={},
                model_counts={},
                model_shares={},
                total_estimated_cost_usd=0.0,
                total_monolithic_flagship_cost_usd=0.0,
                total_savings_usd=0.0,
                savings_percentage=0.0,
                average_complexity_score=0.0,
                assignments=[],
            )

        assignments: list[DynamicModelAssignment] = [
            self.resolve_model_for_block(b, glossary_terms=glossary_terms) for b in blocks
        ]

        tier_counts: dict[str, int] = {t.value: 0 for t in ComplexityTier}
        model_counts: dict[str, int] = {}

        total_est_cost = 0.0
        total_flagship_cost = 0.0
        total_score = 0.0

        for a in assignments:
            tier_counts[a.tier.value] = tier_counts.get(a.tier.value, 0) + 1
            model_counts[a.assigned_model] = model_counts.get(a.assigned_model, 0) + 1
            total_est_cost += a.estimated_cost_usd
            total_flagship_cost += a.monolithic_flagship_cost_usd
            total_score += a.metrics.score

        n = len(assignments)
        model_shares = {m: round(cnt / n, 4) for m, cnt in model_counts.items()}
        total_savings = max(0.0, total_flagship_cost - total_est_cost)
        savings_pct = (
            round((total_savings / total_flagship_cost) * 100.0, 2)
            if total_flagship_cost > 0
            else 0.0
        )
        avg_score = round(total_score / n, 4) if n > 0 else 0.0

        return DynamicRoutingSimulationReport(
            total_blocks=n,
            tier_counts=tier_counts,
            model_counts=model_counts,
            model_shares=model_shares,
            total_estimated_cost_usd=round(total_est_cost, 4),
            total_monolithic_flagship_cost_usd=round(total_flagship_cost, 4),
            total_savings_usd=round(total_savings, 4),
            savings_percentage=savings_pct,
            average_complexity_score=avg_score,
            assignments=assignments,
        )

    def render_simulation_terminal(
        self,
        report: DynamicRoutingSimulationReport,
        console: Console | None = None,
    ) -> None:
        """Render Rich terminal dashboard of simulation results."""
        out = console or Console()

        overview_table = Table(box=None, show_header=False, padding=(0, 2))
        overview_table.add_column("Metric", style="bold cyan")
        overview_table.add_column("Value")

        overview_table.add_row("Total Document Blocks", str(report.total_blocks))
        overview_table.add_row("Average Complexity", f"{report.average_complexity_score:.3f}")
        overview_table.add_row(
            "Monolithic Flagship Cost", f"${report.total_monolithic_flagship_cost_usd:.4f}"
        )
        overview_table.add_row(
            "Dynamic Pareto Cost", f"[bold green]${report.total_estimated_cost_usd:.4f}[/]"
        )
        overview_table.add_row(
            "Estimated Savings",
            f"[bold green]${report.total_savings_usd:.4f} ({report.savings_percentage}%)[/]",
        )

        out.print(
            Panel(
                overview_table,
                title="[bold]Dynamic Pareto Routing Simulation[/bold]",
                subtitle=f"Policy: {self.policy.value.upper()}",
                border_style="green",
            )
        )

        # Tier breakdown table
        tier_table = Table(title="Complexity Tier Distribution", show_header=True)
        tier_table.add_column("Complexity Tier", style="bold")
        tier_table.add_column("Block Count", justify="right")
        tier_table.add_column("Percentage", justify="right")

        for tier, cnt in report.tier_counts.items():
            pct = (cnt / report.total_blocks * 100.0) if report.total_blocks > 0 else 0.0
            tier_table.add_row(tier.upper(), str(cnt), f"{pct:.1f}%")

        out.print(tier_table)

        # Model dispatch table
        model_table = Table(title="Model Dispatch Allocation", show_header=True)
        model_table.add_column("Model Name", style="bold cyan")
        model_table.add_column("Blocks Dispatched", justify="right")
        model_table.add_column("Workload Share", justify="right")

        for model, cnt in report.model_counts.items():
            share = report.model_shares.get(model, 0.0) * 100.0
            model_table.add_row(model, str(cnt), f"{share:.1f}%")

        out.print(model_table)

    def render_simulation_markdown(self, report: DynamicRoutingSimulationReport) -> str:
        """Render GitHub Flavored Markdown summary report."""
        lines = [
            "# Dynamic Pareto Routing Simulation Report",
            "",
            "> [!TIP]",
            f"> Dynamic routing projects **${report.total_savings_usd:.4f}** ({report.savings_percentage}%) "
            f"token cost savings compared to a monolithic {self.flagship_model} run.",
            "",
            "## Executive Summary",
            "",
            f"- **Total Blocks**: {report.total_blocks}",
            f"- **Average Complexity Score**: {report.average_complexity_score:.3f}",
            f"- **Monolithic Flagship Cost**: ${report.total_monolithic_flagship_cost_usd:.4f}",
            f"- **Dynamic Routed Cost**: ${report.total_estimated_cost_usd:.4f}",
            f"- **Net Cost Reduction**: {report.savings_percentage}%",
            "",
            "## Model Dispatch Allocation",
            "",
            "| Model | Block Count | Workload Share |",
            "| :--- | :---: | :---: |",
        ]
        for m, cnt in report.model_counts.items():
            share = report.model_shares.get(m, 0.0) * 100.0
            lines.append(f"| `{m}` | {cnt} | {share:.1f}% |")

        lines.extend(
            [
                "",
                "## Complexity Tier Breakdown",
                "",
                "| Tier | Blocks | Share |",
                "| :--- | :---: | :---: |",
            ]
        )
        for t, cnt in report.tier_counts.items():
            share = (cnt / report.total_blocks * 100.0) if report.total_blocks > 0 else 0.0
            lines.append(f"| `{t.upper()}` | {cnt} | {share:.1f}% |")

        return "\n".join(lines)

    def render_simulation_json(self, report: DynamicRoutingSimulationReport) -> str:
        """Render JSON format simulation results."""
        return json.dumps(report.model_dump(), ensure_ascii=False, indent=2)


__all__ = [
    "BlockComplexityClassifier",
    "BlockComplexityMetrics",
    "ComplexityTier",
    "DynamicModelAssignment",
    "DynamicParetoScheduler",
    "DynamicRoutingPolicy",
    "DynamicRoutingSimulationReport",
]

"""Machine-readable KPI schema for a translation run.

The KPIs are *derived* from artifacts the pipeline already persists
(``*_quality_report.json`` and ``*_visual_report.json``), so collecting them
costs nothing and never re-scans the text. Definitions live here — the single
source of truth — and are versioned by :data:`SCHEMA_VERSION`; the on-disk
artifact carries only values, so a definition change forces a version bump
instead of silently comparing numbers that no longer mean the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: Bump when a KPI is added/removed or its formula/source changes. Golden files
#: and CI comparisons reject mismatched versions rather than comparing
#: incomparable numbers.
SCHEMA_VERSION = 2

KpiUnit = Literal["ratio", "score", "count", "usd_per_1k_chars"]


class Direction(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True)
class KpiDefinition:
    """One KPI: what it means, where it comes from, and how to compare it."""

    name: str
    unit: KpiUnit
    direction: Direction
    definition: str
    source: str
    #: Default absolute tolerance for the regression compare (same unit as the
    #: KPI). A change smaller than this never counts as a regression.
    tolerance: float
    #: Optional absolute gate bound (None = report-only). ``floor`` applies to
    #: higher-is-better KPIs, ``ceiling`` to lower-is-better ones. Only a KPI
    #: whose violation is a *defect* gets one, never a KPI that measures model
    #: appetite: a bound is what stops ``UBT_UPDATE_GOLDENS=1`` from laundering
    #: a regression into the record (the baseline gate checks the golden against
    #: these too, not only the candidate).
    #: Deliberately unbounded: ``pass_rate`` (how loudly a corpus quarantines is
    #: a policy choice, not a defect), ``avg_qe`` / ``repair_rate`` / ``term_*``
    #: (means over a handful of flagged blocks, so corpus-shaped noise), the two
    #: rigid-fidelity keys (0 whenever the render was not rigid), and
    #: ``cost_per_1k_chars`` (0 when the run could not be priced).
    floor: float | None = None
    ceiling: float | None = None


KPI_DEFINITIONS: tuple[KpiDefinition, ...] = (
    KpiDefinition(
        "term_consistency",
        "ratio",
        Direction.HIGHER_IS_BETTER,
        "1 - terms_with_drift / terms_audited; vacuously 1.0 when no term was audited",
        "quality_report.entity_consistency",
        0.02,
    ),
    KpiDefinition(
        "term_precision",
        "ratio",
        Direction.HIGHER_IS_BETTER,
        "WMT term precision: glossary terms whose exact rendering is present",
        "quality_report.terminology.term_precision",
        0.02,
    ),
    KpiDefinition(
        "term_fuzzy_precision",
        "ratio",
        Direction.HIGHER_IS_BETTER,
        "WMT fuzzy term precision (>=80% rendering match)",
        "quality_report.terminology.fuzzy_term_precision",
        0.02,
    ),
    KpiDefinition(
        "term_recall",
        "ratio",
        Direction.HIGHER_IS_BETTER,
        "WMT term recall: canonical renderings actually used",
        "quality_report.terminology.term_recall",
        0.02,
    ),
    KpiDefinition(
        "render_skip_rate",
        "ratio",
        Direction.LOWER_IS_BETTER,
        "fail-closed render skips / total blocks (a translation could not be placed)",
        "quality_report.render_coverage.fail_closed_blocks / summary.total_blocks",
        0.01,
        ceiling=0.0,
    ),
    KpiDefinition(
        "render_preserved_rate",
        "ratio",
        Direction.LOWER_IS_BETTER,
        "intentionally preserved source elements (chrome/non-prose/policy/footer) / total blocks",
        "quality_report.render_coverage.preserved_blocks / summary.total_blocks",
        0.05,
    ),
    KpiDefinition(
        "placeholder_retention",
        "ratio",
        Direction.HIGHER_IS_BETTER,
        "1 - corrupt / masked placeholder spans (math/code/cite round-trip)",
        "quality_report.placeholder.retention_rate",
        0.01,
        floor=1.0,
    ),
    KpiDefinition(
        "pass_rate",
        "ratio",
        Direction.HIGHER_IS_BETTER,
        "completed blocks / total blocks",
        "quality_report.summary.pass_rate",
        0.01,
    ),
    KpiDefinition(
        "avg_qe",
        "score",
        Direction.HIGHER_IS_BETTER,
        "mean QE over the blocks that actually met the gate",
        "quality_report.score_metrics.avg_qe",
        0.02,
    ),
    KpiDefinition(
        "repair_rate",
        "ratio",
        Direction.LOWER_IS_BETTER,
        "repaired blocks / total blocks (higher = more repair work needed)",
        "quality_report.summary.repaired_blocks / summary.total_blocks",
        0.05,
    ),
    KpiDefinition(
        "needs_human_rate",
        "ratio",
        Direction.LOWER_IS_BETTER,
        "NEEDS_HUMAN blocks (shippable, queued for human PE) / total blocks",
        "quality_report.summary.needs_human_blocks / summary.total_blocks",
        0.01,
        ceiling=0.0,
    ),
    KpiDefinition(
        "blocked_human_rate",
        "ratio",
        Direction.LOWER_IS_BETTER,
        "BLOCKED_HUMAN blocks (MQM Critical, must-not-ship) / total blocks",
        "quality_report.summary.blocked_human_blocks / summary.total_blocks",
        0.0,
        ceiling=0.0,
    ),
    KpiDefinition(
        "formula_fidelity",
        "ratio",
        Direction.HIGHER_IS_BETTER,
        "1 - formula witness substitutions / display-formula blocks "
        "(inline math is not counted; see details.formula_blocks)",
        "quality_report.formula_witness_fallbacks / formula_blocks",
        0.02,
        floor=1.0,
    ),
    KpiDefinition(
        "fidelity_non_text_residual",
        "ratio",
        Direction.LOWER_IS_BETTER,
        "fraction of pixels OUTSIDE painted text boxes that differ from the "
        "source on a rigid render (should be ~0; advisory, 0 when not measured)",
        "quality_report.fidelity.non_text_residual",
        0.002,
    ),
    KpiDefinition(
        "rigid_painted_coverage",
        "ratio",
        Direction.HIGHER_IS_BETTER,
        "fraction of page area actually painted as translated prose on a rigid "
        "render (the overlay-coverage metric; low = most page still source text)",
        "quality_report.fidelity.painted_coverage",
        0.03,
    ),
    KpiDefinition(
        "formula_substitutions",
        "count",
        Direction.LOWER_IS_BETTER,
        "display formulas replaced by the source graphic after a witness mismatch",
        "quality_report.formula_witness_fallbacks",
        0.0,
    ),
    KpiDefinition(
        "visual_critical_rate",
        "ratio",
        Direction.LOWER_IS_BETTER,
        "CRITICAL visual-gate findings / pages (0 when no visual report)",
        "visual_report.findings[severity=critical] / route.pages",
        0.0,
        ceiling=0.0,
    ),
    KpiDefinition(
        "visual_major_rate",
        "ratio",
        Direction.LOWER_IS_BETTER,
        "MAJOR visual-gate findings / pages (0 when no visual report)",
        "visual_report.findings[severity=major] / route.pages",
        0.01,
    ),
    KpiDefinition(
        "untranslated_leak_rate",
        "ratio",
        Direction.LOWER_IS_BETTER,
        "blocks flagged untranslated / total blocks (source leaked into target)",
        "quality_report.defect_flags[untranslated:*] / summary.total_blocks",
        0.0,
        ceiling=0.0,
    ),
    KpiDefinition(
        "cost_per_1k_chars",
        "usd_per_1k_chars",
        Direction.LOWER_IS_BETTER,
        "estimated USD cost / (source chars / 1000); 0 when the run could not be "
        "priced at all — the difference is in details.estimated_cost_usd (null)",
        "quality_report.summary.estimated_cost_usd / route.chars",
        0.01,
    ),
)

KPI_BY_NAME: dict[str, KpiDefinition] = {d.name: d for d in KPI_DEFINITIONS}


class KpiSet(BaseModel):
    """The versioned KPI artifact written next to the quality report."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 0  # 0 = legacy/unknown: a missing field must mismatch a
    # real version (the compare gate then flags drift) instead of being stamped
    # with the live version and silently compared.
    job: dict[str, str] = Field(default_factory=dict)
    kpis: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)

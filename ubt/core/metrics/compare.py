"""Compare two KPI sets and gate on regressions or absolute thresholds.

The compare is direction-aware: a KPI that should go up regresses when it
*fails* to rise past its tolerance, and vice versa. A KPI the baseline does not
carry is listed in ``unknown_kpis`` and left out of the comparison, so it can
never regress; pass ``strict_names=True`` to make that absence a violation.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from ubt.core.metrics.schema import KPI_BY_NAME, Direction, KpiSet


class KpiDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    baseline: float
    candidate: float
    delta: float
    direction: str
    tolerance: float
    regressed: bool


class RegressionReport(BaseModel):
    """Outcome of comparing a candidate KPI set against a baseline."""

    model_config = ConfigDict(frozen=True)

    baseline_schema: int
    candidate_schema: int
    deltas: list[KpiDelta] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    unknown_kpis: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations and self.candidate_schema == self.baseline_schema


def compare_kpis(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    tolerances: Mapping[str, float] | None = None,
    strict_names: bool = False,
) -> RegressionReport:
    """Compare candidate KPIs against a baseline, direction- and tolerance-aware.

    ``tolerances`` overrides the per-KPI regression tolerance from the schema
    (e.g. an operator loosening a noisy metric for one gate run).

    ``strict_names`` closes the one hole a golden leaves open by construction: a
    KPI the pipeline started emitting is invisible to a baseline recorded before
    it existed, so a metric can be added and never gate anything. Off by default
    because an operational comparison routinely pits a new run against an
    artifact that predates the metric set.
    """
    overrides = tolerances or {}
    deltas: list[KpiDelta] = []
    violations: list[str] = []
    unknown: list[str] = []

    for name in sorted(set(baseline) | set(candidate)):
        if name not in baseline:
            unknown.append(name)
            if strict_names:
                violations.append(
                    f"{name}: missing from baseline (stale golden — re-record deliberately)"
                )
            continue
        if name not in candidate:
            violations.append(f"{name}: missing from candidate")
            continue
        base_value = baseline[name]
        cand_value = candidate[name]
        definition = KPI_BY_NAME.get(name)
        direction = definition.direction if definition else Direction.HIGHER_IS_BETTER
        tolerance = overrides.get(name, definition.tolerance if definition else 0.0)
        delta = cand_value - base_value
        regressed = (
            cand_value < base_value - tolerance
            if direction is Direction.HIGHER_IS_BETTER
            else cand_value > base_value + tolerance
        )
        deltas.append(
            KpiDelta(
                name=name,
                baseline=base_value,
                candidate=cand_value,
                delta=delta,
                direction=str(direction),
                tolerance=tolerance,
                regressed=regressed,
            )
        )
        if regressed:
            violations.append(
                f"{name}: {base_value:.4g} -> {cand_value:.4g} "
                f"({direction}, tolerance {tolerance:g})"
            )

    return RegressionReport(
        baseline_schema=0,
        candidate_schema=0,
        deltas=deltas,
        violations=violations,
        unknown_kpis=unknown,
    )


def compare_kpi_sets(
    baseline: KpiSet,
    candidate: KpiSet,
    *,
    tolerances: Mapping[str, float] | None = None,
    strict_names: bool = False,
) -> RegressionReport:
    """Version-checked compare of two KPI artifacts."""
    report = compare_kpis(
        baseline.kpis, candidate.kpis, tolerances=tolerances, strict_names=strict_names
    )
    violations = list(report.violations)
    if baseline.schema_version != candidate.schema_version:
        violations.append(
            f"schema_version mismatch: baseline {baseline.schema_version} "
            f"!= candidate {candidate.schema_version} (definitions changed)"
        )
    return report.model_copy(
        update={
            "baseline_schema": baseline.schema_version,
            "candidate_schema": candidate.schema_version,
            "violations": violations,
        }
    )


def check_thresholds(
    kpis: Mapping[str, float],
    *,
    floors: Mapping[str, float] | None = None,
    ceilings: Mapping[str, float] | None = None,
) -> list[str]:
    """Absolute-bound gate. Floors apply to higher-is-better KPIs, ceilings to
    lower-is-better ones; explicit ``floors``/``ceilings`` override the schema."""
    violations: list[str] = []
    for name, definition in KPI_BY_NAME.items():
        if name not in kpis:
            continue
        value = kpis[name]
        floor = floors.get(name) if floors and name in floors else definition.floor
        ceiling = ceilings.get(name) if ceilings and name in ceilings else definition.ceiling
        if floor is not None and value < floor:
            violations.append(f"{name}: {value:.4g} below floor {floor:g}")
        if ceiling is not None and value > ceiling:
            violations.append(f"{name}: {value:.4g} above ceiling {ceiling:g}")
    return violations

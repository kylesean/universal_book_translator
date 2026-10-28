"""Baseline drift computation and regression detection for corpus benchmarks."""

from __future__ import annotations

from ubt.core.benchmark.models import (
    BenchmarkDriftReport,
    CaseBenchmarkResult,
    CaseDrift,
    CorpusBenchmarkReport,
)

DEFAULT_DRIFT_TOLERANCES: dict[str, float] = {
    "delivered_ratio_drop": 0.01,
    "accounted_ratio_drop": 0.01,
    "max_overflow_increase_pt": 2.0,
    "spill_rate_increase": 0.02,
    "contract_errors_increase": 0.0,
    "missing_assets_increase": 0.0,
}


def _compare_single_case(
    cand: CaseBenchmarkResult,
    base: CaseBenchmarkResult,
    tol: dict[str, float],
) -> CaseDrift:
    """Compare one candidate case result against baseline."""
    delta_delivered_ratio = round(cand.contract.delivered_ratio - base.contract.delivered_ratio, 4)
    delta_accounted_ratio = round(cand.contract.accounted_ratio - base.contract.accounted_ratio, 4)
    delta_contract_errors = cand.contract.errors_count - base.contract.errors_count
    delta_missing_assets = cand.contract.missing_assets - base.contract.missing_assets
    delta_spill_rate = round(cand.layout.spill_rate - base.layout.spill_rate, 4)
    delta_max_overflow_pt = round(cand.layout.max_overflow_pt - base.layout.max_overflow_pt, 2)
    delta_runtime_sec = round(cand.runtime_sec - base.runtime_sec, 2)
    delta_total_tokens = cand.cost.total_tokens - base.cost.total_tokens

    delta_cost_usd: float | None = None
    if cand.cost.estimated_cost_usd is not None and base.cost.estimated_cost_usd is not None:
        delta_cost_usd = round(cand.cost.estimated_cost_usd - base.cost.estimated_cost_usd, 4)

    delta_avg_qe: float | None = None
    if cand.quality.qe_measured and base.quality.qe_measured:
        delta_avg_qe = round(cand.quality.avg_qe - base.quality.avg_qe, 4)

    reasons: list[str] = []
    is_regressed = False

    # Check for hard execution regressions
    if cand.status == "fail" and base.status == "pass":
        is_regressed = True
        reasons.append(
            f"Status regressed from PASS to FAIL ({cand.failure_reason or 'execution failure'})"
        )

    # Check contract regressions
    if delta_contract_errors > tol["contract_errors_increase"]:
        is_regressed = True
        reasons.append(
            f"Contract errors increased by {delta_contract_errors} (from {base.contract.errors_count} to {cand.contract.errors_count})"
        )

    if delta_missing_assets > tol["missing_assets_increase"]:
        is_regressed = True
        reasons.append(
            f"Missing assets increased by {delta_missing_assets} (from {base.contract.missing_assets} to {cand.contract.missing_assets})"
        )

    if delta_delivered_ratio < -tol["delivered_ratio_drop"]:
        is_regressed = True
        reasons.append(
            f"Delivered ratio dropped by {abs(delta_delivered_ratio):.1%} (from {base.contract.delivered_ratio:.1%} to {cand.contract.delivered_ratio:.1%})"
        )

    if delta_accounted_ratio < -tol["accounted_ratio_drop"]:
        is_regressed = True
        reasons.append(
            f"Accounted ratio dropped by {abs(delta_accounted_ratio):.1%} (from {base.contract.accounted_ratio:.1%} to {cand.contract.accounted_ratio:.1%})"
        )

    # Check layout regressions
    if delta_spill_rate > tol["spill_rate_increase"]:
        is_regressed = True
        reasons.append(
            f"Spill rate increased by {delta_spill_rate:.1%} (from {base.layout.spill_rate:.1%} to {cand.layout.spill_rate:.1%})"
        )

    if (
        delta_max_overflow_pt > tol["max_overflow_increase_pt"]
        and cand.layout.spill_pages_count > 0
    ):
        is_regressed = True
        reasons.append(
            f"Max layout overflow increased by {delta_max_overflow_pt:.1f}pt (from {base.layout.max_overflow_pt:.1f}pt to {cand.layout.max_overflow_pt:.1f}pt)"
        )

    if is_regressed:
        verdict = "REGRESSED"
    else:
        # Check for improvements
        is_improved = False
        if cand.status == "pass" and base.status == "fail":
            is_improved = True
            reasons.append("Status recovered from FAIL to PASS")
        if delta_contract_errors < 0:
            is_improved = True
            reasons.append(f"Contract errors decreased by {abs(delta_contract_errors)}")
        if delta_missing_assets < 0:
            is_improved = True
            reasons.append(f"Missing assets decreased by {abs(delta_missing_assets)}")
        if delta_delivered_ratio > tol["delivered_ratio_drop"]:
            is_improved = True
            reasons.append(f"Delivered ratio improved by {delta_delivered_ratio:.1%}")
        if delta_spill_rate < -tol["spill_rate_increase"]:
            is_improved = True
            reasons.append(f"Spill rate reduced by {abs(delta_spill_rate):.1%}")

        verdict = "IMPROVED" if is_improved else "STABLE"
        if not reasons:
            reasons.append("Metrics within normal variation thresholds")

    return CaseDrift(
        case_id=cand.id,
        verdict=verdict,
        delta_delivered_ratio=delta_delivered_ratio,
        delta_accounted_ratio=delta_accounted_ratio,
        delta_contract_errors=delta_contract_errors,
        delta_missing_assets=delta_missing_assets,
        delta_spill_rate=delta_spill_rate,
        delta_max_overflow_pt=delta_max_overflow_pt,
        delta_runtime_sec=delta_runtime_sec,
        delta_total_tokens=delta_total_tokens,
        delta_cost_usd=delta_cost_usd,
        delta_avg_qe=delta_avg_qe,
        reasons=reasons,
    )


def compute_benchmark_drift(
    baseline: CorpusBenchmarkReport,
    candidate: CorpusBenchmarkReport,
    tolerances: dict[str, float] | None = None,
) -> BenchmarkDriftReport:
    """Analyze metric drift and regressions of candidate report against baseline."""
    tol = dict(DEFAULT_DRIFT_TOLERANCES)
    if tolerances:
        tol.update(tolerances)

    base_cases = {c.id: c for c in baseline.cases}
    cand_cases = {c.id: c for c in candidate.cases}

    case_drifts: list[CaseDrift] = []
    regressed_count = 0
    improved_count = 0
    stable_count = 0
    new_count = 0
    missing_count = 0

    for case_id, cand in cand_cases.items():
        if case_id in base_cases:
            drift = _compare_single_case(cand, base_cases[case_id], tol)
            case_drifts.append(drift)
            if drift.verdict == "REGRESSED":
                regressed_count += 1
            elif drift.verdict == "IMPROVED":
                improved_count += 1
            else:
                stable_count += 1
        else:
            new_count += 1
            case_drifts.append(
                CaseDrift(
                    case_id=case_id,
                    verdict="NEW",
                    reasons=["Case not present in baseline report"],
                )
            )

    for case_id in base_cases:
        if case_id not in cand_cases:
            missing_count += 1
            case_drifts.append(
                CaseDrift(
                    case_id=case_id,
                    verdict="MISSING",
                    reasons=["Case present in baseline but missing from candidate run"],
                )
            )

    summary_deltas: dict[str, float] = {
        "delta_avg_delivered_ratio": round(
            candidate.avg_delivered_ratio - baseline.avg_delivered_ratio, 4
        ),
        "delta_avg_accounted_ratio": round(
            candidate.avg_accounted_ratio - baseline.avg_accounted_ratio, 4
        ),
        "delta_avg_spill_rate": round(candidate.avg_spill_rate - baseline.avg_spill_rate, 4),
        "delta_contract_errors": float(
            candidate.total_contract_errors - baseline.total_contract_errors
        ),
        "delta_missing_assets": float(
            candidate.total_missing_assets - baseline.total_missing_assets
        ),
        "delta_runtime_sec": round(candidate.total_runtime_sec - baseline.total_runtime_sec, 2),
    }
    if candidate.total_cost_usd is not None and baseline.total_cost_usd is not None:
        summary_deltas["delta_cost_usd"] = round(
            candidate.total_cost_usd - baseline.total_cost_usd, 4
        )

    return BenchmarkDriftReport(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        total_cases=len(cand_cases),
        regressed_count=regressed_count,
        improved_count=improved_count,
        stable_count=stable_count,
        new_count=new_count,
        missing_count=missing_count,
        drift_passed=(regressed_count == 0),
        case_drifts=case_drifts,
        summary_deltas=summary_deltas,
    )

"""Data models for corpus automated benchmark evaluation and drift reporting."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkMode(StrEnum):
    """Execution mode for the corpus benchmark runner."""

    DRY_RUN = "dry-run"
    LIVE = "live"
    EVAL_ONLY = "eval-only"


class ContractBenchmarkMetrics(BaseModel):
    """Reconciled delivery contract metrics for a single corpus case."""

    model_config = ConfigDict(frozen=True)

    total_text: int = 0
    delivered_text: int = 0
    verbatim_text: int = 0
    source_kept_text: int = 0
    skipped_text: int = 0
    delivered_ratio: float = 0.0
    accounted_ratio: float = 0.0
    missing_assets: int = 0
    violations_count: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    contract_passed: bool = True
    expectation_failures: list[str] = Field(default_factory=list)


class LayoutBenchmarkMetrics(BaseModel):
    """Physical PDF layout spill and typography diagnostics."""

    model_config = ConfigDict(frozen=True)

    total_pages: int = 0
    spill_pages_count: int = 0
    spill_rate: float = 0.0
    max_overflow_pt: float = 0.0
    spill_count: int = 0
    type_collapse_count: int = 0
    compiler_warnings_count: int = 0
    layout_passed: bool = True


class QualityBenchmarkMetrics(BaseModel):
    """Translation quality, MTQE score distribution and placeholder metrics."""

    model_config = ConfigDict(frozen=True)

    avg_qe: float = 0.0
    min_qe: float = 0.0
    p50_qe: float = 0.0
    p90_qe: float = 0.0
    scored_blocks: int = 0
    qe_measured: bool = False
    retention_rate: float = 1.0
    term_precision: float = 0.0
    term_recall: float = 0.0
    pass_rate: float = 1.0


class CostBenchmarkMetrics(BaseModel):
    """Token usage, cache hit efficiency, and financial cost estimates."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cache_hit_rate: float = 0.0
    estimated_cost_usd: float | None = None


class CaseBenchmarkResult(BaseModel):
    """Comprehensive benchmark evaluation result for one corpus case."""

    model_config = ConfigDict(frozen=True)

    id: str
    description: str = ""
    document: str | None = None
    artifact_path: str | None = None
    status: str = "pass"  # "pass" | "fail" | "skip"
    failure_reason: str | None = None
    runtime_sec: float = 0.0
    contract: ContractBenchmarkMetrics = Field(default_factory=ContractBenchmarkMetrics)
    layout: LayoutBenchmarkMetrics = Field(default_factory=LayoutBenchmarkMetrics)
    quality: QualityBenchmarkMetrics = Field(default_factory=QualityBenchmarkMetrics)
    cost: CostBenchmarkMetrics = Field(default_factory=CostBenchmarkMetrics)

    @property
    def passed(self) -> bool:
        """A case passes if status is pass, contract passed, and layout passed."""
        return self.status == "pass" and self.contract.contract_passed and self.layout.layout_passed


class CorpusBenchmarkReport(BaseModel):
    """Complete aggregated benchmark report across all evaluated cases."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    timestamp: str
    corpus_dir: str
    mode: str
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    skipped_cases: int = 0
    overall_passed: bool = True
    total_runtime_sec: float = 0.0
    avg_delivered_ratio: float = 0.0
    avg_accounted_ratio: float = 0.0
    total_contract_errors: int = 0
    total_missing_assets: int = 0
    avg_spill_rate: float = 0.0
    total_spill_pages: int = 0
    total_type_collapse: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_tokens: int = 0
    overall_cache_hit_rate: float = 0.0
    total_cost_usd: float | None = None
    cases: list[CaseBenchmarkResult] = Field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to indented JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str | bytes) -> CorpusBenchmarkReport:
        """Deserialize report from JSON."""
        return cls.model_validate_json(data)

    def save(self, path: Path | str) -> None:
        """Write report atomically to specified path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")


class CaseDrift(BaseModel):
    """Metric drift between baseline and candidate for a single case."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    verdict: str  # "REGRESSED" | "IMPROVED" | "STABLE" | "NEW" | "MISSING"
    delta_delivered_ratio: float = 0.0
    delta_accounted_ratio: float = 0.0
    delta_contract_errors: int = 0
    delta_missing_assets: int = 0
    delta_spill_rate: float = 0.0
    delta_max_overflow_pt: float = 0.0
    delta_runtime_sec: float = 0.0
    delta_total_tokens: int = 0
    delta_cost_usd: float | None = None
    delta_avg_qe: float | None = None
    reasons: list[str] = Field(default_factory=list)

    @property
    def is_regression(self) -> bool:
        return self.verdict == "REGRESSED"


class BenchmarkDriftReport(BaseModel):
    """Comparative analysis of candidate benchmark run against baseline."""

    model_config = ConfigDict(frozen=True)

    baseline_run_id: str | None = None
    candidate_run_id: str = ""
    total_cases: int = 0
    regressed_count: int = 0
    improved_count: int = 0
    stable_count: int = 0
    new_count: int = 0
    missing_count: int = 0
    drift_passed: bool = True
    case_drifts: list[CaseDrift] = Field(default_factory=list)
    summary_deltas: dict[str, float] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        """Serialize drift report to JSON string."""
        return self.model_dump_json(indent=indent)

    def save(self, path: Path | str) -> None:
        """Save drift report to file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

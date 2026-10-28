"""Corpus automated benchmark execution runner and metrics collector."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ubt.core.benchmark.models import (
    BenchmarkMode,
    CaseBenchmarkResult,
    ContractBenchmarkMetrics,
    CorpusBenchmarkReport,
    CostBenchmarkMetrics,
    LayoutBenchmarkMetrics,
    QualityBenchmarkMetrics,
)
from ubt.core.content.verify import (
    CorpusCase,
    contract_path_for_artifact,
    evaluate_expectations,
    load_contract_file,
    load_corpus,
)
from ubt.core.job_options import sidecar_path
from ubt.render.typst.layout_diagnostics import PageLayoutSpec, diagnose_pdf_layout

logger = logging.getLogger(__name__)

ARTIFACT_FILE_PATTERNS = ("{id}_mono.pdf", "{id}.pdf", "{id}_mono_v3.pdf")


def default_run_cli_translation(document: Path, case_id: str, mode: str, out_path: Path) -> None:
    """Default runner that invokes `ubt translate` via subprocess."""
    env = dict(os.environ)
    env["UBT_OUTPUT_DIR"] = str(out_path.parent)
    cmd = [
        sys.executable,
        "-m",
        "ubt",
        "translate",
        str(document),
        "--fresh",
        "--yes",
        "-o",
        str(out_path),
    ]
    if mode == BenchmarkMode.DRY_RUN.value:
        cmd.append("--dry-run")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600, check=False)
    if proc.returncode != 0:
        err_msg = (proc.stderr or proc.stdout or "").strip()
        tail = err_msg.splitlines()[-4:]
        raise RuntimeError(f"Translation failed (exit {proc.returncode}): {' / '.join(tail)}")


def select_layout_spec_for_case(case_id: str, tolerance_pt: float = 5.0) -> PageLayoutSpec:
    """Resolve appropriate layout specification based on case identifier."""
    norm = case_id.lower()
    if "twocol" in norm or "dual" in norm:
        return PageLayoutSpec.from_name("academic_dual", tolerance_pt=tolerance_pt)
    if "paper" in norm or "academic" in norm:
        return PageLayoutSpec.from_name("academic_single", tolerance_pt=tolerance_pt)
    return PageLayoutSpec.from_name("book", tolerance_pt=tolerance_pt)


def extract_quality_metrics(report_path: Path) -> QualityBenchmarkMetrics:
    """Parse metrics from a *_quality_report.json sidecar."""
    if not report_path.exists():
        return QualityBenchmarkMetrics()
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        scores = data.get("scores", {})
        retention = data.get("retention", {})
        terminology = data.get("terminology", {})
        return QualityBenchmarkMetrics(
            avg_qe=float(scores.get("avg_qe", 0.0) or 0.0),
            min_qe=float(scores.get("min_qe", 0.0) or 0.0),
            p50_qe=float(scores.get("p50_qe", 0.0) or 0.0),
            p90_qe=float(scores.get("p90_qe", 0.0) or 0.0),
            scored_blocks=int(scores.get("scored_blocks", 0) or 0),
            qe_measured=bool(scores.get("qe_measured", False)),
            retention_rate=float(retention.get("retention_rate", 1.0) or 1.0),
            term_precision=float(terminology.get("term_precision", 0.0) or 0.0),
            term_recall=float(terminology.get("term_recall", 0.0) or 0.0),
            pass_rate=float(summary.get("pass_rate", 1.0) or 1.0),
        )
    except Exception as exc:
        logger.warning("Could not parse quality report %s: %s", report_path, exc)
        return QualityBenchmarkMetrics()


def extract_cost_metrics(
    report_path: Path | None = None,
    usage_dict: dict[str, dict[str, int]] | None = None,
) -> CostBenchmarkMetrics:
    """Extract cost and token consumption metrics from quality report or usage dictionary."""
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    total_tokens = 0
    cache_hit_rate = 0.0
    estimated_cost_usd: float | None = None

    if usage_dict:
        for model_totals in usage_dict.values():
            prompt_tokens += int(model_totals.get("prompt_tokens", 0) or 0)
            completion_tokens += int(model_totals.get("completion_tokens", 0) or 0)
            cached_tokens += int(model_totals.get("cached_tokens", 0) or 0)
            total_tokens += int(model_totals.get("total_tokens", 0) or 0)
        if prompt_tokens > 0:
            cache_hit_rate = round(min(cached_tokens / prompt_tokens, 1.0), 4)

    if report_path and report_path.exists():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            summary = data.get("summary", {})
            if estimated_cost_usd is None and "estimated_cost_usd" in summary:
                estimated_cost_usd = summary["estimated_cost_usd"]
            if cache_hit_rate == 0.0 and "cache_hit_rate" in summary:
                cache_hit_rate = float(summary["cache_hit_rate"] or 0.0)
        except Exception:
            pass

    return CostBenchmarkMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        total_tokens=total_tokens,
        cache_hit_rate=cache_hit_rate,
        estimated_cost_usd=estimated_cost_usd,
    )


class BenchmarkRunner:
    """Automated benchmark runner evaluating corpus cases across contract, layout, and cost dimensions."""

    def __init__(
        self,
        corpus_dir: Path | str = "corpus",
        mode: str | BenchmarkMode = BenchmarkMode.DRY_RUN,
        artifacts_dir: Path | str | None = None,
        tolerance_pt: float = 5.0,
        runner_fn: Callable[[Path, str, str, Path], Any] | None = None,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.mode = mode.value if isinstance(mode, BenchmarkMode) else str(mode)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.tolerance_pt = float(tolerance_pt)
        self.runner_fn = runner_fn or default_run_cli_translation

    def find_artifact_for_case(self, case: CorpusCase) -> Path | None:
        """Locate existing delivered artifact for a given case."""
        search_dirs: list[Path] = []
        if self.artifacts_dir and self.artifacts_dir.exists():
            search_dirs.append(self.artifacts_dir)
        search_dirs.append(self.corpus_dir)

        if case.artifact:
            cand = Path(case.artifact)
            if cand.is_absolute() and cand.exists():
                return cand
            for sdir in search_dirs:
                if (sdir / cand).exists():
                    return sdir / cand

        for sdir in search_dirs:
            for pattern in ARTIFACT_FILE_PATTERNS:
                cand = sdir / pattern.format(id=case.id)
                if cand.exists():
                    return cand

        if case.document:
            stem = Path(case.document).stem
            for sdir in search_dirs:
                for pattern in ARTIFACT_FILE_PATTERNS:
                    cand = sdir / pattern.format(id=stem)
                    if cand.exists():
                        return cand
        return None

    def evaluate_case(self, case: CorpusCase) -> CaseBenchmarkResult:
        """Run or evaluate a single corpus case."""
        t0 = time.perf_counter()
        artifact_path: Path | None = None
        temp_work_dir: Path | None = None

        try:
            if self.mode == BenchmarkMode.EVAL_ONLY.value:
                artifact_path = self.find_artifact_for_case(case)
                if not artifact_path:
                    return CaseBenchmarkResult(
                        id=case.id,
                        description=case.description,
                        document=case.document,
                        status="skip",
                        failure_reason="No artifact found in artifacts-dir or corpus",
                        runtime_sec=0.0,
                    )
            else:
                # Mode is dry-run or live: execute translation
                if not case.document:
                    return CaseBenchmarkResult(
                        id=case.id,
                        description=case.description,
                        document=None,
                        status="skip",
                        failure_reason="Case has no document specified to run",
                        runtime_sec=0.0,
                    )
                doc_path = Path(case.document)
                if not doc_path.is_absolute():
                    doc_path = self.corpus_dir / doc_path
                if not doc_path.exists():
                    return CaseBenchmarkResult(
                        id=case.id,
                        description=case.description,
                        document=str(doc_path),
                        status="skip",
                        failure_reason=f"Document file does not exist: {doc_path}",
                        runtime_sec=0.0,
                    )

                if self.artifacts_dir:
                    self.artifacts_dir.mkdir(parents=True, exist_ok=True)
                    artifact_path = self.artifacts_dir / f"{case.id}_mono.pdf"
                else:
                    temp_work_dir = Path(tempfile.mkdtemp(prefix=f"ubt-bm-{case.id}-"))
                    artifact_path = temp_work_dir / f"{case.id}_mono.pdf"

                self.runner_fn(doc_path, case.id, self.mode, artifact_path)

            runtime_sec = round(time.perf_counter() - t0, 3)

            # 1. Reconciliation & Delivery Contract
            contract_sidecar = contract_path_for_artifact(artifact_path)
            if not contract_sidecar.exists():
                return CaseBenchmarkResult(
                    id=case.id,
                    description=case.description,
                    document=case.document,
                    artifact_path=str(artifact_path),
                    status="fail",
                    failure_reason=f"No contract sidecar found at {contract_sidecar}",
                    runtime_sec=runtime_sec,
                )

            reconciliation = load_contract_file(contract_sidecar)
            failures = evaluate_expectations(reconciliation, case.expect)
            contract_passed = len(failures) == 0

            total_t = reconciliation.total_text
            deliv_t = reconciliation.delivered_text
            verb_t = reconciliation.verbatim_text
            deliv_ratio = round(deliv_t / total_t, 4) if total_t > 0 else 1.0
            acc_ratio = round((deliv_t + verb_t) / total_t, 4) if total_t > 0 else 1.0

            contract_metrics = ContractBenchmarkMetrics(
                total_text=total_t,
                delivered_text=deliv_t,
                verbatim_text=verb_t,
                source_kept_text=reconciliation.source_kept_text,
                skipped_text=reconciliation.skipped_text,
                delivered_ratio=deliv_ratio,
                accounted_ratio=acc_ratio,
                missing_assets=reconciliation.missing_assets,
                violations_count=len(reconciliation.violations),
                errors_count=len(reconciliation.errors),
                warnings_count=len(reconciliation.warnings),
                contract_passed=contract_passed,
                expectation_failures=failures,
            )

            # 2. Physical Layout Diagnostics
            layout_spec = select_layout_spec_for_case(case.id, tolerance_pt=self.tolerance_pt)
            layout_rep = diagnose_pdf_layout(artifact_path, spec=layout_spec)

            total_pages = layout_rep.total_pages
            spill_pages_count = len(layout_rep.spill_pages)
            spill_rate = round(spill_pages_count / max(1, total_pages), 4)
            max_overflow = round(
                max((a.delta_pt for a in layout_rep.all_anomalies), default=0.0), 2
            )
            type_collapses = sum(
                1 for a in layout_rep.all_anomalies if a.anomaly_type.value == "type_collapse"
            )

            layout_metrics = LayoutBenchmarkMetrics(
                total_pages=total_pages,
                spill_pages_count=spill_pages_count,
                spill_rate=spill_rate,
                max_overflow_pt=max_overflow,
                spill_count=len(layout_rep.all_anomalies),
                type_collapse_count=type_collapses,
                compiler_warnings_count=len(layout_rep.compiler_warnings),
                layout_passed=layout_rep.passed,
            )

            # 3. Quality & Terminology Metrics
            quality_sidecar = sidecar_path(artifact_path, "quality_report.json")
            quality_metrics = extract_quality_metrics(quality_sidecar)

            # 4. Cost and Token Metrics
            cost_metrics = extract_cost_metrics(report_path=quality_sidecar)

            overall_status = "pass" if (contract_passed and layout_rep.passed) else "fail"
            failure_reason = None
            if not contract_passed:
                failure_reason = f"Contract expectation failure: {'; '.join(failures)}"
            elif not layout_rep.passed:
                failure_reason = (
                    f"Layout diagnostic failure: {layout_rep.error_count} error-level spills"
                )

            return CaseBenchmarkResult(
                id=case.id,
                description=case.description,
                document=case.document,
                artifact_path=str(artifact_path),
                status=overall_status,
                failure_reason=failure_reason,
                runtime_sec=runtime_sec,
                contract=contract_metrics,
                layout=layout_metrics,
                quality=quality_metrics,
                cost=cost_metrics,
            )

        except Exception as exc:
            logger.error("Exception during case evaluation %s: %s", case.id, exc, exc_info=True)
            return CaseBenchmarkResult(
                id=case.id,
                description=case.description,
                document=case.document,
                status="fail",
                failure_reason=str(exc),
                runtime_sec=round(time.perf_counter() - t0, 3),
            )
        finally:
            if temp_work_dir and temp_work_dir.exists():
                shutil.rmtree(temp_work_dir, ignore_errors=True)

    def run_all(self, case_ids: Sequence[str] | None = None) -> CorpusBenchmarkReport:
        """Execute benchmark across all or filtered corpus cases and return comprehensive report."""
        cases = load_corpus(self.corpus_dir)
        if case_ids:
            wanted = set(case_ids)
            cases = [c for c in cases if c.id in wanted]

        run_id = f"bench_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now(UTC).isoformat()

        results: list[CaseBenchmarkResult] = []
        for case in cases:
            res = self.evaluate_case(case)
            results.append(res)

        total_cases = len(results)
        passed_cases = sum(1 for r in results if r.passed)
        failed_cases = sum(1 for r in results if r.status == "fail")
        skipped_cases = sum(1 for r in results if r.status == "skip")
        overall_passed = failed_cases == 0 and total_cases > 0

        total_runtime_sec = round(sum(r.runtime_sec for r in results), 2)
        evaluated_cases = [r for r in results if r.status != "skip"]
        eval_count = len(evaluated_cases) or 1

        avg_deliv = round(sum(r.contract.delivered_ratio for r in evaluated_cases) / eval_count, 4)
        avg_acc = round(sum(r.contract.accounted_ratio for r in evaluated_cases) / eval_count, 4)
        total_errs = sum(r.contract.errors_count for r in evaluated_cases)
        total_missing = sum(r.contract.missing_assets for r in evaluated_cases)

        avg_spill = round(sum(r.layout.spill_rate for r in evaluated_cases) / eval_count, 4)
        total_spill_pages = sum(r.layout.spill_pages_count for r in evaluated_cases)
        total_type_collapse = sum(r.layout.type_collapse_count for r in evaluated_cases)

        tot_prompt = sum(r.cost.prompt_tokens for r in evaluated_cases)
        tot_completion = sum(r.cost.completion_tokens for r in evaluated_cases)
        tot_cached = sum(r.cost.cached_tokens for r in evaluated_cases)
        tot_tokens = sum(r.cost.total_tokens for r in evaluated_cases)
        overall_cache_hit = round(min(tot_cached / tot_prompt, 1.0), 4) if tot_prompt > 0 else 0.0

        priced = [
            r.cost.estimated_cost_usd
            for r in evaluated_cases
            if r.cost.estimated_cost_usd is not None
        ]
        total_cost_usd = round(sum(priced), 4) if priced else None

        return CorpusBenchmarkReport(
            run_id=run_id,
            timestamp=timestamp,
            corpus_dir=str(self.corpus_dir),
            mode=self.mode,
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            skipped_cases=skipped_cases,
            overall_passed=overall_passed,
            total_runtime_sec=total_runtime_sec,
            avg_delivered_ratio=avg_deliv,
            avg_accounted_ratio=avg_acc,
            total_contract_errors=total_errs,
            total_missing_assets=total_missing,
            avg_spill_rate=avg_spill,
            total_spill_pages=total_spill_pages,
            total_type_collapse=total_type_collapse,
            total_prompt_tokens=tot_prompt,
            total_completion_tokens=tot_completion,
            total_cached_tokens=tot_cached,
            total_tokens=tot_tokens,
            overall_cache_hit_rate=overall_cache_hit,
            total_cost_usd=total_cost_usd,
            cases=results,
        )

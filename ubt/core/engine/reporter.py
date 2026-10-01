"""Standardized book translation quality report generator."""

import ast
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import BlockStatus, BlockType, BookManifest
from ubt.core.qe.defect_taxonomy import INTENTIONAL_PRESERVED_SKIP_PREFIXES

# The QE-scored population policy (placeholder/skip exclusion) lives in
# ubt.core.qe.score_policy, which both this report and the ledger's job stats
# import so their averages cannot drift apart again.
from ubt.core.qe.score_policy import qe_scored_values
from ubt.segment.placeholders import default_placeholder_engine

# The placeholder mask order has exactly one owner (``PlaceholderEngine``); the
# report re-masks through it rather than restating the order, so a change to the
# order cannot leave the report measuring a pipeline that no longer exists.
_PLACEHOLDER_ENGINE = default_placeholder_engine()

#: Bump when QualityReport's serialized shape changes so downstream consumers
#: can validate schema compatibility.
QUALITY_REPORT_SCHEMA_VERSION: int = 2


class ReportSummary(BaseModel):
    """Overall volume, completion rate, and budget metrics."""

    model_config = ConfigDict(frozen=True)

    total_blocks: int
    completed_blocks: int
    repaired_blocks: int
    failed_blocks: int
    pass_rate: float
    # ``None`` means "not priced", not "free": the price table deliberately
    # covers only the models the author benchmarks, so most operators' models
    # are absent, and a 0.0 here would read as a $0 delivery on a paid run.
    estimated_cost_usd: float | None
    # Share of prompt tokens served from the provider cache —
    # verifies the static-prefix TCO lever. 0.0 = unmeasured.
    cache_hit_rate: float = 0.0
    # Human PE (HITL) queue population. Blocked blocks are
    # quarantined Critical findings that must never ship as machine output.
    needs_human_blocks: int = 0
    blocked_human_blocks: int = 0


class ReportScoreMetrics(BaseModel):
    """Statistical distribution of MTQE scores across the QE-scored blocks.

    under the default heuristic engine these are **discrete
    defect-class proxies** (only 12 reachable values; 0.92 means "no
    deterministic invariant failed", NOT a quality estimate). Averages and
    percentiles of such values are routing statistics, not calibrated
    quality measurements.

    The population is *QE-scored blocks only*. Verbatim skips and TM exact
    hits carry a 1.0 placeholder instead of a score and are excluded, so
    ``avg_qe`` never reads as covering every block; ``scored_blocks``
    states the population size it was computed over.
    """

    model_config = ConfigDict(frozen=True)

    avg_qe: float
    min_qe: float
    max_qe: float
    p10_qe: float
    p50_qe: float
    p90_qe: float
    bottom_15_avg_qe: float
    #: Number of blocks the metrics above were computed over (0 = nothing was
    #: scored; every block was a skip/TM hit or never reached the gate).
    scored_blocks: int = 0


class ReportPlaceholderMetrics(BaseModel):
    """Math/code placeholder retention: masked spans vs corrupt restores."""

    model_config = ConfigDict(frozen=True)

    masked_spans: int  # math spans masked pre-draft (deterministically recomputed)
    corrupt_spans: int  # missing + mismatched + mutated restores from flags
    retention_rate: float  # 1 - corrupt/masked; 1.0 when nothing was masked
    masked_blocks: int  # blocks carrying at least one masked span
    corrupt_blocks: int  # blocks carrying a math_token_corrupt flag


class ReportRepairBreakdown(BaseModel):
    """Breakdown of repair iterations and circuit breaker activations."""

    model_config = ConfigDict(frozen=True)

    direct_pass_count: int  # 0 repair rounds (fast-pass or initial draft passed)
    round_1_repaired_count: int  # fixed in round 1
    round_2_repaired_count: int  # fixed in round 2
    exhausted_count: int  # reached max rounds


class ReportRenderCoverage(BaseModel):
    """How much of the pipeline completed, and how much of that was delivered.

    ``render_coverage`` is the *translation-completion* ratio
    (``completed / total``), not a pixel/placement measure: for reflow engines
    (publication/Typst) nothing is left in place so it is also the delivery
    ratio, while the overlay engine (rigid) can keep source text on the page.
    Those delivery gaps are carried by ``fail_closed_blocks`` / ``skip_families``
    (and, for rigid, by the separate per-page render-visibility report) — never
    folded into this ratio, so a caller that wants "was every translation
    placed" must read ``fail_closed_blocks`` too.
    """

    model_config = ConfigDict(frozen=True)

    rendered_blocks: int  # completed blocks (== summary.completed); see fail_closed_blocks
    skipped_blocks: int  # render_skip:* occurrences (fail-closed + intentional)
    fail_closed_blocks: int = 0  # source left visible; a translation could not be placed
    preserved_blocks: int = 0  # chrome/non-prose/policy/footer kept in place by design
    render_coverage: float  # completed / total; 1.0 when every block completed
    skip_families: dict[str, int] = Field(default_factory=dict)


class ReportFidelity(BaseModel):
    """Pixel-level fidelity of a ``rigid`` render (advisory, never blocking).

    ``non_text_residual`` is the fraction of pixels *outside* the painted text
    boxes that differ from the source — it should be ~0 because the rigid route
    keeps every non-text element untouched. ``painted_coverage`` is the fraction
    of page area that was actually painted prose (the inverse of the "most of
    the page still shows source text" failure). ``pages_measured`` is 0 when the
    probe could not run (no rigid render, rasterizer unavailable), in which case
    both ratios are meaningless defaults.
    """

    model_config = ConfigDict(frozen=True)

    non_text_residual: float = 0.0
    painted_coverage: float = 0.0
    pages_measured: int = 0


class ReportRouteInfo(BaseModel):
    """Unified-entry routing decision behind this job (router_mode.decide)."""

    model_config = ConfigDict(frozen=True)

    mode: str  # short | long
    pages: int = 0
    chars: int = 0
    reason: str = ""
    formula_mode: str = ""


class ReportTerminologyMetrics(BaseModel):
    """WMT-style terminology precision/recall over finalized blocks.

    Computed deterministically at export (zero LLM cost): a glossary entry is
    counted only where its source actually occurs. ``TF`` allows a fuzzy (>=80%)
    rendering match. All-zero with ``terms_expected == 0`` means no glossary
    term was exercised by this job.
    """

    model_config = ConfigDict(frozen=True)

    terms_expected: int = 0
    terms_rendered: int = 0
    term_precision: float = 0.0
    fuzzy_term_precision: float = 0.0
    term_recall: float = 0.0


class ReportEntityConsistency(BaseModel):
    """Document-level terminology drift, aggregated per source term.

    Derived from the same deterministic scan as ``ReportTerminologyMetrics``
    (zero extra cost): a drift occurrence is a block where the term's source
    surface occurs but its canonical rendering is absent. Terms without a
    canonical rendering (untranslated mined entries) are not auditable here.
    """

    model_config = ConfigDict(frozen=True)

    terms_audited: int = 0
    terms_with_drift: int = 0
    top_drifted: list[dict[str, Any]] = Field(default_factory=list)


class QualityReport(BaseModel):
    """Complete publication-grade translation quality audit report."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    doc_id: str
    book_title: str
    source_path: str
    output_path: str
    target_lang: str
    #: Schema version of this serialized report (see
    #: :data:`QUALITY_REPORT_SCHEMA_VERSION`). Consumers diffing report
    #: shapes across builds must read this instead of sniffing keys.
    schema_version: int = QUALITY_REPORT_SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    summary: ReportSummary
    score_metrics: ReportScoreMetrics
    repair_breakdown: ReportRepairBreakdown
    terminology: ReportTerminologyMetrics = Field(default_factory=ReportTerminologyMetrics)
    entity_consistency: ReportEntityConsistency = Field(default_factory=ReportEntityConsistency)
    placeholder: ReportPlaceholderMetrics = Field(
        default_factory=lambda: ReportPlaceholderMetrics(
            masked_spans=0,
            corrupt_spans=0,
            retention_rate=1.0,
            masked_blocks=0,
            corrupt_blocks=0,
        )
    )
    defect_flags: dict[str, int] = Field(default_factory=dict)
    #: Spans the deterministic glossary enforcer rewrote before the terminology
    #: metrics ran. The report measures the *delivered* (enforced) text while the
    #: translation memory keeps the unenforced draft, so a reader can discount
    #: this many mechanically-corrected spans when comparing the two.
    enforced_spans: int = 0
    # Render delivery audit (fail-closed overlay skips vs pipeline completion).
    render_coverage: ReportRenderCoverage = Field(
        default_factory=lambda: ReportRenderCoverage(
            rendered_blocks=0,
            skipped_blocks=0,
            fail_closed_blocks=0,
            preserved_blocks=0,
            render_coverage=1.0,
        )
    )
    # Rigid-render pixel fidelity (advisory; 0 pages when not a rigid render).
    fidelity: ReportFidelity = Field(default_factory=ReportFidelity)
    # Unified-entry routing decision (router_mode.decide), when the pipeline
    # recorded one. None for pre-router jobs and non-PDF inputs.
    route: ReportRouteInfo | None = None
    # Bilingual render-mode advisory (policy/bilingual_advisor), when the
    # pipeline computed one. Carries the requested/effective modes, the
    # tier, the per-mode ranking and the fired-rule explanations.
    mode_advisory: dict[str, Any] | None = None
    # Pipeline runtime configuration snapshot (draft_model, repair_model, etc.)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    # Typst self-healing lines commented OUT of the delivered
    # PDF (translated content silently gone). Empty = no content removal.
    syntax_fallbacks: list[str] = Field(default_factory=list)
    # Display formulas whose converted Typst differed structurally
    # from the source equation and were replaced by the source graphic
    # (lossless substitution, must stay visible in the audit).
    formula_witness_fallbacks: list[str] = Field(default_factory=list)
    # Display-formula blocks delivered: the denominator of the
    # metrics-layer ``formula_fidelity`` KPI. Inline math is not counted;
    # 0 means this job carried no display formula.
    formula_blocks: int = 0
    # Exact Typst compiler version that built the artifact
    # (None = unmeasured, e.g. compiler-absent Markdown-companion jobs).
    typst_version: str | None = None
    # Engine the QE subprocess reported for this job's scores
    # ("neural" | "heuristic_fallback"; None = heuristic/LLM runner or
    # nothing scored). A fallback label means every mtqe_score in this report
    # is a discrete defect band, not a calibrated CometKiwi measurement.
    qe_score_source: str | None = None
    # Paid LLM-as-Judge ROI counters (zero when the explicit judge opt-in was
    # off or no suspicious blocks reached the tiered runner).
    qe_judge_calls: int = 0
    qe_judge_errors: int = 0
    # Delivery suitability audit flag (e.g. UNSUITABLE_FOR_DELIVERY when math-dense
    # document was forced through overlay engine). None = normal delivery.
    delivery_status: str | None = None
    delivery_warning: str | None = None
    #: Content/asset ledger reconciliation (ubt.core.content.contract). None for
    #: jobs that predate the contract layer. The standalone ``*.contract.json``
    #: holds the same structure; this is the embedded copy for one-stop audit.
    delivery_contract: dict[str, Any] | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avg_qe(self) -> float:
        """Top-level alias of ``score_metrics.avg_qe``.

        Read-only convenience: consumers used to have to know the nested path
        just to read the book's average score. It mirrors the authoritative
        nested field on every serialization, so the alias and the value can
        never drift; it is not accepted as constructor input.
        """
        return self.score_metrics.avg_qe


def _percentile(values: list[float], p: float) -> float:
    """Calculate the p-th percentile (0.0 to 1.0) of a sorted list of floats."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return round(d0 + d1, 4)


def build_quality_report(
    ledger: SQLiteJobLedger,
    job_id: str,
    manifest: BookManifest,
    output_path: Path | str,
    token_cost_usd: float | None = None,
    cache_hit_rate: float | None = None,
    terminology_metrics: ReportTerminologyMetrics | None = None,
    entity_consistency: ReportEntityConsistency | None = None,
    enforced_spans: int = 0,
    delivery_contract: dict[str, Any] | None = None,
    mode_advisory: dict[str, Any] | None = None,
) -> QualityReport:
    """Analyze all ledger blocks for a job and construct a comprehensive QualityReport.

    cost comes from the router's real per-model token accounting
    (``token_cost_usd``). When the caller cannot supply it — which is the normal
    case, since the price table deliberately covers only the models the author
    benchmarks — the report carries ``None`` and renders "unknown": a fabricated
    0.0 would read as a $0 delivery on a paid run.

    ``cache_hit_rate`` comes from the router's cache accounting and
    is surfaced in the summary + markdown table; 0.0 = unmeasured.
    """
    all_blocks = ledger.get_all_blocks(job_id)
    stats = ledger.get_job_stats(job_id)

    total = int(stats.get("total", 0))
    completed = int(stats.get("completed", 0))
    repaired = int(stats.get("repaired", 0))
    failed = int(stats.get("failed", 0))
    needs_human = int(stats.get("needs_human", 0))
    blocked_human = int(stats.get("blocked_human", 0))
    pass_rate = round(completed / total, 4) if total > 0 else 1.0
    estimated_cost = round(token_cost_usd, 5) if token_cost_usd is not None else None
    # Display-formula blocks, the denominator of the metrics-layer
    # formula_fidelity KPI (see ubt/core/metrics/schema.py).
    formula_blocks = sum(1 for b in all_blocks if b.block_type == BlockType.FORMULA)

    # MTQE scores analysis: only blocks that actually met the QE gate; the
    # 1.0 placeholders on skips/TM hits are not scores (shared policy in
    # ubt.core.qe.score_policy, the same predicate the ledger's status-side
    # average must apply).
    scored: list[float] = qe_scored_values(all_blocks)
    scored_blocks = len(scored)
    if scored:
        avg_qe = round(sum(scored) / len(scored), 4)
        min_qe = round(scored[0], 4)
        max_qe = round(scored[-1], 4)
        p10_qe = _percentile(scored, 0.10)
        p50_qe = _percentile(scored, 0.50)
        p90_qe = _percentile(scored, 0.90)
        b15_cutoff = max(1, int(len(scored) * 0.15))
        bottom_15_avg = round(sum(scored[:b15_cutoff]) / b15_cutoff, 4)
    else:
        avg_qe = min_qe = max_qe = p10_qe = p50_qe = p90_qe = bottom_15_avg = 0.0

    # Repair iterations analysis
    direct_pass = 0
    round_1 = 0
    round_2 = 0
    exhausted = 0
    flag_counts: dict[str, int] = {}

    for b in all_blocks:
        if b.repair_rounds == 0 and b.status in (BlockStatus.MTQE_PASSED, BlockStatus.REPAIRED):
            direct_pass += 1
        elif b.repair_rounds >= 1:
            # A block that spent repair rounds but still ended FAILED (e.g. it
            # hit a rounds_cap of 1) is an exhaustion, not a round-1/2 success.
            # A triage quarantine (NEEDS_HUMAN/BLOCKED_HUMAN) is not a success
            # either — it used to be counted as one.
            if b.status == BlockStatus.FAILED:
                exhausted += 1
            elif b.status not in (BlockStatus.NEEDS_HUMAN, BlockStatus.BLOCKED_HUMAN):
                if b.repair_rounds == 1:
                    round_1 += 1
                else:
                    round_2 += 1

        for flag in b.error_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    # Render delivery audit: fail-closed overlay skips lose a translation;
    # chrome/policy/non-prose keeps are deliberate. Split them so the fail-closed
    # count carries the zero-tolerance ceiling and the keeps are reported apart.
    fail_closed_blocks, preserved_blocks, _skip_verdict = summarize_render_skips(flag_counts)
    skipped_blocks = fail_closed_blocks + preserved_blocks
    skip_families: dict[str, int] = {}
    for flag, count in flag_counts.items():
        reason = _skip_reason(flag)
        if reason is None:
            continue
        family = reason.split("(", 1)[0].strip()
        skip_families[family or flag] = skip_families.get(family or flag, 0) + count
    # ``completed`` counts every terminal MTQE_PASSED/REPAIRED block, but a
    # fail-closed overlay skip KEEPS that status, so those blocks were counted as
    # rendered AND as fail-closed — making rendered+fail_closed exceed the total.
    # Count only completed blocks that were not fail-closed skipped; a
    # preserved skip sits on a chrome/non-prose block that was never completed.
    fail_closed_completed = sum(
        1
        for b in all_blocks
        if b.status in (BlockStatus.MTQE_PASSED, BlockStatus.REPAIRED)
        and any(_is_fail_closed_skip_flag(flag) for flag in b.error_flags)
    )
    rendered_blocks = max(0, completed - fail_closed_completed)
    coverage = round(rendered_blocks / total, 4) if total > 0 else 1.0

    route_info: ReportRouteInfo | None = None
    raw_route = manifest.run.route_decision
    if isinstance(raw_route, dict):
        try:
            route_info = ReportRouteInfo(
                mode=str(raw_route.get("mode", "")),
                pages=int(raw_route.get("pages", 0) or 0),
                chars=int(raw_route.get("chars", 0) or 0),
                reason=str(raw_route.get("reason", "")),
                formula_mode=str(manifest.run.formula_mode or ""),
            )
        except (TypeError, ValueError):
            route_info = None

    cfg_snapshot: dict[str, Any] = {}
    raw_cfg = manifest.run.config_snapshot
    if isinstance(raw_cfg, dict):
        cfg_snapshot = {str(k): v for k, v in raw_cfg.items()}
    else:
        for k in (
            "draft_model",
            "repair_model",
            "qe_engine",
            "qe_threshold",
            "max_repair_rounds",
            "prompt_strategy",
            "rerank_k",
        ):
            if k in manifest.metadata:
                cfg_snapshot[k] = manifest.metadata[k]

    # Compiler pin for reproducibility (None = unmeasured).
    raw_tv = manifest.metadata.get("typst_version")
    typst_version = str(raw_tv) if isinstance(raw_tv, str) and raw_tv else None
    # QE subprocess honesty label recorded by the quality gate (None when the
    # run used a non-subprocess runner or scored nothing).
    raw_qe_src = ledger.get_job_metadata_value(job_id, "qe_score_source")
    qe_score_source = str(raw_qe_src) if isinstance(raw_qe_src, str) and raw_qe_src else None

    def _metadata_counter(name: str) -> int:
        raw = ledger.get_job_metadata_value(job_id, name)
        try:
            return max(0, int(raw)) if raw is not None else 0
        except (TypeError, ValueError):
            return 0

    return QualityReport(
        job_id=job_id,
        doc_id=manifest.doc_id,
        book_title=manifest.title,
        source_path=manifest.source_path,
        output_path=str(output_path),
        target_lang=manifest.target_lang,
        schema_version=QUALITY_REPORT_SCHEMA_VERSION,
        summary=ReportSummary(
            total_blocks=total,
            completed_blocks=completed,
            repaired_blocks=repaired,
            failed_blocks=failed,
            pass_rate=pass_rate,
            estimated_cost_usd=estimated_cost,
            cache_hit_rate=round(cache_hit_rate, 4) if cache_hit_rate is not None else 0.0,
            needs_human_blocks=needs_human,
            blocked_human_blocks=blocked_human,
        ),
        score_metrics=ReportScoreMetrics(
            avg_qe=avg_qe,
            min_qe=min_qe,
            max_qe=max_qe,
            p10_qe=p10_qe,
            p50_qe=p50_qe,
            p90_qe=p90_qe,
            bottom_15_avg_qe=bottom_15_avg,
            scored_blocks=scored_blocks,
        ),
        terminology=terminology_metrics or ReportTerminologyMetrics(),
        entity_consistency=entity_consistency or ReportEntityConsistency(),
        repair_breakdown=ReportRepairBreakdown(
            direct_pass_count=direct_pass,
            round_1_repaired_count=round_1,
            round_2_repaired_count=round_2,
            exhausted_count=exhausted,
        ),
        placeholder=compute_placeholder_metrics(all_blocks),
        defect_flags=flag_counts,
        enforced_spans=enforced_spans,
        render_coverage=ReportRenderCoverage(
            rendered_blocks=rendered_blocks,
            skipped_blocks=skipped_blocks,
            fail_closed_blocks=fail_closed_blocks,
            preserved_blocks=preserved_blocks,
            render_coverage=coverage,
            skip_families=skip_families,
        ),
        fidelity=ReportFidelity(
            non_text_residual=float(
                (manifest.metadata.get("fidelity") or {}).get("non_text_diff_ratio", 0.0)
            ),
            painted_coverage=float(
                (manifest.metadata.get("fidelity") or {}).get("masked_coverage_ratio", 0.0)
            ),
            pages_measured=int((manifest.metadata.get("fidelity") or {}).get("pages_measured", 0)),
        ),
        route=route_info,
        mode_advisory=mode_advisory,
        config_snapshot=cfg_snapshot,
        syntax_fallbacks=[
            str(item) for item in (manifest.metadata.get("typst_syntax_fallbacks") or []) if item
        ],
        formula_witness_fallbacks=[
            str(item) for item in (manifest.metadata.get("formula_witness_findings") or []) if item
        ],
        formula_blocks=formula_blocks,
        typst_version=typst_version,
        qe_score_source=qe_score_source,
        qe_judge_calls=_metadata_counter("qe_judge_calls"),
        qe_judge_errors=_metadata_counter("qe_judge_errors"),
        delivery_status=manifest.run.delivery_status,
        delivery_warning=manifest.run.delivery_warning,
        delivery_contract=delivery_contract,
    )


PASS_RATE_THRESHOLD = 0.98

#: Prefixes the export stage uses for fail-closed render skips.
#: Supports both ``inplace_skip:`` and ``render_skip:`` prefixes for full compatibility.
RENDER_SKIP_PREFIXES = ("inplace_skip:", "render_skip:")


def _skip_reason(flag: str) -> str | None:
    for prefix in RENDER_SKIP_PREFIXES:
        if flag.startswith(prefix):
            return flag[len(prefix) :]
    return None


def _is_fail_closed_skip_flag(flag: str) -> bool:
    """True when ``flag`` is a fail-closed render skip (not an intentional keep)."""
    return _skip_reason(flag) is not None and not flag.startswith(
        INTENTIONAL_PRESERVED_SKIP_PREFIXES
    )


def summarize_render_skips(defect_flags: dict[str, int]) -> tuple[int, int, str]:
    """Aggregate ``render_skip:{reason}`` flags by family.

    Returns ``(fail_closed, preserved, verdict)``. Families strip the
    parenthesized fit params (``overflow(base=10.0)`` -> ``overflow``) so per-size
    variants do not fragment the audit. A family is *preserved* when every flag
    is an intentional keep (chrome/non-prose/policy/footer); anything else is
    fail-closed and is what the zero-ceiling KPI and the review verdict track.
    """
    fail_closed_families: dict[str, int] = {}
    preserved_families: dict[str, int] = {}
    for flag, count in defect_flags.items():
        reason = _skip_reason(flag)
        if reason is None:
            continue
        family = reason.split("(", 1)[0].strip() or reason
        if flag.startswith(INTENTIONAL_PRESERVED_SKIP_PREFIXES):
            preserved_families[family] = preserved_families.get(family, 0) + count
        else:
            fail_closed_families[family] = fail_closed_families.get(family, 0) + count
    fail_closed = sum(fail_closed_families.values())
    preserved = sum(preserved_families.values())
    if fail_closed:
        detail = ", ".join(f"{fam}×{n}" for fam, n in sorted(fail_closed_families.items()))
        return fail_closed, preserved, f"{detail} — review required"
    if preserved:
        detail = ", ".join(f"{fam}×{n}" for fam, n in sorted(preserved_families.items()))
        return 0, preserved, f"{preserved} source element(s) intentionally preserved ({detail})"
    return 0, 0, "No source-visible skips recorded"


# `math_token_corrupt missing=[..] mismatched=[..] mutated=[..] [reordered=[..]]
# [duplicated=[..]]` (draft stage). ``reordered``/``duplicated`` are optional:
# a block whose masked spans came back out of order or doubled is just as corrupt
# as one with a missing span, so both must count or the KDP audit prints "Full
# retention" while the ledger flags corruption.
_CORRUPT_FLAG_RE = re.compile(
    r"(?:math|cite|code)_token_corrupt missing=(\[.*?\]) mismatched=(\[.*?\]) mutated=(\[.*?\])"
    r"(?: reordered=(\[.*?\]))?(?: duplicated=(\[.*?\]))?"
)


def _parse_corrupt_count(flag: str) -> int:
    """Count corrupt spans encoded in one math_token_corrupt error flag."""
    match = _CORRUPT_FLAG_RE.search(flag)
    if match is None:
        return 0
    total = 0
    for group in match.groups():
        if group is None:
            continue
        try:
            total += len(ast.literal_eval(group))
        except (SyntaxError, ValueError):
            continue
    return total


def compute_placeholder_metrics(blocks: list[Any]) -> ReportPlaceholderMetrics:
    """Recompute placeholder retention without any pipeline or schema change.

    The maskers are deterministic pure functions, so re-masking each block's
    source through the one ``PlaceholderEngine`` (whose order the draft stage
    also uses) reproduces the pre-draft masked-span count. Corrupt restores are
    read back from the persisted ``math_token_corrupt`` flags, capped per block
    at its masked count so re-finalized stages can never double-count the same
    event.
    """
    masked_total = 0
    corrupt_total = 0
    masked_blocks = 0
    corrupt_blocks = 0
    for block in blocks:
        source = block.source_text or ""
        if not source:
            continue
        masked = _PLACEHOLDER_ENGINE.mask(source)
        # Count every masked span type, not just math: code/citation
        # corruption is MQM-Critical, so omitting it left retention at 1.0.
        masked_count = (
            len(masked.code_map)
            + len(masked.math_map)
            + len(masked.soup_map)
            + len(masked.cite_map)
        )
        if masked_count:
            masked_blocks += 1
            masked_total += masked_count
        corrupt_count = sum(_parse_corrupt_count(flag) for flag in (block.error_flags or []))
        if corrupt_count:
            corrupt_blocks += 1
            corrupt_total += min(corrupt_count, masked_count)
    retention = 1.0 if masked_total == 0 else max(0.0, 1.0 - corrupt_total / masked_total)
    return ReportPlaceholderMetrics(
        masked_spans=masked_total,
        corrupt_spans=corrupt_total,
        retention_rate=round(retention, 4),
        masked_blocks=masked_blocks,
        corrupt_blocks=corrupt_blocks,
    )


def _render_advisory_markdown(advisory: dict[str, Any] | None) -> str:
    """Render the bilingual render-mode advisory as a report section (or empty)."""
    if not advisory:
        return ""
    ranking = advisory.get("ranking", [])
    rank_lines = "\n".join(
        f"- **{entry.get('mode')}:** suitability `{entry.get('score')}`"
        + (
            f" (penalized: {', '.join(entry.get('penalties', []))})"
            if entry.get("penalties")
            else ""
        )
        for entry in ranking
    )
    reason_lines = "\n".join(f"- {reason}" for reason in advisory.get("reasons", []))
    rendered = advisory.get("rendered_modes", [])
    rendered_line = f"\nRendered artifacts: `{', '.join(rendered)}`." if rendered else ""
    downgrade = advisory.get("difficulty_downgrade")
    downgrade_line = f"\nPhase-2 difficulty downgrade applied: {downgrade}." if downgrade else ""
    return f"""
---

## 5. Render-Mode Advisory (Bilingual Suitability)

- **Requested mode:** `{advisory.get("requested")}` → **effective mode:** `{advisory.get("effective")}`
- **Advisory tier:** `{advisory.get("tier")}` (enforcement: `{advisory.get("enforcement")}`)
- **Recommended mode:** `{advisory.get("recommended")}`{rendered_line}{downgrade_line}

### Per-mode suitability
{rank_lines}

### Fired rules
{reason_lines if reason_lines else "- (none — document is interleave-friendly)"}
"""


def render_kdp_audit_markdown(report: QualityReport) -> str:
    """Render an Amazon KDP-ready AI translation and quality audit compliance statement.

    All compliance verdicts are derived directly from the report metrics.
    """
    pass_rate_pct = report.summary.pass_rate * 100
    pass_verdict = (
        f"Exceeds {PASS_RATE_THRESHOLD * 100:.1f}% threshold"
        if report.summary.pass_rate >= PASS_RATE_THRESHOLD
        else f"BELOW {PASS_RATE_THRESHOLD * 100:.1f}% threshold — review required"
    )
    if report.summary.failed_blocks == 0:
        failure_verdict = "Zero Critical Failures"
    else:
        failure_verdict = f"{report.summary.failed_blocks} failed segment(s) — remediation required"
    # The Aho-Corasick enforcer is constructed ONLY when the pipeline enables
    # deterministic enforcement, which it does for the short chain alone
    # (pipeline sets ``deterministic_glossary_enforce=short_chain``); the long
    # chain — the default for any book over the short-page cut-off — validates
    # without enforcing. The compliance claim must therefore be gated on the
    # route the job actually took (which the report already carries), never
    # asserted unconditionally.
    if report.route is None or report.route.mode != "short":
        consistency_verdict = (
            "Glossary validated against the enforced table; deterministic "
            "Aho-Corasick enforcement applies to the short chain only "
            f"(this job routed {report.route.mode if report.route else 'unknown'})"
        )
    elif report.repair_breakdown.exhausted_count == 0 and report.summary.failed_blocks == 0:
        consistency_verdict = (
            "Enforced deterministically via Aho-Corasick glossary enforcer at export"
        )
    else:
        consistency_verdict = (
            "Enforcement completed with unresolved segments — manual review required"
        )
    placeholder = report.placeholder
    if placeholder.masked_spans == 0:
        placeholder_verdict = "No masked math spans in source (metric not applicable)"
    elif placeholder.corrupt_spans == 0:
        placeholder_verdict = "Full retention"
    else:
        placeholder_verdict = (
            f"{placeholder.corrupt_spans} corrupt span(s) in "
            f"{placeholder.corrupt_blocks} block(s) — review required"
        )
    fail_closed_total, preserved_total, skip_verdict = summarize_render_skips(report.defect_flags)
    skip_total = fail_closed_total + preserved_total
    cost_line = (
        f"${report.summary.estimated_cost_usd:.5f}"
        if report.summary.estimated_cost_usd is not None
        else "unknown (this model has no price-table entry — unknown is not $0)"
    )
    advisory_section = _render_advisory_markdown(report.mode_advisory)
    coverage_pct = report.render_coverage.render_coverage * 100
    if fail_closed_total:
        coverage_verdict = (
            f"{fail_closed_total} block(s) left source-visible without a translation "
            f"({skip_verdict}) — review required"
        )
    elif preserved_total:
        coverage_verdict = f"{preserved_total} source element(s) intentionally preserved"
    else:
        coverage_verdict = "Full delivery — nothing left source-visible"
    route_line = ""
    if report.route is not None:
        route_line = (
            f"| **Route** | `{report.route.mode}` ({report.route.pages}pp, "
            f"formula `{report.route.formula_mode or 'n/a'}`) | {report.route.reason} |\n"
        )
    # Compiler pin for reproducibility (None = version never recorded).
    toolchain_line = (
        f"| **Typst Compiler** | `{report.typst_version}` | Pinned in CI (upgrade = golden rerun) |\n"
        if report.typst_version
        else "| **Typst Compiler** | unknown | Version unrecorded or compiler-absent run |\n"
    )
    drift_rows = report.entity_consistency.top_drifted
    if drift_rows:
        drift_lines = [
            "| Term (source) | Expected rendering | Occurrences | Exact | Drift rate |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for row in drift_rows:
            drift_lines.append(
                f"| `{row.get('source', '')}` | `{row.get('expected', '')}` | "
                f"{row.get('occurrences', 0)} | {row.get('exact_renderings', 0)} | "
                f"{float(row.get('drift_rate', 0.0)):.1%} |"
            )
        drift_table = "\n".join(drift_lines)
    else:
        drift_table = (
            "No source term drifted across the document (or no glossary term was exercised)."
        )

    cfg_parts: list[str] = []
    if report.config_snapshot:
        cfg = report.config_snapshot
        if "draft_model" in cfg:
            cfg_parts.append(f"Draft: `{cfg['draft_model']}`")
        if "repair_model" in cfg:
            cfg_parts.append(f"Repair: `{cfg['repair_model']}`")
        if "qe_engine" in cfg:
            qe_text = f"QE: `{cfg['qe_engine']}`"
            if "qe_threshold" in cfg:
                qe_text += f" (threshold: `{cfg['qe_threshold']}`)"
            cfg_parts.append(qe_text)
        if "max_repair_rounds" in cfg:
            cfg_parts.append(f"Max Repair Rounds: `{cfg['max_repair_rounds']}`")
        if "prompt_strategy" in cfg:
            cfg_parts.append(f"Prompt Strategy: `{cfg['prompt_strategy']}`")
    provenance_line = (
        f"**Configuration Provenance:** {' · '.join(cfg_parts)}\n" if cfg_parts else ""
    )
    # Typst syntax fallbacks deleted translated lines from the
    # delivered PDF — this must be visible in the audit, not just in logs.
    syntax_fallback_line = ""
    if report.syntax_fallbacks:
        preview = " · ".join(report.syntax_fallbacks[:3])
        more = (
            f" (+{len(report.syntax_fallbacks) - 3} more)"
            if len(report.syntax_fallbacks) > 3
            else ""
        )
        syntax_fallback_line = (
            f"> ⚠️ **Content removal warning:** the Typst self-healing loop "
            f"commented out **{len(report.syntax_fallbacks)}** translated line(s) "
            f"from the delivered PDF ({preview}{more}). "
            f"Human review required — see the .typ source markers "
            f"`[UBT_SYNTAX_FALLBACK]`.\n"
        )

    # Witness substitutions are lossless (the source equation
    # graphic ships instead), but the reader must still be told which
    # equations were swapped and why.
    witness_line = ""
    if report.formula_witness_fallbacks:
        preview = " · ".join(report.formula_witness_fallbacks[:3])
        more = (
            f" (+{len(report.formula_witness_fallbacks) - 3} more)"
            if len(report.formula_witness_fallbacks) > 3
            else ""
        )
        witness_line = (
            f"> **Formula witness substitutions:** **{len(report.formula_witness_fallbacks)}** "
            f"display formula(s) differed structurally from the source equation after "
            f"conversion and were rendered from their source graphics instead "
            f"({preview}{more}).\n"
        )

    delivery_warning_line = ""
    delivery_table_line = ""
    if report.delivery_status:
        # An advisory is a tradeoff the reader weighs; only a genuine
        # UNSUITABLE verdict is a delivery stop. Auto routing sends
        # formula-dense documents to overlay on purpose, so that pairing is
        # never a failure.
        is_advisory = report.delivery_status.startswith("LAYOUT_TRADEOFF_ADVISORY")
        glyph = "⚠️" if is_advisory else "🛑"
        verdict = (
            "ADVISORY — informed layout tradeoff"
            if is_advisory
            else "ACTION REQUIRED — Layout constraint mismatch"
        )
        delivery_warning_line = (
            f"> {glyph} **DELIVERY STATUS: {report.delivery_status}**\n"
            f"> {report.delivery_warning or ''}\n\n"
        )
        delivery_table_line = (
            f"| **Delivery Status** | `{report.delivery_status}` | {glyph} {verdict} |\n"
        )

    return f"""# AI Translation Quality & Compliance Audit Report

**Book Title:** {report.book_title}
**Source Path:** `{report.source_path}`
**Output Target:** `{report.output_path}`
**Target Language:** {report.target_lang}
**Audit Timestamp:** {report.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")}
**Job ID:** `{report.job_id}`
{provenance_line}
{delivery_warning_line}{syntax_fallback_line}{witness_line}---

## 1. Amazon KDP AI Content Disclosure Statement
This publication was translated using the **Universal Book Translator (UBT)** asymmetric dual-tier pipeline.
In accordance with Amazon Kindle Direct Publishing (KDP) guidelines on AI-assisted and AI-generated content:
- **Translation Category:** AI-Assisted with Substantial Human/Automated Multi-Layer Quality Verification.
- **Consistency Enforcement:** {consistency_verdict}
- **Defect Remediation:** Automated 2-round targeted repair loop on suspicious segments via high-reasoning flagship models.

---

## 2. Quality & Integrity Metrics Summary

| Metric Name | Audited Value | Status / Benchmark |
| :--- | :--- | :--- |
{delivery_table_line}| **Total Translated Segments** | {report.summary.total_blocks} | 100% Extracted |
| **Successfully Completed** | {report.summary.completed_blocks} | Pipeline completion (≠ delivery) |
| **Pass Rate** | {pass_rate_pct:.1f}% | {pass_verdict} |
| **Render Coverage** | {coverage_pct:.1f}% ({report.render_coverage.rendered_blocks}/{report.summary.total_blocks} on page) | {coverage_verdict} |
{route_line}| **Average MTQE Score** | {report.score_metrics.avg_qe:.4f} | Over {report.score_metrics.scored_blocks} QE-scored block(s) — verbatim skips and TM hits carry no score and are excluded; discrete defect-class proxy, uncalibrated |
| **LLM Judge Calls / Errors** | {report.qe_judge_calls} / {report.qe_judge_errors} | Paid second opinions / unusable replies; explicit opt-in only |

> **MTQE score semantics **: under the default heuristic engine the
> MTQE score is a deterministic **defect-class proxy**, not a calibrated
> quality measurement. Only twelve discrete values are reachable:
> `0.92` pass (no deterministic invariant failed) · `0.70` other structural
> rejection · `0.60` length anomaly · `0.55` numeric fidelity · `0.40`
> script density · `0.35` omission · `0.30` HTML delta · `0.25` glossary
> term violation · `0.20` hallucination loop · `0.15`
> untranslated/fabricated · `0.10` leak · `0.0` empty. Averages/percentiles
> of these values are routing statistics over the scored population only;
> calibrated numbers require the CometKiwi L2 runner.

| **Bottom 15% Score Average** | {report.score_metrics.bottom_15_avg_qe:.4f} | Protected Quality Floor |
| **Failed Segments** | {report.summary.failed_blocks} | {failure_verdict} |
| **HITL Remediation Queue** | {report.summary.needs_human_blocks} needs human, {report.summary.blocked_human_blocks} blocked | Quarantined from machine output |
| **Masked Math Spans** | {report.placeholder.masked_spans} | Across {report.placeholder.masked_blocks} block(s) |
| **Placeholder Retention** | {report.placeholder.retention_rate:.2%} | {placeholder_verdict} |
| **Terminology Precision (TP)** | {report.terminology.term_precision:.2%} | Exact term renderings over the occurrences exercised by this job ({report.terminology.terms_expected} distinct term(s)) |
| **Fuzzy Terminology Precision (TF)** | {report.terminology.fuzzy_term_precision:.2%} | Allows a fuzzy (≥80%) rendering match |
| **Terminology Recall** | {report.terminology.term_recall:.2%} | {report.terminology.terms_rendered}/{report.terminology.terms_expected} term(s) rendered exactly at least once |
| **Terminology Consistency** | {report.entity_consistency.terms_with_drift}/{report.entity_consistency.terms_audited} term(s) drifted | Canonical rendering absent in at least one block |
| **Render-Skipped Segments** | {skip_total} | {skip_verdict} |
| **Estimated Token Cost** | {cost_line} | From provider usage accounting |
| **Cache Hit Rate** | {report.summary.cache_hit_rate:.2%} | static-prefix TCO lever (0% = unmeasured) |
{toolchain_line}

---

## 3. Score Distribution (Quantiles)

- **Scored Population:** `{report.score_metrics.scored_blocks}` block(s) that reached QE scoring
  (verbatim skips and TM exact hits are excluded — they carry no score)

- **Minimum Score:** `{report.score_metrics.min_qe:.4f}`
- **10th Percentile (P10):** `{report.score_metrics.p10_qe:.4f}`
- **Median (P50):** `{report.score_metrics.p50_qe:.4f}`
- **90th Percentile (P90):** `{report.score_metrics.p90_qe:.4f}`
- **Maximum Score:** `{report.score_metrics.max_qe:.4f}`

---

## 3.5 Terminology Consistency by Term

{drift_table}

---

## 4. Multi-Layer Remediation Breakdown

- **Direct Pass (0 Repair Rounds):** {report.repair_breakdown.direct_pass_count} blocks
- **Resolved in Round 1 Repair:** {report.repair_breakdown.round_1_repaired_count} blocks
- **Resolved in Round 2 Repair:** {report.repair_breakdown.round_2_repaired_count} blocks
- **Max Rounds Exhausted:** {report.repair_breakdown.exhausted_count} blocks
- **HITL Quarantined (Needs Human Review):** {report.summary.needs_human_blocks} blocks
- **Critical Blocked (Quarantined Machine Output):** {report.summary.blocked_human_blocks} blocks
{advisory_section}
*Report certified by Universal Book Translator (UBT) Quality Engine.*
"""


def save_quality_report(
    report: QualityReport, report_path: Path | str, *, write_markdown: bool = False
) -> Path:
    """Serialize QualityReport to JSON, optionally with a KDP audit Markdown companion.

    The Markdown audit is opt-in: it is a human review / KDP artifact, not part
    of the machine contract, and writing it unconditionally put a second file in
    every output dir that no stale-report sweep knows about.
    """
    out_file = Path(report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    if write_markdown:
        md_file = out_file.with_suffix(".md")
        md_file.write_text(render_kdp_audit_markdown(report), encoding="utf-8")
    return out_file

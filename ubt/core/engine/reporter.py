"""Standardized book translation quality report generator."""

import ast
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import BlockStatus, BlockType, BookManifest
from ubt.core.ports import placeholder_engine
from ubt.core.qe.defect_taxonomy import INTENTIONAL_PRESERVED_SKIP_PREFIXES

# The QE-scored population policy (placeholder/skip exclusion) lives in
# ubt.core.qe.score_policy, which both this report and the ledger's job stats
# import so their averages cannot drift apart again.
from ubt.core.qe.score_policy import qe_scored_values

# The placeholder mask order has exactly one owner (``PlaceholderEngine``); the
# report re-masks through it rather than restating the order, so a change to the
# order cannot leave the report measuring a pipeline other than the one that ran.
# The engine is named in ``ubt.segment``, so core reaches it only through the port.
_PLACEHOLDER_ENGINE = placeholder_engine()

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
    (``completed / total``), not a pixel/placement measure: while the overlay
    engine keeps the source page as canvas, translatable prose is typeset into
    the page. Those delivery gaps are carried by ``fail_closed_blocks`` /
    ``skip_families`` (and by the separate per-page render-visibility report) —
    never folded into this ratio, so a caller that wants "was every translation
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
    """Pixel-level fidelity of an ``overlay`` render (advisory, never blocking).

    ``non_text_residual`` is the fraction of pixels *outside* the painted text
    boxes that differ from the source — it should be ~0 because the overlay route
    keeps every non-text element untouched. ``painted_coverage`` is the fraction
    of page area that was actually painted prose (the inverse of the "most of
    the page still shows source text" failure). ``pages_measured`` is 0 when the
    probe could not run (rasterizer unavailable), in which case both ratios are
    meaningless defaults.
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
    # Overlay-render pixel fidelity (advisory; 0 pages when not measured).
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
    # Display-formula blocks delivered. Inline math is not counted; 0 means
    # this job carried no display formula.
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

        Read-only convenience: a consumer need not know the nested path to read
        the book's average score. It mirrors the authoritative
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
    # Display-formula blocks delivered (inline math is not counted).
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
            # either.
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
# [duplicated=[..]]` (draft stage). One rule for all four maskers -- math, cite,
# code AND soup -- so the alternation must list every family the draft stage can
# emit: dropping ``soup`` made a corrupt soup span invisible to the counter and
# the KDP audit printed "Full retention" while the ledger flagged corruption.
# ``reordered``/``duplicated`` are optional: a block whose masked spans came
# back out of order or doubled is just as corrupt as one with a missing span, so
# both must count.
_CORRUPT_FLAG_RE = re.compile(
    r"(?:email|math|cite|code|soup)_token_corrupt missing=(\[.*?\]) mismatched=(\[.*?\]) mutated=(\[.*?\])"
    r"(?: reordered=(\[.*?\]))?(?: duplicated=(\[.*?\]))?"
)


def _parse_corrupt_count(flag: str) -> int:
    """Count corrupt spans encoded in one ``*_token_corrupt`` error flag."""
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
            len(masked.email_map)
            + len(masked.code_map)
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
    downgrade_line = f"\nDifficulty downgrade applied: {downgrade}." if downgrade else ""
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


def save_quality_report(report: QualityReport, report_path: Path | str) -> Path:
    """Serialize QualityReport to JSON."""
    out_file = Path(report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return out_file

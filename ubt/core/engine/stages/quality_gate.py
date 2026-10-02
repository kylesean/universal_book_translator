import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.stage_context import StageContext
from ubt.core.exceptions import MTQEEvaluationError
from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.comet_runner import (
    HeuristicQERunner,
    glossary_violation_flag,
)
from ubt.core.qe.defect_taxonomy import has_structural_defect
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.validators.consistency import GlossaryConsistencyValidator
from ubt.pipeline.facts import Scoring, Terminology
from ubt.pipeline.services import RunServices

logger = logging.getLogger(__name__)


def _fast_pass_screen(
    blocks: list[IRBlock],
    *,
    fast_pass: FastPassFilter,
    glossary_validator: GlossaryConsistencyValidator | None,
    glossary_terms: list[str] | None = None,
) -> tuple[list[IRBlock], list[dict[str, Any]]]:
    """Split drafted blocks into auto-passes and scoring candidates.

    Runs in a worker thread: FastPass plus a terminology walk over every drafted
    block of a book is a long synchronous stretch, and the loop it would
    otherwise hold serves SSE subscribers and other concurrent jobs.
    """
    suspicious_blocks: list[IRBlock] = []
    passed_updates: list[dict[str, Any]] = []

    for b in blocks:
        # Record the enforced glossary terms present in this block's
        # target, so ``glossary_hits`` is populated rather than always [].
        terms = glossary_terms or []
        b.glossary_hits = [t for t in terms if t and t in (b.target_text or "")]
        decision = fast_pass.evaluate(
            b.source_text,
            b.target_text or "",
            block_type=b.block_type,
            skip_translate=b.skip_translate,
        )
        # Terminology runs before the verdict. Verbatim ships are exempt
        # (the export stage skips them too: substituting terms into a kept
        # bibliography produces Chinglish mash).
        if glossary_validator is not None and b.target_text and not b.skip_translate:
            glossary_flag = glossary_violation_flag(
                b.source_text, b.target_text, glossary_validator
            )
            if glossary_flag is not None and glossary_flag not in b.error_flags:
                b.error_flags.append(glossary_flag)
        # A draft-stage structural marker (e.g. masked-token corruption) is
        # fatal even when the text happens to clear FastPass: the restoration
        # mismatch means the target is wrong, not merely awkward.
        if decision.passed and not has_structural_defect(b.error_flags):
            b.status = BlockStatus.MTQE_PASSED
            passed_updates.append(
                {
                    "block_id": b.id,
                    "target_text": b.target_text or "",
                    "status": BlockStatus.MTQE_PASSED,
                    "glossary_hits": b.glossary_hits,
                }
            )
        else:
            b.status = BlockStatus.REPAIR_PENDING
            if not decision.passed:
                # Keep the fast-pass reason at index 0 (the heuristic scorer
                # reads it) but retain the draft stage's evidence. The former
                # blanket overwrite erased exactly the structural flag that must
                # keep the block out of auto-pass.
                prior = [f for f in b.error_flags if f and f != decision.reason]
                b.error_flags = [decision.reason, *prior]
            suspicious_blocks.append(b)

    return suspicious_blocks, passed_updates


async def _audit_pass_sample(
    qe_runner: BaseQERunner,
    blocks: list[IRBlock],
    threshold: float,
) -> list[IRBlock]:
    """FastPass-passing blocks whose sampled QE score falls below ``threshold``.

    FastPass-passing blocks used to be written straight to ``MTQE_PASSED``, so
    the ``pass_sample`` mechanism meant to audit a fraction of clean passes
    (``TieredQERunner``) could never see them — the knob was inert. When the
    runner samples passes, score them here and route any below-threshold block
    back into repair. A plain heuristic/COMET runner has no ``pass_sample``, so
    the default configuration is unchanged.
    """
    if getattr(qe_runner, "pass_sample", 0.0) <= 0.0 or not blocks:
        return []
    pairs = [{"src": b.source_text, "mt": b.target_text or ""} for b in blocks]
    scores = await qe_runner.score_pairs(pairs)
    if len(scores) != len(blocks):
        raise MTQEEvaluationError(
            f"QE runner returned {len(scores)} score(s) for {len(blocks)} pass-sample block(s)",
            details={"expected": len(blocks), "got": len(scores)},
        )
    return [b for b, score in zip(blocks, scores, strict=True) if score < threshold]


async def run_quality_gate_stage(
    ctx: StageContext,
    services: RunServices,
    terminology: Terminology,
    scoring: Scoring,
    chapter_id: str | None = None,
) -> AsyncIterator[TranslationProgressEvent]:
    """Evaluate drafted blocks with FastPassFilter and score suspicious blocks via QE runner.

    Terminology first: the run's enforced terminology is bound to the heuristic runner
    here (and to the repair loop that shares it) because this is the first stage
    that scores, and only the bible stage could have produced a glossary. An
    explicitly injected runner keeps its own scoring policy.

    The run's enforced terminology comes from the ``terminology`` parameter. A target
    that drops/alters a term gets a flag carrying
    ``GLOSSARY_VIOLATION_MARKER``, which is a registered structural marker —
    the block is routed into the scoring branch below instead of auto-passing
    on structural invariants alone. An empty sheet keeps the previous behaviour
    for callers that have no glossary.
    """
    ledger = ctx.ledger
    actual_job_id = ctx.job_id
    fast_pass = services.fast_pass
    create_event_fn = ctx.create_event
    glossary_dicts = terminology.glossary_dicts or None
    # The gate produces the run's scoring collaborators (explicit
    # stage execution context): it binds the glossary onto the QE runner and builds the repair
    # loop around it, so repair, triage and consistency re-score terminology with
    # the same runner. A runner/loop the caller injected is kept.
    qe_runner = scoring.qe_runner or services.qe_runner
    bind_glossary = getattr(qe_runner, "with_glossary", None)
    if not services.qe_runner_explicit and glossary_dicts and callable(bind_glossary):
        # Both the gate's scoring branch and the repair loop's re-score must
        # classify a terminology violation at its own band instead of the 0.92
        # pass value. Heuristic and tiered runners expose ``with_glossary``
        # (the tier forwards it to its heuristic leg, the only leg that scores
        # terminology); a subprocess/neural runner has none and keeps its own
        # scoring.
        qe_runner = bind_glossary(glossary_dicts)
        scoring.qe_runner = qe_runner
        scoring.repair_loop = services.repair_loop_for(
            qe_runner, config=ctx.config, router=ctx.router
        )
    drafted_blocks = await asyncio.to_thread(
        ledger.fetch_blocks_by_status, actual_job_id, BlockStatus.DRAFTED, chapter_id=chapter_id
    )
    glossary_validator = (
        GlossaryConsistencyValidator(glossary=glossary_dicts) if glossary_dicts else None
    )
    glossary_terms = sorted(
        {str(g.get("translation", "")).strip() for g in (glossary_dicts or [])} - {""}
    )
    suspicious_blocks, passed_updates = await asyncio.to_thread(
        _fast_pass_screen,
        drafted_blocks,
        fast_pass=fast_pass,
        glossary_validator=glossary_validator,
        glossary_terms=glossary_terms,
    )

    # Audit a sample of FastPass passes when the runner supports it
    # (``pass_sample`` > 0, i.e. TieredQERunner), so a clean-but-wrong block the
    # judge lowers is routed to repair instead of shipping as MTQE_PASSED.
    if passed_updates:
        by_id = {b.id: b for b in drafted_blocks}
        pass_blocks = [by_id[u["block_id"]] for u in passed_updates if u["block_id"] in by_id]
        demoted = await _audit_pass_sample(qe_runner, pass_blocks, ctx.config.qe_threshold)
        if demoted:
            demoted_ids = {b.id for b in demoted}
            passed_updates = [u for u in passed_updates if u["block_id"] not in demoted_ids]
            suspicious_blocks.extend(demoted)

    if passed_updates:
        await asyncio.to_thread(ledger.save_checkpoints_batch, passed_updates)

    if suspicious_blocks:
        if isinstance(qe_runner, HeuristicQERunner):
            # score_from_flags caps a glossary violation at its own band
            # while every other defect class keeps its existing value.
            scores = [HeuristicQERunner.score_from_flags(b.error_flags) for b in suspicious_blocks]
        else:
            pairs = [{"src": b.source_text, "mt": b.target_text or ""} for b in suspicious_blocks]
            scores = await qe_runner.score_pairs(pairs)

        # A runner that violates its one-score-per-pair contract must fail the
        # stage loudly: silently truncating leaves blocks unscored in the
        # ledger while the job reports progress.
        if len(scores) != len(suspicious_blocks):
            raise MTQEEvaluationError(
                f"QE runner returned {len(scores)} score(s) for {len(suspicious_blocks)} block(s)",
                details={"expected": len(suspicious_blocks), "got": len(scores)},
            )

        engine = getattr(qe_runner, "last_engine", None)
        if engine is not None:
            # Honesty provenance: a COMET subprocess that fell back to its
            # built-in heuristic must say so in the audit artifact, instead of
            # shipping twelve discrete bands as if they were calibrated
            # CometKiwi measurements.
            recorded = await asyncio.to_thread(
                ledger.get_job_metadata_value, actual_job_id, "qe_score_source"
            )
            if recorded != engine:
                await asyncio.to_thread(
                    ledger.set_job_metadata_value, actual_job_id, "qe_score_source", engine
                )

        qe_updates: list[dict[str, Any]] = []
        for b, score in zip(suspicious_blocks, scores, strict=True):
            # Anchor provenance weighting (pure domain data, no adapter imports)
            # Dual-witness agreement boost (+0.05) when matched and no review needed.
            # Disagreement (needs_review or high vlm_only discrepancy) routes to REPAIR_PENDING.
            prov = b.provenance or {}
            anchor_prov = str(prov.get("anchor_provenance", ""))
            needs_review = bool(prov.get("needs_review", False))
            stats = prov.get("anchor_stats")
            if isinstance(stats, dict):
                matched = stats.get("matched", 0)
                vlm_only = stats.get("vlm_only", 0)
                pdfium_only = stats.get("pdfium_only", 0)
                total = matched + vlm_only + pdfium_only
                if total > 0 and (matched / total) >= 0.8 and not needs_review:
                    score = min(1.0, score + 0.05)
                elif (
                    needs_review or (total > 0 and (vlm_only / total) > 0.5)
                ) and "Visual witness discrepancy" not in b.error_flags:
                    b.error_flags.append("Visual witness discrepancy")
            elif "proofread" in anchor_prov and not needs_review:
                score = min(1.0, score + 0.05)
            elif needs_review:
                if "Visual witness discrepancy" not in b.error_flags:
                    b.error_flags.append("Visual witness discrepancy")

            b.mtqe_score = score
            # Suspicious blocks always enter repair. Every reason the FastPass
            # filter can emit maps to a fatal structural marker, so no measured
            # QE score may release one — the old
            # ``score >= qe_threshold and not has_structural_defect`` branch was
            # therefore unreachable, and its dead ``qe_threshold`` comparison
            # only obscured the fact. The score (with the provenance boost
            # above) is persisted for repair ranking and audit, not release.
            b.status = BlockStatus.REPAIR_PENDING

            qe_updates.append(
                {
                    "block_id": b.id,
                    "target_text": b.target_text or "",
                    "status": b.status,
                    "mtqe_score": score,
                    "error_flags": b.error_flags,
                    "glossary_hits": b.glossary_hits,
                }
            )

        if qe_updates:
            await asyncio.to_thread(ledger.save_checkpoints_batch, qe_updates)

    # TieredQERunner keeps paid second-opinion ROI counters on the runner. Store
    # them in the durable job metadata so the final report can expose judge
    # spend and silent fallback errors even after the per-run object is gone.
    for key, value in (
        ("qe_judge_calls", getattr(qe_runner, "judge_calls", None)),
        ("qe_judge_errors", getattr(qe_runner, "judge_errors", None)),
    ):
        if isinstance(value, int):
            await asyncio.to_thread(ledger.set_job_metadata_value, actual_job_id, key, str(value))

    event = await create_event_fn(
        EventType.MTQE_EVALUATED,
        actual_job_id,
        ledger,
        message="Quality evaluation completed for all drafted blocks",
    )
    yield event

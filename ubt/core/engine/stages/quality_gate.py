import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.facts import Scoring, Terminology
from ubt.core.engine.services import RunServices
from ubt.core.engine.stage_context import StageContext
from ubt.core.exceptions import MTQEEvaluationError
from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.core.memory.cjk_matcher import count_term_in_text
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.defect_taxonomy import has_structural_defect
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.qe.heuristic import glossary_violation_flag
from ubt.core.qe.score_policy import shippable_threshold
from ubt.core.validators.consistency import GlossaryConsistencyValidator

logger = logging.getLogger(__name__)


def _fast_pass_screen(
    blocks: list[IRBlock],
    *,
    fast_pass: FastPassFilter,
    glossary_validator: GlossaryConsistencyValidator | None,
    glossary_terms: list[str] | None = None,
    rehearsal: bool = False,
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
        # Boundary-aware matching (not ``t in target``) so "cat" is not
        # recorded as a hit inside "category".
        terms = glossary_terms or []
        b.glossary_hits = [
            t for t in terms if t and count_term_in_text(str(t), b.target_text or "") > 0
        ]
        decision = fast_pass.evaluate(
            b.source_text,
            b.target_text or "",
            block_type=b.block_type,
            skip_translate=b.skip_translate,
            rehearsal=rehearsal,
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
            # Non-blocking findings on a passing draft (a numeric order
            # inversion, say) become review flags, never a rejection: they are
            # the same advisory tier as the HTML delta validator's formatting
            # drift. Recorded before the checkpoint so the flag is persisted.
            advisory_flags = [
                f"numeric_advisory: {advisory}"
                for advisory in decision.advisories
                if f"numeric_advisory: {advisory}" not in b.error_flags
            ]
            b.error_flags.extend(advisory_flags)
            b.status = BlockStatus.MTQE_PASSED
            update: dict[str, Any] = {
                "block_id": b.id,
                "target_text": b.target_text or "",
                "status": BlockStatus.MTQE_PASSED,
                "glossary_hits": b.glossary_hits,
            }
            # Only widen the checkpoint's columns when an advisory actually
            # landed, so a plain pass keeps the shape it always had.
            if advisory_flags:
                update["error_flags"] = b.error_flags
            passed_updates.append(update)
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


def screen_drafted_blocks(
    blocks: list[IRBlock],
    *,
    fast_pass: FastPassFilter,
    glossary_validator: GlossaryConsistencyValidator | None,
    glossary_terms: list[str] | None = None,
    rehearsal: bool = False,
) -> tuple[list[IRBlock], list[dict[str, Any]]]:
    """Public name for the deterministic FastPass screen used by Agent mode.

    The provider pipeline's gate and the Agent finish path must apply the same
    invariant checks. Keeping the implementation in one place prevents an Agent
    submission from bypassing a rule the normal pipeline would enforce.
    """
    return _fast_pass_screen(
        blocks,
        fast_pass=fast_pass,
        glossary_validator=glossary_validator,
        glossary_terms=glossary_terms,
        rehearsal=rehearsal,
    )


async def _audit_pass_sample(
    qe_runner: BaseQERunner,
    blocks: list[IRBlock],
    threshold: float,
) -> tuple[list[IRBlock], list[IRBlock]]:
    """Score a sample of FastPass passes; split into (below the line, scored).

    A FastPass-passing block written straight to ``MTQE_PASSED`` would be
    invisible to the ``pass_sample`` mechanism meant to audit a fraction of clean
    passes (``TieredQERunner``), leaving that knob inert. When the runner samples
    passes, they are scored here and any below-threshold block is routed back
    into repair. A plain heuristic/COMET runner has no ``pass_sample``, so
    the default configuration is unchanged.

    Both halves are returned. Reporting only the demoted ones meant a block the
    judge scored *above* the line kept ``mtqe_score=None``, so the ledger could
    not distinguish "measured and good" from "never measured" — measured on a
    4-page run, 28 of 34 passing prose blocks looked unmeasured while the judge
    had in fact scored them.

    Blocks that carry no prose are skipped before scoring. Measured on arXiv
    2606.23344: four asset blocks (``img_*``/``fml_*``) reached the judge with a
    *file path* as both source and target. There is no translation to judge, and
    the judge's opinion of a path is noise — it scored them 1.0, spending four
    calls and reporting a meaningless pass. Images, formulas and verbatim-skipped
    blocks are excluded here rather than trusted to score themselves well.
    """
    if getattr(qe_runner, "pass_sample", 0.0) <= 0.0 or not blocks:
        return [], []
    scorable = [b for b in blocks if not _is_unscorable(b)]
    if not scorable:
        return [], []
    pairs = [
        {"src": b.source_text, "mt": b.target_text or "", "block_type": str(b.block_type)}
        for b in scorable
    ]
    scores = await qe_runner.score_pairs(pairs)
    if len(scores) != len(scorable):
        raise MTQEEvaluationError(
            f"QE runner returned {len(scores)} score(s) for {len(scorable)} pass-sample block(s)",
            details={"expected": len(scorable), "got": len(scores)},
        )
    scored: list[IRBlock] = []
    demoted: list[IRBlock] = []
    # A number is only a *measurement* for the entries the runner can vouch for:
    # every entry when it is calibrated, and — for an uncalibrated tier — only
    # the entries a judge actually returned an opinion for. A tier whose judge
    # died still returns its heuristic leg's flat 0.92 for a clean pass, and
    # recording that as ``mtqe_score`` made "the judge never answered"
    # indistinguishable from "the judge measured this at 0.92", which is the
    # audit's entire output. A partly-failed batch has the same shape one level
    # down: one answered pair must not license the bands of the rest.
    measured = _measured_indices(qe_runner, len(scorable))
    for index, (block, score) in enumerate(zip(scorable, scores, strict=True)):
        if index not in measured:
            continue
        block.mtqe_score = float(score)
        scored.append(block)
        if score < threshold:
            demoted.append(block)
    return demoted, scored


def _measured_indices(runner: Any, count: int) -> frozenset[int]:
    """Indexes of the last batch whose scores are evidence, not fallback bands.

    One definition, read by both the pass-sample audit and the suspicious-block
    scoring path, so "is this number evidence" cannot drift between them.
    """
    is_calibrated = getattr(runner, "is_calibrated", None)
    if is_calibrated is not None and is_calibrated():
        return frozenset(range(count))
    judged_indices = getattr(runner, "judged_indices", None)
    if judged_indices is not None:
        return frozenset(judged_indices(count))
    # A runner that reports batch-level judgement only: keep its behaviour.
    if getattr(runner, "judged_this_run", False):
        return frozenset(range(count))
    return frozenset()


#: Block types whose text is not a translation the judge can read. Assets are
#: paths; a figure/equation is placed as an image. Tables and prose are NOT here:
#: their text is content the reader actually reads.
_UNSCORABLE_BLOCK_TYPES = frozenset({"figure", "equation"})


def _is_unscorable(block: IRBlock) -> bool:
    """Whether judging this block's target against its source would mean nothing.

    Two shapes qualify, and both were measured on a real run:

    * an **asset block** — its text is a cache path (``…/assets/6331…png``), so
      source and target are identical paths and the judge scores them 1.0;
    * a **verbatim-skipped block** — a bibliography entry or an author list the
      pipeline deliberately keeps in the source language, so a low score there
      would ask repair to "fix" a correct decision.

    Tables and prose are scored: their text is content the reader reads.
    """
    if block.skip_translate:
        return True
    block_type = str(getattr(block, "block_type", "") or "")
    if block_type in _UNSCORABLE_BLOCK_TYPES:
        return True
    source = (block.source_text or "").strip()
    target = (block.target_text or "").strip()
    # An asset path is the same string on both sides and points at a file.
    return bool(source) and source == target and _looks_like_asset_path(source)


def _looks_like_asset_path(text: str) -> bool:
    """Whether a block's text is a generated asset path rather than prose."""
    if len(text) > 512 or "\n" in text:
        return False
    lowered = text.lower()
    if not lowered.endswith((".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf")):
        return False
    return "/assets/" in lowered or "\\assets\\" in lowered


async def run_quality_gate_stage(
    ctx: StageContext,
    services: RunServices,
    terminology: Terminology,
    scoring: Scoring,
    chapter_id: str | None = None,
) -> AsyncGenerator[TranslationProgressEvent, None]:
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
    # The document context used to be re-bound here onto the runner's own
    # (private) omission gate — the patch for the inner/outer filter split.
    # The runner now scores through the run's single filter (the pipeline
    # hands it over via ``with_fast_pass``), and the draft stage is the one
    # place that binds the glossary, document context and vocabulary onto it,
    # so this stage has nothing left to synchronise: whatever the draft bound
    # is, by construction, what ``score_pairs`` scores with.
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
        rehearsal=bool(getattr(ctx, "is_mock_run", False)),
    )

    # Audit a sample of FastPass passes when the runner supports it
    # (``pass_sample`` > 0, i.e. TieredQERunner), so a clean-but-wrong block the
    # judge lowers is routed to repair instead of shipping as MTQE_PASSED.
    #
    # The audit *scores* every sampled block, but only the ones below the line
    # were acted on: a block the judge scored 0.82 was left with
    # ``mtqe_score=None`` in the ledger. Measured on a 4-page run: 28 of 34
    # passing prose blocks had no score at all, so the report could not show a
    # distribution and "how good is this delivery" was unanswerable. The scores
    # are now carried back for every sampled block.
    audited_demotions: list[dict[str, Any]] = []
    if passed_updates:
        by_id = {b.id: b for b in drafted_blocks}
        pass_blocks = [by_id[u["block_id"]] for u in passed_updates if u["block_id"] in by_id]
        # The *shippable* bar, not the raw config value: releasing a sampled
        # pass below SHIPPABLE_FLOOR would ship a draft that every other
        # release path (repair, triage) refuses. One definition, one floor.
        audited = await _audit_pass_sample(
            qe_runner, pass_blocks, shippable_threshold(ctx.config.qe_threshold)
        )
        demoted, scored = audited
        if demoted:
            demoted_ids = {b.id for b in demoted}
            passed_updates = [u for u in passed_updates if u["block_id"] not in demoted_ids]
            # A demoted block was *just* scored by the audit. Re-adding it to
            # ``suspicious_blocks`` sent it through ``score_pairs`` a second
            # time, and because the sampling is a stable ``crc32`` of the source
            # the same pairs hit the judge again — the same opinion, billed
            # twice. Persist what the audit measured instead.
            #
            # Status and score are exactly what the second pass would have
            # produced (the audit keeps ``min(heuristic, judge)``); only the
            # duplicate call disappears. No flag is added: a judge-only doubt is
            # SEMANTIC_DOUBT and triage reads the score, not a new marker.
            audited_demotions = [
                {
                    "block_id": b.id,
                    "target_text": b.target_text or "",
                    "status": BlockStatus.REPAIR_PENDING,
                    "mtqe_score": b.mtqe_score,
                    "error_flags": b.error_flags,
                    "glossary_hits": b.glossary_hits,
                }
                for b in demoted
            ]
        if scored:
            # Record what the audit measured for the blocks it let through, so a
            # passing block is not indistinguishable from an unmeasured one.
            scored_ids = {b.id: b.mtqe_score for b in scored}
            for update in passed_updates:
                score = scored_ids.get(update["block_id"])
                if score is not None:
                    update["mtqe_score"] = score

    if passed_updates:
        await asyncio.to_thread(ledger.save_checkpoints_batch, passed_updates)
    if audited_demotions:
        await asyncio.to_thread(ledger.save_checkpoints_batch, audited_demotions)

    if suspicious_blocks:
        # ``may_be_calibrated`` (not ``is_calibrated``): a dynamic or deferred
        # runner reports uncalibrated until its *first* batch answers, so gating the
        # scoring call on ``is_calibrated`` meant the first call was never made
        # and every block got ``mtqe_score=None`` forever. Ask whether a call is
        # worth attempting, make it, then read the settled verdict.
        runner = qe_runner
        attempt = getattr(runner, "may_be_calibrated", runner.is_calibrated)()
        if attempt:
            pairs = [
                {"src": b.source_text, "mt": b.target_text or "", "block_type": str(b.block_type)}
                for b in suspicious_blocks
            ]
            raw_scores = await runner.score_pairs(pairs)
            if len(raw_scores) != len(suspicious_blocks):
                raise MTQEEvaluationError(
                    f"QE runner returned {len(raw_scores)} score(s) "
                    f"for {len(suspicious_blocks)} block(s)",
                    details={"expected": len(suspicious_blocks), "got": len(raw_scores)},
                )
        # The authoritative verdict after any scoring attempt: a subprocess that
        # answered with its heuristic fallback is still uncalibrated, and its
        # discrete bands must not become pseudo-scores.
        #
        # A judge opinion is different from a heuristic band: it is a graded
        # answer, so it is kept even though ``is_calibrated`` stays False. Without
        # this, every verdict the judge produced was thrown away — measured on
        # arXiv 2606.23344, a judge scored 0.78 for a block that had dropped a
        # word and the gate stored ``None``, so the block could not reach repair
        # on the judgement it had just been given.
        #
        # The predicate is shared with ``_audit_pass_sample`` so the two paths
        # cannot disagree about whether the same batch's numbers are evidence.
        # It is applied per entry: a batch where the judge answered some pairs
        # keeps those opinions and stores ``None`` for the rest, rather than
        # promoting every unanswered entry's heuristic band to a "judgement".
        measured = _measured_indices(runner, len(suspicious_blocks)) if attempt else frozenset()
        if measured:
            scores: list[float | None] = [
                float(s) if index in measured else None for index, s in enumerate(raw_scores)
            ]
        else:
            # Bi-modal Gate Track 1: Structural Integrity.
            # When uncalibrated (HeuristicQERunner, or a COMET subprocess whose
            # torch import failed), do NOT fabricate continuous pseudo-scores.
            # Explicitly mark mtqe_score = None and route to REPAIR_PENDING based on structural defect flags.
            scores = [None] * len(suspicious_blocks)

        engine = qe_runner.score_source()
        if engine:
            # Honesty provenance: the audit artifact must say which engine
            # produced these numbers, because a discrete heuristic band and a
            # graded judge opinion read identically once stored as a float.
            # Every concrete runner implements ``score_source`` — the previous
            # probe asked for ``last_engine``, an attribute only the retired
            # CometKiwi subprocess ever set, so this label was always absent.
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
            prov = b.provenance
            anchor_prov = str(prov.anchor_provenance or "")
            needs_review = bool(prov.needs_review)
            stats = prov.anchor_stats
            if isinstance(stats, dict):
                matched = stats.get("matched", 0)
                vlm_only = stats.get("vlm_only", 0)
                pdfium_only = stats.get("pdfium_only", 0)
                total = matched + vlm_only + pdfium_only
                if total > 0 and (matched / total) >= 0.8 and not needs_review:
                    if score is not None:
                        score = min(1.0, score + 0.05)
                elif (
                    needs_review or (total > 0 and (vlm_only / total) > 0.5)
                ) and "Visual witness discrepancy" not in b.error_flags:
                    b.error_flags.append("Visual witness discrepancy")
            elif "proofread" in anchor_prov and not needs_review:
                if score is not None:
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

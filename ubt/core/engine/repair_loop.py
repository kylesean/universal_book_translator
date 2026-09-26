import asyncio
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

from ubt.core.exceptions import MTQEEvaluationError
from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.comet_runner import GLOSSARY_VIOLATION_MARKER
from ubt.core.qe.defect_taxonomy import is_format_only
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.router.router import ModelRouter
from ubt.core.validators.span_repair import MQMSpanAnnotator, SpanRepairSplicer

logger = logging.getLogger(__name__)


def _out_of_span_content_preserved(draft: str, spans: list[Any], candidate: str) -> bool:
    """Whether a wholesale <final_translation> kept the unflagged text intact.

    ``SpanRepairSplicer`` falls back to the model's whole-segment replacement
    when no per-span ``<correction>`` was spliced; the structural gates prove
    the text is *sound*, not that it is still the *same translation*. A
    character-multiset overlap over the draft's out-of-span remainder is a
    cheap, language-agnostic proxy for "nothing unflagged was rewritten".
    Short remainders (headers, lone terms re-render freely) never trip the
    guard.
    """
    cursor = 0
    pieces: list[str] = []
    for span in sorted(spans, key=lambda s: s.start_pos):
        if span.start_pos >= span.end_pos:
            continue  # insertion: removes no out-of-span text
        start = max(cursor, span.start_pos)
        end = max(cursor, min(span.end_pos, len(draft)))
        pieces.append(draft[cursor:start])
        cursor = end
    pieces.append(draft[cursor:])
    outside = Counter(ch for ch in "".join(pieces) if not ch.isspace())
    total = sum(outside.values())
    if total < 40:
        return True
    kept = sum((outside & Counter(ch for ch in candidate if not ch.isspace())).values())
    return kept / total >= 0.8


class RepairLoop:
    """Targeted repair orchestrator focusing expensive models strictly on degraded blocks."""

    def __init__(
        self,
        router: ModelRouter,
        qe_runner: BaseQERunner,
        max_rounds: int = 2,
        qe_threshold: float = 0.75,
        bottom_percentile: float = 0.15,
        fast_pass: FastPassFilter | None = None,
        escalation_extra_rounds: int = 1,
        rerank_k: int = 1,
    ) -> None:
        self.router = router
        self.qe_runner = qe_runner
        self.max_rounds = max_rounds
        self.qe_threshold = qe_threshold
        self.bottom_percentile = bottom_percentile
        self.fast_pass = fast_pass
        # MQM Critical escalation: flagship repair tier gets
        # extra rounds beyond the standard circuit breaker before a block is
        # declared BLOCKED_HUMAN.
        self.escalation_extra_rounds = escalation_extra_rounds
        # Shippable bar for a circuit-breaker exit. Floored at 0.5 so a
        # deliberately low qe_threshold cannot promote a barely-passable draft,
        # but it tracks the configured threshold upward so repair and triage
        # never disagree about what "shippable" means.
        self.terminal_pass_floor = max(0.5, qe_threshold)
        # Best-of-n repair candidates (MBR-lite). 1 = single candidate;
        # >1 samples and keeps the best structurally valid one, ranked by a
        # neural QE score.
        self.rerank_k = max(1, int(rerank_k))
        self.span_annotator = MQMSpanAnnotator()
        self.span_splicer = SpanRepairSplicer()

    def _rerank_enabled(self) -> bool:
        """Whether best-of-n reranking is active and meaningful.

        Ranking needs a utility that measures quality; the heuristic runner
        emits discrete defect classes, so it is excluded -- and so is a COMET
        subprocess whose reply says it fell back to the heuristic this run
        (``is_calibrated``), which an isinstance check could never see.
        """
        return self.rerank_k > 1 and self.qe_runner.is_calibrated()

    def select_repair_candidates(self, blocks: list[IRBlock]) -> list[IRBlock]:
        """Select blocks eligible for repair, capped strictly to bottom 15% and defect flags."""
        eligible: list[IRBlock] = []
        for b in blocks:
            if b.skip_translate or b.is_finalized:
                continue
            if b.repair_rounds >= self.max_rounds:
                continue
            eligible.append(b)

        if not eligible:
            return []

        # Find scored blocks
        scored = [b for b in eligible if b.mtqe_score is not None]
        if not scored:
            # If none scored, select blocks with error flags or REPAIR_PENDING
            unscored_candidates = [
                b for b in eligible if b.error_flags or b.status == BlockStatus.REPAIR_PENDING
            ]
            for c in unscored_candidates:
                c.status = BlockStatus.REPAIR_PENDING
            return unscored_candidates

        scored.sort(key=lambda x: x.mtqe_score if x.mtqe_score is not None else 0.0)
        cutoff_count = max(1, math.ceil(len(scored) * self.bottom_percentile))
        lowest_ids = {b.id for b in scored[:cutoff_count]}

        candidates: list[IRBlock] = []
        for b in eligible:
            score = b.mtqe_score or 0.0
            # Condition: low score within bottom percentile or below threshold, or severe error flags
            is_low_quality = (b.id in lowest_ids and score < self.qe_threshold) or (
                score < self.qe_threshold * 0.8
            )
            has_defects = bool(b.error_flags)
            if is_low_quality or has_defects:
                b.status = BlockStatus.REPAIR_PENDING
                candidates.append(b)

        return candidates

    async def repair_single_block(
        self,
        block: IRBlock,
        glossary_table: str = "",
        target_lang: str = "zh",
        source_lang: str = "en",
        fast_pass: FastPassFilter | None = None,
        glossary_entries: list[dict[str, Any]] | None = None,
        escalated: bool = False,
        source_pdf_path: Path | None = None,
    ) -> IRBlock:
        """Execute one targeted repair iteration with hard circuit breaker at max_rounds.

        With ``escalated=True`` (MQM Critical triage) the
        circuit breaker extends by ``escalation_extra_rounds`` and the reasoning
        effort is pinned to the strongest setting regardless of format-only
        shortcuts.
        """
        rounds_cap = self.max_rounds + (self.escalation_extra_rounds if escalated else 0)
        if block.repair_rounds >= rounds_cap:
            # Circuit breaker triggered. Same terminal criteria as the regular
            # path below: a shippable REPAIRED verdict additionally requires no
            # residual error_flags. Without this, a FAILED block escalated by
            # triage with stale "Repair error:" markers could be promoted to
            # REPAIRED (shippable) with its defects never re-checked.
            if (block.mtqe_score or 0.0) >= self.terminal_pass_floor and not block.error_flags:
                block.status = BlockStatus.REPAIRED
            else:
                block.status = BlockStatus.FAILED
            return block

        draft_text = block.target_text or block.draft_text or ""

        # Dynamic Test-Time Compute Allocation (2026 EACL):
        # Format-only defects bypass expensive CoT to avoid overthinking & hallucinations;
        # semantic/glossary/low-confidence defects receive targeted reasoning ("low").
        format_only = is_format_only(block.error_flags)
        score = block.mtqe_score or 0.0
        if escalated:
            effort = "high"
        elif format_only and (score >= 0.70 or block.mtqe_score is None):
            effort = "low"
        else:
            effort = self.router.repair_reasoning_effort

        # Fine-grained error span annotation (GEMBA-MQM Infilling Protocol)
        annotated_draft, error_spans = self.span_annotator.annotate_draft(
            source_text=block.source_text,
            draft_text=draft_text,
            error_flags=block.error_flags,
            glossary_entries=glossary_entries,
        )
        has_spans = bool(error_spans)

        # L4 Visual Scalpel precision cropping for hard blocks (formula, table, low QE)
        image_b64: str | None = None
        if source_pdf_path is not None:
            try:
                from ubt.core.ports import crop_block_image, is_visual_scalpel_applicable

                if is_visual_scalpel_applicable(block, source_pdf_path=source_pdf_path):
                    # Renders the page at 150 dpi and PNG-optimizes the crop: CPU
                    # work that would freeze the event loop (SSE, queue
                    # heartbeat) for every formula/table repair candidate.
                    image_b64 = await asyncio.to_thread(crop_block_image, source_pdf_path, block)
            except Exception as exc:
                logger.debug("Visual scalpel crop skipped for block %s: %s", block.id, exc)

        async def _repair_call() -> str:
            return await self.router.repair(
                block=block,
                draft_text=draft_text,
                error_flags=block.error_flags,
                glossary_table=glossary_table,
                target_lang=target_lang,
                source_lang=source_lang,
                reasoning_effort=effort,
                annotated_draft=annotated_draft,
                has_error_spans=has_spans,
                image_b64=image_b64,
            )

        # Best-of-n repair (MBR-lite): when enabled and the QE runner can rank,
        # sample k repairs concurrently and keep the best structurally-valid one.
        # k=1 evaluates a single repair candidate.
        if self._rerank_enabled():
            gathered = await asyncio.gather(
                *[_repair_call() for _ in range(self.rerank_k)],
                return_exceptions=True,
            )
            raw_candidates = [item for item in gathered if isinstance(item, str)]
            if not raw_candidates:
                failure = next((item for item in gathered if isinstance(item, BaseException)), None)
                if failure is not None:
                    raise failure
        else:
            raw_candidates = [await _repair_call()]

        block.repair_rounds += 1
        active_fast_pass = fast_pass or self.fast_pass

        # Closed-loop invariant check: 0-Token structural validation when a
        # filter is configured. Returns (valid, flags); valid implies no flags.
        def _structural_check(candidate_text: str) -> tuple[bool, list[str]]:
            if active_fast_pass is None:
                return True, []
            # evaluate() already runs validate_structural_invariants first and
            # returns that decision on structural failure, so calling both would be
            # a redundant double pass for every repair candidate. Run evaluate once; only on a *factual* failure do we
            # re-ask the structural check to classify the reason — and a
            # structural failure is recognized from evaluate's own early return
            # by running the cheap classifier once on the rare failure path.
            full_check = active_fast_pass.evaluate(
                block.source_text,
                candidate_text,
                block_type=block.block_type,
                skip_translate=block.skip_translate,
            )
            if full_check.passed:
                return True, []
            structural_check = active_fast_pass.validate_structural_invariants(
                block.source_text,
                candidate_text,
                block_type=block.block_type,
                skip_translate=block.skip_translate,
            )
            if not structural_check.passed:
                return False, [f"repair_structural_failure: {structural_check.reason}"]
            return False, [full_check.reason]

        # Tiered repair candidate collection:
        # Tier 1: Targeted in-place spliced repairs (preserving unflagged context)
        # Tier 2: Wholesale rewrites (fallback only when in-place repair fails/degrades,
        #         provided it strictly satisfies 0-Token structural invariants and scores higher)
        from ubt.core.validators.consistency import (
            GlossaryConsistencyValidator,
            NumericConsistencyValidator,
        )

        tier1_candidates: list[tuple[str, bool, list[str]]] = []
        tier2_candidates: list[tuple[str, bool, list[str]]] = []

        fast_pass_guard = active_fast_pass or FastPassFilter(
            source_lang=source_lang, target_lang=target_lang
        )
        num_validator = NumericConsistencyValidator()
        glossary_validator = (
            GlossaryConsistencyValidator(glossary=glossary_entries) if glossary_entries else None
        )

        for raw in raw_candidates:
            candidate_text, is_in_place = self.span_splicer.splice_repairs(
                original_draft=draft_text,
                spans=error_spans,
                model_output=raw,
            )
            valid, flags = _structural_check(candidate_text)
            is_preserved = (
                is_in_place
                or (not error_spans)
                or _out_of_span_content_preserved(draft_text, error_spans, candidate_text)
            )

            if is_preserved:
                tier1_candidates.append((candidate_text, valid, flags))
            else:
                ws_valid = valid
                ws_flags = list(flags)
                if ws_valid:
                    fp_dec = fast_pass_guard.evaluate(
                        block.source_text,
                        candidate_text,
                        block_type=block.block_type,
                        skip_translate=block.skip_translate,
                    )
                    if not fp_dec.passed:
                        ws_valid = False
                        ws_flags.append(f"repair_structural_failure: {fp_dec.reason}")

                    if ws_valid:
                        num_res = num_validator.validate(block.source_text, candidate_text)
                        if not num_res.is_valid:
                            ws_valid = False
                            ws_flags.append(f"repair_structural_failure: {num_res.message}")

                    if ws_valid and glossary_validator is not None:
                        gloss_res = glossary_validator.validate(block.source_text, candidate_text)
                        if not gloss_res.is_valid:
                            ws_valid = False
                            ws_flags.append(f"repair_structural_failure: {gloss_res.message}")

                if not ws_valid:
                    ws_flags.append(
                        "repair_overreach: wholesale replacement rewrote out-of-span content"
                    )
                tier2_candidates.append((candidate_text, ws_valid, ws_flags))

        valid_tier1 = [entry for entry in tier1_candidates if entry[1]]
        valid_tier2 = [entry for entry in tier2_candidates if entry[1]]
        old_score = block.mtqe_score or 0.0

        chosen_text: str
        _chosen_flags: list[str]
        new_score: float | None = None

        if valid_tier1:
            if self._rerank_enabled() and len(valid_tier1) >= 2:
                scores = await self.qe_runner.score_pairs(
                    [{"src": block.source_text, "mt": entry[0]} for entry in valid_tier1]
                )
                if len(scores) != len(valid_tier1):
                    raise MTQEEvaluationError(
                        f"QE runner returned {len(scores)} score(s) for "
                        f"{len(valid_tier1)} repair candidate(s)",
                        details={"expected": len(valid_tier1), "got": len(scores)},
                    )
                best_t1_idx = max(range(len(valid_tier1)), key=lambda i: scores[i])
                t1_chosen_text, _, t1_chosen_flags = valid_tier1[best_t1_idx]
                t1_score: float | None = scores[best_t1_idx]
            else:
                t1_chosen_text, _, t1_chosen_flags = valid_tier1[0]
                t1_score = await self.qe_runner.score(block.source_text, t1_chosen_text)

            t1_passed = (t1_score is not None) and (
                t1_score >= self.qe_threshold or t1_score >= old_score
            )

            if t1_passed or not valid_tier2:
                chosen_text, _chosen_flags, new_score = t1_chosen_text, t1_chosen_flags, t1_score
            else:
                # Spliced repair degraded quality; evaluate wholesale fallback
                if self._rerank_enabled() and len(valid_tier2) >= 2:
                    scores_t2 = await self.qe_runner.score_pairs(
                        [{"src": block.source_text, "mt": entry[0]} for entry in valid_tier2]
                    )
                    best_t2_idx = max(range(len(valid_tier2)), key=lambda i: scores_t2[i])
                    t2_chosen_text, _, t2_chosen_flags = valid_tier2[best_t2_idx]
                    t2_score: float | None = scores_t2[best_t2_idx]
                else:
                    t2_chosen_text, _, t2_chosen_flags = valid_tier2[0]
                    t2_score = await self.qe_runner.score(block.source_text, t2_chosen_text)

                if (
                    t2_score is not None
                    and (t2_score >= self.qe_threshold or t2_score > old_score)
                    and (t1_score is None or t2_score > t1_score)
                ):
                    chosen_text, _chosen_flags, new_score = (
                        t2_chosen_text,
                        t2_chosen_flags,
                        t2_score,
                    )
                else:
                    chosen_text, _chosen_flags, new_score = (
                        t1_chosen_text,
                        t1_chosen_flags,
                        t1_score,
                    )

        elif valid_tier2:
            if self._rerank_enabled() and len(valid_tier2) >= 2:
                scores_t2 = await self.qe_runner.score_pairs(
                    [{"src": block.source_text, "mt": entry[0]} for entry in valid_tier2]
                )
                if len(scores_t2) != len(valid_tier2):
                    raise MTQEEvaluationError(
                        f"QE runner returned {len(scores_t2)} score(s) for "
                        f"{len(valid_tier2)} repair candidate(s)",
                        details={"expected": len(valid_tier2), "got": len(scores_t2)},
                    )
                best_t2_idx = max(range(len(valid_tier2)), key=lambda i: scores_t2[i])
                t2_chosen_text, _, t2_chosen_flags = valid_tier2[best_t2_idx]
                t2_score = scores_t2[best_t2_idx]
            else:
                t2_chosen_text, _, t2_chosen_flags = valid_tier2[0]
                t2_score = await self.qe_runner.score(block.source_text, t2_chosen_text)

            if t2_score is not None and (t2_score >= self.qe_threshold or t2_score > old_score):
                chosen_text, _chosen_flags, new_score = t2_chosen_text, t2_chosen_flags, t2_score
            else:
                chosen_text = draft_text
                _chosen_flags = ["repair_overreach: wholesale replacement failed quality threshold"]
                new_score = None
        else:
            all_fallback = tier1_candidates + tier2_candidates
            if all_fallback:
                chosen_text, _, _chosen_flags = all_fallback[0]
            else:
                chosen_text, _chosen_flags = (
                    draft_text,
                    ["repair_structural_failure: no repair candidate"],
                )
            new_score = None

        if new_score is not None:
            # Clearing error_flags claims the defects are gone, so it needs
            # evidence the scorer actually produced: either a better class than
            # the draft, or a candidate at/above the pass line. Every structural
            # flag caps its band at 0.70 (``score_from_flags``), so a candidate
            # that ties the draft at 0.70 had neither -- and the old ``>=`` did
            # both: it adopted the text (right: the consistency stage fixes
            # terminology drift no QE band measures) *and* wiped the evidence,
            # leaving a block whose defect was never re-checked to read as clean
            # and be re-accepted by the next resume's FastPass.
            cleaned = new_score > old_score or new_score >= self.qe_threshold
            if new_score >= old_score or new_score >= self.qe_threshold:
                block.target_text = chosen_text
                block.mtqe_score = new_score
                if cleaned:
                    # Cleared only upon passing both structural and semantic
                    # criteria. A glossary-violation marker survives ONLY when
                    # the scorer cannot see terminology: COMET / LLM judge score
                    # fluency, so a high re-score is no evidence the dropped or
                    # aliased term is back, and the flag must keep the block out
                    # of REPAIRED. A glossary-aware runner re-checks the term
                    # itself (a surviving violation caps the score below the
                    # pass line, so ``cleaned`` is False), which means its clean
                    # re-score *is* evidence — clearing the marker there is what
                    # lets a genuinely fixed term reach REPAIRED.
                    if self.qe_runner.is_glossary_aware() and new_score >= self.qe_threshold:
                        # A glossary-aware re-score is evidence only when it
                        # reaches the pass line. A surviving violation caps the
                        # score at its own band (0.25), which can still exceed a
                        # *lower* hard-defect band carried by the draft (leak
                        # 0.10, fabricated 0.15, repetition 0.20) — so `cleaned`
                        # alone is not evidence and must not erase the violation
                        # before triage can classify it Major.
                        block.error_flags = []
                    else:
                        block.error_flags = [
                            f for f in block.error_flags if f.startswith(GLOSSARY_VIOLATION_MARKER)
                        ]
        else:
            # The fallback keeps the draft, so every defect the draft already
            # carried is still present: merge the rejection reason into the
            # existing flags instead of replacing them. Replacing dropped a
            # Critical marker (e.g. "Numeric fidelity failure") and left only
            # the Minor ``repair_overreach`` note, so triage resolved the block
            # to "minor" and the gate could ship a draft that lost a figure.
            merged = list(block.error_flags)
            for flag in _chosen_flags or [
                "repair_structural_failure: no structurally valid repair"
            ]:
                if flag not in merged:
                    merged.append(flag)
            block.error_flags = merged

        # Update lifecycle status
        if (block.mtqe_score or 0.0) >= self.qe_threshold and not block.error_flags:
            block.status = BlockStatus.REPAIRED
        elif block.repair_rounds >= rounds_cap:
            block.status = (
                BlockStatus.REPAIRED
                if ((block.mtqe_score or 0.0) >= self.terminal_pass_floor and not block.error_flags)
                else BlockStatus.FAILED
            )
        else:
            block.status = BlockStatus.REPAIR_PENDING

        return block

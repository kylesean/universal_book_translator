"""Stage 5.5: MQM severity triage with human PE (HITL) queue routing.

After the targeted repair stage, unresolved blocks (REPAIR_PENDING leftovers
plus repair-path FAILED blocks) are classified by MQM severity derived from
deterministic error-span annotation and error flags:

- **Critical** (numeric distortion, unresolved structural corruption) → one
  escalated repair round on the flagship tier (extended circuit breaker +
  ``high`` reasoning effort). If still unresolved the block is quarantined as
  ``BLOCKED_HUMAN`` and its machine draft is stripped from the rendered output
  (source-only placeholder) — nothing unreviewed ships. **Critical escape
  rate: 0.**
- **Major** (terminology violations, generic repair failures) →
  ``NEEDS_HUMAN``: the draft is preserved for human post-editing via the PE
  queue (CSV / XLIFF 2.1 export).
- **Minor** → auto-pass when the QE score permits, otherwise conservative
  escalation to ``NEEDS_HUMAN``.

Severity and serialized spans are persisted on the ledger (``mqm_severity`` /
``mqm_spans_json``) so exporters and auditors can rely on them later.
"""

import asyncio
import html
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.stage_context import StageContext
from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.core.memory.glossary_table import build_chunk_glossary_table
from ubt.core.qe.defect_taxonomy import (
    FLAG_MQM_CRITICAL_BLOCKED as _FLAG_MQM_CRITICAL_BLOCKED,
)
from ubt.core.qe.defect_taxonomy import (
    FLAG_NEEDS_HUMAN_REVIEW as _FLAG_NEEDS_HUMAN_REVIEW,
)
from ubt.core.qe.defect_taxonomy import (
    has_critical_defect,
    has_structural_defect,
    is_transient_lifecycle_only,
)
from ubt.core.validators.span_repair import max_severity, span_to_dict

logger = logging.getLogger(__name__)

#: Heartbeat cadence for the per-block loops below: at most one progress event
#: per N blocks, and at least one per interval seconds to keep subscribers informed during long fan-outs.
TRIAGE_PROGRESS_EVERY_N = 20
TRIAGE_PROGRESS_INTERVAL_S = 5.0


class _ProgressThrottle:
    """Decide when a long per-block loop may spend one progress event.

    Emitting per block would flood every consumer (the CLI prints one line per
    event), so a heartbeat is due after ``every`` units *or* once ``interval``
    seconds have passed since the previous one, whichever comes first.
    """

    def __init__(
        self,
        every: int = TRIAGE_PROGRESS_EVERY_N,
        interval: float = TRIAGE_PROGRESS_INTERVAL_S,
    ) -> None:
        self._every = max(1, every)
        self._interval = interval
        self._since = 0
        self._last = time.monotonic()

    def due(self) -> bool:
        """True when this unit may spend a progress event; consumes the slot."""
        self._since += 1
        if self._since >= self._every or (time.monotonic() - self._last) >= self._interval:
            self._since = 0
            self._last = time.monotonic()
            return True
        return False


def _flags_severity(flags: list[str]) -> str:
    """Severity floor from the shared defect taxonomy.

    Critical: unresolved structural/factual corruption (protected-token loss,
    numeric fidelity, omission, fabrication, repetition) — the draft cannot be
    trusted and must be quarantined if escalation fails. Major: other
    structural defects (terminology, format, provenance). Minor: everything
    else. This is the contract the module docstring states.
    """
    if has_critical_defect(flags):
        return "critical"
    if has_structural_defect(flags):
        return "major"
    return "minor"


def _needs_human_flags(block: IRBlock) -> list[str]:
    """Error flags for a block escalated to NEEDS_HUMAN (idempotent append)."""
    flags = list(block.error_flags)
    if _FLAG_NEEDS_HUMAN_REVIEW not in flags:
        flags.append(_FLAG_NEEDS_HUMAN_REVIEW)
    return flags


def _blocked_human_target(source_text: str) -> str:
    """Quarantine placeholder: source text only, no machine output ships."""
    return (
        '<mark class="ubt-blocked-human" title="MQM Critical unresolved after escalated '
        'repair; quarantined for human post-editing">'
        f"【待人工审校 | Human review required】{html.escape(source_text)}</mark>"
    )


async def run_triage_stage(
    ctx: StageContext,
) -> AsyncIterator[TranslationProgressEvent]:
    """Classify unresolved blocks by MQM severity and route them accordingly."""
    ledger = ctx.ledger
    actual_job_id = ctx.job_id
    repair_loop = ctx.repair_loop
    glossary_dicts = ctx.glossary_dicts
    abbreviation_entries = ctx.abbreviation_entries
    target_lang = ctx.target_lang
    source_lang = ctx.source_lang
    fast_pass = ctx.fast_pass
    concurrency_sem = ctx.concurrency_sem
    create_event_fn = ctx.create_event
    qe_threshold = ctx.config.qe_threshold
    repair_pending = await asyncio.to_thread(
        ledger.fetch_blocks_by_status, actual_job_id, BlockStatus.REPAIR_PENDING
    )
    failed = await asyncio.to_thread(
        ledger.fetch_blocks_by_status, actual_job_id, BlockStatus.FAILED
    )
    candidates: list[IRBlock] = [
        b
        for b in [*repair_pending, *failed]
        if not b.skip_translate
        # A transient drafting/repair failure is a provider problem, not a
        # quality verdict: leave it FAILED/REPAIR_PENDING so resume can retry it
        # (see ``is_transient_lifecycle_only``). Permanent "Drafting
        # unrecoverable" failures are still triaged.
        and not is_transient_lifecycle_only(b.error_flags)
        # FAILED is terminal per IRBlock.is_finalized, but this stage is the
        # Documented last net for repair-path failures: classify
        # them into the human queue instead of leaving raw FAILED blocks.
        and b.status in (BlockStatus.REPAIR_PENDING, BlockStatus.FAILED)
    ]
    if not candidates:
        return

    counters = {
        "critical_resolved": 0,
        "critical_blocked": 0,
        "needs_human": 0,
        "auto_passed": 0,
    }

    async def _progress_heartbeat(message: str) -> TranslationProgressEvent:
        """One throttled in-stage progress event on the shared channels.

        Reuses ``EventType.REPAIR_BATCH_COMPLETED`` (the escalated repair this
        stage runs *is* repair; adding a TRIAGE_PROGRESS enum would widen the
        published event contract for one internal heartbeat) and reaches the
        consumer through the generator's own yield, exactly like the batch
        emits of the draft stage. Counters on the event still come from the
        real stats query, so the heartbeat can never misreport ledger state —
        only ``message`` carries the loop progress.
        """
        event = await create_event_fn(
            EventType.REPAIR_BATCH_COMPLETED,
            actual_job_id,
            ledger,
            message=message,
        )
        return event

    triaged: list[tuple[IRBlock, str, list[dict[str, Any]]]] = []
    critical: list[IRBlock] = []
    classify_tick = _ProgressThrottle()
    for idx, cand in enumerate(candidates, start=1):
        draft_text = cand.target_text or cand.draft_text or ""
        _, spans = repair_loop.span_annotator.annotate_draft(
            source_text=cand.source_text,
            draft_text=draft_text,
            error_flags=cand.error_flags,
            glossary_entries=glossary_dicts,
        )
        span_dicts = [span_to_dict(s) for s in spans]
        severity = max_severity([*(s.severity for s in spans), _flags_severity(cand.error_flags)])
        triaged.append((cand, severity, span_dicts))
        if severity == "critical":
            critical.append(cand)
        if idx < len(candidates) and classify_tick.due():
            yield await _progress_heartbeat(
                f"MQM triage: classified {idx}/{len(candidates)} blocks"
            )

    # ------------------------------------------------------------------
    # Critical: one escalated repair attempt on the flagship tier before
    # quarantine. Uses the same per-block glossary tables as the repair
    # stage and extends the circuit breaker by escalation_extra_rounds.
    # ------------------------------------------------------------------
    async def _escalate(cand: IRBlock) -> dict[str, Any]:
        glossary_table = build_chunk_glossary_table(
            glossary_dicts, abbreviation_entries, cand.source_text
        )
        async with concurrency_sem:
            repaired = await repair_loop.repair_single_block(
                block=cand,
                glossary_table=glossary_table,
                target_lang=target_lang,
                source_lang=source_lang,
                fast_pass=fast_pass,
                glossary_entries=glossary_dicts,
                escalated=True,
            )
        return {
            "block_id": repaired.id,
            "target_text": repaired.target_text or "",
            "status": repaired.status,
            "mtqe_score": repaired.mtqe_score,
            "repair_rounds": repaired.repair_rounds,
            "error_flags": repaired.error_flags,
        }

    escalation_results: dict[str, dict[str, Any]] = {}
    escalation_errors: dict[str, str] = {}
    if critical:
        # Drive the fan-out task by task instead of one opaque ``gather`` so
        # the long silent stretch (E2E: 4m56s between REPAIR_BATCH_COMPLETED
        # and TRIAGE_COMPLETED while 310 escalated repairs ran) can spend a
        # throttled heartbeat between completions.
        tasks: dict[asyncio.Task[dict[str, Any]], IRBlock] = {
            asyncio.create_task(_escalate(c)): c for c in critical
        }
        pending: set[asyncio.Task[dict[str, Any]]] = set(tasks)
        finished = 0
        escalate_tick = _ProgressThrottle()
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    cand = tasks[task]
                    finished += 1
                    if task.cancelled():
                        escalation_errors[cand.id] = "escalation cancelled"
                        logger.warning("Escalated repair cancelled for block %s", cand.id)
                        continue
                    exc = task.exception()
                    if exc is None:
                        result = task.result()
                        escalation_results[cand.id] = result
                        # Persist the paid repair as soon as it finishes. The
                        # final batch runs only after every critical block, so a
                        # cancel/crash in between would drop it and re-escalate
                        # (re-bill) the same block on resume. MQM fields are
                        # added by that final batch.
                        await asyncio.to_thread(ledger.save_checkpoints_batch, [result])
                    else:
                        # The block still lands in BLOCKED_HUMAN below, but without a
                        # marker nobody can tell "escalation never ran" apart from
                        # "escalation ran and failed" in the PE queue.
                        escalation_errors[cand.id] = str(exc)[:200]
                        logger.warning("Escalated repair failed for block %s: %s", cand.id, exc)
                if pending and escalate_tick.due():
                    yield await _progress_heartbeat(
                        f"MQM triage: escalated repair {finished}/{len(critical)} blocks"
                    )
        finally:
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    # ------------------------------------------------------------------
    # Persist triage decisions (single transactional batch).
    # ------------------------------------------------------------------
    updates: list[dict[str, Any]] = []
    for cand, severity, span_dicts in triaged:
        mqm_update: dict[str, Any] = {
            "mqm_severity": severity,
            "mqm_spans": span_dicts,
        }

        if severity == "critical":
            escalation = escalation_results.get(cand.id)
            if escalation is not None and escalation["status"] == BlockStatus.REPAIRED:
                counters["critical_resolved"] += 1
                updates.append(
                    {
                        "block_id": cand.id,
                        "status": BlockStatus.REPAIRED,
                        "target_text": escalation["target_text"],
                        "mtqe_score": escalation["mtqe_score"],
                        "repair_rounds": escalation["repair_rounds"],
                        "error_flags": escalation["error_flags"],
                        **mqm_update,
                    }
                )
                continue
            counters["critical_blocked"] += 1
            flags = list(cand.error_flags)
            if cand.id in escalation_errors:
                flags.append(f"escalation error: {escalation_errors[cand.id]}")
            if _FLAG_MQM_CRITICAL_BLOCKED not in flags:
                flags.append(_FLAG_MQM_CRITICAL_BLOCKED)
            updates.append(
                {
                    "block_id": cand.id,
                    "status": BlockStatus.BLOCKED_HUMAN,
                    "target_text": _blocked_human_target(cand.source_text),
                    "mtqe_score": cand.mtqe_score,
                    "repair_rounds": cand.repair_rounds,
                    "error_flags": flags,
                    **mqm_update,
                }
            )
            continue

        if severity == "major":
            counters["needs_human"] += 1
            updates.append(
                {
                    "block_id": cand.id,
                    "status": BlockStatus.NEEDS_HUMAN,
                    "mtqe_score": cand.mtqe_score,
                    "repair_rounds": cand.repair_rounds,
                    "error_flags": _needs_human_flags(cand),
                    **mqm_update,
                }
            )
            continue

        # Minor: auto-pass when the QE score is shippable, otherwise keep the
        # draft for human review (never silently ship low-score output). The
        # floor tracks the run's qe_threshold so tightening it (e.g. 0.9 with
        # a neural judge) actually holds triage to the same bar.
        if (cand.mtqe_score or 0.0) >= max(0.5, qe_threshold):
            counters["auto_passed"] += 1
            updates.append(
                {
                    "block_id": cand.id,
                    "status": BlockStatus.MTQE_PASSED,
                    "mtqe_score": cand.mtqe_score,
                    "repair_rounds": cand.repair_rounds,
                    "error_flags": cand.error_flags,
                    **mqm_update,
                }
            )
        else:
            counters["needs_human"] += 1
            updates.append(
                {
                    "block_id": cand.id,
                    "status": BlockStatus.NEEDS_HUMAN,
                    "mtqe_score": cand.mtqe_score,
                    "repair_rounds": cand.repair_rounds,
                    "error_flags": _needs_human_flags(cand),
                    **mqm_update,
                }
            )

    await asyncio.to_thread(ledger.save_checkpoints_batch, updates)
    logger.info(
        "MQM triage for job %s: %d candidates -> %d critical-resolved, %d blocked-human, "
        "%d needs-human, %d auto-passed",
        actual_job_id,
        len(candidates),
        counters["critical_resolved"],
        counters["critical_blocked"],
        counters["needs_human"],
        counters["auto_passed"],
    )

    event = await create_event_fn(
        EventType.TRIAGE_COMPLETED,
        actual_job_id,
        ledger,
        message=(
            f"MQM severity triage: {counters['critical_resolved']} critical resolved, "
            f"{counters['critical_blocked']} blocked for human, "
            f"{counters['needs_human']} needs human, {counters['auto_passed']} auto-passed"
        ),
    )
    yield event

"""Stage 6.5: promote this run's accepted translations into the shared TM.

The translation memory is shared across jobs and books, so writing to it is the
one place where a run can affect *other* runs. Two rules follow from that:

- a mock run never writes: its ``[模拟翻译]`` text is not a translation, and
  simulated rows would pollute the TM that every other job reads;
- a block only qualifies when the added-content gate clears it, because a
  terminal status is not a correctness proof — ``MTQE_PASSED`` can mean "no
  structural defect was registered for this failure mode".
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ubt.core.engine.facts import Terminology
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.services import RunServices
from ubt.core.engine.stage_context import StageContext
from ubt.core.ir.models import BlockStatus
from ubt.core.memory.glossary_table import build_global_glossary_table
from ubt.core.memory.tm import (
    PROVENANCE_MACHINE,
    TMPendingEntry,
    TranslationMemory,
)
from ubt.core.qe.added_content import AddedContentGate

logger = logging.getLogger(__name__)

#: Only accepted translations are promoted; a quarantined or failed block must
#: not become a few-shot example for the next book.
TM_WRITEBACK_STATUSES = frozenset({BlockStatus.MTQE_PASSED, BlockStatus.REPAIRED})

_added_content_gate = AddedContentGate()


def tm_writeback_eligible(block: Any) -> bool:
    """Whether a finalized block may be promoted into the shared TM."""
    if block.skip_translate:
        # A skip row echoes its source, and the read side permanently refuses
        # identity rows — storing one is dead weight in the fuzzy candidate
        # pool and inflates the entry count for every later lookup.
        return False
    draft = tm_writeback_text(block)
    decision = _added_content_gate.evaluate(block.source_text or "", draft)
    if decision.passed:
        return True
    logger.warning(
        "Refusing to write TM entry for block %s: %s",
        getattr(block, "id", "?"),
        decision.reason,
    )
    return False


def tm_writeback_text(block: Any) -> str:
    """The text a TM entry should carry: the draft, not the shipped artifact.

    The writeback runs after export, and export rewrites ``target_text`` in
    place — CJK spacing normalization, opt-in glossary enforcement, and the
    ``<mark>`` wrapper the HTML-delta validator adds to a failed draft. Those
    are typesetting products of *this* run, not translations: storing them makes
    the markup (and the flag it encodes) the few-shot example for every later
    book, so a defect would propagate across runs instead of being re-detected.

    ``draft_text`` is the first-draft backup the draft stage writes
    (``stages/draft.py``) and export never touches, so it is the run's own
    translation as produced. Falling back to ``target_text`` covers blocks from
    a resume whose draft column predates the backup.
    """
    if getattr(block, "status", None) == BlockStatus.REPAIRED:
        return str(getattr(block, "target_text", "") or "")
    return str(getattr(block, "draft_text", None) or getattr(block, "target_text", "") or "")


def writeback_tm_from_ledger(
    ledger: SQLiteJobLedger,
    job_id: str,
    tm: TranslationMemory,
    source_lang: str,
    target_lang: str,
    tm_context: str = "",
    domain: str | None = None,
) -> int:
    """Flush finalized block translations into the shared TM (best-effort)."""
    finalized = [
        b
        for b in ledger.fetch_blocks_by_statuses(job_id, TM_WRITEBACK_STATUSES)
        if tm_writeback_text(b) and tm_writeback_eligible(b)
    ]
    if not finalized:
        return 0
    entries = [
        TMPendingEntry(
            src_lang=source_lang,
            tgt_lang=target_lang,
            source_text=b.source_text,
            # The draft, deliberately: see tm_writeback_text. Raises nothing
            # because the filter above already established it is non-empty.
            target_text=tm_writeback_text(b),
            provenance=PROVENANCE_MACHINE,
            domain=domain,
            context_hash=tm_context,
        )
        for b in finalized
    ]
    return tm.writeback(entries)


async def run_tm_writeback_stage(
    ctx: StageContext, services: RunServices, terminology: Terminology
) -> None:
    """Flush this run's accepted blocks into the shared TM. Yields no events."""
    from ubt.core.memory.abbreviation_miner import format_abbreviations_markdown_table
    from ubt.core.memory.tm import PROMPT_VERSION, compute_tm_context

    tm = services.tm
    if tm is None or ctx.is_mock_run:
        return
    try:
        # Same context fingerprint as the draft-stage lookup. domain stays None
        # so context_hash is the only machine gate: a prompt/glossary/profile
        # change invalidates old entries instead of being overridden by a
        # domain match.
        writeback_context = compute_tm_context(
            PROMPT_VERSION,
            ctx.profile_name,
            # The capped sheet the draft prompts actually carried — hashing the
            # uncapped table here would let a glossary change go undetected by
            # the TM fingerprint.
            build_global_glossary_table(
                terminology.glossary_dicts, ctx.config.glossary_max_global_entries
            ),
            ctx.source_lang,
            ctx.target_lang,
            # Same abbreviation channel the draft stage hashed.
            format_abbreviations_markdown_table(terminology.abbreviation_entries),
        )
        written = await asyncio.to_thread(
            writeback_tm_from_ledger,
            ctx.ledger,
            ctx.job_id,
            tm,
            ctx.source_lang,
            ctx.target_lang,
            writeback_context,
            # domain=None: context_hash is the sole machine gate, so a changed
            # glossary / prompt invalidates old hits.
            None,
        )
        if written:
            logger.info("TM writeback stored %d entries for job %s", written, ctx.job_id)
    except Exception as exc:
        logger.warning("TM writeback failed for job %s: %s", ctx.job_id, exc)

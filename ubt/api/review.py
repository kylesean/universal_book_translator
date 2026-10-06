"""Review-workbench helpers: segment projection and human post-editing.

The L3 workbench shows the finalized blocks of a job, groups their defects for
the fault ribbon, and applies one human revision at a time. Two rules keep it
consistent with the rest of the engine:

* defect grouping is derived from the *shared* marker tables in
  :mod:`ubt.core.qe.defect_taxonomy`, not a second hand-written list, so a
  block the gate calls a terminology defect is grouped the same way here;
* a human revision goes through the exact write path the file-based PE import
  uses (:func:`ubt.core.engine.pe_import.import_pe_revisions`): the ledger row is
  promoted to ``REPAIRED`` with a ``human_pe_imported`` flag, its stale machine
  verdict is cleared in the same transaction, and the accepted pair is fed back
  into the shared TM with ``provenance='human_pe'`` — the one provenance machine
  writeback can never downgrade.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.exceptions import LedgerWriterLockConflictError
from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.core.memory.tm import PROVENANCE_HUMAN_PE, TMPendingEntry, TranslationMemory
from ubt.core.qe.defect_taxonomy import (
    ECHO_MARKERS,
    FLAG_MQM_CRITICAL_BLOCKED,
    FLAG_NEEDS_HUMAN_REVIEW,
    INTENTIONAL_PRESERVED_SKIP_PREFIXES,
)

logger = logging.getLogger(__name__)

#: The flag a human revision carries (same literal the file-based PE import
#: writes, so the two human-edit channels are indistinguishable downstream).
FLAG_HUMAN_PE_IMPORTED = "human_pe_imported"

#: Workbench issue kind -> the engine's own defect markers. Grouping is by
#: marker *substring*, matching ``defect_taxonomy``'s helpers.
_ISSUE_MARKERS: dict[str, tuple[str, ...]] = {
    "terminology": ("Glossary term violation",),
    "formula": (
        "Math span mismatch",
        "Undelimited math",
        "Hallucinated LaTeX",
        "math_token_corrupt",
        "soup_token_corrupt",
        "Visual witness discrepancy",
    ),
    "numeric": ("Numeric fidelity",),
    "omission": (
        "Omission suspected",
        "Table dropped",
        "Empty target text",
        "Target text suspiciously truncated",
    ),
    "fabrication": (
        "Added reference",
        "Prompt scaffold",
        "Prompt template XML artifacts",
        "Target text suspiciously inflated",
    ),
    "repetition": ("Repetitive loop hallucination", "loop hallucination", "repetition"),
    "structure": (
        "html_tag_mismatch",
        "HTML delta failure",
        "Table grid mismatch",
        "repair_structural_failure",
    ),
    "echo": ECHO_MARKERS,
    "review": (FLAG_NEEDS_HUMAN_REVIEW,),
    "critical": (FLAG_MQM_CRITICAL_BLOCKED,),
}

#: Prefixes that mark a fail-closed render skip (a translation that could not be
#: placed). The intentional-preserve prefixes are excluded — those are chrome /
#: non-prose kept on purpose, not faults.
_RENDER_SKIP_PREFIXES = ("render_skip:", "inplace_skip:")

#: Every issue kind, in ribbon display order.
ISSUE_KINDS: tuple[str, ...] = (*_ISSUE_MARKERS, "render")


def segment_issue_kinds(block: IRBlock) -> list[str]:
    """Classify a block's flags into workbench issue kinds (order-stable)."""
    kinds: list[str] = []
    for flag in block.error_flags or ():
        if not flag:
            continue
        if any(flag.startswith(prefix) for prefix in INTENTIONAL_PRESERVED_SKIP_PREFIXES):
            continue
        matched = False
        for kind, markers in _ISSUE_MARKERS.items():
            if any(marker in flag for marker in markers):
                if kind not in kinds:
                    kinds.append(kind)
                matched = True
                break
        if not matched and flag.startswith(_RENDER_SKIP_PREFIXES) and "render" not in kinds:
            kinds.append("render")
    return kinds


def serialize_segment(block: IRBlock) -> dict[str, Any]:
    """Project a block into the workbench's segment shape."""
    bbox = block.bbox
    return {
        "block_id": block.id,
        "spine_index": block.spine_index,
        "page": bbox.page if bbox is not None else None,
        "block_type": block.block_type.value,
        "status": block.status.value,
        "source_text": block.source_text or "",
        "target_text": block.target_text or "",
        "mtqe_score": block.mtqe_score,
        "error_flags": list(block.error_flags or ()),
        "issues": segment_issue_kinds(block),
        "human_verified": FLAG_HUMAN_PE_IMPORTED in (block.error_flags or ()),
    }


def segment_matches_filter(block: IRBlock, status_filter: str | None) -> bool:
    """Whether a block belongs in a workbench view.

    ``status_filter`` is one of ``all``/``None`` (everything), ``issues``
    (anything the ribbon counts, plus PE-queue members), or a concrete
    ``BlockStatus`` value.
    """
    if status_filter in (None, "", "all"):
        return True
    if status_filter == "issues":
        return bool(segment_issue_kinds(block)) or block.status in (
            BlockStatus.NEEDS_HUMAN,
            BlockStatus.BLOCKED_HUMAN,
        )
    return block.status.value == status_filter


class ReviewEditError(ValueError):
    """A human edit could not be applied (empty text)."""


class ReviewBlockNotFound(ReviewEditError):
    """The block id is unknown to this job's ledger."""


class ReviewEditConflict(ReviewEditError):
    """The job's ledger is locked by another writer (e.g. a running resume)."""


def apply_human_edit(
    db_path: Path,
    job_id: str,
    block_id: str,
    target_text: str,
    *,
    tm_path: Path | None = None,
) -> dict[str, Any]:
    """Apply one human revision to the ledger and (optionally) the shared TM.

    Takes the job's writer lock, so an interactive edit cannot race a concurrent
    resume. An unchanged revision is a no-op (``changed=False``). The language
    pair for the TM entry is read from the ledger — a caller-supplied default
    would file human verdicts under the wrong pair permanently.
    """
    text = target_text.strip()
    if not text:
        raise ReviewEditError("target_text must be non-empty")

    ledger = SQLiteJobLedger(db_path)
    lock = LedgerWriterLock(db_path, job_id)
    tm: TranslationMemory | None = None
    try:
        try:
            lock.acquire()
        except LedgerWriterLockConflictError as exc:
            raise ReviewEditConflict(f"job is being written by another process: {exc}") from exc

        block = ledger.get_block(block_id, job_id=job_id)
        if block is None:
            raise ReviewBlockNotFound(f"unknown block: {block_id}")
        if text == (block.target_text or ""):
            return {"block_id": block_id, "changed": False, "tm_written": 0}

        # One transaction: the human text and the cleared machine verdict land
        # together (see import_pe_revisions for the same reasoning).
        ledger.save_checkpoints_batch(
            [
                {
                    "block_id": block_id,
                    "status": BlockStatus.REPAIRED,
                    "target_text": text,
                    "error_flags": [FLAG_HUMAN_PE_IMPORTED],
                }
            ],
            clear_verdict_for=[block_id],
            job_id=job_id,
        )

        tm_written = 0
        if tm_path is not None and block.source_text:
            src = ledger.get_job_metadata_value(job_id, "source_lang")
            tgt = ledger.get_job_target_lang(job_id)
            if src and tgt:
                tm = TranslationMemory(tm_path)
                tm_written = tm.writeback(
                    [
                        TMPendingEntry(
                            src_lang=str(src),
                            tgt_lang=str(tgt),
                            source_text=block.source_text,
                            target_text=text,
                            provenance=PROVENANCE_HUMAN_PE,
                        )
                    ]
                )
            else:
                logger.warning(
                    "Human edit on %s: TM write skipped, language pair unresolved", block_id
                )

        return {"block_id": block_id, "changed": True, "tm_written": tm_written}
    finally:
        if tm is not None:
            tm.close()
        lock.release()
        ledger.close()

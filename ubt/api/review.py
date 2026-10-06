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
from collections import Counter
from pathlib import Path
from typing import Any

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.bible import _BIBLE_CACHE_KEY
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
from ubt.core.qe.term_drift import detect_target_term_violations, replace_term_surface

logger = logging.getLogger(__name__)

#: Upper bound on how many blocks one cascade may rewrite. A runaway surface
#: (a term whose "wrong" form is a common word) is bounded rather than allowed
#: to rewrite an entire book in one click; the caller is told how many were
#: capped so the UI can be honest about it.
MAX_CASCADE_BLOCKS = 5000

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


# --------------------------------------------------------------------------- #
# Global term propagation (PRD §5.2.2)
# --------------------------------------------------------------------------- #


def job_glossary(ledger: SQLiteJobLedger, job_id: str) -> list[dict[str, Any]]:
    """The glossary a job actually enforced, from its persisted bible cache.

    The run's effective glossary (external file + curated seeds + mined terms)
    is stored under ``bible_cache``; using it here — rather than re-reading
    ``config.glossary_path`` — means the workbench recommends exactly the
    renderings the quality gate judged, so a fix cannot disagree with the flag.
    """
    payload = ledger.get_job_metadata_value(job_id, _BIBLE_CACHE_KEY)
    if not isinstance(payload, dict):
        return []
    return [dict(entry) for entry in payload.get("glossary_dicts", []) if isinstance(entry, dict)]


def block_term_violations(block: IRBlock, glossary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Offending term surfaces in one block, with per-surface occurrence counts."""
    violations = []
    for violation in detect_target_term_violations(block.target_text or "", glossary):
        violations.append(
            {
                "source": violation.source,
                "expected": violation.expected,
                "surface": violation.surface,
                "kind": violation.kind,
                "occurrences": len(violation.hits),
            }
        )
    return violations


def term_cascade_report(db_path: Path, job_id: str, block_id: str) -> dict[str, Any] | None:
    """Terminology findings for one block plus the cascade size each implies.

    ``None`` when the block is unknown; an empty ``violations`` list when the
    job has no glossary or the block is clean. The cascade counts come from a
    *single* pass over the book (not one scan per finding), because the
    workbench requests this for every visible card.
    """
    ledger = SQLiteJobLedger(db_path, read_only=True)
    try:
        block = ledger.get_block(block_id, job_id=job_id)
        if block is None:
            return None
        glossary = job_glossary(ledger, job_id)
        if not glossary:
            return {"block_id": block_id, "glossary_size": 0, "violations": []}
        findings = block_term_violations(block, glossary)
        if not findings:
            return {"block_id": block_id, "glossary_size": len(glossary), "violations": []}
        blocks = ledger.get_all_blocks(job_id)
    finally:
        ledger.close()

    # Only the pivot's own findings can cascade, so restrict the book-wide scan
    # to their (surface, expected) keys and prefilter blocks by the surfaces'
    # first characters — the matcher never drops a character, so a block without
    # any of those characters cannot carry a match.
    pivot_keys = {(finding["surface"], finding["expected"]) for finding in findings}
    probe_chars = {surface[0] for surface, _ in pivot_keys}
    counts_all: Counter[tuple[str, str]] = Counter()
    counts_subsequent: Counter[tuple[str, str]] = Counter()
    for other in blocks:
        if other.id == block.id:
            continue  # the pivot is always in scope; the checkbox counts the rest
        text = other.target_text or ""
        if not text or not any(char in text for char in probe_chars):
            continue
        keys_here = {
            (violation.surface, violation.expected)
            for violation in detect_target_term_violations(text, glossary)
            if (violation.surface, violation.expected) in pivot_keys
        }
        for key in keys_here:
            counts_all[key] += 1
            if other.spine_index >= block.spine_index:
                counts_subsequent[key] += 1

    violations = [
        {
            **finding,
            "cascade_all": counts_all[(finding["surface"], finding["expected"])],
            "cascade_subsequent": counts_subsequent[(finding["surface"], finding["expected"])],
        }
        for finding in findings
    ]
    return {"block_id": block_id, "glossary_size": len(glossary), "violations": violations}


def apply_term_propagation(
    db_path: Path,
    job_id: str,
    block_id: str,
    surface: str,
    expected: str,
    *,
    scope: str = "all",
    tm_path: Path | None = None,
) -> dict[str, Any]:
    """Replace one offending surface with its canonical rendering.

    ``scope`` is ``block`` (only the selected block), ``subsequent`` (the block
    and every later one), or ``all`` (default). Every rewritten block goes
    through the same write path as a manual revision: ledger promoted to
    ``REPAIRED`` with ``human_pe_imported``, stale machine verdict cleared, and
    the accepted pair fed back to the shared TM as ``human_pe``.
    """
    if not surface or not expected:
        raise ReviewEditError("surface and expected must be non-empty")
    if scope not in ("block", "subsequent", "all"):
        raise ReviewEditError(f"unknown scope: {scope}")

    ledger = SQLiteJobLedger(db_path)
    lock = LedgerWriterLock(db_path, job_id)
    tm: TranslationMemory | None = None
    try:
        try:
            lock.acquire()
        except LedgerWriterLockConflictError as exc:
            raise ReviewEditConflict(f"job is being written by another process: {exc}") from exc

        blocks = ledger.get_all_blocks(job_id)
        pivot = next((b for b in blocks if b.id == block_id), None)
        if pivot is None:
            raise ReviewBlockNotFound(f"unknown block: {block_id}")
        glossary = job_glossary(ledger, job_id)
        if not glossary:
            raise ReviewEditError("this job has no glossary to propagate terms from")

        plans: list[tuple[IRBlock, str, int]] = []
        for block in blocks:
            if scope == "block" and block.id != block_id:
                continue
            if scope == "subsequent" and block.spine_index < pivot.spine_index:
                continue
            text = block.target_text or ""
            # The matcher never drops a character, so a block without the
            # surface's first character cannot carry a match — skip its scan.
            if not text or surface[0] not in text:
                continue
            rewritten, replacements = replace_term_surface(
                text, glossary, surface=surface, expected=expected
            )
            if replacements:
                plans.append((block, rewritten, replacements))
        if not plans:
            raise ReviewEditError(f"term not found in scope '{scope}': {surface}")

        capped = len(plans) > MAX_CASCADE_BLOCKS
        if capped:
            plans = plans[:MAX_CASCADE_BLOCKS]

        updated = ledger.save_checkpoints_batch(
            [
                {
                    "block_id": block.id,
                    "status": BlockStatus.REPAIRED,
                    "target_text": rewritten,
                    "error_flags": [FLAG_HUMAN_PE_IMPORTED],
                }
                for block, rewritten, _ in plans
            ],
            clear_verdict_for=[block.id for block, _, _ in plans],
            job_id=job_id,
        )

        tm_written = 0
        if tm_path is not None:
            src = ledger.get_job_metadata_value(job_id, "source_lang")
            tgt = ledger.get_job_target_lang(job_id)
            if src and tgt:
                entries = [
                    TMPendingEntry(
                        src_lang=str(src),
                        tgt_lang=str(tgt),
                        source_text=block.source_text,
                        target_text=rewritten,
                        provenance=PROVENANCE_HUMAN_PE,
                    )
                    for block, rewritten, _ in plans
                    if block.source_text
                ]
                if entries:
                    tm = TranslationMemory(tm_path)
                    tm_written = tm.writeback(entries)
            else:
                logger.warning(
                    "Term propagation on %s: TM write skipped, language pair unresolved", job_id
                )

        return {
            "block_id": block_id,
            "surface": surface,
            "expected": expected,
            "scope": scope,
            "blocks_updated": updated,
            "blocks_planned": len(plans),
            "replacements": sum(n for _, _, n in plans),
            "capped": capped,
            "block_ids": [block.id for block, _, _ in plans],
            "tm_written": tm_written,
        }
    finally:
        if tm is not None:
            tm.close()
        lock.release()
        ledger.close()

"""Human PE revision re-import.

Accepts a post-edited PE queue file (CSV round-trip table or XLIFF 2.1),
applies accepted revisions to the job ledger (``target_text`` updated, status
promoted to ``REPAIRED``), and feeds accepted pairs back into the shared
translation memory with ``provenance='human_pe'`` — the asset that later powers
TM exact-skip and fuzzy few-shot for future jobs (self-evolution groundwork).

CSV convention: humans fill the ``revised_translation`` column of a CSV
round-trip table (``block_id`` / ``source_text`` / ``target_text`` /
``revised_translation``); empty cells and rows equal to the exported target are
skipped.

XLIFF convention: any ``<unit>`` whose ``<segment><target>`` text is non-empty
and differs from the ledger's current target counts as a revision.

Both formats must carry the job binding they were exported with (the CSV
``job_id`` column / the XLIFF ``ubt-job-id`` note). Block ids are structural
names (``ch_003#b012``) that repeat across the ledgers of the same book, so
an unbound or foreign-bound file cannot be told apart from a same-named block
in another language's job — it is refused.
"""

from __future__ import annotations

import csv
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import BlockStatus
from ubt.core.memory.tm import PROVENANCE_HUMAN_PE, TMPendingEntry, TranslationMemory

logger = logging.getLogger(__name__)

PE_QUEUE_STATUSES: frozenset[BlockStatus] = frozenset(
    {BlockStatus.NEEDS_HUMAN, BlockStatus.BLOCKED_HUMAN}
)

_XLIFF_NS = "urn:oasis:names:tc:xliff:document:2.1"

_FLAG_HUMAN_PE_IMPORTED = "human_pe_imported"


class PEImportError(ValueError):
    """Raised when a PE queue file cannot be parsed or does not round-trip."""


@dataclass(frozen=True)
class PEImportResult:
    """Outcome of a re-import pass."""

    total_rows: int
    imported: int
    skipped: int
    tm_written: int
    revised_block_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedRevisions:
    """Revisions plus the job bindings the file declares (empty = unbound)."""

    revisions: dict[str, str]
    job_ids: set[str]


def parse_csv_revisions(file_path: Path) -> ParsedRevisions:
    """Extract ``block_id -> revised translation`` and ``job_id`` bindings from a CSV."""
    revisions: dict[str, str] = {}
    job_ids: set[str] = set()
    with file_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if "block_id" not in fieldnames or "revised_translation" not in fieldnames:
            raise PEImportError(
                "PE queue CSV must contain 'block_id' and 'revised_translation' columns; "
                f"found: {fieldnames}"
            )
        for row in reader:
            block_id = (row.get("block_id") or "").strip()
            revised = (row.get("revised_translation") or "").strip()
            job = (row.get("job_id") or "").strip()
            if job:
                job_ids.add(job)
            if block_id and revised:
                revisions[block_id] = revised
    return ParsedRevisions(revisions=revisions, job_ids=job_ids)


def parse_xliff_revisions(file_path: Path) -> ParsedRevisions:
    """Extract ``unit id -> target`` and ``ubt-job-id`` notes from an XLIFF 2.1 document."""
    try:
        root = ET.parse(file_path).getroot()
    except ET.ParseError as exc:
        raise PEImportError(f"Invalid XLIFF XML: {exc}") from exc
    if root.tag != f"{{{_XLIFF_NS}}}xliff":
        raise PEImportError(
            f"Not an XLIFF 2.1 document (expected namespace {_XLIFF_NS}, got root <{root.tag}>)"
        )
    revisions: dict[str, str] = {}
    job_ids: set[str] = set()
    for unit in root.iter(f"{{{_XLIFF_NS}}}unit"):
        unit_id = (unit.get("id") or "").strip()
        if not unit_id:
            continue
        for note in unit.iter(f"{{{_XLIFF_NS}}}note"):
            if note.get("category") == "ubt-job-id":
                bound_job = (note.text or "").strip()
                if bound_job:
                    job_ids.add(bound_job)
        for segment in unit.iter(f"{{{_XLIFF_NS}}}segment"):
            target_el = segment.find(f"{{{_XLIFF_NS}}}target")
            text = (target_el.text or "").strip() if target_el is not None else ""
            if text:
                revisions[unit_id] = text
    return ParsedRevisions(revisions=revisions, job_ids=job_ids)


def parse_revisions(file_path: Path) -> ParsedRevisions:
    """Dispatch on file suffix: ``.csv`` table vs ``.xliff``/``.xlf`` document."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return parse_csv_revisions(file_path)
    if suffix in (".xliff", ".xlf", ".xml"):
        return parse_xliff_revisions(file_path)
    raise PEImportError(
        f"Unsupported PE queue file extension '{suffix}' (expected .csv, .xliff, .xlf or .xml)"
    )


def import_pe_revisions(
    job_id: str,
    file_path: Path,
    ledger: SQLiteJobLedger,
    tm: TranslationMemory | None = None,
    source_lang: str | None = None,
    target_lang: str | None = None,
) -> PEImportResult:
    """Apply human PE revisions to the ledger and (optionally) the shared TM.

    The file must be bound to this ledger's job (CSV ``job_id`` column / XLIFF
    ``ubt-job-id`` note, both written by the exporter): block ids repeat across
    the ledgers of the same book, so an unbound or foreign-bound file cannot be
    distinguished from same-named blocks in another language's job and is
    refused outright.

    Rows whose revision is unknown to the ledger or identical to the current
    target are skipped. Accepted revisions are promoted to ``REPAIRED`` with a
    ``human_pe_imported`` flag appended, and written back into the TM with
    ``provenance='human_pe'`` (machine writeback can never downgrade that
    provenance afterwards). The TM language pair is read from the ledger when
    not given explicitly — a caller-supplied "en/zh" default would otherwise
    file human verdicts under the wrong pair permanently.
    """
    stored_id = ledger.resolve_job_id(job_id)
    if stored_id is None:
        raise PEImportError(
            f"Ledger at this path has no job '{job_id}' (nor a doc_id resolving to it)"
        )

    parsed = parse_revisions(file_path)
    if not parsed.job_ids:
        raise PEImportError(
            "PE queue file carries no job binding (CSV 'job_id' column / XLIFF "
            "'ubt-job-id' note); refusing an unbound file. Re-export the queue "
            "with the current UBT version and re-apply the edits."
        )
    if parsed.job_ids != {stored_id}:
        raise PEImportError(
            f"PE queue file is bound to job(s) {sorted(parsed.job_ids)}, not this "
            f"ledger's job '{stored_id}' — refusing a cross-job import."
        )

    tm_pair: tuple[str, str] | None = None
    if tm is not None:
        if source_lang is None:
            source_lang = ledger.get_job_metadata_value(stored_id, "source_lang")
        if target_lang is None:
            target_lang = ledger.get_job_target_lang(stored_id)
        if not source_lang or not target_lang:
            raise PEImportError(
                f"Cannot write TM entries: language pair unresolved for job "
                f"'{stored_id}' (source={source_lang!r}, target={target_lang!r}). "
                "Re-import with --no-write-tm to update the ledger only."
            )
        tm_pair = (str(source_lang), str(target_lang))

    revisions = parsed.revisions
    blocks = {b.id: b for b in ledger.get_all_blocks(job_id)}

    updates: list[dict[str, Any]] = []
    tm_entries: list[TMPendingEntry] = []
    revised_ids: list[str] = []
    skipped = 0

    for block_id, revised in revisions.items():
        block = blocks.get(block_id)
        if block is None:
            logger.warning("PE import: unknown block_id '%s' skipped", block_id)
            skipped += 1
            continue
        if revised == (block.target_text or ""):
            skipped += 1
            continue
        if block.status not in PE_QUEUE_STATUSES:
            # Only PE-queue members (NEEDS_HUMAN / BLOCKED_HUMAN) accept
            # revisions: a stray row must never overwrite an already-good
            # block with unreviewed text.
            logger.warning(
                "PE import: block '%s' is %s (not in the PE queue) — revision skipped",
                block_id,
                block.status,
            )
            skipped += 1
            continue
        # Human verdict supersedes every machine verdict: stale flags
        # (critical / needs_human_review) and scores measured against the
        # old draft are dropped, not carried onto the revised text.
        updates.append(
            {
                "block_id": block_id,
                "status": BlockStatus.REPAIRED,
                "target_text": revised,
                "error_flags": [_FLAG_HUMAN_PE_IMPORTED],
            }
        )
        if tm_pair is not None and block.source_text:
            tm_entries.append(
                TMPendingEntry(
                    src_lang=tm_pair[0],
                    tgt_lang=tm_pair[1],
                    source_text=block.source_text,
                    target_text=revised,
                    provenance=PROVENANCE_HUMAN_PE,
                )
            )
        revised_ids.append(block_id)

    if updates:
        # One transaction: the human text and the cleared machine verdict must
        # land together, or a crash between them leaves a human-fixed block
        # still reporting its superseded "critical" severity to auditors.
        ledger.save_checkpoints_batch(updates, clear_verdict_for=revised_ids)

    tm_written = 0
    if tm_entries and tm is not None:
        tm_written = tm.writeback(tm_entries)

    imported = len(revised_ids)
    logger.info(
        "PE import for job %s: %d/%d revision(s) imported, %d skipped, %d TM entries written",
        job_id,
        imported,
        len(revisions),
        skipped,
        tm_written,
    )
    return PEImportResult(
        total_rows=len(revisions),
        imported=imported,
        skipped=skipped,
        tm_written=tm_written,
        revised_block_ids=revised_ids,
    )

"""Book patch application engine and incremental ledger seeding."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import (
    TERMINAL_STATUSES,
    BlockStatus,
    BookManifest,
    ChapterIR,
    IRBlock,
)
from ubt.core.patch.differ import DocumentSemanticDiffer
from ubt.core.patch.models import BlockDiffItem, PatchBlockStatus, PatchSummary
from ubt.core.ports import resolve_adapter

logger = logging.getLogger(__name__)


def resolve_job_ledger_path(job_id: str, db_dir: Path | None = None) -> Path:
    """Resolve SQLite ledger file path for a given job ID."""
    base_dir = db_dir or UBTConfig.from_env().db_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".sqlite", ".db"):
        cand = base_dir / f"{job_id}{ext}"
        if cand.exists():
            return cand
    return base_dir / f"{job_id}.sqlite"


async def load_document_blocks(
    source: Path | str,
    ledger: SQLiteJobLedger | None = None,
    db_dir: Path | None = None,
) -> tuple[str, list[IRBlock], BookManifest | None]:
    """Load blocks and manifest from an existing ledger job ID or a file path."""
    source_str = str(source)

    # 1. Attempt to resolve via provided ledger
    if ledger is not None:
        resolved_job_id = ledger.resolve_job_id(source_str)
        if resolved_job_id:
            blocks = ledger.get_all_blocks(resolved_job_id)
            if blocks:
                logger.info(
                    "Loaded %d blocks from base ledger job '%s'", len(blocks), resolved_job_id
                )
                return resolved_job_id, blocks, None

    # 2. Check if source matches an existing job ledger file in db_dir
    cand_db = resolve_job_ledger_path(source_str, db_dir)
    if cand_db.exists():
        with SQLiteJobLedger(cand_db, read_only=True) as leg:
            resolved_job_id = leg.resolve_job_id(source_str) or source_str
            blocks = leg.get_all_blocks(resolved_job_id)
            if blocks:
                logger.info("Loaded %d blocks from job ledger at '%s'", len(blocks), cand_db)
                return resolved_job_id, blocks, None

    # 3. Treat as file path
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source '{source}' is neither an active ledger job ID nor an existing file."
        )

    adapter = resolve_adapter(source_path)
    manifest = await adapter.extract_manifest(source_path)
    chapters: list[ChapterIR] = []
    async for chapter in adapter.parse_stream(source_path):
        chapters.append(chapter)

    blocks = [b for ch in chapters for b in ch.blocks]
    logger.info("Parsed %d blocks from file '%s'", len(blocks), source_path)
    return str(source_path), blocks, manifest


class BookPatchEngine:
    """Orchestrates document diffing, incremental patching, and ledger seeding."""

    def __init__(
        self,
        differ: DocumentSemanticDiffer | None = None,
        ledger: SQLiteJobLedger | None = None,
        db_dir: Path | None = None,
    ) -> None:
        self.differ = differ or DocumentSemanticDiffer()
        self.ledger = ledger
        self.db_dir = db_dir

    async def diff_documents(
        self,
        base: Path | str | Sequence[IRBlock],
        revised: Path | str | Sequence[IRBlock],
        *,
        base_job_id: str | None = None,
    ) -> PatchSummary:
        """Diff two book versions, accepting files, job IDs, or block sequences."""
        if isinstance(base, (Path, str)):
            base_src, base_blocks, _ = await load_document_blocks(base, self.ledger, self.db_dir)
        else:
            base_src = base_job_id or "memory_base"
            base_blocks = list(base)

        if isinstance(revised, (Path, str)):
            rev_src, rev_blocks, _ = await load_document_blocks(revised, self.ledger, self.db_dir)
        else:
            rev_src = "memory_revised"
            rev_blocks = list(revised)

        resolved_id = None
        if self.ledger is not None:
            resolved_id = self.ledger.resolve_job_id(base_src)
        elif resolve_job_ledger_path(base_src, self.db_dir).exists():
            resolved_id = base_src

        return self.differ.diff_blocks(
            base_blocks,
            rev_blocks,
            base_job_id=base_job_id or resolved_id,
            base_source=base_src,
            revised_source=rev_src,
        )

    async def apply_patch(
        self,
        base_source: Path | str,
        revised_path: Path,
        *,
        new_job_id: str | None = None,
        target_lang: str = "zh",
    ) -> tuple[str, PatchSummary]:
        """Apply patch from base job/file onto revised document, seeding a new job ledger."""
        revised_file = Path(revised_path)
        if not revised_file.exists():
            raise FileNotFoundError(f"Revised document '{revised_path}' does not exist.")

        # Load base
        base_src, base_blocks, _ = await load_document_blocks(base_source, self.ledger, self.db_dir)

        # Ingest revised file
        adapter = resolve_adapter(revised_file)
        manifest = await adapter.extract_manifest(revised_file)
        revised_chapters: list[ChapterIR] = []
        async for chapter in adapter.parse_stream(revised_file):
            revised_chapters.append(chapter)

        all_revised_blocks = [b for ch in revised_chapters for b in ch.blocks]

        # Check resolved base id
        base_job_id_found = None
        if self.ledger is not None:
            base_job_id_found = self.ledger.resolve_job_id(base_src)
        elif resolve_job_ledger_path(base_src, self.db_dir).exists():
            base_job_id_found = base_src

        # Compute semantic diff
        summary = self.differ.diff_blocks(
            base_blocks,
            all_revised_blocks,
            base_job_id=base_job_id_found,
            base_source=base_src,
            revised_source=str(revised_file),
        )

        # Generate target job id if not provided
        if not new_job_id:
            raw_hash = hashlib.sha256(revised_file.name.encode()).hexdigest()[:8]
            actual_job_id = f"patch_{revised_file.stem}_{raw_hash}"
        else:
            actual_job_id = new_job_id

        # Index diff items and base blocks
        diff_by_rev_id: dict[str, BlockDiffItem] = {
            item.revised_block_id: item
            for item in summary.diff_items
            if item.revised_block_id is not None
        }
        base_by_id: dict[str, IRBlock] = {b.id: b for b in base_blocks}

        # Apply transformations per chapter
        patched_chapters: list[ChapterIR] = []
        for ch in revised_chapters:
            new_blocks: list[IRBlock] = []
            for b in ch.blocks:
                diff_item = diff_by_rev_id.get(b.id)
                if diff_item and diff_item.status == PatchBlockStatus.UNCHANGED:
                    base_b = (
                        base_by_id.get(diff_item.base_block_id) if diff_item.base_block_id else None
                    )
                    reused_target = base_b.target_text if base_b else None
                    reused_status = (
                        base_b.status
                        if (base_b and base_b.status in TERMINAL_STATUSES)
                        else BlockStatus.MTQE_PASSED
                    )
                    prov = b.provenance.model_copy()
                    if diff_item.base_block_id:
                        prov.fused_sources.append(f"patch_base:{diff_item.base_block_id}")

                    updated_block = b.model_copy(
                        update={
                            "target_text": reused_target,
                            "draft_text": base_b.draft_text if base_b else None,
                            "status": reused_status,
                            "mtqe_score": base_b.mtqe_score if base_b else 1.0,
                            "repair_rounds": base_b.repair_rounds if base_b else 0,
                            "error_flags": list(base_b.error_flags) if base_b else [],
                            "provenance": prov,
                        }
                    )
                    new_blocks.append(updated_block)
                elif diff_item and diff_item.status == PatchBlockStatus.MODIFIED:
                    base_b = (
                        base_by_id.get(diff_item.base_block_id) if diff_item.base_block_id else None
                    )
                    prov = b.provenance.model_copy()
                    if diff_item.base_block_id:
                        prov.fused_sources.append(f"patch_modified:{diff_item.base_block_id}")

                    updated_block = b.model_copy(
                        update={
                            "status": BlockStatus.PENDING,
                            "target_text": None,
                            "error_flags": [],
                            "provenance": prov,
                        }
                    )
                    new_blocks.append(updated_block)
                else:
                    updated_block = b.model_copy(
                        update={
                            "status": BlockStatus.PENDING,
                            "target_text": None,
                            "error_flags": [],
                        }
                    )
                    new_blocks.append(updated_block)

            patched_chapters.append(ch.model_copy(update={"blocks": new_blocks}))

        # Resolve target ledger instance
        target_ledger = self.ledger
        if target_ledger is None:
            target_db = resolve_job_ledger_path(actual_job_id, self.db_dir)
            target_ledger = SQLiteJobLedger(target_db)

        # Persist to ledger atomically
        target_ledger.init_job_placeholder(
            job_id=actual_job_id,
            source_path=str(revised_file),
            target_lang=target_lang,
            title=manifest.title,
        )
        target_ledger.init_job_from_manifest(actual_job_id, manifest)

        for ch in patched_chapters:
            target_ledger.append_chapter(actual_job_id, ch)

        target_ledger.set_job_metadata_value(
            actual_job_id,
            "patch_provenance",
            summary.model_dump(),
        )

        logger.info(
            "Seeded patched job '%s': %d unchanged (reused), %d modified, %d added",
            actual_job_id,
            summary.unchanged_count,
            summary.modified_count,
            summary.added_count,
        )

        return actual_job_id, summary

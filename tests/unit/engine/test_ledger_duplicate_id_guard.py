from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.exceptions import LedgerError
from ubt.core.ir.models import (
    BlockProvenance,
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    IRBlock,
    make_element,
)

pytestmark = pytest.mark.fast


def _block(block_id: str, text: str, spine: int) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=spine,
            block_type=BlockType.NARRATIVE,
            source_text=text,
        ),
        status=BlockStatus.PENDING,
        provenance=BlockProvenance(source_page=1),
    )


def _chapter(blocks: list[IRBlock]) -> ChapterIR:
    return ChapterIR(
        doc_id="doc",
        chapter_id="c0001",
        title="ch",
        spine_index=1,
        blocks=blocks,
    )


def _ledger(tmp_path: Path) -> SQLiteJobLedger:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.init_job_from_manifest("job", BookManifest(doc_id="doc", title="t", source_path="s.pdf"))
    return ledger


def test_a_duplicate_id_in_one_append_raises_instead_of_overwriting(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    try:
        chapter_blocks = [
            _block("pdf_main#b0001", "the title page", 1),
            _block("pdf_main#b0002", "the abstract", 2),
            _block("pdf_main#b0002", "1 Introduction (toc row)", 2),
        ]
        with pytest.raises(LedgerError, match="duplicate block id"):
            ledger.append_chapter("job", _chapter(chapter_blocks))
        stats = ledger.get_job_stats("job")
        assert int(stats.get("total", 0)) == 0
    finally:
        ledger.close()


def test_distinct_ids_in_one_append_still_write(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    try:
        ledger.append_chapter(
            "job",
            _chapter(
                [
                    _block("pdf_main#b0001", "first", 1),
                    _block("pdf_main#b0002", "second", 2),
                ]
            ),
        )
        stats = ledger.get_job_stats("job")
        assert int(stats.get("total", 0)) == 2
    finally:
        ledger.close()


def test_rewriting_a_known_id_across_appends_is_still_supported(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    try:
        ledger.append_chapter("job", _chapter([_block("pdf_main#b0001", "first", 1)]))
        ledger.append_chapter("job", _chapter([_block("pdf_main#b0001", "first, resumed", 1)]))
        stats = ledger.get_job_stats("job")
        assert int(stats.get("total", 0)) == 1
    finally:
        ledger.close()

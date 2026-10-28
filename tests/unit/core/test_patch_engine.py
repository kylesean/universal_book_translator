"""Unit tests for the semantic AST diffing and incremental patch engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    IRBlock,
    make_element,
)
from ubt.core.patch.differ import (
    DocumentSemanticDiffer,
    estimate_block_tokens,
    normalize_source_text,
)
from ubt.core.patch.engine import BookPatchEngine
from ubt.core.patch.models import PatchBlockStatus

pytestmark = pytest.mark.fast


def _make_block(
    b_id: str,
    spine_index: int,
    text: str,
    block_type: BlockType = BlockType.NARRATIVE,
    target_text: str | None = None,
    status: BlockStatus = BlockStatus.PENDING,
    score: float | None = None,
) -> IRBlock:
    element = make_element(
        id=b_id,
        spine_index=spine_index,
        block_type=block_type,
        source_text=text,
    )
    return IRBlock(
        element=element,
        target_text=target_text,
        status=status,
        mtqe_score=score,
    )


def test_normalize_and_token_estimation():
    """Verify text normalization and token count heuristics."""
    assert normalize_source_text("  hello \n\t world  ") == "hello world"
    assert normalize_source_text(None) == ""

    # English token estimation
    en_tokens = estimate_block_tokens(
        "This is a simple sentence describing distributed algorithms."
    )
    assert en_tokens > 5

    # CJK token estimation
    zh_tokens = estimate_block_tokens("这是一个关于分布式系统一致性协议的简短描述。")
    assert zh_tokens >= 20


def test_semantic_differ_identical_documents():
    """Identical block sequences must result in 100% reuse and UNCHANGED status."""
    differ = DocumentSemanticDiffer()
    blocks = [
        _make_block(
            "b1",
            1,
            "# Chapter 1: Introduction",
            BlockType.HEADING,
            "第一章：简介",
            BlockStatus.MTQE_PASSED,
        ),
        _make_block(
            "b2",
            2,
            "Distributed systems are resilient.",
            BlockType.NARRATIVE,
            "分布式系统具备容错性。",
            BlockStatus.MTQE_PASSED,
        ),
    ]

    summary = differ.diff_blocks(blocks, blocks)
    assert summary.total_base_blocks == 2
    assert summary.total_revised_blocks == 2
    assert summary.unchanged_count == 2
    assert summary.modified_count == 0
    assert summary.added_count == 0
    assert summary.deleted_count == 0
    assert summary.reuse_ratio == 1.0
    assert summary.estimated_tokens_saved > 0
    assert all(item.status == PatchBlockStatus.UNCHANGED for item in summary.diff_items)


def test_semantic_differ_empty_and_extremes():
    """Empty or one-sided block diffs should be handled without exceptions."""
    differ = DocumentSemanticDiffer()
    empty_summary = differ.diff_blocks([], [])
    assert empty_summary.total_base_blocks == 0
    assert empty_summary.total_revised_blocks == 0
    assert empty_summary.reuse_ratio == 0.0

    # Base only -> pure deletion
    blocks = [_make_block("b1", 1, "Deleted block", BlockType.NARRATIVE)]
    del_summary = differ.diff_blocks(blocks, [])
    assert del_summary.deleted_count == 1
    assert del_summary.unchanged_count == 0

    # Revised only -> pure addition
    add_summary = differ.diff_blocks([], blocks)
    assert add_summary.added_count == 1
    assert add_summary.unchanged_count == 0


def test_semantic_differ_mixed_revisions():
    """Diffing modified, added, deleted, and unchanged blocks."""
    differ = DocumentSemanticDiffer(min_similarity=0.40)

    base = [
        _make_block("b1", 1, "# Chapter 1", BlockType.HEADING, "第一章", BlockStatus.MTQE_PASSED),
        _make_block(
            "b2",
            2,
            "Paxos is an ancient consensus protocol.",
            BlockType.NARRATIVE,
            "Paxos 是一种古老的一致性协议。",
            BlockStatus.MTQE_PASSED,
        ),
        _make_block(
            "b3",
            3,
            "This paragraph will be deleted.",
            BlockType.NARRATIVE,
            "该段将被删除。",
            BlockStatus.MTQE_PASSED,
        ),
        _make_block(
            "b4", 4, "Conclusion notes.", BlockType.NARRATIVE, "结语说明。", BlockStatus.MTQE_PASSED
        ),
    ]

    revised = [
        _make_block("r1", 1, "# Chapter 1", BlockType.HEADING),  # UNCHANGED
        _make_block(
            "r2", 2, "Paxos is a classic distributed consensus protocol.", BlockType.NARRATIVE
        ),  # MODIFIED
        _make_block(
            "r3", 3, "Raft was designed for understandability.", BlockType.NARRATIVE
        ),  # ADDED
        _make_block("r4", 4, "Conclusion notes.", BlockType.NARRATIVE),  # UNCHANGED
    ]

    summary = differ.diff_blocks(base, revised)
    assert summary.total_base_blocks == 4
    assert summary.total_revised_blocks == 4
    assert summary.unchanged_count == 2  # r1 and r4
    assert summary.modified_count == 1  # r2
    assert summary.added_count == 1  # r3
    assert summary.deleted_count == 1  # b3
    assert summary.reuse_ratio == 0.50

    # Markdown and JSON rendering
    md = differ.render_diff_markdown(summary)
    assert "### Universal Book Translator: Semantic Revision Patch Summary" in md
    assert "Asset Reuse Rate" in md

    js = differ.render_diff_json(summary)
    data = json.loads(js)
    assert data["unchanged_count"] == 2
    assert data["reuse_ratio"] == 0.5


@pytest.mark.asyncio
async def test_patch_engine_apply_patch(tmp_path: Path):
    """Verify BookPatchEngine applies patch to seed a new SQLite ledger job with preserved translations."""
    db_file = tmp_path / "test_ledger.sqlite"
    ledger = SQLiteJobLedger(db_file)

    # 1. Create base document & revised document
    base_file = tmp_path / "base_book.md"
    base_file.write_text(
        "# Introduction\n\n"
        "Distributed systems require fault tolerance.\n\n"
        "Chapter one concludes here.\n",
        encoding="utf-8",
    )

    revised_file = tmp_path / "revised_book.md"
    revised_file.write_text(
        "# Introduction\n\n"
        "Modern distributed systems require Byzantine fault tolerance.\n\n"
        "Here is a brand new section.\n\n"
        "Chapter one concludes here.\n",
        encoding="utf-8",
    )

    differ = DocumentSemanticDiffer(min_similarity=0.40)
    engine = BookPatchEngine(differ=differ, ledger=ledger)

    # 2. Diff documents
    diff_summary = await engine.diff_documents(base_file, revised_file)
    assert diff_summary.total_base_blocks == 3
    assert diff_summary.total_revised_blocks == 4
    assert diff_summary.unchanged_count == 2  # Heading + Conclusion
    assert diff_summary.modified_count == 1  # Modern distributed...
    assert diff_summary.added_count == 1  # Brand new section

    # 3. Simulate existing base job with translations
    base_job_id = "job_base_v1"
    from ubt.core.ports import resolve_adapter

    ad = resolve_adapter(base_file)
    m = await ad.extract_manifest(base_file)
    ledger.init_job_from_manifest(base_job_id, m)
    async for ch in ad.parse_stream(base_file):
        # Give translation to base blocks
        for b in ch.blocks:
            b.target_text = f"已译: {b.source_text}"
            b.status = BlockStatus.MTQE_PASSED
            b.mtqe_score = 0.95
        ledger.append_chapter(base_job_id, ch)

    # 4. Apply patch onto revised book
    new_job_id = "job_revised_v2"
    actual_id, patch_res = await engine.apply_patch(
        base_source=base_job_id,
        revised_path=revised_file,
        new_job_id=new_job_id,
        target_lang="zh",
    )

    assert actual_id == new_job_id
    assert patch_res.unchanged_count == 2
    assert patch_res.modified_count == 1
    assert patch_res.added_count == 1

    # 5. Inspect seeded blocks in new ledger
    all_blocks = ledger.get_all_blocks(actual_id)
    assert len(all_blocks) == 4

    unchanged_blocks = [b for b in all_blocks if b.status == BlockStatus.MTQE_PASSED]
    pending_blocks = [b for b in all_blocks if b.status == BlockStatus.PENDING]

    # Two blocks must be already finalized/reused with preserved target text!
    assert len(unchanged_blocks) == 2
    for ub in unchanged_blocks:
        assert ub.target_text is not None
        assert ub.target_text.startswith("已译: ")

    # Two blocks must be pending for translation!
    assert len(pending_blocks) == 2
    for pb in pending_blocks:
        assert pb.target_text is None

    # Check patch provenance metadata in ledger
    meta_prov = ledger.get_job_metadata_value(actual_id, "patch_provenance")
    assert meta_prov is not None
    assert meta_prov["unchanged_count"] == 2
    assert meta_prov["reuse_ratio"] == 0.5

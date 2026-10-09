"""Unit tests for the stdio MCP server (ubt.mcp.server).

Covers:
- ubt_list_issues tool for human-in-the-loop (HITL) review.
- ubt_edit_segment tool for applying human post-edits.
- Adapter render skip / flag contract properties.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.base import BaseDocumentAdapter
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    IRBlock,
    make_element,
)
from ubt.core.ports import DocumentAdapter, get_last_render_flags, get_last_render_skips

pytestmark = pytest.mark.fast


def test_document_adapter_render_properties() -> None:
    class DummyAdapter(BaseDocumentAdapter):
        async def extract_manifest(self, input_path: Path):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def parse_stream(self, input_path: Path, pages=None):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    adapter = DummyAdapter()
    assert isinstance(adapter, DocumentAdapter)
    assert adapter.last_render_skips == []
    assert adapter.last_render_flags == []
    assert get_last_render_skips(adapter) == []
    assert get_last_render_flags(adapter) == []

    # Verify property setter
    adapter.last_render_skips = [("blk_1", "render_skip:overflow")]
    adapter.last_render_flags = [("blk_1", "low_legibility_font")]
    assert adapter.last_render_skips == [("blk_1", "render_skip:overflow")]
    assert adapter.last_render_flags == [("blk_1", "low_legibility_font")]
    assert get_last_render_skips(adapter) == [("blk_1", "render_skip:overflow")]
    assert get_last_render_flags(adapter) == [("blk_1", "low_legibility_font")]


@pytest.mark.asyncio
async def test_mcp_hitl_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ubt.mcp.server import ubt_edit_segment, ubt_list_issues

    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(tmp_path))

    job_id = "test_hitl_job"
    db_path = tmp_path / f"{job_id}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest(
            job_id,
            BookManifest(
                doc_id=job_id,
                title="t",
                source_path=str(tmp_path / "in.md"),
                source_lang="en",
                target_lang="zh",
            ),
        )
        b1 = IRBlock(
            element=make_element(
                id="blk_1",
                spine_index=0,
                block_type=BlockType.NARRATIVE,
                source_text="Patient took 50mg of Drug A.",
            ),
            target_text="患者服用了 5mg 的药物 A。",
            status=BlockStatus.MTQE_PASSED,
            error_flags=["Numeric fidelity mismatch"],
        )
        b2 = IRBlock(
            element=make_element(
                id="blk_2",
                spine_index=1,
                block_type=BlockType.NARRATIVE,
                source_text="Normal text.",
            ),
            target_text="正常文本。",
            status=BlockStatus.MTQE_PASSED,
            error_flags=[],
        )
        ledger.append_chapter(
            job_id,
            ChapterIR(doc_id=job_id, chapter_id="c1", title="C1", spine_index=0, blocks=[b1, b2]),
        )

    # Test ubt_list_issues
    issues_res = await ubt_list_issues(job_id=job_id, db_dir=str(tmp_path))
    assert issues_res["job_id"] == job_id
    assert issues_res["total_issues"] >= 1
    assert issues_res["counts"]["numeric"] >= 1
    assert len(issues_res["segments"]) == 1
    assert issues_res["segments"][0]["block_id"] == "blk_1"

    # Test filter by issue_kind
    filtered_res = await ubt_list_issues(job_id=job_id, issue_kind="numeric", db_dir=str(tmp_path))
    assert len(filtered_res["segments"]) == 1
    assert filtered_res["segments"][0]["block_id"] == "blk_1"

    unmatched_res = await ubt_list_issues(job_id=job_id, issue_kind="formula", db_dir=str(tmp_path))
    assert len(unmatched_res["segments"]) == 0

    # Test ubt_edit_segment
    new_text = "患者服用了 50mg 的药物 A。"
    edit_res = await ubt_edit_segment(
        job_id=job_id,
        block_id="blk_1",
        target_text=new_text,
        db_dir=str(tmp_path),
    )
    assert edit_res["job_id"] == job_id
    assert edit_res["block_id"] == "blk_1"
    assert edit_res["changed"] is True
    assert edit_res["segment"] is not None
    assert edit_res["segment"]["target_text"] == new_text
    assert edit_res["segment"]["status"] == "repaired"

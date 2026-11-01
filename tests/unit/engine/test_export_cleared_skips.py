from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.export import _apply_render_skip_ledger_pass
from ubt.core.ir.models import BlockProvenance, BlockStatus, IRBlock

pytestmark = pytest.mark.fast


@pytest.mark.asyncio
async def test_apply_render_skip_ledger_pass_preserves_flipped_status(tmp_path: Path) -> None:
    saved_checkpoints: list[list[dict[str, Any]]] = []

    class FakeLedger:
        def save_checkpoints_batch(self, batch: list[dict[str, Any]]) -> None:
            saved_checkpoints.append(batch)

    class FakeAdapter:
        last_render_skips = [("b1", "table_unparseable")]
        last_render_flags: list[tuple[str, str]] = []

    from ubt.core.ir.models import BlockType, BoundingBox, make_element

    block_b1 = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Resume bullet text",
            bbox=BoundingBox(x0=54.0, y0=700.0, x1=81.6, y1=708.3, page=1),
        ),
        target_text="简历要点文本",
        status=BlockStatus.MTQE_PASSED,
        error_flags=["render_skip:old_reason"],
        provenance=BlockProvenance(page_kind="resume_dense"),
    )

    ctx = StageContext(
        config=cast("UBTConfig", SimpleNamespace()),
        router=cast("Any", SimpleNamespace()),
        ledger=cast("Any", FakeLedger()),
        manifest=cast("Any", SimpleNamespace(title="Test", source_path="test.pdf")),
        job_id="test-job",
        input_path=tmp_path / "test.pdf",
        source_lang="en",
        target_lang="zh",
        profile_name="general",
        create_event=cast("Any", lambda *a, **k: None),
    )

    await _apply_render_skip_ledger_pass(
        ctx=ctx,
        adapter=cast("Any", FakeAdapter()),
        final_blocks=[block_b1],
    )

    assert len(saved_checkpoints) == 1
    batch = saved_checkpoints[0]
    b1_records = [c for c in batch if c["block_id"] == "b1"]
    assert len(b1_records) == 1
    rec = b1_records[0]
    assert rec["status"] == BlockStatus.MTQE_PASSED
    assert "render_skip:table_unparseable" in rec["error_flags"]
    assert "render_skip:old_reason" not in rec["error_flags"]
    assert not any("length_overflow:" in flag for flag in rec["error_flags"])


@pytest.mark.asyncio
async def test_a_re_export_does_not_write_the_skip_set_back(tmp_path: Path) -> None:
    saved: list[list[dict[str, Any]]] = []

    class FakeLedger:
        def save_checkpoints_batch(self, batch: list[dict[str, Any]]) -> None:
            saved.append(batch)

    class FakeAdapter:
        last_render_skips = [("b1", "asset_not_found")]
        last_render_flags: list[tuple[str, str]] = []

    from ubt.core.ir.models import BlockType, make_element

    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Figure text",
        ),
        target_text="插图文本",
        status=BlockStatus.REPAIRED,
        error_flags=[],
        provenance=BlockProvenance(),
    )
    ctx = StageContext(
        config=cast("UBTConfig", SimpleNamespace()),
        router=cast("Any", SimpleNamespace()),
        ledger=cast("Any", FakeLedger()),
        manifest=cast("Any", SimpleNamespace(title="Test", source_path="test.pdf")),
        job_id="test-job",
        input_path=tmp_path / "test.pdf",
        source_lang="en",
        target_lang="zh",
        profile_name="general",
        create_event=cast("Any", lambda *a, **k: None),
    )

    await _apply_render_skip_ledger_pass(
        ctx=ctx,
        adapter=cast("Any", FakeAdapter()),
        final_blocks=[block],
        persist=False,
    )

    assert saved == []
    assert "render_skip:asset_not_found" in block.error_flags

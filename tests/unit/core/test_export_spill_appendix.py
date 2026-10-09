"""Tests for the layout spill appendix and graceful degradation export companion.

When translations overflow their target boxes even after font shrinking (spill/overflow),
the 'appendix' policy ensures that:
1. Space failures are downgraded to WARNING in the reconciliation report with an audit trail tag.
2. A companion .spill_appendix.json and .spill_appendix.md are generated for the displaced blocks.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ubt.core.config import UBTConfig
from ubt.core.content.contract import (
    ReconciliationReport,
    Severity,
    Violation,
    ViolationKind,
)
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.export import _write_spill_appendix
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


def _make_block(block_id: str, *, page: int = 1, src: str = "Hello", tgt: str = "你好") -> IRBlock:
    block = IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text=src,
            bbox=BoundingBox(x0=54.0, y0=700.0, x1=81.6, y1=708.3, page=page),
        )
    )
    block.target_text = tgt
    block.status = BlockStatus.MTQE_PASSED
    return block


def _make_ctx(tmp_path: Path, spill_policy: str = "appendix") -> StageContext:
    config = SimpleNamespace(spill_policy=spill_policy)
    return StageContext(
        config=cast("UBTConfig", config),
        router=cast("Any", SimpleNamespace()),
        ledger=cast("Any", SimpleNamespace()),
        manifest=cast("Any", SimpleNamespace(title="Test Book", source_path="test.pdf")),
        job_id="test-spill-job-123",
        input_path=tmp_path / "test.pdf",
        source_lang="en",
        target_lang="zh",
        profile_name="general",
        create_event=cast("Any", lambda *a, **k: None),
    )


def test_write_spill_appendix_generates_json_and_md(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, spill_policy="appendix")
    rendered_path = tmp_path / "output.pdf"
    rendered_path.write_bytes(b"%PDF-1.4 dummy")

    block = _make_block("b1", page=3, src="English text that overflowed", tgt="溢出的中文内容")
    contract = ReconciliationReport(
        total_text=1,
        source_kept_text=1,
        violations=(
            Violation(
                kind=ViolationKind.TEXT_SOURCE_KEPT,
                severity=Severity.WARNING,
                node_id="b1",
                detail="render:target text overflow after 4pt [spill_degraded_to_warning]",
            ),
        ),
    )

    md_path = _write_spill_appendix(ctx, rendered_path, [block], contract)
    assert md_path is not None
    assert md_path.exists()
    assert md_path.suffix == ".md"

    md_content = md_path.read_text(encoding="utf-8")
    assert "# 排版溢出优雅降级对照附录" in md_content
    assert "test-spill-job-123" in md_content
    assert "b1" in md_content
    assert "P3" in md_content
    assert "English text that overflowed" in md_content
    assert "溢出的中文内容" in md_content

    from ubt.core.job_options import companion_path

    json_path = companion_path(rendered_path, ".spill_appendix.json")
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["job_id"] == "test-spill-job-123"
    assert len(data["spills"]) == 1
    assert data["spills"][0]["block_id"] == "b1"
    assert data["spills"][0]["page"] == 3
    assert data["spills"][0]["source_text"] == "English text that overflowed"
    assert data["spills"][0]["target_text"] == "溢出的中文内容"


def test_write_spill_appendix_noop_when_no_spill_violations(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, spill_policy="appendix")
    rendered_path = tmp_path / "output.pdf"
    rendered_path.write_bytes(b"%PDF-1.4 dummy")

    block = _make_block("b1")
    contract = ReconciliationReport(
        total_text=1,
        violations=(),
    )

    md_path = _write_spill_appendix(ctx, rendered_path, [block], contract)
    assert md_path is None

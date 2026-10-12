"""Unit tests for translated table appendix generation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.export import _write_table_appendix
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


def _make_table_block(
    block_id: str,
    *,
    page: int = 1,
    src: str = "| A | B |\n| --- | --- |\n| 1 | 2 |",
    tgt: str = "| 甲 | 乙 |\n| --- | --- |\n| 一 | 二 |",
    is_complex: bool = False,
) -> IRBlock:
    if is_complex:
        src = "| C1 | C2 | C3 | C4 | C5 | C6 | C7 |\n| --- | --- | --- | --- | --- | --- | --- |\n| 1 | 2 | 3 | 4 | 5 | 6 | 7 |"
        tgt = "| 栏1 | 栏2 | 栏3 | 栏4 | 栏5 | 栏6 | 栏7 |\n| --- | --- | --- | --- | --- | --- | --- |\n| 1 | 2 | 3 | 4 | 5 | 6 | 7 |"
    block = IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=BlockType.TABLE,
            source_text=src,
            bbox=BoundingBox(x0=50.0, y0=500.0, x1=400.0, y1=600.0, page=page),
        )
    )
    block.target_text = tgt
    block.status = BlockStatus.MTQE_PASSED
    return block


def _make_ctx(tmp_path: Path, table_render_mode: str = "auto") -> StageContext:
    config = SimpleNamespace(table_render_mode=table_render_mode)
    return StageContext(
        config=cast("UBTConfig", config),
        router=cast("Any", SimpleNamespace()),
        ledger=cast("Any", SimpleNamespace()),
        manifest=cast("Any", SimpleNamespace(title="Test Document", source_path="test.pdf")),
        job_id="test-table-job-456",
        input_path=tmp_path / "test.pdf",
        source_lang="en",
        target_lang="zh",
        profile_name="general",
        create_event=cast("Any", lambda *a, **k: None),
    )


def test_write_table_appendix_auto_mode_complex_table(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, table_render_mode="auto")
    rendered_path = tmp_path / "output.pdf"
    rendered_path.write_bytes(b"%PDF-1.4 dummy")

    simple_block = _make_table_block("tbl_simple", page=2, is_complex=False)
    complex_block = _make_table_block("tbl_complex", page=5, is_complex=True)

    md_path = _write_table_appendix(ctx, rendered_path, [simple_block, complex_block])
    assert md_path is not None
    assert md_path.exists()
    assert md_path.suffix == ".md"

    md_content = md_path.read_text(encoding="utf-8")
    assert "核心图表与数据矩阵对照附录" in md_content
    assert "test-table-job-456" in md_content
    assert "tbl_complex" in md_content
    # A simple grid is rebuilt in the page, so it needs no appendix companion;
    # ``auto`` only appends the complex ones.
    assert "tbl_simple" not in md_content

    from ubt.core.job_options import companion_path

    json_path = companion_path(rendered_path, ".table_appendix.json")
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["job_id"] == "test-table-job-456"
    assert len(data["tables"]) == 1
    assert data["tables"][0]["block_id"] == "tbl_complex"
    assert data["tables"][0]["page"] == 5


def test_write_table_appendix_all_in_appendix_mode(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, table_render_mode="appendix")
    rendered_path = tmp_path / "output.pdf"
    rendered_path.write_bytes(b"%PDF-1.4 dummy")

    simple_block = _make_table_block("tbl_simple", page=2, is_complex=False)
    complex_block = _make_table_block("tbl_complex", page=5, is_complex=True)

    md_path = _write_table_appendix(ctx, rendered_path, [simple_block, complex_block])
    assert md_path is not None
    assert md_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "tbl_simple" in md_content
    assert "tbl_complex" in md_content


def test_write_table_appendix_skip_mode_noop(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, table_render_mode="skip")
    rendered_path = tmp_path / "output.pdf"
    rendered_path.write_bytes(b"%PDF-1.4 dummy")

    block = _make_table_block("tbl_1", page=1, is_complex=True)
    md_path = _write_table_appendix(ctx, rendered_path, [block])
    assert md_path is None


def test_write_table_appendix_no_tables_noop(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, table_render_mode="auto")
    rendered_path = tmp_path / "output.pdf"
    rendered_path.write_bytes(b"%PDF-1.4 dummy")

    md_path = _write_table_appendix(ctx, rendered_path, [])
    assert md_path is None

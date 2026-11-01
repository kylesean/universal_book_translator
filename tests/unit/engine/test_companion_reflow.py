from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.export import _write_reflow_companion
from ubt.core.ir.render_plan import RenderPlan

pytestmark = pytest.mark.fast


def _ctx(tmp_path: Path, *, enabled: bool = True) -> StageContext:
    config = SimpleNamespace(companion_reflow=enabled)
    input_file = tmp_path / "sample.pdf"
    input_file.write_bytes(b"%PDF-1.4 mock")
    return StageContext(
        config=cast("Any", config),
        router=cast("Any", SimpleNamespace()),
        ledger=cast("Any", SimpleNamespace(set_job_metadata_value=MagicMock())),
        manifest=cast("Any", SimpleNamespace(title="Book", doc_id="d", metadata={})),
        job_id="job_reflow_companion",
        input_path=input_file,
        source_lang="en",
        target_lang="zh",
        profile_name="general",
        create_event=cast("Any", lambda *a, **k: None),
    )


def _adapter(
    tmp_path: Path,
    *,
    delivery_mode: str = "facsimile",
    rendered: Path | None = None,
    raises: Exception | None = None,
) -> tuple[Any, list[str]]:
    strategy = SimpleNamespace(delivery_mode=delivery_mode)
    seen_modes: list[str] = []
    adapter = SimpleNamespace(_renderer=strategy, engine_name="docling")

    async def _render_blocks(**kwargs: Any) -> Path:
        seen_modes.append(strategy.delivery_mode)
        if raises is not None:
            raise raises
        out = Path(kwargs["output_path"])
        out.write_bytes(b"%PDF-1.4 reflow")
        return out

    adapter.render_blocks = _render_blocks
    return adapter, seen_modes


async def test_the_companion_is_off_by_default(tmp_path: Path) -> None:
    adapter, seen = _adapter(tmp_path)

    result = await _write_reflow_companion(
        _ctx(tmp_path, enabled=False), adapter, [], tmp_path / "book.pdf", RenderPlan()
    )

    assert result.kind == "reflow"
    assert result.status == "skipped"
    assert result.reason is not None and "companion_reflow" in result.reason
    assert seen == [], "a disabled companion must not render anything"


async def test_the_companion_renders_through_the_reflow_channel(tmp_path: Path) -> None:
    adapter, seen = _adapter(tmp_path)
    primary = tmp_path / "book_mono.pdf"
    primary.write_bytes(b"%PDF-1.4 facsimile")

    result = await _write_reflow_companion(
        _ctx(tmp_path), adapter, [], primary, RenderPlan(bilingual_mode="monolingual")
    )

    assert result.status == "generated"
    assert result.path is not None and result.path.exists()
    assert seen == ["reflow"], "the channel must be swapped for the companion render"
    assert result.path.name == "book_mono_reflow.pdf"
    assert result.path != primary, "the companion must not overwrite the delivery"
    assert result.size_bytes and result.size_bytes > 0


async def test_the_primary_delivery_mode_is_restored_afterwards(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    primary = tmp_path / "book.pdf"
    primary.write_bytes(b"%PDF")

    await _write_reflow_companion(_ctx(tmp_path), adapter, [], primary, RenderPlan())

    assert adapter._renderer.delivery_mode == "facsimile"


async def test_the_mode_is_restored_even_when_the_render_fails(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path, raises=RuntimeError("typst died"))
    primary = tmp_path / "book.pdf"
    primary.write_bytes(b"%PDF")

    result = await _write_reflow_companion(_ctx(tmp_path), adapter, [], primary, RenderPlan())

    assert result.status == "failed"
    assert adapter._renderer.delivery_mode == "facsimile"


async def test_a_failure_is_recorded_with_its_reason(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path, raises=RuntimeError("typst died"))
    primary = tmp_path / "book.pdf"
    primary.write_bytes(b"%PDF")

    result = await _write_reflow_companion(_ctx(tmp_path), adapter, [], primary, RenderPlan())

    assert result.status == "failed"
    assert result.error is not None
    assert "typst died" in result.error


async def test_a_reflow_primary_skips_instead_of_duplicating(tmp_path: Path) -> None:
    adapter, seen = _adapter(tmp_path, delivery_mode="reflow")
    primary = tmp_path / "book.pdf"
    primary.write_bytes(b"%PDF")

    result = await _write_reflow_companion(_ctx(tmp_path), adapter, [], primary, RenderPlan())

    assert result.status == "skipped"
    assert result.reason is not None and "already reflow" in result.reason
    assert seen == [], "no second render should run"


async def test_a_render_that_writes_nothing_is_a_failure_not_a_success(
    tmp_path: Path,
) -> None:
    strategy = SimpleNamespace(delivery_mode="facsimile")
    adapter = SimpleNamespace(_renderer=strategy, engine_name="docling")

    async def _render_blocks(**kwargs: Any) -> Path:
        return Path(kwargs["output_path"])

    adapter.render_blocks = _render_blocks
    primary = tmp_path / "book.pdf"
    primary.write_bytes(b"%PDF")

    result = await _write_reflow_companion(_ctx(tmp_path), adapter, [], primary, RenderPlan())

    assert result.status == "failed"
    assert result.error is not None and "without writing" in result.error


def test_the_companion_is_wired_into_the_export_stage() -> None:
    import inspect

    from ubt.core.engine.stages import export as export_module

    stage = inspect.getsource(export_module.run_export_stage)
    assert "_write_reflow_companion" in stage, "the exporter is never called"
    assert "companion_results.append" in stage, "the result never reaches the summary"
    assert "companion_summary" in stage, "the summary is never built/persisted"

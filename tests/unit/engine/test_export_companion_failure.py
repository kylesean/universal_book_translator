"""A failed complementary render must be auditable, not silently omitted."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pikepdf
import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.facts import Terminology
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.engine.services import RunServices
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages import export as export_module
from ubt.core.engine.stages.export import run_export_stage
from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    BoundingBox,
    ChapterIR,
    IRBlock,
    make_element,
)
from ubt.core.ir.render_plan import RenderPlan
from ubt.core.policy.adaptive_policy import AdaptivePolicy
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.fast_pass import FastPassFilter

pytestmark = pytest.mark.fast

_JOB = "companion-failure-job"


def _block() -> IRBlock:
    element = make_element(
        id="b001",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Done sentence.",
        bbox=BoundingBox(page=1, x0=50.0, y0=700.0, x1=350.0, y1=712.0),
    )
    block = IRBlock(element=element)
    block.target_text = "已完成。"
    block.status = BlockStatus.MTQE_PASSED
    return block


def _write_pdf(path: Path) -> Path:
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(300, 400))
    pdf.save(path)
    pdf.close()
    return path


async def test_a_failed_complementary_render_is_recorded_and_the_job_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "ledger.sqlite"
    lock = LedgerWriterLock(db_path, _JOB, lease_seconds=60.0)
    lock.acquire()
    ledger = SQLiteJobLedger(db_path)
    ledger.bind_writer_lock(lock)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4 source")
    manifest = BookManifest(doc_id=_JOB, title="t", source_path=str(source), target_lang="zh")
    ledger.init_job_from_manifest(_JOB, manifest)
    ledger.append_chapter(
        _JOB,
        ChapterIR(doc_id=_JOB, chapter_id="c1", title="C1", spine_index=0, blocks=[_block()]),
    )

    output = tmp_path / "book.pdf"

    class _Adapter:
        engine_name = "docling"
        last_render_skips: list[tuple[str, str]] = []
        last_render_flags: list[tuple[str, str]] = []
        last_render_reconstructions: list[Any] = []
        last_page_geometry: list[dict[str, Any]] = []
        last_render_outcome = None

        async def render_blocks(self, **kwargs: Any) -> Path:
            return _write_pdf(Path(kwargs["output_path"]))

    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        enable_pdf_optimization=False,
        visual_gate_enabled=False,
        emit_xliff_companion=False,
        emit_html_companion=False,
        emit_epub_companion=False,
        emit_attestation_shadow=False,
        table_render_mode="skip",
    )
    services = RunServices(
        fast_pass=FastPassFilter(source_lang="en", target_lang="zh"),
        qe_runner=cast(BaseQERunner, object()),
        repair_loop=cast(RepairLoop, object()),
        adaptive_policy=AdaptivePolicy(
            fast_lane_bible=False,
            visual_blocking=False,
            deterministic_glossary=False,
            reason="test",
        ),
        concurrency_sem=asyncio.Semaphore(1),
    )

    async def _create_event(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace()

    ctx = StageContext(
        config=config,
        router=cast(Any, SimpleNamespace(provider=SimpleNamespace(is_mock=True))),
        ledger=ledger,
        manifest=manifest,
        adapter=cast(Any, _Adapter()),
        job_id=_JOB,
        input_path=source,
        output_path=output,
        source_lang="en",
        target_lang="zh",
        profile_name="general",
        create_event=cast(Any, _create_event),
    )

    async def _fake_gate(*_args: Any, **_kwargs: Any) -> tuple[Path, None, None]:
        return output, None, None

    def _ok_contract(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            passed=True,
            errors=[],
            model_dump=lambda **_kw: {"violations": []},
            summary_line=lambda: "ok",
        )

    def _no_delivery(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _fake_reports(*_args: Any, **_kwargs: Any) -> tuple[Any, Path]:
        return SimpleNamespace(), tmp_path / "quality_report.json"

    async def _boom(*_args: Any, **_kwargs: Any) -> tuple[None, str]:
        return None, "boom"

    monkeypatch.setattr(export_module, "_run_visual_gate", _fake_gate)
    monkeypatch.setattr(export_module, "_deliver_contract", _ok_contract)
    monkeypatch.setattr(export_module, "_evaluate_delivery", _no_delivery)
    monkeypatch.setattr(export_module, "_build_reports", _fake_reports)
    monkeypatch.setattr(export_module, "_render_complementary_artifact", _boom)

    try:
        async for _event in run_export_stage(
            ctx,
            services,
            RenderPlan(bilingual_mode="monolingual", effective_dual_mode="mono"),
            Terminology(),
        ):
            pass

        stored = ledger.get_job_metadata_value(_JOB, "companion_artifacts")
        assert isinstance(stored, dict), "companion artifacts were not persisted"
        results = [r for r in stored["results"] if r["kind"] == "complementary_pdf"]
        assert results, "the failed complementary render was not recorded"
        assert results[0]["status"] == "failed"
        assert "boom" in results[0]["error"]
        assert ledger.get_job_status(_JOB) == "completed"
    finally:
        ledger.close()
        lock.release()

"""Regression tests for faithful->readable degradation and native contract evidence."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pikepdf
import pytest

from ubt.codecs.protocol import OverflowItem
from ubt.core.config import UBTConfig
from ubt.core.engine.facts import Terminology
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.engine.services import RunServices
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages import export as export_module
from ubt.core.engine.stages.export import _delivery_contract_evidence, run_export_stage
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


def _write_pdf(path: Path, pages: int = 1) -> Path:
    pdf = pikepdf.Pdf.new()
    for _ in range(pages):
        pdf.add_blank_page(page_size=(300, 400))
    pdf.save(path)
    pdf.close()
    return path


def _make_export_ctx(
    tmp_path: Path,
    *,
    job_id: str,
    adapter: Any,
    bilingual_mode: str = "monolingual",
    effective_dual_mode: str = "mono",
    allow_degrade: bool = True,
) -> tuple[StageContext, SQLiteJobLedger, LedgerWriterLock, RunServices, RenderPlan, Path]:
    db_path = tmp_path / f"{job_id}.sqlite"
    lock = LedgerWriterLock(db_path, job_id, lease_seconds=60.0)
    lock.acquire()
    ledger = SQLiteJobLedger(db_path)
    ledger.bind_writer_lock(lock)

    source = _write_pdf(tmp_path / "source.pdf", pages=1)
    manifest = BookManifest(
        doc_id=job_id,
        title="Test Book",
        source_path=str(source),
        target_lang="zh",
        metadata={
            "delivery_mode": "faithful",
            "delivery_mode_effective": "faithful",
            "delivery_mode_requested": "faithful",
            "delivery_plan": {
                "schema_version": 1,
                "requested_mode": "faithful",
                "contract": "faithful",
                "degrade_policy": "allow_degrade" if allow_degrade else "fail",
                "reasons": ["eligible"],
            },
        },
    )
    ledger.init_job_from_manifest(job_id, manifest)
    b_ok = IRBlock(
        element=make_element(
            id="b001",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Done sentence.",
            bbox=BoundingBox(page=1, x0=50.0, y0=700.0, x1=350.0, y1=712.0),
        ),
        status=BlockStatus.MTQE_PASSED,
        target_text="已完成译文。",
    )
    ledger.append_chapter(
        job_id, ChapterIR(doc_id=job_id, chapter_id="c1", title="C1", spine_index=0, blocks=[b_ok])
    )

    output = tmp_path / "book.pdf"
    config = UBTConfig(
        db_dir=tmp_path,
        allowed_dirs=str(tmp_path),
        allow_degrade=allow_degrade,
        enable_pdf_optimization=False,
        artifact_gate_enabled=False,
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
            artifact_blocking=False,
            deterministic_glossary=False,
            reason="test",
        ),
        concurrency_sem=asyncio.Semaphore(1),
    )

    async def _create_event(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(event_type=SimpleNamespace(value="event"))

    ctx = StageContext(
        config=config,
        router=cast(
            Any,
            SimpleNamespace(
                provider=SimpleNamespace(is_mock=False),
                usage_totals_by_model=lambda: {},
            ),
        ),
        ledger=ledger,
        manifest=manifest,
        adapter=cast(Any, adapter),
        job_id=job_id,
        input_path=source,
        output_path=output,
        source_lang="en",
        target_lang="zh",
        profile_name="general",
        create_event=cast(Any, _create_event),
    )
    render_plan = RenderPlan(bilingual_mode=bilingual_mode, effective_dual_mode=effective_dual_mode)
    return ctx, ledger, lock, services, render_plan, output


@pytest.mark.asyncio
async def test_export_degrade_switches_to_docling_and_persists_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class _NativeAdapter:
        engine_name = "native"
        last_render_skips: list[tuple[str, str]] = []
        last_render_flags: list[tuple[str, str]] = []
        last_render_reconstructions: list[Any] = []
        last_page_geometry: list[dict[str, Any]] = []
        last_overflows: tuple[OverflowItem, ...] = ()
        last_units: tuple[Any, ...] = ()
        last_render_outcome = None

        async def render_blocks(self, **kwargs: Any) -> Path:
            calls.append("native")
            return _write_pdf(Path(kwargs["output_path"]))

    class _DoclingReflowAdapter:
        engine_name = "docling"
        last_render_skips: list[tuple[str, str]] = []
        last_render_flags: list[tuple[str, str]] = []
        last_render_reconstructions: list[Any] = []
        last_page_geometry: list[dict[str, Any]] = []
        last_render_outcome = None

        async def render_blocks(self, **kwargs: Any) -> Path:
            calls.append("docling")
            return _write_pdf(Path(kwargs["output_path"]), pages=2)

    native_adapter = _NativeAdapter()
    ctx, ledger, lock, services, render_plan, output = _make_export_ctx(
        tmp_path, job_id="job-degrade-real", adapter=native_adapter
    )
    gate_calls: list[bool] = []

    async def _fake_gate(*args: Any, **kwargs: Any) -> tuple[Path, None, None]:
        del kwargs
        gate_calls.append(export_module._is_faithful_delivery(args[0]))
        return output, None, None

    def _ok_contract(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            passed=True,
            errors=[],
            summary_line=lambda: "ok",
            model_dump=lambda mode="json": {"passed": True, "violations": []},
        )

    eval_contracts: list[str] = []

    def _fake_eval(
        _ctx: Any,
        adapter_arg: Any,
        _blocks: Any,
        _rendered: Any,
        _dc: Any,
        _pg: Any = None,
    ) -> Any:
        eval_contracts.append(str(getattr(adapter_arg, "engine_name", "")))
        if getattr(adapter_arg, "engine_name", "") == "native":
            return SimpleNamespace(
                passed=False,
                deliverable=False,
                contract="faithful",
                findings=(),
                summary_line=lambda: "[BLOCKED] faithful",
            )
        return SimpleNamespace(
            passed=True,
            deliverable=True,
            contract="readable",
            findings=(),
            summary_line=lambda: "[PASSED] readable",
        )

    monkeypatch.setattr(export_module, "_run_artifact_gate", _fake_gate)
    monkeypatch.setattr(export_module, "_deliver_contract", _ok_contract)
    monkeypatch.setattr(export_module, "_evaluate_delivery", _fake_eval)
    monkeypatch.setattr(
        "ubt.core.engine.document_services.resolve_adapter",
        lambda _path, pdf_engine="auto": _DoclingReflowAdapter(),
    )

    try:
        events = [
            ev
            async for ev in run_export_stage(
                ctx, services, render_plan, Terminology(glossary_dicts=[], abbreviation_entries=[])
            )
        ]
        assert events
        assert calls == ["native", "docling"]
        assert eval_contracts == ["native", "docling"]
        assert gate_calls == [True, False]
        assert ledger.get_job_metadata_value(ctx.job_id, "delivery_mode") == "readable"
        assert ledger.get_job_metadata_value(ctx.job_id, "delivery_mode_effective") == "readable"
        stored_plan = ledger.get_job_metadata_value(ctx.job_id, "delivery_plan")
        assert isinstance(stored_plan, dict)
        assert stored_plan["contract"] == "readable"
        assert stored_plan["degraded"] is True
    finally:
        ledger.close()
        lock.release()


@pytest.mark.asyncio
async def test_export_does_not_fake_readable_when_docling_degrade_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class _NativeAdapter:
        engine_name = "native"
        last_render_skips: list[tuple[str, str]] = []
        last_render_flags: list[tuple[str, str]] = []
        last_render_reconstructions: list[Any] = []
        last_page_geometry: list[dict[str, Any]] = []
        last_overflows: tuple[OverflowItem, ...] = ()
        last_units: tuple[Any, ...] = ()
        last_render_outcome = None

        async def render_blocks(self, **kwargs: Any) -> Path:
            calls.append("native")
            return _write_pdf(Path(kwargs["output_path"]))

    ctx, ledger, lock, services, render_plan, output = _make_export_ctx(
        tmp_path, job_id="job-degrade-fail", adapter=_NativeAdapter()
    )

    async def _fake_gate(*_args: Any, **_kwargs: Any) -> tuple[Path, None, None]:
        return output, None, None

    def _ok_contract(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            passed=True,
            errors=[],
            summary_line=lambda: "ok",
            model_dump=lambda mode="json": {"passed": True, "violations": []},
        )

    def _failed_eval(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            passed=False,
            deliverable=False,
            contract="faithful",
            findings=(),
            summary_line=lambda: "[BLOCKED] faithful",
        )

    monkeypatch.setattr(export_module, "_run_artifact_gate", _fake_gate)
    monkeypatch.setattr(export_module, "_deliver_contract", _ok_contract)
    monkeypatch.setattr(export_module, "_evaluate_delivery", _failed_eval)
    monkeypatch.setattr(
        "ubt.core.engine.document_services.resolve_adapter",
        lambda _path, pdf_engine="auto": (_ for _ in ()).throw(RuntimeError("no docling")),
    )

    try:
        async for _ in run_export_stage(
            ctx, services, render_plan, Terminology(glossary_dicts=[], abbreviation_entries=[])
        ):
            pass
        assert calls == ["native"]
        assert ctx.manifest.metadata["delivery_mode"] == "faithful"
        assert ledger.get_job_metadata_value(ctx.job_id, "delivery_mode") != "readable"
    finally:
        ledger.close()
        lock.release()


def test_bilingual_native_evidence_expects_double_page_count_and_reads_overflows(
    tmp_path: Path,
) -> None:
    source = _write_pdf(tmp_path / "src.pdf", pages=2)
    rendered = _write_pdf(tmp_path / "out.pdf", pages=4)
    manifest = BookManifest(
        doc_id="bi-native",
        title="Bi",
        source_path=str(source),
        target_lang="zh",
        metadata={
            "delivery_mode": "faithful",
            "delivery_plan": {"contract": "faithful"},
            "render_plan": {"bilingual_mode": "bilingual", "effective_dual_mode": "dual"},
        },
    )
    adapter = SimpleNamespace(
        engine_name="native",
        last_page_geometry=[],
        last_units=(),
        last_overflows=(
            OverflowItem(
                unit_id="u1",
                page_no=1,
                overflow_ratio=1.3,
                target_char_budget=10,
                applied_font_scale=0.7,
                clipped=True,
            ),
        ),
    )
    ctx = SimpleNamespace(
        manifest=manifest,
        config=UBTConfig(db_dir=tmp_path, allowed_dirs=str(tmp_path)),
    )
    evidence = _delivery_contract_evidence(
        cast(StageContext, ctx),
        cast(Any, adapter),
        [],
        rendered,
        None,
    )
    assert evidence.expected_page_count == 4
    assert len(evidence.overflow_records) == 1
    assert evidence.overflow_records[0].block_id == "u1"
    assert evidence.overflow_records[0].reason == "native_clipped"


def test_export_contract_gate_strict_vs_advisory() -> None:
    from ubt.core.exceptions import IntegrityViolationError

    failed_report = SimpleNamespace(
        passed=False,
        deliverable=False,
        contract="faithful",
        summary_line=lambda: "[BLOCKED] faithful: 2 critical, 0 major",
    )

    def _simulate_gate(strict: bool) -> bool:
        contract_ok = bool(failed_report.deliverable)
        if not contract_ok and strict:
            raise IntegrityViolationError(
                f"Export blocked: delivery contract failed — {failed_report.summary_line()}"
            )
        return contract_ok

    assert _simulate_gate(strict=False) is False
    with pytest.raises(IntegrityViolationError, match="delivery contract failed"):
        _simulate_gate(strict=True)


@pytest.mark.asyncio
async def test_export_does_not_degrade_native_on_major_only_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Major-only advisories (0 critical) must keep the intact 1:1 native PDF."""
    calls: list[str] = []

    class _NativeAdapter:
        engine_name = "native"
        last_render_skips: list[tuple[str, str]] = []
        last_render_flags: list[tuple[str, str]] = []
        last_render_reconstructions: list[Any] = []
        last_page_geometry: list[dict[str, Any]] = []
        last_overflows: tuple[OverflowItem, ...] = ()
        last_units: tuple[Any, ...] = ()
        last_render_outcome = None

        async def render_blocks(self, **kwargs: Any) -> Path:
            calls.append("native")
            return _write_pdf(Path(kwargs["output_path"]))

    ctx, ledger, lock, services, render_plan, output = _make_export_ctx(
        tmp_path, job_id="job-major-no-degrade", adapter=_NativeAdapter()
    )

    async def _fake_gate(*_args: Any, **_kwargs: Any) -> tuple[Path, None, None]:
        return output, None, None

    def _ok_contract(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            passed=True,
            errors=[],
            summary_line=lambda: "ok",
            model_dump=lambda mode="json": {"passed": True, "violations": []},
        )

    def _major_only_eval(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            passed=True,  # 0 critical findings
            deliverable=False,  # major findings only
            contract="faithful",
            findings=(),
            summary_line=lambda: "[MAJOR] faithful: 0 critical, 3 major, 3 total finding(s)",
        )

    monkeypatch.setattr(export_module, "_run_artifact_gate", _fake_gate)
    monkeypatch.setattr(export_module, "_deliver_contract", _ok_contract)
    monkeypatch.setattr(export_module, "_evaluate_delivery", _major_only_eval)
    monkeypatch.setattr(
        "ubt.core.engine.document_services.resolve_adapter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("docling degrade must not be called on major-only findings")
        ),
    )

    try:
        async for _ in run_export_stage(
            ctx, services, render_plan, Terminology(glossary_dicts=[], abbreviation_entries=[])
        ):
            pass
        assert calls == ["native"]
        assert ctx.manifest.metadata["delivery_mode"] == "faithful"
    finally:
        ledger.close()
        lock.release()


@pytest.mark.asyncio
async def test_export_degrade_sanitizes_native_inline_anchors_before_reflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When critical faithful failure degrades to docling reflow, <f1/> and <c1> must be resolved."""
    from ubt.codecs.protocol import InlineToken, TranslationUnit, UnitRole

    captured_texts: list[str] = []

    class _NativeAdapter:
        engine_name = "native"
        last_render_skips: list[tuple[str, str]] = []
        last_render_flags: list[tuple[str, str]] = []
        last_render_reconstructions: list[Any] = []
        last_page_geometry: list[dict[str, Any]] = []
        last_overflows: tuple[OverflowItem, ...] = ()
        last_units = (
            TranslationUnit(
                unit_id="b001",
                chapter_id="c1",
                role=UnitRole.PARAGRAPH,
                source_marked="Value <f1/> at <c1>https://example.com</c1>.",
                source_plain="Value H at https://example.com.",
                tokens=(
                    InlineToken(tag_id="f1", kind="rigid", payload={"raw_text": "H"}),
                    InlineToken(tag_id="c1", kind="pair", payload={"color_hex": "#0000ff"}),
                ),
            ),
        )
        last_render_outcome = None

        async def render_blocks(self, **kwargs: Any) -> Path:
            return _write_pdf(Path(kwargs["output_path"]))

    class _DoclingReflowAdapter:
        engine_name = "docling"
        last_render_skips: list[tuple[str, str]] = []
        last_render_flags: list[tuple[str, str]] = []
        last_render_reconstructions: list[Any] = []
        last_page_geometry: list[dict[str, Any]] = []
        last_render_outcome = None

        async def render_blocks(self, **kwargs: Any) -> Path:
            for b in kwargs["blocks"]:
                captured_texts.append(b.target_text or "")
            return _write_pdf(Path(kwargs["output_path"]), pages=2)

    ctx, ledger, lock, services, render_plan, output = _make_export_ctx(
        tmp_path, job_id="job-degrade-anchors", adapter=_NativeAdapter()
    )
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "b001",
                "target_text": "数值 <f1/> 见 <c1>https://example.com</c1>。",
                "status": BlockStatus.MTQE_PASSED,
                "error_flags": [],
            }
        ],
        job_id=ctx.job_id,
    )

    async def _fake_gate(*_args: Any, **_kwargs: Any) -> tuple[Path, None, None]:
        return output, None, None

    def _ok_contract(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            passed=True,
            errors=[],
            summary_line=lambda: "ok",
            model_dump=lambda mode="json": {"passed": True, "violations": []},
        )

    def _fake_eval(
        _ctx: Any, adapter_arg: Any, _blocks: Any, _rendered: Any, _dc: Any, _pg: Any = None
    ) -> Any:
        if getattr(adapter_arg, "engine_name", "") == "native":
            return SimpleNamespace(
                passed=False,
                deliverable=False,
                contract="faithful",
                findings=(),
                summary_line=lambda: "[BLOCKED] faithful",
            )
        return SimpleNamespace(
            passed=True,
            deliverable=True,
            contract="readable",
            findings=(),
            summary_line=lambda: "[PASSED] readable",
        )

    monkeypatch.setattr(export_module, "_run_artifact_gate", _fake_gate)
    monkeypatch.setattr(export_module, "_deliver_contract", _ok_contract)
    monkeypatch.setattr(export_module, "_evaluate_delivery", _fake_eval)
    monkeypatch.setattr(
        "ubt.core.engine.document_services.resolve_adapter",
        lambda _path, pdf_engine="auto": _DoclingReflowAdapter(),
    )

    try:
        async for _ in run_export_stage(
            ctx, services, render_plan, Terminology(glossary_dicts=[], abbreviation_entries=[])
        ):
            pass
        assert captured_texts == ["数值 H 见 https://example.com。"]
    finally:
        ledger.close()
        lock.release()

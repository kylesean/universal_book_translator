"""Unit tests for render-engine routing on short documents and adaptive policy."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.stage_ctx_factory import build_stage_ctx
from ubt.adapters.base import BaseDocumentAdapter
from ubt.adapters.markdown.adapter import MarkdownAdapter
from ubt.core.config import UBTConfig
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.ir.models import BlockStatus, BlockType, BookManifest, BoundingBox, FlowID, IRBlock
from ubt.core.policy.adaptive_policy import Granularity, resolve_adaptive_policy, resolve_pdf_engine
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter
from ubt.core.router_mode import RouteDecision


def test_adaptive_policy_defaults_to_micro_granularity() -> None:
    """Default configuration selects micro granularity with frozen math protection."""
    config = UBTConfig(render_engine="publication", exec_mode="short")
    route_dec = RouteDecision(
        mode="short",
        pages=5,
        chars=1000,
        has_scan=False,
        formula_heavy=False,
        reason="short test",
    )
    manifest = BookManifest(doc_id="doc1", title="test", source_path="test.pdf")
    policy = resolve_adaptive_policy(manifest, route_dec, config)

    assert policy.granularity == Granularity.MICRO
    assert policy.render_engine == "publication"
    assert policy.fast_lane_bible is True
    assert policy.deterministic_glossary is True


def test_adaptive_policy_accepts_anchored_engine() -> None:
    """Explicit anchored engine passes through the adaptive policy unchanged."""
    config = UBTConfig(render_engine="rigid", exec_mode="short")
    route_dec = RouteDecision(
        mode="short",
        pages=2,
        chars=500,
        has_scan=False,
        formula_heavy=False,
        reason="short test",
    )
    manifest = BookManifest(doc_id="doc2", title="test2", source_path="test2.pdf")
    policy = resolve_adaptive_policy(manifest, route_dec, config)

    assert policy.granularity == Granularity.MICRO
    assert policy.render_engine == "rigid"


def test_adaptive_policy_long_document_routing() -> None:
    """Long documents use micro granularity without fast-lane shortcuts."""
    config = UBTConfig(render_engine="publication", exec_mode="long")
    route_dec = RouteDecision(
        mode="long",
        pages=120,
        chars=50000,
        has_scan=False,
        formula_heavy=True,
        reason="long book test",
    )
    manifest = BookManifest(doc_id="doc_long", title="Long Book", source_path="book.pdf")
    policy = resolve_adaptive_policy(manifest, route_dec, config)

    assert policy.granularity == Granularity.MICRO
    assert policy.fast_lane_bible is False


@pytest.mark.asyncio
async def test_pipeline_publication_executes_full_stages(tmp_path: Path) -> None:
    """Short documents run through the unified 5-stage pipeline under micro granularity."""
    input_pdf = tmp_path / "short.pdf"
    input_pdf.write_bytes(b"%PDF-1.4 mock short")
    output_pdf = tmp_path / "out.pdf"

    config = UBTConfig(
        render_engine="publication",
        exec_mode="short",
        visual_gate_enabled=False,
        db_dir=tmp_path / "db",
    )
    router = ModelRouter(provider=MockModelProvider(), draft_model="mock")
    orchestrator = PipelineOrchestrator(config=config, router=router)

    mock_manifest = BookManifest(
        doc_id="doc1",
        title="技术报告",
        source_path=str(input_pdf),
        metadata={"total_pages": 5},
    )

    with (
        patch("ubt.core.engine.pipeline.resolve_adapter") as mock_resolve,
        patch("ubt.core.engine.pipeline.decide") as mock_decide,
    ):
        mock_adapter = MagicMock(spec=BaseDocumentAdapter)
        mock_adapter.extract_manifest = AsyncMock(return_value=mock_manifest)
        mock_adapter.render_blocks = AsyncMock(return_value=output_pdf)
        mock_resolve.return_value = mock_adapter

        mock_decide.return_value = RouteDecision(
            mode="short",
            pages=5,
            chars=1000,
            has_scan=False,
            formula_heavy=False,
            reason="short born-digital test",
        )

        events = []
        async for ev in orchestrator.run(input_path=input_pdf, output_path=output_pdf):
            events.append(ev)

        policy = mock_manifest.run.adaptive_policy
        assert policy is not None, "the run never recorded an adaptive policy"
        assert policy["granularity"] == "micro"
        assert policy["render_engine"] == "publication"
        assert events[-1].event_type == EventType.EXPORT_COMPLETED


@pytest.mark.asyncio
async def test_pipeline_anchored_downgrades_to_monolingual(tmp_path: Path) -> None:
    """render_engine='anchored' is preserved and its bilingual request is downgraded."""
    input_pdf = tmp_path / "resume.pdf"
    input_pdf.write_bytes(b"%PDF-1.4 mock resume")
    output_pdf = tmp_path / "out_anchored.pdf"

    config = UBTConfig(
        render_engine="rigid",
        exec_mode="short",
        db_dir=tmp_path / "db",
        tm_enabled=False,
    )
    router = ModelRouter(provider=MockModelProvider(), draft_model="mock")
    orchestrator = PipelineOrchestrator(config=config, router=router)

    mock_manifest = BookManifest(
        doc_id="doc_resume",
        title="个人简历",
        source_path=str(input_pdf),
        metadata={"total_pages": 1},
    )

    with (
        patch("ubt.core.engine.pipeline.resolve_adapter") as mock_resolve,
        patch("ubt.core.engine.pipeline.decide") as mock_decide,
        patch("ubt.core.engine.pipeline.run_ingest_stage") as mock_ingest,
        patch("ubt.core.engine.pipeline.run_bible_stage") as mock_bible,
        patch("ubt.core.engine.pipeline.run_draft_stage") as mock_draft,
        patch("ubt.core.engine.pipeline.run_quality_gate_stage") as mock_qg,
        patch("ubt.core.engine.pipeline.run_repair_stage") as mock_repair,
        patch("ubt.core.engine.pipeline.run_triage_stage") as mock_triage,
        patch("ubt.core.engine.pipeline.run_export_stage") as mock_export,
    ):
        mock_adapter = MagicMock(spec=BaseDocumentAdapter)
        mock_adapter.extract_manifest = AsyncMock(return_value=mock_manifest)
        mock_resolve.return_value = mock_adapter

        mock_decide.return_value = RouteDecision(
            mode="short",
            pages=1,
            chars=800,
            has_scan=False,
            formula_heavy=False,
            reason="short resume",
        )

        async def _empty_async_gen(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[TranslationProgressEvent]:
            if False:
                yield TranslationProgressEvent(
                    event_type=EventType.JOB_STARTED,
                    job_id="dummy",
                    total_blocks=0,
                    completed_blocks=0,
                )

        mock_ingest.side_effect = _empty_async_gen
        mock_bible.side_effect = _empty_async_gen
        mock_draft.side_effect = _empty_async_gen
        mock_qg.side_effect = _empty_async_gen
        mock_repair.side_effect = _empty_async_gen
        mock_triage.side_effect = _empty_async_gen

        export_event = TranslationProgressEvent(
            event_type=EventType.EXPORT_COMPLETED,
            job_id="test_job_anchored",
            total_blocks=1,
            completed_blocks=1,
            message="Export complete",
        )

        async def _export_async_gen(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[TranslationProgressEvent]:
            yield export_event

        mock_export.side_effect = _export_async_gen

        events = []
        async for ev in orchestrator.run(input_path=input_pdf, output_path=output_pdf):
            events.append(ev)

        assert mock_ingest.called
        assert mock_manifest.run.render_engine == "rigid"
        assert mock_manifest.run.effective_dual_mode == "monolingual"
        assert mock_manifest.run.bilingual_mode == "monolingual"
        assert mock_manifest.run.dual_mode_downgraded == "inline"
        assert events[-1].event_type == EventType.EXPORT_COMPLETED


@pytest.mark.asyncio
async def test_pipeline_overlay_formula_heavy_marked_advisory(tmp_path: Path) -> None:
    """A formula_heavy document forced onto overlay is an informed tradeoff,
    not a delivery failure: the status must be a LAYOUT_TRADEOFF_ADVISORY,
    never an UNSUITABLE stop (auto routing sends these to overlay on purpose)."""
    input_pdf = tmp_path / "paper.pdf"
    input_pdf.write_bytes(b"%PDF-1.4 mock paper")
    output_pdf = tmp_path / "out_overlay.pdf"

    config = UBTConfig(
        render_engine="rigid",
        exec_mode="short",
        db_dir=tmp_path / "db",
        tm_enabled=False,
    )
    router = ModelRouter(provider=MockModelProvider(), draft_model="mock")
    orchestrator = PipelineOrchestrator(config=config, router=router)

    mock_manifest = BookManifest(
        doc_id="doc_paper",
        title="学术论文",
        source_path=str(input_pdf),
        metadata={"total_pages": 5},
    )

    with (
        patch("ubt.core.engine.pipeline.resolve_adapter") as mock_resolve,
        patch("ubt.core.engine.pipeline.decide") as mock_decide,
        patch("ubt.core.engine.pipeline.run_ingest_stage") as mock_ingest,
        patch("ubt.core.engine.pipeline.run_bible_stage") as mock_bible,
        patch("ubt.core.engine.pipeline.run_draft_stage") as mock_draft,
        patch("ubt.core.engine.pipeline.run_quality_gate_stage") as mock_qg,
        patch("ubt.core.engine.pipeline.run_repair_stage") as mock_repair,
        patch("ubt.core.engine.pipeline.run_triage_stage") as mock_triage,
        patch("ubt.core.engine.pipeline.run_export_stage") as mock_export,
    ):
        mock_adapter = MagicMock(spec=BaseDocumentAdapter)
        mock_adapter.extract_manifest = AsyncMock(return_value=mock_manifest)
        mock_resolve.return_value = mock_adapter

        mock_decide.return_value = RouteDecision(
            mode="short",
            pages=5,
            chars=3000,
            has_scan=False,
            formula_heavy=True,
            reason="short paper with heavy math",
        )

        async def _empty_async_gen(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[TranslationProgressEvent]:
            if False:
                yield TranslationProgressEvent(
                    event_type=EventType.JOB_STARTED,
                    job_id="dummy",
                    total_blocks=0,
                    completed_blocks=0,
                )

        mock_ingest.side_effect = _empty_async_gen
        mock_bible.side_effect = _empty_async_gen
        mock_draft.side_effect = _empty_async_gen
        mock_qg.side_effect = _empty_async_gen
        mock_repair.side_effect = _empty_async_gen
        mock_triage.side_effect = _empty_async_gen
        mock_export.side_effect = _empty_async_gen

        events = []
        async for ev in orchestrator.run(input_path=input_pdf, output_path=output_pdf):
            events.append(ev)

        status = mock_manifest.run.delivery_status
        warning = mock_manifest.run.delivery_warning
        assert status is not None and status.startswith("LAYOUT_TRADEOFF_ADVISORY")
        assert "UNSUITABLE" not in status
        assert warning is not None and "formula-dense" in warning


async def test_export_stage_non_destructive_glossary_validation(tmp_path: Path) -> None:
    """Verify export stage performs non-destructive validation by default instead of destructive string replacement."""
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.stages.export import run_export_stage
    from ubt.core.ir.models import DocumentIR
    from ubt.core.validators.html_delta import HTMLDeltaValidator

    db_path = tmp_path / "ledger.db"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_test_glossary"

    # Glossary specifies "network" -> "网络", but translation produced "计算机网络系统"
    # A destructive Aho-Corasick canonicalizer might corrupt it or overwrite it unexpectedly.
    # Non-destructive mode should preserve the LLM's draft and flag inconsistency.
    glossary = [{"source": "neural network", "translation": "神经网络", "target": "神经网络"}]
    doc_ir = DocumentIR(
        doc_id=job_id,
        source_path=str(tmp_path / "book.md"),
        format_type="markdown",
        blocks=[
            IRBlock(
                id="b1",
                spine_index=1,
                flow_id=FlowID.MAIN_STORY,
                block_type=BlockType.NARRATIVE,
                source_text="We implement a deep neural network.",
                target_text="我们实现了一个深层网络模型。",  # Misses "神经网络"
                status=BlockStatus.MTQE_PASSED,
            )
        ],
    )
    ledger.init_job(job_id, doc_ir, target_lang="zh")

    # Mark block as MTQE_PASSED in ledger
    ledger.save_checkpoint(
        "b1",
        BlockStatus.MTQE_PASSED,
        target_text="我们实现了一个深层网络模型。",
    )

    manifest = BookManifest(
        doc_id=job_id,
        title="Test Book",
        source_path=str(tmp_path / "book.md"),
        target_lang="zh",
        source_lang="en",
    )
    adapter = MarkdownAdapter()
    out_path = tmp_path / "output.md"

    async def dummy_event(*args: Any, **kwargs: Any) -> None:
        return None

    # Run default non-destructive export
    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id=job_id,
        manifest=manifest,
        adapter=adapter,
        output_path=out_path,
        input_path=tmp_path / "book.md",
        target_lang="zh",
        source_lang="en",
        glossary_dicts=glossary,
        html_validator=HTMLDeltaValidator(),
        create_event=dummy_event,
    )
    _events = [e async for e in run_export_stage(ctx)]

    all_blocks = ledger.get_all_blocks(job_id)
    # Target text should be preserved (NOT destroyed by mechanical string replacement)
    assert all_blocks[0].target_text == "我们实现了一个深层网络模型。"
    # Inconsistency should be recorded in error_flags
    assert any("glossary_inconsistency" in flag for flag in all_blocks[0].error_flags)


async def test_export_stage_reraises_cancellation_from_visual_gate(tmp_path: Path) -> None:
    """P0 regression: a cancel landing inside the visual gate must abort export.

    ``ReflowControlLoop.run`` raises ``JobInterruptedError``; the gate's
    blanket ``except Exception`` used to log it as "non-fatal" and continue,
    so the run rendered, wrote the report and stamped the job "completed"
    while the queue said cancelled.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.stages.export import run_export_stage
    from ubt.core.exceptions import JobInterruptedError
    from ubt.core.ir.models import DocumentIR

    job_id = "job_cancel_gate"
    ledger = SQLiteJobLedger(tmp_path / "ledger_cancel.db")
    doc_ir = DocumentIR(
        doc_id=job_id,
        source_path=str(tmp_path / "book.md"),
        format_type="markdown",
        blocks=[
            IRBlock(
                id="b1",
                spine_index=1,
                flow_id=FlowID.MAIN_STORY,
                block_type=BlockType.NARRATIVE,
                source_text="Hello world.",
                target_text="你好，世界。",
                status=BlockStatus.MTQE_PASSED,
            )
        ],
    )
    ledger.init_job(job_id, doc_ir, target_lang="zh")

    manifest = BookManifest(
        doc_id=job_id,
        title="Cancel Gate",
        source_path=str(tmp_path / "book.md"),
        source_lang="en",
        target_lang="zh",
    )
    fake_pdf = tmp_path / "out.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n")

    async def fake_render(*args: Any, **kwargs: Any) -> Path:
        return fake_pdf

    class GateThatRaisesCancel:
        """Stand-in for ReflowControlLoop: cancel arrives mid-gate, as in production."""

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def run(
            self, *, rendered_path: Path, blocks: list[IRBlock]
        ) -> tuple[Path, Path, Any]:
            raise JobInterruptedError("cancelled inside the visual gate")

    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id=job_id,
        manifest=manifest,
        adapter=MarkdownAdapter(),
        output_path=tmp_path / "out.md",
        input_path=tmp_path / "book.md",
        target_lang="zh",
        source_lang="en",
        config=UBTConfig(visual_gate_enabled=True),
    )
    with (
        patch("ubt.core.engine.stages.export._render_adapter_output", fake_render),
        patch("ubt.core.engine.reflow_loop.ReflowControlLoop", GateThatRaisesCancel),
        pytest.raises(JobInterruptedError),
    ):
        async for _ in run_export_stage(ctx):
            pass
    # The cancelled run must not have shipped a completed export artifact.
    assert not (tmp_path / "out.md").exists()
    ledger.close()


# --- auto dispatch: resolve_pdf_engine -------------------------------------
# These had no tests when the dispatch direction was flipped in cc8da7e, which
# is how "formula-dense -> reflow" replaced "formula-dense -> anchored" with the
# module docstring still describing the latter. arXiv 2609.20519 settled which
# way is right: reflow shattered its multi-row tables and lost 4 of 6 figures.


def _blocks(*specs: tuple[str, BlockType]) -> list[IRBlock]:
    return [
        IRBlock(
            id=f"ch01#b{i:03d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            source_text=text,
            block_type=btype,
            bbox=BoundingBox(page=i, x0=72.0, y0=100.0, x1=540.0, y1=140.0),
        )
        for i, (text, btype) in enumerate(specs, start=1)
    ]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("publication", "publication"),
        ("reflow", "publication"),
        ("rigid", "rigid"),
    ],
)
def test_explicit_render_engine_passes_through(requested: str, expected: str) -> None:
    blocks = _blocks(("anything", BlockType.NARRATIVE))
    assert resolve_pdf_engine(requested, blocks) == expected


def test_auto_sends_math_dense_document_to_overlay() -> None:
    """A $ in prose is enough: reflow must rebuild only what it can extract."""
    blocks = _blocks(
        ("The field $E_{th}$ governs the regime.", BlockType.NARRATIVE),
        ("Ordinary narrative prose about the method.", BlockType.NARRATIVE),
    )
    assert resolve_pdf_engine("auto", blocks) == "rigid"


def test_auto_sends_table_heavy_document_to_overlay() -> None:
    blocks = _blocks(
        ("| a | b |", BlockType.TABLE),
        ("| c | d |", BlockType.TABLE),
        ("Some prose between the tables.", BlockType.NARRATIVE),
        ("More prose here.", BlockType.NARRATIVE),
    )
    assert resolve_pdf_engine("auto", blocks) == "rigid"


def test_auto_keeps_plain_prose_on_reflow() -> None:
    blocks = _blocks(
        ("It was the best of times, it was the worst of times.", BlockType.NARRATIVE),
        ("Chapter One: The Period", BlockType.HEADING),
        ("A century of change followed.", BlockType.NARRATIVE),
    )
    assert resolve_pdf_engine("auto", blocks) == "publication"


def test_auto_falls_back_to_reflow_without_blocks_and_on_unknown_names() -> None:
    assert resolve_pdf_engine("auto", []) == "publication"
    assert resolve_pdf_engine("quantum-typesetting", _blocks(("x", BlockType.NARRATIVE))) == (
        "publication"
    )


def test_auto_routes_geometryless_fallback_blocks_to_reflow() -> None:
    """Plain-text fallback extraction stamps zero-area bboxes: rigid cannot
    zone a single block, so auto must not dispatch there and fail at export
    after the whole translation has been paid for."""
    zero_blocks = [
        IRBlock(
            id=f"ch01#b{i:03d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            source_text=text,
            block_type=btype,
            bbox=BoundingBox(page=i, x0=0.0, y0=0.0, x1=0.0, y1=0.0),
        )
        for i, (text, btype) in enumerate(
            (
                ("The field $E_{th}$ governs the regime.", BlockType.NARRATIVE),
                ("| a | b |", BlockType.TABLE),
            ),
            start=1,
        )
    ]
    assert resolve_pdf_engine("auto", zero_blocks) == "publication"
    # Explicit rigid intent is still honored (the fail-closed guard in the
    # typesetter owns that refusal, not the dispatcher).
    assert resolve_pdf_engine("rigid", zero_blocks) == "rigid"


# --- forced render-engine disagreement warning (docling_render) -------------


def _geo_blocks(*specs: tuple[str, Any]) -> list[Any]:
    from ubt.core.ir.models import BoundingBox

    return [
        IRBlock(
            id=f"ch01#b{i:03d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            source_text=text,
            block_type=btype,
            bbox=BoundingBox(page=i, x0=72.0, y0=100.0, x1=540.0, y1=140.0),
        )
        for i, (text, btype) in enumerate(specs, start=1)
    ]


def test_forced_reflow_on_structure_dense_document_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from ubt.adapters.pdf.docling_render import _warn_forced_engine

    blocks = _geo_blocks(
        ("| a | b |", BlockType.TABLE),
        ("| c | d |", BlockType.TABLE),
        ("Prose.", BlockType.NARRATIVE),
    )
    with caplog.at_level("WARNING", logger="ubt.adapters.pdf.docling_render"):
        _warn_forced_engine("reflow", "publication", blocks)
    assert "auto dispatch would route this document to 'rigid'" in caplog.text


def test_auto_or_agreeing_forced_engine_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from ubt.adapters.pdf.docling_render import _warn_forced_engine

    prose = _geo_blocks(("Plain narrative prose.", BlockType.NARRATIVE))
    with caplog.at_level("WARNING", logger="ubt.adapters.pdf.docling_render"):
        _warn_forced_engine("auto", "publication", prose)
        _warn_forced_engine("reflow", "publication", prose)  # agrees with auto
    assert caplog.text == ""


@pytest.mark.fast
def test_adaptive_policy_ignores_currency_dollars() -> None:
    from ubt.core.ir.models import BoundingBox

    blocks = [
        IRBlock(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="The subscription fee is $50 per month or $100 per year.",
            bbox=BoundingBox(page=1, x0=10, y0=10, x1=500, y1=100),
        ),
        IRBlock(
            id="b2",
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text="Prices may vary by region.",
            bbox=BoundingBox(page=1, x0=10, y0=110, x1=500, y1=200),
        ),
    ]
    # Should choose "publication" for prose with dollar prices, NOT "rigid"
    engine = resolve_pdf_engine("auto", blocks)
    assert engine == "publication", f"Expected publication engine for currency, got {engine}"


@pytest.mark.fast
def test_resolve_pdf_engine_respects_manifest_formula_heavy() -> None:
    """When IR blocks do not have explicit BlockType.FORMULA (e.g. fast ingest without VLM),
    resolve_pdf_engine must consult manifest.run.route_decision to prevent split-brain routing.

    Regression: In 2608.25512v1, page profiler detected formula_heavy=True in discovery,
    but docling_render recomputed resolve_pdf_engine on IR blocks (which had 0 formula blocks),
    concluded auto would choose 'publication', and falsely warned that forced 'rigid' was wrong.
    """
    from ubt.core.ir.models import BookManifest, BoundingBox

    blocks = [
        IRBlock(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="This paper introduces a programming paradigm for spatiotemporal composability.",
            bbox=BoundingBox(page=1, x0=10, y0=10, x1=500, y1=100),
        ),
    ]

    manifest = BookManifest(
        doc_id="test_doc",
        title="Test Doc",
        source_path="/fake/path.pdf",
    )
    manifest.run.route_decision = {"formula_heavy": True, "mode": "standard", "pages": 92}

    # With formula_heavy=True on manifest, auto must resolve to 'rigid'
    engine = resolve_pdf_engine("auto", blocks, manifest=manifest)
    assert engine == "rigid", f"Expected rigid for formula_heavy manifest, got {engine}"

    # Without manifest or formula_heavy=False, it resolves to 'publication'
    engine_plain = resolve_pdf_engine("auto", blocks)
    assert engine_plain == "publication"

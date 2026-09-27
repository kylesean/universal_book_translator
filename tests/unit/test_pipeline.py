"""Unit and integration tests for 6-stage PipelineOrchestrator."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.mock_providers import TokenEchoMockProvider
from tests.stage_ctx_factory import build_stage_ctx
from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.adapters.markdown.adapter import MarkdownAdapter
from ubt.core.config import UBTConfig
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.pipeline import PipelineOrchestrator, derive_job_id
from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.engine.stages.export import run_export_stage
from ubt.core.exceptions import IntegrityViolationError
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterMeta,
    FlowID,
    IRBlock,
)
from ubt.core.job_options import sidecar_path
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.router.provider import BaseModelProvider, MockModelProvider, OpenAICompatibleProvider
from ubt.core.router.rate_limiter import AdaptiveTokenBucket
from ubt.core.router.router import ModelRouter
from ubt.core.validators.html_delta import HTMLDeltaValidator


@pytest.fixture
def sample_markdown(tmp_path: Path) -> Path:
    md_file = tmp_path / "sample_book.md"
    content = """# Chapter 1: The Beginning

It was a bright cold day in April, and the clocks were striking thirteen.
Winston Smith slipped quickly through the glass doors.

```python
print("Protected code block")
```

# Chapter 2: The Ministry of Truth

The Ministry of Truth contained three thousand rooms above ground.
"""
    md_file.write_text(content, encoding="utf-8")
    return md_file


@pytest.mark.asyncio
async def test_pipeline_orchestrator_end_to_end(sample_markdown: Path, tmp_path: Path) -> None:
    db_dir = tmp_path / "ledgers"
    output_file = tmp_path / "bilingual_sample.md"

    config = UBTConfig(
        db_dir=db_dir,
        draft_model="mock-draft",
        repair_model="mock-repair",
        rate_limit_rpm=600,
    )

    mock_provider = MockModelProvider(
        default_response="这是用于流水线全流程测试的中文双语对照测试翻译。",
        custom_responses={
            "Translate\n# Chapter 1": "# 第一章 1：开端\n",
            "Translate\nIt was a bright": "四月里一个晴朗而寒冷的日子，时钟敲了十三下。温斯顿·史密斯快步溜进了玻璃门中。这是中文测试翻译。",
            "Translate\n# Chapter 2": "# 第二章 2：真理部\n",
            "Translate\nThe Ministry of Truth": "真理部在地面上总共有三千个房间。这是中文测试翻译。",
        },
    )
    router = ModelRouter(
        provider=mock_provider, draft_model="mock-draft", repair_model="mock-repair"
    )
    qe = MockQERunner(default_score=0.88)

    orchestrator = PipelineOrchestrator(
        config=config,
        router=router,
        qe_runner=qe,
    )

    events: list[TranslationProgressEvent] = []
    async for event in orchestrator.run(
        input_path=sample_markdown,
        output_path=output_file,
        target_lang="zh",
        job_id="test_job_e2e",
    ):
        events.append(event)

    # 1. Verify event lifecycle sequence
    event_types = [e.event_type for e in events]
    assert EventType.JOB_STARTED in event_types
    assert EventType.PREPROCESSING_DONE in event_types
    assert EventType.BIBLE_EXTRACTED in event_types
    assert EventType.DRAFT_BATCH_COMPLETED in event_types
    assert EventType.MTQE_EVALUATED in event_types
    assert EventType.EXPORT_COMPLETED in event_types

    # 2. Verify rendered file
    assert output_file.exists()
    rendered_text = output_file.read_text(encoding="utf-8")
    assert "Chapter 1: The Beginning" in rendered_text
    assert "这是中文测试翻译。" in rendered_text
    # Verify code block preserved
    assert 'print("Protected code block")' in rendered_text

    # 3. Verify final event stats
    last_event = events[-1]
    assert last_event.completed_blocks > 0
    assert last_event.failed_blocks == 0
    # FastPass auto-cleared the narrative blocks and the code block's 1.0
    # placeholder is excluded per the shared QE score policy (empty scored population = 0.0).
    assert last_event.current_avg_qe >= 0.0


@pytest.mark.asyncio
async def test_pipeline_orchestrator_idempotent_resumption(
    sample_markdown: Path, tmp_path: Path
) -> None:
    db_dir = tmp_path / "ledgers_resume"
    output_file = tmp_path / "bilingual_resumed.md"

    config = UBTConfig(db_dir=db_dir, rate_limit_rpm=600)
    mock_provider = MockModelProvider(default_response="这是断点续传翻译。")
    router = ModelRouter(provider=mock_provider)

    orchestrator = PipelineOrchestrator(config=config, router=router)

    # First run completes everything
    events1: list[TranslationProgressEvent] = []
    async for event in orchestrator.run(
        input_path=sample_markdown,
        output_path=output_file,
        job_id="job_resume_01",
    ):
        events1.append(event)

    call_count_run1 = len(mock_provider.call_history)
    assert call_count_run1 > 0

    # Second run with same job_id should skip all already translated blocks!
    events2: list[TranslationProgressEvent] = []
    async for event in orchestrator.run(
        input_path=sample_markdown,
        output_path=output_file,
        job_id="job_resume_01",
    ):
        events2.append(event)

    # Provider call history length should NOT increase because all blocks were MTQE_PASSED
    assert len(mock_provider.call_history) == call_count_run1


def test_orchestrator_repair_loop_has_no_language_blind_filter(
    tmp_path: Path,
) -> None:
    """The orchestrator must NOT bake a FastPassFilter (which
    silently defaults to ZH thresholds) into the repair loop. The per-run
    filter is built in run() with the job's source/target langs and passed to
    every stage; the fallback should be None, never a wrong-language default.
    """
    config = UBTConfig(db_dir=tmp_path / "ledgers")
    router = ModelRouter(
        provider=MockModelProvider(),
        draft_model="mock-draft",
        repair_model="mock-repair",
    )
    orchestrator = PipelineOrchestrator(config=config, router=router)
    assert orchestrator.repair_loop.fast_pass is None


@pytest.mark.asyncio
async def test_pipeline_invokes_triage_under_default_config(tmp_path: Path) -> None:
    """With default config (pe_queue_enabled=False) the MQM triage
    stage must still be invoked by the pipeline. Previously the orchestrator
    gated ``run_triage_stage`` behind ``pe_queue_enabled``, so the "Critical
    escape rate 0" guarantee silently did not hold. We spy on the stage to
    prove it runs end-to-end regardless of that flag (the stage's own routing
    to NEEDS_HUMAN / BLOCKED_HUMAN is covered by tests/unit/test_mqm_triage.py).
    """
    from unittest.mock import patch

    from ubt.core.engine.stages.triage import run_triage_stage as _real_run_triage

    md = tmp_path / "book.md"
    md.write_text(
        "# Heading\n\nThe reactor outputs 42 megawatts daily.\n\n```python\nx = 1\n```\n",
        encoding="utf-8",
    )
    db_dir = tmp_path / "ledgers_ubt011"
    output_file = tmp_path / "out_ubt011.md"

    config = UBTConfig(db_dir=db_dir, rate_limit_rpm=600, pe_queue_enabled=False)
    # Leaking mock: returns the narrative source verbatim -> untranslated leak
    # (score 0.10). Under the OLD wiring this block was force-finalized and
    # triage never ran; it must now be isolated (FAILED / human review).
    mock_provider = MockModelProvider(default_response="The reactor outputs 42 megawatts daily.")
    router = ModelRouter(provider=mock_provider, draft_model="mock", repair_model="mock")

    orchestrator = PipelineOrchestrator(config=config, router=router)

    called = {"v": False}

    async def _spy_run_triage(*args: Any, **kwargs: Any) -> Any:
        called["v"] = True
        agen = _real_run_triage(*args, **kwargs)
        async for ev in agen:
            yield ev

    with patch("ubt.core.engine.pipeline.run_triage_stage", _spy_run_triage):
        async for _ in orchestrator.run(
            input_path=md,
            output_path=output_file,
            target_lang="zh",
            job_id="job_ubt011",
        ):
            pass

    assert called["v"] is True, "run_triage_stage was not invoked under default config"

    # The defective (number-dropping) block must NOT be silently shipped as a
    # clean pass under default config — it is isolated (FAILED / NEEDS_HUMAN /
    # BLOCKED_HUMAN) instead.
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BlockStatus

    db = SQLiteJobLedger(db_dir / "job_ubt011.sqlite")
    with db as ledger:
        statuses = [b.status for b in ledger.get_all_blocks("job_ubt011")]
    isolated = {BlockStatus.FAILED, BlockStatus.NEEDS_HUMAN, BlockStatus.BLOCKED_HUMAN}
    assert any(s in isolated for s in statuses), statuses


def test_runtime_qe_runner_binds_run_languages(tmp_path: Path) -> None:
    """Regression: the shared default runner stays language-agnostic; each run
    derives a language-bound runner and repair loop instead of mutating it."""
    from ubt.core.qe.comet_runner import HeuristicQERunner

    config = UBTConfig(db_dir=tmp_path / "ledgers")
    router = ModelRouter(provider=MockModelProvider(), draft_model="mock-draft")
    orchestrator = PipelineOrchestrator(config=config, router=router)

    runtime = orchestrator._runtime_qe_runner("en", "fr")
    assert isinstance(runtime, HeuristicQERunner)
    assert (runtime.source_lang, runtime.target_lang) == ("en", "fr")
    # Construction-time default is untouched (safe for concurrent runs).
    assert isinstance(orchestrator.qe_runner, HeuristicQERunner)
    assert orchestrator.qe_runner.target_lang == "zh"

    runtime_loop = orchestrator._runtime_repair_loop(runtime)
    assert runtime_loop.qe_runner is runtime
    assert orchestrator.repair_loop.qe_runner is orchestrator.qe_runner


@pytest.mark.asyncio
async def test_language_bound_heuristic_qe_scores_actual_target_script() -> None:
    """Regression: a ZH-bound heuristic scores French output 0.40 (script
    density), which made every non-ZH repair result unadoptable in the repair
    loop; the run's target language must drive the filter."""
    from ubt.core.qe.comet_runner import HeuristicQERunner

    src = "The gradient channel approximation describes the charge distribution."
    fr = "L'approximation du canal graduel décrit la distribution de charge."
    assert await HeuristicQERunner(target_lang="zh").score(src, fr) == 0.40
    assert await HeuristicQERunner(target_lang="fr").score(src, fr) == 0.92


@pytest.mark.asyncio
async def test_pipeline_anchored_advisory_emits_mono_downgrade(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Stage 1.5 downgrades a bilingual request on the monolingual anchored engine."""
    from collections.abc import AsyncIterator

    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BlockType, BookManifest, ChapterIR, IRBlock

    class MockPDFAdapter:
        async def extract_manifest(self, path: Path) -> BookManifest:
            return BookManifest(doc_id="bk1", title="Test", source_path=str(path))

        async def parse_stream(
            self, path: Path, pages: set[int] | None = None
        ) -> AsyncIterator[ChapterIR]:
            yield ChapterIR(
                doc_id="bk1",
                chapter_id="c1",
                title="C1",
                spine_index=1,
                blocks=[
                    IRBlock(
                        id="b1", spine_index=1, block_type=BlockType.FORMULA, source_text="E = mc^2"
                    ),
                    IRBlock(
                        id="b2",
                        spine_index=2,
                        block_type=BlockType.NARRATIVE,
                        source_text="Equation.",
                    ),
                ],
            )

        async def render_blocks(
            self,
            manifest: BookManifest,
            blocks: list[IRBlock],
            target_lang: str,
            output_path: Path,
            bilingual_mode: str | None = None,
            render_engine: str | None = None,
        ) -> Path:
            return output_path

        async def render_output(
            self,
            manifest: BookManifest,
            ledger: SQLiteJobLedger,
            target_lang: str,
            output_path: Path,
            job_id: str | None = None,
            bilingual_mode: str | None = None,
        ) -> Path:
            return output_path

        def apply_config(self, runtime_config: object) -> None:
            return None

    config = UBTConfig(
        db_dir=tmp_path / "ledgers",
        render_engine="rigid",
    )
    router = ModelRouter(provider=MockModelProvider(), draft_model="mock-draft")
    orchestrator = PipelineOrchestrator(config=config, router=router, adapter=MockPDFAdapter())

    pdf_file = tmp_path / "test_math.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 mock")

    events = []
    with caplog.at_level("WARNING", logger="ubt.core.engine.stages.advisory"):
        async for event in orchestrator.run(
            input_path=pdf_file,
            output_path=tmp_path / "out.pdf",
            job_id="job_pdf_advisory",
        ):
            events.append(event)
            if event.event_type == EventType.MODE_ADVISED:
                break

    advisory_events = [e for e in events if e.event_type == EventType.MODE_ADVISED]
    assert len(advisory_events) == 1
    assert "Render-engine advisory" in advisory_events[0].message
    assert "downgraded to 'monolingual'" in advisory_events[0].message
    # The drop is a WARNING, not INFO: the default `auto` engine can route a
    # dense PDF to rigid without the CLI's explicit-rigid warning, so the runtime
    # log is the only place a silently lost bilingual mode is visible.
    assert "Render-engine advisory: 'rigid' is monolingual" in caplog.text


# ---------------------------------------------------------------------------
# Resume identity: the chapter window must namespace the ledger
# ---------------------------------------------------------------------------


def test_derive_job_id_default_window_keeps_historical_id() -> None:
    """The default window must NOT change the id, or existing ledgers orphan."""
    assert (
        derive_job_id(
            doc_id="abcdef0123456789",
            target_lang="zh",
            pages=None,
            start_chapter=1,
            max_chapters=None,
        )
        == "job_abcdef012345_zh"
    )


def test_derive_job_id_namespaces_language_window_and_pages() -> None:
    """Language, page selection and chapter window each get their own ledger."""
    doc = "abcdef0123456789"
    assert (
        derive_job_id(doc_id=doc, target_lang="de", pages=None, start_chapter=1, max_chapters=None)
        == "job_abcdef012345_de"
    )
    # A windowed run must not collide with the full-book run.
    assert (
        derive_job_id(doc_id=doc, target_lang="zh", pages=None, start_chapter=1, max_chapters=2)
        == "job_abcdef012345_zh_c1-2"
    )
    # --start-chapter without --max-chapters runs to the end.
    assert (
        derive_job_id(doc_id=doc, target_lang="zh", pages=None, start_chapter=3, max_chapters=None)
        == "job_abcdef012345_zh_c3-end"
    )
    # A mid-book window reports its real last chapter.
    assert (
        derive_job_id(doc_id=doc, target_lang="zh", pages=None, start_chapter=3, max_chapters=2)
        == "job_abcdef012345_zh_c3-4"
    )
    # Page selection keeps its historical id shape.
    assert (
        derive_job_id(doc_id=doc, target_lang="zh", pages="7-9", start_chapter=1, max_chapters=None)
        == "job_abcdef012345_zh_p7_9"
    )
    # Both selectors together stay distinguishable from either alone.
    assert (
        derive_job_id(doc_id=doc, target_lang="zh", pages="7-9", start_chapter=2, max_chapters=1)
        == "job_abcdef012345_zh_p7_9_c2-2"
    )


def test_chapter_window_guard_records_then_refuses_a_different_window(tmp_path: Path) -> None:
    """A windowed ledger must refuse to be resumed as a different window.

    Regression for the silent truncation: ingest skips parsing whenever the
    ledger already holds blocks, so without this guard a full-book run resumed
    into a 2-chapter ledger, drafted nothing, exported 2 chapters and reported
    "completed".
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.stages.ingest import _guard_chapter_window
    from ubt.core.exceptions import DocumentParseError

    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    try:
        seed_job(
            ledger,
            "job_x",
            SeedDoc(doc_id="doc", source_path="book.md", format_type="md"),
            "zh",
        )

        # First call records the window.
        _guard_chapter_window(ledger, "job_x", 1, 2)
        assert ledger.get_job_metadata_value("job_x", "chapter_window") == [1, 2]

        # Same window again is a legitimate resume.
        _guard_chapter_window(ledger, "job_x", 1, 2)

        # A different window is refused rather than silently truncated.
        with pytest.raises(DocumentParseError, match="silently truncated"):
            _guard_chapter_window(ledger, "job_x", 1, None)
        with pytest.raises(DocumentParseError, match="chapter window"):
            _guard_chapter_window(ledger, "job_x", 3, 2)

        # The stored window survives the refusals.
        assert ledger.get_job_metadata_value("job_x", "chapter_window") == [1, 2]
    finally:
        ledger.close()


def test_page_selection_guard_records_then_refuses_a_different_selection(tmp_path: Path) -> None:
    """``--pages`` needs the same resume guard as the chapter window.

    Regression: ingest skips parsing whenever the
    ledger already holds blocks, so resuming a ``--pages 7-9`` ledger with
    ``--pages 10-12`` (or without ``--pages`` at all) silently re-exported the
    OLD selection while reporting "completed".
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.stages.ingest import _guard_selected_pages
    from ubt.core.exceptions import DocumentParseError

    ledger = SQLiteJobLedger(tmp_path / "p.sqlite")
    try:
        seed_job(
            ledger,
            "job_p",
            SeedDoc(doc_id="doc", source_path="book.md", format_type="md"),
            "zh",
        )

        # A first run without --pages records the EMPTY list. It must never
        # record None: None is what get_job_metadata_value returns while the
        # key is absent, so None would be indistinguishable from a first call
        # and the job would refuse to ingest itself.
        _guard_selected_pages(ledger, "job_p", None)
        assert ledger.get_job_metadata_value("job_p", "selected_pages") == []

        # The same whole-book resume stays legitimate.
        _guard_selected_pages(ledger, "job_p", None)

        # Adding a selection is a mismatch, refused rather than truncated.
        with pytest.raises(DocumentParseError, match="page selection"):
            _guard_selected_pages(ledger, "job_p", {1, 2})
    finally:
        ledger.close()

    ledger = SQLiteJobLedger(tmp_path / "q.sqlite")
    try:
        seed_job(
            ledger,
            "job_q",
            SeedDoc(doc_id="doc", source_path="book.md", format_type="md"),
            "zh",
        )
        # A page-ranged job records its sorted selection.
        _guard_selected_pages(ledger, "job_q", {9, 7, 8})
        assert ledger.get_job_metadata_value("job_q", "selected_pages") == [7, 8, 9]

        # Same selection again is a legitimate resume.
        _guard_selected_pages(ledger, "job_q", {7, 8, 9})

        # A different selection is refused rather than silently truncated.
        with pytest.raises(DocumentParseError, match="silently truncated"):
            _guard_selected_pages(ledger, "job_q", {10, 11, 12})
        # Dropping the selection (a whole-book resume) is also a mismatch.
        with pytest.raises(DocumentParseError, match="page selection"):
            _guard_selected_pages(ledger, "job_q", None)
    finally:
        ledger.close()


@pytest.mark.asyncio
async def test_pipeline_run_refuses_second_live_writer(
    sample_markdown: Path, tmp_path: Path
) -> None:
    """Schema v7 removed the per-block lease without a job-level replacement;
    the writer lock must make a second live writer fail before any spend."""
    from ubt.core.engine.writer_lock import LedgerWriterLock
    from ubt.core.exceptions import LedgerError

    db_dir = tmp_path / "ledgers_writerlock"
    config = UBTConfig(db_dir=db_dir, rate_limit_rpm=600)
    orchestrator = PipelineOrchestrator(
        config=config, router=ModelRouter(provider=MockModelProvider())
    )
    held = LedgerWriterLock(db_dir / "locked_job.sqlite", "locked_job")
    held.acquire()
    try:
        with pytest.raises(LedgerError, match="already being written"):
            async for _ in orchestrator.run(
                input_path=sample_markdown,
                output_path=tmp_path / "out.md",
                job_id="locked_job",
            ):
                pass
    finally:
        held.release()


@pytest.mark.asyncio
async def test_run_budget_exceeded_stops_job_and_keeps_ledger(
    sample_markdown: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--budget-usd must fail the job at the next priced event but leave the
    ledger resumable (money spent had no ceiling)."""
    from ubt.core.exceptions import UBTError

    db_dir = tmp_path / "ledgers_budget"
    config = UBTConfig(db_dir=db_dir, rate_limit_rpm=600, budget_usd=0.0001)
    router = ModelRouter(
        provider=MockModelProvider(), draft_model="mock-draft", repair_model="mock-repair"
    )
    # Cumulative usage crosses the cap only after ingest has stored its
    # blocks: reads 1-3 are baseline/JOB_STARTED, the 4th (PREPROCESSING_DONE
    # and later) reports 1M prompt tokens (~$0.27) — proving the stop keeps a
    # resumable ledger rather than an empty one.
    reads = {"n": 0}

    def _usage() -> dict[str, dict[str, int]]:
        reads["n"] += 1
        if reads["n"] <= 3:
            return {}
        return {"deepseek-chat": {"prompt_tokens": 1_000_000, "completion_tokens": 100}}

    monkeypatch.setattr(router, "usage_totals_by_model", _usage)
    orchestrator = PipelineOrchestrator(config=config, router=router)

    failure_message = ""
    raised: Exception | None = None
    try:
        async for event in orchestrator.run(
            input_path=sample_markdown,
            output_path=tmp_path / "out.md",
            job_id="budget_job",
        ):
            if event.event_type == EventType.PIPELINE_FAILED:
                failure_message = event.message
    except Exception as exc:  # re-raised form is equally acceptable
        raised = exc

    budget_hit = "Budget exceeded" in failure_message or (
        isinstance(raised, UBTError) and "Budget exceeded" in str(raised)
    )
    assert budget_hit, (failure_message, raised)

    # The ledger must still carry the ingested blocks for a resume.
    from ubt.core.engine.ledger import SQLiteJobLedger

    ledger = SQLiteJobLedger(db_dir / "budget_job.sqlite")
    try:
        assert ledger.get_job_stats("budget_job")["total"] >= 1
    finally:
        ledger.close()


@pytest.mark.asyncio
async def test_progress_event_prices_the_job_across_resumes(tmp_path: Path) -> None:
    """A restarted process must report the book's whole bill, not this run's.

    Provider counters are per-process, so before the ledger carried usage a job
    resumed N times reported only the N-th sitting's cost — and --budget-usd was
    a per-restart allowance rather than a cap on the job.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BookManifest, ChapterMeta
    from ubt.core.router.pricing import estimate_cost_usd

    config = UBTConfig(db_dir=tmp_path / "ledgers", rate_limit_rpm=600)
    router = ModelRouter(
        provider=MockModelProvider(), draft_model="mock-draft", repair_model="mock-repair"
    )
    prior = {"deepseek-chat": {"prompt_tokens": 1_000_000, "completion_tokens": 100}}
    ledger = SQLiteJobLedger(tmp_path / "ledgers" / "resume_job.sqlite")
    try:
        ledger.init_job_from_manifest(
            "resume_job",
            BookManifest(
                doc_id="doc_resume",
                title="Resume Bill",
                source_path="book.md",
                chapters=[ChapterMeta(chapter_id="ch01", title="Chapter 1", spine_index=1)],
            ),
        )
        ledger.record_job_usage("resume_job", prior)
        orchestrator = PipelineOrchestrator(config=config, router=router)
        # This run added one token: a new process whose counters started at zero.
        orchestrator._run_usage_baseline = {}
        router.usage_totals_by_model = lambda: {  # type: ignore[method-assign]
            "deepseek-chat": {"prompt_tokens": 1, "completion_tokens": 0}
        }

        event = await orchestrator._create_progress_event(
            EventType.MTQE_EVALUATED, "resume_job", ledger
        )

        assert event.estimated_cost_usd == pytest.approx(estimate_cost_usd(prior) or 0.0, rel=1e-3)
        # And the merge was written back, so the next restart inherits it.
        stored = ledger.get_job_usage("resume_job")
        assert stored["deepseek-chat"]["prompt_tokens"] == 1_000_001
    finally:
        ledger.close()


@pytest.mark.asyncio
async def test_repeated_progress_events_do_not_double_count_spend(tmp_path: Path) -> None:
    """Billing is idempotent against a cumulative snapshot.

    ``_run_usage()`` returns this run's *cumulative* usage while
    ``record_job_usage`` writes an absolute figure, so folding the whole of it
    in on every progress event re-added every earlier event's tokens. A book
    that emitted ~10 events inflated its stored bill super-linearly, and
    ``budget_violation`` then aborted it for spend it never made.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BookManifest, ChapterMeta

    config = UBTConfig(db_dir=tmp_path / "ledgers", rate_limit_rpm=600)
    router = ModelRouter(
        provider=MockModelProvider(), draft_model="mock-draft", repair_model="mock-repair"
    )
    ledger = SQLiteJobLedger(tmp_path / "ledgers" / "cumulative_job.sqlite")
    seen: list[int] = []
    try:
        ledger.init_job_from_manifest(
            "cumulative_job",
            BookManifest(
                doc_id="doc_cumulative",
                title="Cumulative Bill",
                source_path="book.md",
                chapters=[ChapterMeta(chapter_id="ch01", title="Chapter 1", spine_index=1)],
            ),
        )
        orchestrator = PipelineOrchestrator(config=config, router=router)
        orchestrator._run_usage_baseline = {}
        router.usage_totals_by_model = lambda: {  # type: ignore[method-assign]
            "deepseek-chat": {"prompt_tokens": seen[-1], "completion_tokens": 0}
        }

        # Three events whose cumulative totals are 10, 25 and 40 tokens.
        for total in (10, 25, 40):
            seen.append(total)
            await orchestrator._create_progress_event(
                EventType.MTQE_EVALUATED, "cumulative_job", ledger
            )

        stored = ledger.get_job_usage("cumulative_job")
        # The final cumulative total, not the 75 an additive merge would store.
        assert stored["deepseek-chat"]["prompt_tokens"] == 40
    finally:
        ledger.close()


@pytest.mark.asyncio
async def test_pre_flight_estimate_refuses_an_impossible_budget(
    sample_markdown: Path, tmp_path: Path
) -> None:
    """A budget the book cannot fit must stop at zero tokens, not mid-chapter."""
    from ubt.core.engine.ledger import SQLiteJobLedger

    db_dir = tmp_path / "ledgers_preflight"
    config = UBTConfig(
        db_dir=db_dir,
        rate_limit_rpm=600,
        draft_model="deepseek-chat",
        repair_model="deepseek-chat",
        budget_usd=0.00001,
    )
    router = ModelRouter(
        provider=MockModelProvider(), draft_model="deepseek-chat", repair_model="deepseek-chat"
    )
    orchestrator = PipelineOrchestrator(config=config, router=router)

    message = ""
    try:
        async for event in orchestrator.run(
            input_path=sample_markdown,
            output_path=tmp_path / "out.md",
            job_id="preflight_job",
        ):
            if event.event_type == EventType.PIPELINE_FAILED:
                message = event.message
    except Exception as exc:  # re-raised form is equally acceptable
        message = str(exc)

    assert "Refused before the first request" in message, message
    ledger = SQLiteJobLedger(db_dir / "preflight_job.sqlite")
    try:
        stats = ledger.get_job_stats("preflight_job")
        assert stats["total"] >= 1, "ingest must have completed before the refusal"
        # Nothing was drafted: the only terminal rows are the verbatim blocks
        # ingest ships untranslated by design (a cover heading, say).
        assert stats["drafted"] == 0, message
        assert stats["failed"] == 0, message
    finally:
        ledger.close()


class DummyProvider(BaseModelProvider):
    @property
    def provider_name(self) -> str:
        return "dummy"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = "",
        model: str | None = "mock",
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        if "こんにちは" in prompt or "日" in prompt:
            return "これは日本語のテスト翻訳です。"
        if "Bonjour" in prompt:
            return "Ceci est une traduction de test en français."
        return "这是翻译内容。"


async def test_pipeline_dynamic_profile_injection_and_concurrency(tmp_path: Path) -> None:
    """Fix 1 & Fix 4: Verify pipeline injects Japanese profile and executes concurrently."""
    src_file = tmp_path / "japanese_book.md"
    src_file.write_text(
        "# 日本语测试\n\nこんにちは世界。\n\n今日は良い天気ですね。\n", encoding="utf-8"
    )
    out_file = tmp_path / "japanese_bilingual.md"

    config = UBTConfig(
        db_dir=tmp_path / "db",
        max_concurrency=4,
    )
    router = ModelRouter(provider=DummyProvider())
    qe = MockQERunner(default_score=0.92)
    orchestrator = PipelineOrchestrator(config=config, router=router, qe_runner=qe)

    # Run for Japanese target language
    async for _ in orchestrator.run(
        input_path=src_file,
        output_path=out_file,
        target_lang="ja",
    ):
        pass

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "これは日本語のテスト翻訳です。" in content


async def test_pipeline_no_chapter_title_self_lock(tmp_path: Path) -> None:
    """Verify that chapter titles are NOT self-locked as English translations in the translation bible."""
    md_file = tmp_path / "book.md"
    md_file.write_text(
        "# Chapter 1: Neural Networks\n\nDeep learning is powerful.\n", encoding="utf-8"
    )
    out_file = tmp_path / "out.md"

    config = UBTConfig(db_dir=tmp_path / "ledgers")
    mock_router = MagicMock()
    mock_router.draft = AsyncMock(return_value="这是翻译")
    mock_router.complete_raw = AsyncMock(return_value="这是摘要")
    mock_router.repair = AsyncMock(return_value="这是修复")
    mock_router.repair_reasoning_effort = "none"
    orchestrator = PipelineOrchestrator(config=config, router=mock_router)

    async for _ in orchestrator.run(
        input_path=md_file,
        output_path=out_file,
        target_lang="zh",
        job_id="job_title_test",
    ):
        pass

    # Verify that in all draft calls, glossary tables do NOT self-lock "Neural Networks | Neural Networks"
    for call in mock_router.draft.call_args_list:
        glossary_table = call.kwargs.get("glossary_table", "")
        assert "Neural Networks | Neural Networks" not in glossary_table
        global_glossary = call.kwargs.get("global_glossary", "")
        assert "Neural Networks | Neural Networks" not in global_glossary


@pytest.mark.asyncio
async def test_usage_sinks_split_a_shared_provider_between_concurrent_jobs() -> None:
    """Two runs over ONE provider must each bill only their own spend.

    The API server shares a single provider across concurrent jobs, and cost came
    from process-wide counters diffed against a start-of-run snapshot: every job
    was charged the blended spend of all of them, which both overstated reports
    and tripped --budget-usd on money another job spent.
    """
    import asyncio

    from ubt.core.router.provider import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(api_key="test-key")

    async def spend(*amounts: int) -> dict[str, dict[str, int]]:
        sink = provider.begin_usage_sink()

        def record(amount: int) -> None:
            provider._record_usage(
                "deepseek-chat", {"prompt_tokens": amount, "completion_tokens": 1}
            )

        async def record_in_child_task(amount: int) -> None:
            record(amount)

        # The sink has to follow the run through every hop its work takes:
        # this frame, a worker thread, and a spawned child task.
        record(amounts[0])
        await asyncio.to_thread(record, amounts[1])
        await asyncio.create_task(record_in_child_task(amounts[2]))
        return sink

    sink_a, sink_b = await asyncio.gather(spend(40, 30, 30), spend(100, 100, 50))

    assert sink_a["deepseek-chat"]["prompt_tokens"] == 100
    assert sink_b["deepseek-chat"]["prompt_tokens"] == 250
    # The process-wide view must stay complete, not steal into the per-run views.
    assert provider.usage_totals_by_model["deepseek-chat"]["prompt_tokens"] == 350


def test_run_usage_prefers_the_sink_and_falls_back_without_attribution(
    tmp_path: Path,
) -> None:
    """A provider that can attribute wins; one that can't keeps the snapshot delta."""
    config = UBTConfig(db_dir=tmp_path / "ledgers", rate_limit_rpm=600)

    class AttributingProvider(MockModelProvider):
        def begin_usage_sink(self) -> dict[str, dict[str, int]]:
            sink: dict[str, dict[str, int]] = {"m": {"prompt_tokens": 5, "calls": 1}}
            self.sink = sink
            return sink

    provider = AttributingProvider()
    router = ModelRouter(provider=provider, draft_model="m")
    orchestrator = PipelineOrchestrator(config=config, router=router)
    router.usage_totals_by_model = lambda: {"m": {"prompt_tokens": 500}}  # type: ignore[method-assign]

    orchestrator._run_usage_sink = router.begin_usage_sink()
    assert orchestrator._run_usage() == {"m": {"prompt_tokens": 5, "calls": 1}}
    # Reading must not hand out the dict the provider is still writing into.
    assert orchestrator._run_usage() is not provider.sink

    plain = ModelRouter(provider=MockModelProvider(), draft_model="m")
    assert plain.begin_usage_sink() is None
    fallback = PipelineOrchestrator(config=config, router=plain)
    fallback._run_usage_baseline = {"m": {"prompt_tokens": 400}}
    plain.usage_totals_by_model = lambda: {"m": {"prompt_tokens": 500}}  # type: ignore[method-assign]
    assert fallback._run_usage() == {"m": {"prompt_tokens": 100}}


@pytest.mark.asyncio
async def test_billing_starts_over_after_fresh_discards_the_old_bill(tmp_path: Path) -> None:
    """A --fresh reset must not be undone by a bill read earlier in the same run.

    ``--fresh`` zeroes the job's recorded usage during ingest; the prior bill used
    to be snapshotted at run() entry, so the first priced event resurrected the
    numbers the run had just thrown away (the read moved to billing time, after
    ingest).
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.usage import bill_job_run
    from ubt.core.ir.models import BookManifest, ChapterMeta

    ledger = SQLiteJobLedger(tmp_path / "ledgers" / "fresh_job.sqlite")
    try:
        ledger.init_job_from_manifest(
            "fresh_job",
            BookManifest(
                doc_id="doc_fresh",
                title="Fresh Bill",
                source_path="book.md",
                chapters=[ChapterMeta(chapter_id="ch01", title="Chapter 1", spine_index=1)],
            ),
        )
        ledger.record_job_usage(
            "fresh_job", {"m": {"prompt_tokens": 9_000, "completion_tokens": 9}}
        )
        bill = await bill_job_run(ledger, "fresh_job", {})
        assert bill.lifetime_usage["m"]["prompt_tokens"] == 9_000

        ledger.record_job_usage("fresh_job", {})  # what --fresh does during ingest
        bill = await bill_job_run(ledger, "fresh_job", {"m": {"prompt_tokens": 10, "calls": 1}})
        assert bill.lifetime_usage == {"m": {"prompt_tokens": 10, "calls": 1}}
    finally:
        ledger.close()


class _BillableMockProvider(MockModelProvider):
    """Same simulated text, but presents as a real (billable) provider.

    The budget preflight deliberately skips mock runs; exercising it needs a
    provider whose ``is_mock`` says otherwise without any network access.
    """

    is_mock = False


def _preflight_orchestrator(tmp_path: Path, **cfg_over: object) -> PipelineOrchestrator:
    from ubt.core.config import UBTConfig as _Cfg

    config = _Cfg(db_dir=tmp_path / "ledgers", budget_usd=5.0, **cfg_over)  # type: ignore[arg-type]
    router = ModelRouter(
        provider=_BillableMockProvider(),
        draft_model="unheard-of-llm-v9",
        repair_model="unheard-of-llm-v9",
    )
    return PipelineOrchestrator(config=config, router=router)


def test_budget_preflight_refuses_unpriced_models(tmp_path: Path) -> None:
    """A cap that can never fire must stop the run, not warn once and drift.

    With no price entry every cost is None; ``budget_violation`` reads None as
    "not exceeded", so an unpriced model silently bypasses UBT_BUDGET_USD.
    """
    from ubt.core.exceptions import UBTError

    orchestrator = _preflight_orchestrator(tmp_path)
    with pytest.raises(UBTError, match="unheard-of-llm-v9"):
        orchestrator._preflight_budget_pricing()


def test_budget_preflight_allows_a_self_hosted_endpoint(tmp_path: Path) -> None:
    """A local endpoint cannot spend, so the "cap that can never fire" refusal
    must not apply to it.

    Without this, the project's own self-hosted path (Ollama / llama.cpp /
    llama-swap, whose model names are deliberately absent from the price table)
    refuses to start the moment a user also sets ``--budget-usd`` — the most
    natural combination for someone running the whole thing locally.
    """
    orchestrator = _preflight_orchestrator(tmp_path, base_url="http://127.0.0.1:9090/v1")
    orchestrator._preflight_budget_pricing()


def test_budget_preflight_still_refuses_a_cloud_channel_of_a_local_run(tmp_path: Path) -> None:
    """Only the channels that actually reach the local endpoint are exempt.

    A local LLM with cloud OCR still spends real money through ``ocr_endpoint``;
    exempting the run wholesale would put that channel outside the cap — the
    exact bug the OCR enumeration was added to fix.
    """
    from ubt.core.exceptions import UBTError

    orchestrator = _preflight_orchestrator(
        tmp_path,
        base_url="http://127.0.0.1:9090/v1",
        ocr_mode="cloud",
        ocr_model="unheard-of-vision-v9",
    )
    with pytest.raises(UBTError, match="unheard-of-vision-v9"):
        orchestrator._preflight_budget_pricing()


def _priced_preflight_orchestrator(tmp_path: Path, **cfg_over: object) -> PipelineOrchestrator:
    """Like ``_preflight_orchestrator`` but with priced headline models.

    Isolates the model under test: an unpriced draft/repair would already
    refuse and mask which additional path (judge, fallback) the preflight
    picked up.
    """
    from ubt.core.config import UBTConfig as _Cfg

    config = _Cfg(db_dir=tmp_path / "ledgers", budget_usd=5.0, **cfg_over)  # type: ignore[arg-type]
    router = ModelRouter(
        provider=_BillableMockProvider(),
        draft_model="deepseek-chat",
        repair_model="deepseek-chat",
        # Wired the way the pipeline builds its own router, so the preflight
        # reads the fallback list the run will actually retry onto.
        fallback_models=list(config.fallback_models),
    )
    return PipelineOrchestrator(config=config, router=router)


def test_budget_preflight_covers_the_judge_and_fallbacks(tmp_path: Path) -> None:
    """Every model the run can bill must be priced, not just draft/repair.

    The judge is enabled by ``qe_judge_enabled`` OR ``qe_engine=tiered``, and a
    429/5xx retry lands on ``fallback_models``. An unpriced model on either path
    makes ``estimate_cost_usd`` return None for the rest of the run, so the cap
    silently stops enforcing — the exact failure the preflight exists to stop.
    """
    from ubt.core.exceptions import UBTError

    judge = _priced_preflight_orchestrator(
        tmp_path, qe_judge_enabled=True, qe_judge_model="unheard-of-judge-v1"
    )
    with pytest.raises(UBTError, match="unheard-of-judge-v1"):
        judge._preflight_budget_pricing()

    fallback = _priced_preflight_orchestrator(tmp_path, fallback_models=["unheard-of-fallback-v1"])
    with pytest.raises(UBTError, match="unheard-of-fallback-v1"):
        fallback._preflight_budget_pricing()

    # Judge enabled but its model defaults to the priced repair model: passes.
    _priced_preflight_orchestrator(tmp_path, qe_judge_enabled=True)._preflight_budget_pricing()

    # The OCR channel bills its own model through its own httpx client, so an
    # explicit cloud/vlm pick must be priced or the cap never sees that spend.
    ocr = _priced_preflight_orchestrator(
        tmp_path, ocr_mode="vlm", ocr_model="unheard-of-vision-v9", allow_page_upload=True
    )
    with pytest.raises(UBTError, match="unheard-of-vision-v9"):
        ocr._preflight_budget_pricing()

    # ``auto`` counts only with page egress on: without it the probe can only
    # resolve to a free local engine, and refusing there would block runs that
    # cannot bill anything through the OCR channel.
    _priced_preflight_orchestrator(
        tmp_path, ocr_mode="auto", ocr_model="unheard-of-vision-v9", allow_page_upload=False
    )._preflight_budget_pricing()


def test_budget_preflight_opt_out_and_bypasses(tmp_path: Path) -> None:
    from ubt.core.config import UBTConfig

    # Explicit opt-out keeps the warning-only behavior.
    _preflight_orchestrator(tmp_path, allow_unpriced_budget=True)._preflight_budget_pricing()
    # Uncapped runs and mock (dry-run) runs are untouched.
    uncapped = PipelineOrchestrator(
        config=UBTConfig(db_dir=tmp_path / "ledgers", budget_usd=None),
        router=ModelRouter(provider=_BillableMockProvider(), draft_model="unheard-of-llm-v9"),
    )
    uncapped._preflight_budget_pricing()
    mocked = _preflight_orchestrator(tmp_path)
    mocked.router.provider.is_mock = True
    mocked._preflight_budget_pricing()
    # A priced model passes normally.
    priced = PipelineOrchestrator(
        config=UBTConfig(db_dir=tmp_path / "ledgers", budget_usd=5.0),
        router=ModelRouter(provider=_BillableMockProvider(), draft_model="gpt-4o"),
    )
    priced._preflight_budget_pricing()


@pytest.mark.asyncio
async def test_concurrent_streaming_billing_is_serialized(tmp_path: Path) -> None:
    """Chapter-streaming bills from two workers at once (draft + QE/repair).

    Snapshot-delta, the ledger read-merge-write and the baseline write-back
    sit behind awaits: with an unordered completion the second worker billed
    against the first's stale baseline, and the absolute merge stored the
    overlap twice — inflating spend enough to trip budget_usd on tokens the
    job never spent. The billing section must behave as one critical section.
    """
    from ubt.core.engine import pipeline as pipeline_module
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BookManifest, ChapterMeta

    config = UBTConfig(db_dir=tmp_path / "ledgers", rate_limit_rpm=600)
    router = ModelRouter(
        provider=MockModelProvider(), draft_model="mock-draft", repair_model="mock-repair"
    )
    ledger = SQLiteJobLedger(tmp_path / "ledgers" / "stream_job.sqlite")
    from ubt.core.engine.usage import bill_job_run as real_bill

    active = 0
    max_active = 0
    entered_first = asyncio.Event()
    release = asyncio.Event()

    async def _spy_bill(*args: Any, **kwargs: Any) -> Any:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        entered_first.set()
        await release.wait()
        try:
            return await real_bill(*args, **kwargs)
        finally:
            active -= 1

    try:
        ledger.init_job_from_manifest(
            "stream_job",
            BookManifest(
                doc_id="doc_stream",
                title="Stream Bill",
                source_path="book.md",
                chapters=[ChapterMeta(chapter_id="ch01", title="Chapter 1", spine_index=1)],
            ),
        )
        orchestrator = PipelineOrchestrator(config=config, router=router)
        orchestrator._run_usage_baseline = {}
        cumulative = [100]
        router.usage_totals_by_model = lambda: {  # type: ignore[method-assign]
            "deepseek-chat": {"prompt_tokens": cumulative[0], "completion_tokens": 0}
        }
        # setattr, not assignment: bill_job_run is an implicit re-export the
        # strict checker refuses to see as a bindable module attribute.
        setattr(pipeline_module, "bill_job_run", _spy_bill)  # noqa: B010
        try:
            task_a = asyncio.create_task(
                orchestrator._create_progress_event(EventType.MTQE_EVALUATED, "stream_job", ledger)
            )
            await asyncio.wait_for(entered_first.wait(), timeout=5)
            # Provider spend advances while worker A is mid-bill; worker B
            # starts and must not interleave its own billing.
            cumulative[0] = 200
            task_b = asyncio.create_task(
                orchestrator._create_progress_event(
                    EventType.DRAFT_BATCH_COMPLETED, "stream_job", ledger
                )
            )
            await asyncio.sleep(0.05)
            assert max_active == 1, "worker B billed concurrently with A"
            release.set()
            await asyncio.gather(task_a, task_b)
        finally:
            setattr(pipeline_module, "bill_job_run", real_bill)  # noqa: B010

        assert max_active == 1
        stored = ledger.get_job_usage("stream_job")
        # A billed 100, B then billed the 100-token delta (not the stale-
        # baseline 200): the ledger ends at the exact cumulative total.
        assert stored["deepseek-chat"]["prompt_tokens"] == 200
    finally:
        ledger.close()


@pytest.mark.fast
@pytest.mark.asyncio
async def test_pipeline_finally_releases_writer_lock_on_ledger_close_failure(
    tmp_path: Path,
) -> None:
    from unittest.mock import MagicMock, patch

    doc = tmp_path / "sample.md"
    doc.write_text("# Test\nBody", encoding="utf-8")

    cfg = UBTConfig(db_dir=tmp_path)
    orchestrator = PipelineOrchestrator(config=cfg)

    # Mock ledger.close to raise an exception
    with patch("ubt.core.engine.pipeline.SQLiteJobLedger") as mock_ledger_cls:
        mock_instance = MagicMock()
        mock_instance.get_job_target_lang.return_value = "zh"
        mock_instance.get_job_fingerprint.return_value = None
        mock_instance.close.side_effect = RuntimeError("SQLite close exploded")
        mock_ledger_cls.return_value = mock_instance

        # Also mock writer_lock to verify release is called even when ledger.close explodes
        with patch("ubt.core.engine.pipeline.LedgerWriterLock") as mock_lock_cls:
            mock_lock = MagicMock()
            mock_lock_cls.return_value = mock_lock

            with pytest.raises(Exception):  # noqa: B017
                async for _ in orchestrator.run(input_path=doc):
                    pass

            assert mock_lock.release.called


@pytest.mark.fast
@pytest.mark.asyncio
async def test_pipeline_orchestrator_marks_cancelled_on_keyboard_interrupt(tmp_path: Path) -> None:
    """KeyboardInterrupt during pipeline execution must mark job status as 'cancelled'."""
    cfg = UBTConfig(db_dir=tmp_path / "ledgers")
    cfg.db_dir.mkdir(parents=True, exist_ok=True)
    input_file = tmp_path / "book.txt"
    input_file.write_text("Hello world\n", encoding="utf-8")

    orchestrator = PipelineOrchestrator(config=cfg)
    ledger = SQLiteJobLedger(cfg.db_dir / "job_kb_int.sqlite")
    manifest = BookManifest(
        doc_id="doc_kb",
        title="KB Test",
        source_path=str(input_file),
        chapters=[ChapterMeta(chapter_id="ch1", title="Ch1", spine_index=0)],
    )
    ledger.init_job_from_manifest("job_kb_int", manifest)
    ledger.close()

    async def raise_kb_interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt("user pressed Ctrl+C")
        yield  # make it an async generator

    with (
        patch("ubt.core.engine.pipeline.run_ingest_stage", side_effect=raise_kb_interrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        async for _ in orchestrator.run(
            input_path=input_file,
            output_path=tmp_path / "out.txt",
            job_id="job_kb_int",
        ):
            pass

    check_ledger = SQLiteJobLedger(cfg.db_dir / "job_kb_int.sqlite", read_only=True)
    assert check_ledger.get_job_status("job_kb_int") == "cancelled"
    check_ledger.close()


def _a0920_ledger_with_targets(tmp_path: Path, *, translated: int, total: int) -> SQLiteJobLedger:
    job_id = "job_coverage"
    ledger = SQLiteJobLedger(tmp_path / f"{job_id}.sqlite")
    blocks = [
        IRBlock(
            id=f"b{idx}",
            spine_index=idx,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text=f"Source paragraph number {idx}.",
        )
        for idx in range(1, total + 1)
    ]
    seed_job(
        ledger,
        job_id,
        SeedDoc(
            doc_id=job_id,
            source_path=str(tmp_path / "book.md"),
            format_type="markdown",
            blocks=blocks,
        ),
        target_lang="zh",
    )
    for block in blocks[:translated]:
        ledger.save_checkpoint(
            block_id=block.id,
            status=BlockStatus.MTQE_PASSED,
            target_text=f"译文 {block.id}",
        )
    for block in blocks[translated:]:
        ledger.save_checkpoint(block_id=block.id, status=BlockStatus.FAILED)
    return ledger


async def _a0920_run_export(ledger: SQLiteJobLedger, tmp_path: Path, *, ratio: float) -> Path:
    job_id = "job_coverage"
    manifest = BookManifest(
        doc_id=job_id,
        title="Coverage",
        source_path=str(tmp_path / "book.md"),
        target_lang="zh",
        source_lang="en",
    )
    (tmp_path / "book.md").write_text(
        "# Coverage\n\nSource paragraph number 1.\n", encoding="utf-8"
    )

    async def _event(*args: Any, **kwargs: Any) -> None:
        return None

    ctx = build_stage_ctx(
        tmp_path,
        # The coverage floor is a config knob now; the stage has no shadow
        # default of its own to keep in step with it.
        config=UBTConfig(db_dir=tmp_path, export_min_completion_ratio=ratio),
        ledger=ledger,
        job_id=job_id,
        manifest=manifest,
        adapter=MarkdownAdapter(),
        output_path=tmp_path / "out.md",
        input_path=tmp_path / "book.md",
        target_lang="zh",
        source_lang="en",
        glossary_dicts=[],
        html_validator=HTMLDeltaValidator(),
        create_event=_event,
    )
    _events = [e async for e in run_export_stage(ctx)]
    assert ctx.output_path is not None
    return Path(ctx.output_path)


@pytest.mark.asyncio
async def test_export_refuses_a_book_where_most_blocks_have_no_target(tmp_path: Path) -> None:
    """All-failed used to render source text and finalize the job as completed."""
    ledger = _a0920_ledger_with_targets(tmp_path, translated=1, total=4)
    with pytest.raises(IntegrityViolationError, match="carry a translation"):
        await _a0920_run_export(ledger, tmp_path, ratio=0.5)
    assert ledger.get_job_stats("job_coverage")["completed"] == 1


@pytest.mark.asyncio
async def test_coverage_gate_floor_is_configurable(tmp_path: Path) -> None:
    """The gate must be liftable to 0 for a knowingly-partial delivery."""
    ledger = _a0920_ledger_with_targets(tmp_path, translated=1, total=4)
    out = await _a0920_run_export(ledger, tmp_path, ratio=0.0)
    assert out.is_file()


def test_orchestrator_honours_an_injected_shared_rate_limiter() -> None:
    from pydantic import SecretStr

    from ubt.core.config import MOCK_API_KEY, UBTConfig
    from ubt.core.engine.pipeline import PipelineOrchestrator
    from ubt.core.router.rate_limiter import AdaptiveTokenBucket

    shared = AdaptiveTokenBucket(initial_rpm=7, max_rpm=7)
    orchestrator = PipelineOrchestrator(
        config=UBTConfig(api_key=SecretStr(MOCK_API_KEY)), rate_limiter=shared
    )
    assert orchestrator.router.rate_limiter is shared


def _r0918b_counting_router(calls: list[object]) -> ModelRouter:
    class _Counting(MockModelProvider):
        async def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> str:
            calls.append(prompt)
            return "[TRANSLATED]"

    return ModelRouter(
        provider=_Counting(),
        rate_limiter=AdaptiveTokenBucket(initial_rpm=60, max_rpm=60),
        draft_model="counting",
    )


async def _r0918b_drain(orchestrator: PipelineOrchestrator, source: Path, out: Path) -> None:
    async for _ in orchestrator.run(
        input_path=source, output_path=out, target_lang="fr", source_lang="en"
    ):
        pass


def test_orchestrator_detects_a_simulated_run_from_the_provider() -> None:
    """``--dry-run`` keeps a real ``api_key`` in config and swaps only the
    router's provider, so a config-based check would miss every dry run."""
    limiter = AdaptiveTokenBucket(initial_rpm=60, max_rpm=60)
    mock = ModelRouter(provider=MockModelProvider(), rate_limiter=limiter, draft_model="m")
    real = ModelRouter(
        provider=OpenAICompatibleProvider(api_key="sk-real", base_url="http://127.0.0.1:9/v1"),
        rate_limiter=limiter,
        draft_model="m",
    )
    assert PipelineOrchestrator(config=UBTConfig(), router=mock)._is_mock_run is True
    assert PipelineOrchestrator(config=UBTConfig(), router=real)._is_mock_run is False


def test_pipeline_refuses_a_contradicting_output_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.exceptions import UnsupportedDocumentFormatError
    from ubt.core.ports import resolve_adapter

    source = tmp_path / "book.md"
    source.write_text("# Chapter One\n\nHello world.\n", encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(
        "ubt.core.engine.pipeline.resolve_adapter",
        lambda *a, **k: resolve_adapter(*a, **k),
    )
    orchestrator = PipelineOrchestrator(
        config=UBTConfig(db_dir=tmp_path / "db"),
        router=_r0918b_counting_router(calls),
    )
    with pytest.raises(UnsupportedDocumentFormatError) as exc:
        asyncio.run(_r0918b_drain(orchestrator, source, tmp_path / "book.pdf"))
    assert ".md" in str(exc.value)
    assert calls == [], "the refusal must happen before the first model call"


_r0922_BOOK = """# Chapter One

The mill ran through the night, and the river carried the noise away.

# Chapter Two

By morning the wheel had stopped, and the builder came down to look at it.
"""


def _r0922_sidecars(out: Path) -> list[Path]:
    return [
        sidecar_path(out, "quality_report.json"),
        sidecar_path(out, "metrics.json"),
        sidecar_path(out, "visual_report.json"),
    ]


async def _r0922_translate(
    src: Path, out: Path, db_dir: Path, job_id: str, *, reply: str = "离线译文段落。"
) -> None:
    router = ModelRouter(
        provider=TokenEchoMockProvider(default_response=reply),
        draft_model="mock-draft",
        repair_model="mock-repair",
        rate_limiter=AdaptiveTokenBucket(
            initial_rpm=1_000_000,
            max_rpm=1_000_000,
            initial_tpm=1_000_000_000,
            max_tpm=1_000_000_000,
        ),
    )
    orchestrator = PipelineOrchestrator(
        config=UBTConfig(db_dir=db_dir, rate_limit_rpm=100_000, tm_enabled=False),
        router=router,
        qe_runner=MockQERunner(default_score=0.92),
    )
    async for _ in orchestrator.run(
        input_path=src, output_path=out, target_lang="zh", job_id=job_id
    ):
        pass


@pytest.mark.asyncio
async def test_interrupted_export_leaves_no_report_for_a_deliverable_it_never_saw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that dies after the render must not leave the last run's reports.

    Reproduction of the round-5 finding: the deliverable is written first and the
    reports later, so an interrupt (or the blocking visual gate raising) in that
    window left a ``*_quality_report.json`` claiming a finished book whose text it
    had never seen -- and ``artifact_and_report_paths`` re-attaches those names to
    later runs, so the stale report outlived the process that orphaned it.
    """
    src = tmp_path / "book.md"
    src.write_text(_r0922_BOOK, encoding="utf-8")
    out = tmp_path / "out_bilingual.md"

    await _r0922_translate(src, out, tmp_path / "db1", "job_sidecar_first")
    written = [path for path in _r0922_sidecars(out) if path.exists()]
    assert written, "the first run wrote no sidecar reports; the test proves nothing"
    first_deliverable = out.read_bytes()

    async def _die(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt("operator stopped the run after the render")

    monkeypatch.setattr(
        "ubt.core.engine.stages.export._build_reports",
        _die,
        raising=True,
    )
    with pytest.raises(BaseException):  # noqa: B017 - KeyboardInterrupt is the point
        await _r0922_translate(
            src, out, tmp_path / "db2", "job_sidecar_second", reply="第二轮离线译文。"
        )

    assert out.read_bytes() != first_deliverable, "the second run never replaced the deliverable"
    survivors = [path.name for path in _r0922_sidecars(out) if path.exists()]
    assert not survivors, f"stale reports describe the new deliverable: {survivors}"


@pytest.mark.asyncio
async def test_pipeline_stateless_fast_pass_and_repair() -> None:
    """Verify PipelineOrchestrator does not mutate shared state during execution."""
    router = ModelRouter(provider=MockModelProvider())
    repair_loop = RepairLoop(router=router, qe_runner=MockQERunner())

    initial_fast_pass = FastPassFilter(source_lang="en", target_lang="zh")
    repair_loop.fast_pass = initial_fast_pass

    orchestrator = PipelineOrchestrator(
        router=router,
        repair_loop=repair_loop,
        qe_runner=MockQERunner(),
    )
    # The orchestrator no longer carries a language-blind
    # fast_pass attribute at all — filters are per-stage, per-language.
    assert not hasattr(orchestrator, "fast_pass")

    # Ensure repair_single_block accepts explicit fast_pass
    custom_fast_pass = FastPassFilter(source_lang="ja", target_lang="en")
    block = IRBlock(
        id="repair_b1",
        spine_index=1,
        source_text="こんにちは世界",
        target_text="Bonjour le monde",
        status=BlockStatus.REPAIR_PENDING,
        repair_rounds=0,
    )

    repaired = await repair_loop.repair_single_block(
        block=block,
        glossary_table="",
        target_lang="en",
        source_lang="ja",
        fast_pass=custom_fast_pass,
    )
    assert repaired.repair_rounds == 1

    # Verify instance fast_pass was not clobbered
    assert repair_loop.fast_pass is initial_fast_pass


def test_completion_floor_counts_blocked_human_placeholders_as_untranslated() -> None:
    """A quarantined book must not pass the completion floor on placeholders.

    ``BLOCKED_HUMAN`` carries a non-empty ``<mark>`` source wrapper, so counting
    "has a target" let an all-quarantined book finalize as completed.
    """
    from ubt.core.engine.stages.export import _check_completion_ratio

    blocks = [
        IRBlock(
            id="a",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="One.",
            target_text="一。",
            status=BlockStatus.MTQE_PASSED,
        ),
        IRBlock(
            id="b",
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text="Two.",
            target_text="<mark>Two.</mark>",
            status=BlockStatus.BLOCKED_HUMAN,
        ),
    ]
    with pytest.raises(IntegrityViolationError):
        _check_completion_ratio("job", blocks, 0.9)


def test_derive_job_id_namespaces_profile_and_engine_knobs() -> None:
    """A different genre profile or preset must not resume the other's drafts."""
    kwargs = {
        "doc_id": "abcdef0123456789",
        "target_lang": "zh",
        "pages": None,
        "start_chapter": 1,
        "max_chapters": None,
    }
    default = derive_job_id(**kwargs)  # type: ignore[arg-type]
    assert default == "job_abcdef012345_zh"
    profiled = derive_job_id(**kwargs, profile_name="academic")  # type: ignore[arg-type]
    assert profiled != default and profiled.endswith("_pracademic")
    engine = derive_job_id(**kwargs, engine_signature="psrich-mbimage")  # type: ignore[arg-type]
    assert engine != default and engine.endswith("_engpsrichmbimage")
    # Defaults add nothing, so historical ledgers stay resumable.
    assert (
        derive_job_id(**kwargs, profile_name="general", engine_signature="")  # type: ignore[arg-type]
        == default
    )


def test_engine_signature_is_empty_for_defaults_and_tags_overrides() -> None:
    from ubt.core.config import UBTConfig
    from ubt.core.engine.pipeline import engine_signature

    assert engine_signature(UBTConfig()) == ""
    assert "psrich" in engine_signature(UBTConfig(prompt_strategy="rich"))
    assert "mbimage" in engine_signature(UBTConfig(math_backend="image"))

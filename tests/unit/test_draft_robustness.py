"""High-ROI robustness tests for the draft stage and ledger resume."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.stage_ctx_factory import build_stage_ctx, inert_event
from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import TARGET_SCHEMA_VERSION, SQLiteJobLedger
from ubt.core.engine.stages.draft import run_draft_stage
from ubt.core.exceptions import ModelProviderError
from ubt.core.ir.models import (
    BlockStatus,
    BookManifest,
    ChapterMeta,
    DocumentIR,
    FlowID,
    IRBlock,
)
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


def _make_doc(n: int, poison_idx: int | None = None) -> DocumentIR:
    blocks = []
    for i in range(1, n + 1):
        text = f"Test sentence number {i} for robustness drafting."
        if poison_idx is not None and i == poison_idx:
            text = "POISON block that always fails translation."
        blocks.append(
            IRBlock(
                id=f"ch01#b{i:03d}",
                flow_id=FlowID.MAIN_STORY,
                spine_index=i,
                source_text=text,
            )
        )
    return DocumentIR(
        doc_id="robust_doc",
        source_path="/tmp/robust.epub",
        format_type="epub",
        metadata={},
        blocks=blocks,
    )


def _make_manifest() -> BookManifest:
    return BookManifest(
        doc_id="robust_doc",
        title="Robust",
        source_path="/tmp/robust.epub",
        source_lang="en",
        target_lang="zh",
        chapters=[ChapterMeta(chapter_id="ch01", title="One", spine_index=1)],
        metadata={},
    )


def _base_config(**overrides: Any) -> UBTConfig:
    defaults: dict[str, Any] = {
        "batch_limit": 30,
        "max_concurrency": 4,
        "batch_enabled": False,
        "tm_enabled": False,
        "draft_max_retries": 2,
        "draft_retry_base_delay": 0.01,
    }
    defaults.update(overrides)
    return UBTConfig(**defaults)


_CTX_RENAMES = {
    "actual_job_id": "job_id",
    "create_event_fn": "create_event",
    "all_blocks_count": "block_count",
}


async def _drain(**kwargs: Any) -> None:
    """Old-style keywords in, one StageContext out (see the draft stage's signature)."""
    ctx = build_stage_ctx(**{_CTX_RENAMES.get(k, k): v for k, v in kwargs.items()})
    async for _ in run_draft_stage(ctx):
        pass


class FlakyOnceProvider(MockModelProvider):
    """Fails the first generate call per prompt, then succeeds."""

    def __init__(self) -> None:
        super().__init__(default_response="好的翻译")
        self.seen: set[str] = set()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        if prompt not in self.seen:
            self.seen.add(prompt)
            # NOTE: deliberately NOT a 429/rate-limit message: those trigger
            # the router's 429 backoff (1s+) and rate-limiter penalty, which
            # would make this unit test sleep ~50s. A plain timeout exercises
            # the draft-stage retry (H3) without the slow path.
            raise ModelProviderError("transient timeout")
        return await super().generate(
            prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )


class PoisonProvider(MockModelProvider):
    """Succeeds for all prompts (poison routing is done at router level)."""

    def __init__(self) -> None:
        super().__init__(default_response="好的翻译")


class PoisonRouter(ModelRouter):
    """Fails exactly one block id to simulate a permanent provider failure."""

    def __init__(self, provider: MockModelProvider, poison_id: str = "ch01#b002") -> None:
        super().__init__(provider=provider, draft_model="mock", max_retries=0)
        self.poison_id = poison_id

    async def draft(self, block: IRBlock, *args: Any, **kwargs: Any) -> str:
        # *args/**kwargs passthrough: ModelRouter.draft gains params over time
        # (13 and counting); forwarding explicitly would silently drop new ones
        # while the test stays green.
        if block.id == self.poison_id:
            raise ModelProviderError("permanent provider failure")
        return await super().draft(block, *args, **kwargs)


def test_draft_retries_transient_failures(tmp_path: Path) -> None:
    """Fault injection: transient timeouts are retried, no block is lost."""
    ledger = SQLiteJobLedger(tmp_path / "flaky.sqlite")
    ledger.init_job("job_flaky", _make_doc(4), target_lang="zh")
    provider = FlakyOnceProvider()
    # max_retries=0: exercise the draft-stage retry (H3), not the router's.
    router = ModelRouter(provider=provider, draft_model="mock", max_retries=0)

    asyncio.run(
        _drain(
            ledger=ledger,
            actual_job_id="job_flaky",
            manifest=_make_manifest(),
            profile_name="general",
            target_lang="zh",
            source_lang="en",
            router=router,
            code_masker=CodeMasker(),
            citation_masker=CitationMasker(),
            config=_base_config(),
            all_blocks_count=4,
            glossary_dicts=[],
            abbreviation_entries=[],
            concurrency_sem=asyncio.Semaphore(4),
            create_event_fn=inert_event,
            tm=None,
        )
    )
    blocks = ledger.get_all_blocks("job_flaky")
    assert all(b.status == BlockStatus.DRAFTED for b in blocks)
    # Each block failed once then succeeded: provider saw > n prompts.
    assert len(provider.seen) == 4
    assert len(provider.call_history) == 4


def test_draft_poison_block_becomes_failed_not_lost(tmp_path: Path) -> None:
    """Fault injection: permanent failures land in FAILED with error flags."""
    ledger = SQLiteJobLedger(tmp_path / "poison.sqlite")
    ledger.init_job("job_poison", _make_doc(4, poison_idx=2), target_lang="zh")
    provider = PoisonProvider()
    router = PoisonRouter(provider=provider, poison_id="ch01#b002")

    asyncio.run(
        _drain(
            ledger=ledger,
            actual_job_id="job_poison",
            manifest=_make_manifest(),
            profile_name="general",
            target_lang="zh",
            source_lang="en",
            router=router,
            code_masker=CodeMasker(),
            citation_masker=CitationMasker(),
            config=_base_config(),
            all_blocks_count=4,
            glossary_dicts=[],
            abbreviation_entries=[],
            concurrency_sem=asyncio.Semaphore(4),
            create_event_fn=inert_event,
            tm=None,
        )
    )
    blocks = {b.id: b for b in ledger.get_all_blocks("job_poison")}
    failed = [b for b in blocks.values() if b.status == BlockStatus.FAILED]
    assert len(failed) == 1
    assert failed[0].id == "ch01#b002"
    assert any("Drafting error" in f for f in failed[0].error_flags)


def test_crash_kill_resume_excludes_human_queue(tmp_path: Path) -> None:
    """Crash recovery: NEEDS_HUMAN/BLOCKED_HUMAN survive restart unmodified."""
    db = tmp_path / "crash.sqlite"
    ledger = SQLiteJobLedger(db)
    ledger.init_job("job_crash", _make_doc(6), target_lang="zh")
    ledger.save_checkpoint(
        block_id="ch01#b001",
        target_text="译文一",
        status=BlockStatus.MTQE_PASSED,
        mtqe_score=0.95,
    )
    ledger.save_checkpoint(
        block_id="ch01#b002",
        target_text="译文二",
        status=BlockStatus.NEEDS_HUMAN,
        mqm_severity="major",
    )
    ledger.save_checkpoint(
        block_id="ch01#b003",
        target_text="源文占位",
        status=BlockStatus.BLOCKED_HUMAN,
        mqm_severity="critical",
    )
    # Simulate kill -9: drop the connection without finalize.
    ledger.close()

    reopened = SQLiteJobLedger(db)
    try:
        resume = reopened.fetch_pending_blocks("job_crash", limit=50)
        resume_ids = {b.id for b in resume}
        assert "ch01#b001" not in resume_ids
        assert "ch01#b002" not in resume_ids
        assert "ch01#b003" not in resume_ids
        assert resume_ids == {"ch01#b004", "ch01#b005", "ch01#b006"}
        # Human-queue states are intact after reopen: the transient-error
        # reset only re-queues blocks with transient drafting/repair markers,
        # never quality escalations.
        assert reopened.reset_transient_failures("job_crash") == []
        by_id = {b.id: b for b in reopened.get_all_blocks("job_crash")}
        assert by_id["ch01#b002"].status == BlockStatus.NEEDS_HUMAN
        assert by_id["ch01#b003"].status == BlockStatus.BLOCKED_HUMAN
    finally:
        reopened.close()


def test_schema_version_matches_migrations() -> None:
    """Guard against TARGET_SCHEMA_VERSION drift (must equal latest migration)."""
    assert TARGET_SCHEMA_VERSION == 10


class CountingProvider(MockModelProvider):
    """Tracks peak concurrent generate calls."""

    def __init__(self) -> None:
        super().__init__(default_response="好的翻译内容足够长以通过校验")
        self.inflight = 0
        self.peak = 0
        self.total = 0

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        self.inflight += 1
        self.total += 1
        self.peak = max(self.peak, self.inflight)
        try:
            await asyncio.sleep(0.01)
            return await super().generate(
                prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
        finally:
            self.inflight -= 1


class FailFastRouter(ModelRouter):
    """Raises a non-retryable 401 on every draft call, counting attempts."""

    def __init__(self, provider: MockModelProvider) -> None:
        super().__init__(provider=provider, draft_model="mock", max_retries=0)
        self.calls = 0

    async def draft(self, block: IRBlock, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        raise ModelProviderError("Unauthorized", details={"status_code": 401})


def test_draft_does_not_retry_fail_fast_errors(tmp_path: Path) -> None:
    """A 401 is unrecoverable; the outer draft loop must not multiply it (N4)."""
    ledger = SQLiteJobLedger(tmp_path / "failfast.sqlite")
    ledger.init_job("job_ff", _make_doc(1), target_lang="zh")
    provider = PoisonProvider()
    router = FailFastRouter(provider=provider)

    asyncio.run(
        _drain(
            ledger=ledger,
            actual_job_id="job_ff",
            manifest=_make_manifest(),
            profile_name="general",
            target_lang="zh",
            source_lang="en",
            router=router,
            code_masker=CodeMasker(),
            citation_masker=CitationMasker(),
            config=_base_config(draft_max_retries=3),
            all_blocks_count=1,
            glossary_dicts=[],
            abbreviation_entries=[],
            concurrency_sem=asyncio.Semaphore(4),
            create_event_fn=inert_event,
            tm=None,
        )
    )
    # Retrying would be draft_max_retries+1 calls; fail-fast means exactly one.
    assert router.calls == 1
    blocks = ledger.get_all_blocks("job_ff")
    assert blocks[0].status == BlockStatus.FAILED


def test_draft_respects_concurrency_limit(tmp_path: Path) -> None:
    """Backpressure: peak concurrency never exceeds the semaphore."""
    n = 12
    ledger = SQLiteJobLedger(tmp_path / "conc.sqlite")
    ledger.init_job("job_conc", _make_doc(n), target_lang="zh")
    provider = CountingProvider()
    router = ModelRouter(provider=provider, draft_model="mock", max_retries=0)

    asyncio.run(
        _drain(
            ledger=ledger,
            actual_job_id="job_conc",
            manifest=_make_manifest(),
            profile_name="general",
            target_lang="zh",
            source_lang="en",
            router=router,
            code_masker=CodeMasker(),
            citation_masker=CitationMasker(),
            config=_base_config(max_concurrency=4),
            all_blocks_count=n,
            glossary_dicts=[],
            abbreviation_entries=[],
            concurrency_sem=asyncio.Semaphore(4),
            create_event_fn=inert_event,
            tm=None,
        )
    )
    assert provider.total == n
    assert provider.peak <= 4
    assert provider.peak > 1  # genuinely concurrent, not serial


class JobLevelFailFastRouter(ModelRouter):
    """Every draft() call fails with a non-retryable 401 (bad credential)."""

    def __init__(self) -> None:
        super().__init__(
            provider=MockModelProvider(default_response="x"), draft_model="mock", max_retries=0
        )
        self.calls = 0

    async def draft(self, block: IRBlock, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        raise ModelProviderError("invalid api key", details={"status_code": 401})


def test_draft_fail_fast_circuit_aborts_job(tmp_path: Path) -> None:
    """P1-2: consecutive non-retryable failures must trip a job-level breaker.

    Without it, a bad credential makes every one of a book's blocks walk the
    draft retry chain for nothing; with it, the draft stage aborts after the
    threshold and the pipeline marks the job failed with the provider reason."""
    from ubt.core.exceptions import UBTError

    ledger = SQLiteJobLedger(tmp_path / "ff.sqlite")
    ledger.init_job("job_ff", _make_doc(40), target_lang="zh")
    router = JobLevelFailFastRouter()

    with pytest.raises(UBTError, match="fail-fast circuit"):
        asyncio.run(
            _drain(
                ledger=ledger,
                actual_job_id="job_ff",
                manifest=_make_manifest(),
                profile_name="general",
                target_lang="zh",
                source_lang="en",
                router=router,
                code_masker=CodeMasker(),
                citation_masker=CitationMasker(),
                config=_base_config(macro_chunk_size=1, batch_limit=30),
                all_blocks_count=40,
                glossary_dicts=[],
                abbreviation_entries=[],
                concurrency_sem=asyncio.Semaphore(4),
                create_event_fn=inert_event,
                tm=None,
            )
        )
    # Aborted after the first 30-block batch; the remaining 10 were never drafted.
    assert router.calls == 30
    ledger.close()


@pytest.mark.fast
async def test_draft_stage_restores_memory_on_fresh_job(tmp_path: Path) -> None:
    """run_draft_stage restores rolling memory even if config.fresh is True."""
    from ubt.core.engine.stages.draft import run_draft_stage

    config = UBTConfig(fresh=True, enable_rolling_summary=True)
    manifest = BookManifest(
        doc_id="doc1",
        title="Title",
        source_path=str(tmp_path / "input.epub"),
        chapters=[
            ChapterMeta(chapter_id="ch1", title="Chapter 1", spine_index=0),
            ChapterMeta(chapter_id="ch2", title="Chapter 2", spine_index=1),
        ],
    )
    ledger = SQLiteJobLedger(tmp_path / "test.sqlite")
    ledger.init_job_from_manifest("job_test", manifest)

    ctx = build_stage_ctx(
        tmp_path,
        job_id="job_test",
        input_path=tmp_path / "input.epub",
        config=config,
        manifest=manifest,
        ledger=ledger,
    )

    restore_mock = MagicMock()
    with (
        patch("ubt.core.engine.stages.draft._restore_memory_state", restore_mock),
        patch("ubt.core.engine.stages.draft.resolve_draft_policy", return_value=(True, False, 10)),
        patch.object(ledger, "fetch_pending_blocks", return_value=[]),
    ):
        async for _ in run_draft_stage(ctx, chapter_id="ch2"):
            pass

    restore_mock.assert_called_once()

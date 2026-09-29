"""Integration tests for Batch API drafting and the TM funnel."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.stage_ctx_factory import build_stage_ctx, inert_event
from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.draft import run_draft_stage
from ubt.core.engine.stages.tm_writeback import writeback_tm_from_ledger
from ubt.core.exceptions import BudgetExceededError, ModelProviderError
from ubt.core.ir.models import (
    BlockStatus,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    FlowID,
    IRBlock,
)
from ubt.core.memory.tm import PROMPT_VERSION, TMPendingEntry, TranslationMemory, compute_tm_context
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.router.capabilities import ExtractionStrategy
from ubt.core.router.provider import MockModelProvider, OpenAICompatibleProvider
from ubt.core.router.registry import ModelCapabilityRegistry
from ubt.core.router.router import (
    BatchDraftRequest,
    BatchTranslationError,
    ModelRouter,
)

# ---------------------------------------------------------------------------
# Provider-level Batch API flow (HTTP mocked via transport)
# ---------------------------------------------------------------------------


def _batch_transport(state: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/files":
            return httpx.Response(200, json={"id": "file-1"})
        if path == "/v1/batches" and request.method == "POST":
            state["created"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"id": "batch-1", "status": "validating"})
        if path == "/v1/batches/batch-1":
            state["polls"] += 1
            if state["polls"] < 2:
                return httpx.Response(200, json={"id": "batch-1", "status": "in_progress"})
            return httpx.Response(
                200,
                json={
                    "id": "batch-1",
                    "status": "completed",
                    "output_file_id": "out-1",
                },
            )
        if path == "/v1/files/out-1/content":
            line = {
                "custom_id": "b-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": "batch-test-model",
                        "choices": [{"message": {"content": "<translation>你好</translation>"}}],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 10,
                            "prompt_cache_hit_tokens": 80,
                        },
                    },
                },
            }
            return httpx.Response(200, text=json.dumps(line) + "\n")
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


def test_openai_provider_batch_flow_records_usage() -> None:
    """Submit → poll → parse, with cache-hit usage captured in the accounting."""
    state: dict[str, Any] = {"polls": 0, "created": None}
    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="http://test/v1",
        transport=_batch_transport(state),
        sanitize_output=False,
    )
    batch_id = asyncio.run(
        provider.create_batch_job(
            [{"custom_id": "b-1", "method": "POST", "url": "/v1/chat/completions", "body": {}}]
        )
    )
    assert batch_id == "batch-1"
    assert state["created"] is not None
    assert state["created"]["endpoint"] == "/v1/chat/completions"

    # Poll sequence: first poll in_progress, second poll completed.
    job1 = asyncio.run(provider.get_batch_job("batch-1"))
    assert job1["status"] == "in_progress"
    job2 = asyncio.run(provider.get_batch_job("batch-1"))
    assert job2["status"] == "completed"

    results = asyncio.run(provider.fetch_batch_results("batch-1"))
    assert results["b-1"]["content"] == "<translation>你好</translation>"
    assert results["b-1"]["error"] is None

    totals = provider.usage_totals
    assert totals["calls"] == 1
    assert totals["prompt_tokens"] == 100
    assert totals["completion_tokens"] == 10
    assert provider.usage_log[0]["prompt_cache_hit_tokens"] == 80


def test_mock_provider_has_no_batch_support() -> None:
    """Default contract: providers without a Batch API report unsupported."""
    provider = MockModelProvider()
    assert provider.supports_batch_api is False
    with pytest.raises(NotImplementedError):
        asyncio.run(provider.create_batch_job([]))


# ---------------------------------------------------------------------------
# Router-level batch orchestration
# ---------------------------------------------------------------------------


def test_router_draft_batch_extracts_and_reports_missing_lines() -> None:
    class RouterBatchProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        def __init__(self) -> None:
            super().__init__()
            self.submitted: list[list[dict[str, Any]]] = []

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            self.submitted.append(requests)
            return "batch-x"

        async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
            return {"status": "completed", "output_file_id": "out"}

        async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
            # Only answer the first request; the second line goes missing.
            first = self.submitted[-1][0]["custom_id"]
            return {
                first: {"content": "<translation>译文一</translation>", "error": None},
            }

    provider = RouterBatchProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")
    requests = [
        BatchDraftRequest(custom_id="req-1", source_text="Hello."),
        BatchDraftRequest(custom_id="req-2", source_text="World."),
    ]
    results = asyncio.run(router.draft_batch(requests, poll_interval=0.01))

    assert provider.submitted[0][0]["custom_id"] == "req-1"
    body = provider.submitted[0][0]["body"]
    assert body["model"] == "batch-test-model"
    # A bounded output budget mirrors the interactive path, so a long block is
    # not silently truncated at the provider's default cap.
    assert body["max_tokens"] >= 1024
    assert "Hello." in body["messages"][-1]["content"]
    assert results[0].text == "译文一"
    assert results[1].text is None
    assert results[1].error == "Missing from batch output"


def test_router_draft_batch_rides_out_a_transient_result_fetch_failure() -> None:
    """A blip on the result-file fetch must not discard a completed batch.

    The status poll already rides out retryable transport errors; the result
    fetch is the same kind of read. Treating a single 5xx there as a dead batch
    reaches the caller, which cancels the submitted job and re-drafts every
    block interactively — paying for the same work twice.
    """

    class FlakyFetchProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        def __init__(self) -> None:
            super().__init__()
            self.fetches = 0

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            return "batch-fetch"

        async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
            return {"status": "completed", "output_file_id": "out"}

        async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
            self.fetches += 1
            if self.fetches == 1:
                raise ModelProviderError(
                    "Batch result download failed (503)", details={"status_code": 503}
                )
            return {"req-1": {"content": "<translation>译文一</translation>", "error": None}}

    provider = FlakyFetchProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")
    results = asyncio.run(
        router.draft_batch(
            [BatchDraftRequest(custom_id="req-1", source_text="Hello.")],
            poll_interval=0.01,
        )
    )

    assert provider.fetches == 2, "the result fetch must be retried once"
    assert results[0].text == "译文一"
    assert results[0].error is None


def test_router_batch_body_forwards_chat_template_kwargs() -> None:
    """The batch body must carry chat_template_kwargs, matching the interactive path.

    A llama.cpp/vLLM server configured with {"enable_thinking": false} would
    otherwise emit thinking traces in batch mode only, and extraction would
    then consume the wrong content.
    """
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/files":
            captured["upload"] = request.content.decode("utf-8", errors="replace")
            return httpx.Response(200, json={"id": "file-1"})
        if path == "/v1/batches" and request.method == "POST":
            return httpx.Response(200, json={"id": "batch-1", "status": "validating"})
        if path == "/v1/batches/batch-1":
            return httpx.Response(
                200, json={"id": "batch-1", "status": "completed", "output_file_id": "out-1"}
            )
        if path == "/v1/files/out-1/content":
            line = {
                "custom_id": "b-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": "<translation>你好</translation>"}}]
                    },
                },
            }
            return httpx.Response(200, text=json.dumps(line) + "\n")
        return httpx.Response(404, text="not found")

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="http://test/v1",
        transport=httpx.MockTransport(handler),
        sanitize_output=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    router = ModelRouter(provider=provider, draft_model="batch-test-model")
    asyncio.run(
        router.draft_batch(
            [BatchDraftRequest(custom_id="b-1", source_text="Hello.")],
            poll_interval=0.01,
            cleanup_files=False,
        )
    )
    assert "chat_template_kwargs" in captured["upload"]
    assert "enable_thinking" in captured["upload"]


def test_router_batch_unsupported_provider_raises_translating_error() -> None:
    router = ModelRouter(provider=MockModelProvider(), draft_model="anything")
    with pytest.raises(BatchTranslationError):
        asyncio.run(router.draft_batch([BatchDraftRequest(custom_id="x", source_text="y")]))


def test_router_batch_wraps_provider_errors() -> None:
    class FailingBatchProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            raise ModelProviderError("endpoint has no /batches")

    router = ModelRouter(provider=FailingBatchProvider(), draft_model="batch-test-model")
    with pytest.raises(BatchTranslationError):
        asyncio.run(router.draft_batch([BatchDraftRequest(custom_id="x", source_text="y")]))


def test_router_batch_transport_failure_keeps_batch_id_for_abandon() -> None:
    """When a batch was already created and a later transport call fails,
    the repacked BatchTranslationError must carry batch_id, or draft.py cannot
    abandon_batch and the submitted job keeps billing while interactive drafting
    re-translates (double-charges) the same blocks."""

    class LateFailingBatchProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            return "batch-keep"

        async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
            return {"status": "completed", "output_file_id": "out"}

        async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
            raise ModelProviderError("network blip fetching results")

    router = ModelRouter(provider=LateFailingBatchProvider(), draft_model="batch-test-model")
    with pytest.raises(BatchTranslationError) as excinfo:
        asyncio.run(router.draft_batch([BatchDraftRequest(custom_id="x", source_text="y")]))
    assert excinfo.value.batch_id == "batch-keep"


class _CancelRecordingProvider(MockModelProvider):
    @property
    def supports_batch_api(self) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__()
        self.cancelled: list[str] = []

    async def cancel_batch_job(self, batch_id: str) -> None:
        self.cancelled.append(batch_id)


def test_abandon_batch_leaves_a_finished_batch_resumable(tmp_path: Path) -> None:
    """Abandoning must not cancel or relabel a batch that already finished.

    The poll loop records the provider's terminal status. Overwriting it with
    ``cancelled`` dropped the resumable state of a run whose results were
    already paid for, so a later run could no longer harvest them.
    """
    ledger = SQLiteJobLedger(tmp_path / "ledger.sqlite")
    ledger.register_batch_job("batch-done", "job-1", "idem-1", status="completed")
    provider = _CancelRecordingProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    asyncio.run(router.abandon_batch("batch-done", ledger=ledger, job_id="job-1"))

    assert provider.cancelled == [], "a finished batch has nothing to cancel"
    # Still resumable: the paid results can be fetched on the next run.
    assert ledger.find_live_batch_by_idempotency_key("idem-1") == "batch-done"


def test_abandon_batch_cancels_a_live_batch(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "ledger.sqlite")
    ledger.register_batch_job("batch-live", "job-1", "idem-1", status="in_progress")
    provider = _CancelRecordingProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    asyncio.run(router.abandon_batch("batch-live", ledger=ledger, job_id="job-1"))

    assert provider.cancelled == ["batch-live"]
    assert ledger.find_live_batch_by_idempotency_key("idem-1") is None


# ---------------------------------------------------------------------------
# Draft-stage integration: batch success, batch fallback, TM funnel
# ---------------------------------------------------------------------------


def _make_doc_ir(n_blocks: int) -> SeedDoc:
    # Deliberately dissimilar sentences so TM fuzzy matching only hits the
    # intended block (default test threshold is 0.85).
    contents = [
        "The old lighthouse keeper climbed the spiral staircase every evening.",
        "Modern parsers extract structured text from complicated PDF layouts.",
        "Paragraph 3 source content for testing.",
        "Quantum entanglement remains one of physics most counterintuitive phenomena.",
        "She folded the letter carefully before sliding it into the envelope.",
        "The committee published its final recommendations on Tuesday morning.",
    ][:n_blocks]
    blocks = [
        IRBlock(
            id=f"ch01#b{i:03d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            source_text=contents[i - 1],
        )
        for i in range(1, n_blocks + 1)
    ]
    return SeedDoc(
        doc_id="test_doc_sha256",
        source_path="/tmp/test_book.epub",
        format_type="epub",
        metadata={"title": "Test Book"},
        blocks=blocks,
    )


def _make_manifest() -> BookManifest:
    return BookManifest(
        doc_id="test_doc_sha256",
        title="Test Book",
        source_path="/tmp/test_book.epub",
        chapters=[ChapterMeta(chapter_id="ch01", title="One", spine_index=1)],
        metadata={},
    )


_CTX_RENAMES = {
    "actual_job_id": "job_id",
    "create_event_fn": "create_event",
    "all_blocks_count": "block_count",
}


async def _drain_stage(**kwargs: Any) -> None:
    """Old-style keywords in, one StageContext out.

    The draft stage takes the run's context now; this keeps the ~11 call sites in
    this file written the way they read before, and any keyword that is not a
    context field fails in the constructor rather than being ignored.
    """
    ctx = build_stage_ctx(**{_CTX_RENAMES.get(k, k): v for k, v in kwargs.items()})
    async for _event in run_draft_stage(ctx):
        pass


def _base_config(**overrides: Any) -> UBTConfig:
    defaults: dict[str, Any] = {
        "batch_limit": 30,
        "max_concurrency": 4,
        "batch_enabled": False,
        "batch_min_blocks": 2,
        "batch_poll_interval": 0.01,
        "batch_poll_timeout": 5.0,
    }
    defaults.update(overrides)
    return UBTConfig(**defaults)


class BatchSuccessProvider(MockModelProvider):
    """Trivially-completing Batch API returning a fixed translation per line."""

    @property
    def supports_batch_api(self) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__(default_response="[TRANSLATED]")
        self.batch_submissions: list[list[dict[str, Any]]] = []

    async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
        self.batch_submissions.append(requests)
        return "batch-1"

    async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
        return {"status": "completed", "output_file_id": "out"}

    async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
        return {
            r["custom_id"]: {"content": "[BATCH-TRANSLATED]", "error": None}
            for r in self.batch_submissions[-1]
        }


class BatchBrokenProvider(MockModelProvider):
    """Advertises batch support but fails at submission (e.g. no /batches)."""

    @property
    def supports_batch_api(self) -> bool:
        return True

    async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
        raise ModelProviderError("endpoint has no /batches support")


def test_draft_stage_batch_success_path(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_batch", _make_doc_ir(6), target_lang="zh")
    provider = BatchSuccessProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    asyncio.run(
        _drain_stage(
            ledger=ledger,
            actual_job_id="job_batch",
            manifest=_make_manifest(),
            router=router,
            config=_base_config(batch_enabled=True),
            all_blocks_count=6,
            concurrency_sem=asyncio.Semaphore(4),
        )
    )

    blocks = {b.id: b for b in ledger.get_all_blocks("job_batch")}
    assert all(b.status == BlockStatus.DRAFTED for b in blocks.values())
    assert len(provider.batch_submissions) == 1
    assert len(provider.batch_submissions[0]) == 6  # one line per block
    assert provider.call_history == []  # interactive path never invoked
    ledger.close()


def test_draft_stage_batch_failure_falls_back_interactively(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_fb", _make_doc_ir(6), target_lang="zh")
    provider = BatchBrokenProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    asyncio.run(
        _drain_stage(
            ledger=ledger,
            actual_job_id="job_fb",
            manifest=_make_manifest(),
            router=router,
            config=_base_config(batch_enabled=True),
            all_blocks_count=6,
            concurrency_sem=asyncio.Semaphore(4),
        )
    )

    blocks = ledger.get_all_blocks("job_fb")
    assert all(b.status == BlockStatus.DRAFTED for b in blocks)
    assert len(provider.call_history) == 6  # every block drafted interactively
    ledger.close()


def test_draft_stage_batch_cancelled_in_status_callback_raises(tmp_path: Path) -> None:
    """When a job is cancelled, status_callback inside try_batch_draft must detect it and raise JobInterruptedError
    immediately instead of proceeding to interactive fallback."""
    from ubt.core.exceptions import JobInterruptedError

    ledger = SQLiteJobLedger(tmp_path / "job_cancel.sqlite")
    seed_job(ledger, "job_cancel", _make_doc_ir(6), target_lang="zh")

    cancel_token = asyncio.Event()

    class InProgressBatchProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            return "batch-in-progress"

        async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
            # Trigger cancellation during status polling
            cancel_token.set()
            return {"status": "in_progress"}

    provider = InProgressBatchProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    with pytest.raises(JobInterruptedError):
        asyncio.run(
            _drain_stage(
                ledger=ledger,
                actual_job_id="job_cancel",
                manifest=_make_manifest(),
                router=router,
                config=_base_config(batch_enabled=True, batch_poll_interval=0.01),
                all_blocks_count=6,
                concurrency_sem=asyncio.Semaphore(4),
                cancel_token=cancel_token,
            )
        )
    # Interactive path must not have been invoked
    assert provider.call_history == []
    ledger.close()


def test_draft_stage_batch_budget_exceeded_in_status_callback_raises(tmp_path: Path) -> None:
    """When budget is exceeded, try_batch_draft must not fall back to interactive drafting."""
    ledger = SQLiteJobLedger(tmp_path / "job_budget.sqlite")
    seed_job(ledger, "job_budget", _make_doc_ir(6), target_lang="zh")

    class InProgressBatchProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            return "batch-in-progress"

        async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
            return {"status": "in_progress"}

    provider = InProgressBatchProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    async def _failing_create_event(*args: Any, **kwargs: Any) -> Any:
        raise BudgetExceededError("batch draft exceeded budget")

    with pytest.raises(BudgetExceededError, match="batch draft exceeded budget"):
        asyncio.run(
            _drain_stage(
                ledger=ledger,
                actual_job_id="job_budget",
                manifest=_make_manifest(),
                router=router,
                config=_base_config(batch_enabled=True, batch_poll_interval=0.01),
                all_blocks_count=6,
                concurrency_sem=asyncio.Semaphore(4),
                create_event_fn=_failing_create_event,
            )
        )
    assert provider.call_history == []
    ledger.close()


def test_draft_stage_batch_disabled_uses_interactive(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_off", _make_doc_ir(6), target_lang="zh")
    provider = BatchSuccessProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    asyncio.run(
        _drain_stage(
            ledger=ledger,
            actual_job_id="job_off",
            manifest=_make_manifest(),
            router=router,
            config=_base_config(batch_enabled=False),
            all_blocks_count=6,
            concurrency_sem=asyncio.Semaphore(4),
        )
    )

    assert provider.batch_submissions == []
    assert len(provider.call_history) == 6
    ledger.close()


def test_draft_stage_tm_exact_hit_skips_llm(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_tm", _make_doc_ir(5), target_lang="zh")
    tm = TranslationMemory(tmp_path / "tm.sqlite")
    # The staged entry must carry the same context fingerprint the draft
    # stage computes (PROMPT_VERSION + profile + glossary table + abbreviation
    # table + langs).
    tm.writeback(
        [
            TMPendingEntry(
                "en",
                "zh",
                "Paragraph 3 source content for testing.",
                "第三段测试内容。",
                context_hash=compute_tm_context(PROMPT_VERSION, "general", "", "en", "zh"),
            )
        ]
    )
    provider = MockModelProvider(default_response="[TRANSLATED]")
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    try:
        asyncio.run(
            _drain_stage(
                ledger=ledger,
                actual_job_id="job_tm",
                manifest=_make_manifest(),
                router=router,
                config=_base_config(),
                all_blocks_count=5,
                concurrency_sem=asyncio.Semaphore(4),
                tm=tm,
            )
        )

        blocks = {b.id: b for b in ledger.get_all_blocks("job_tm")}
        tm_block = blocks["ch01#b003"]
        assert tm_block.status == BlockStatus.MTQE_PASSED
        assert tm_block.target_text == "第三段测试内容。"
        assert tm_block.mtqe_score == 1.0

        # The other four blocks went through the LLM; the TM block did not.
        assert len(provider.call_history) == 4
        with ledger._get_conn() as conn:
            row = conn.execute("SELECT tm_hit FROM blocks WHERE block_id = 'ch01#b003'").fetchone()
            assert row is not None and row["tm_hit"] == 1
    finally:
        ledger.close()
        tm.close()


def test_draft_stage_tm_fuzzy_hit_injects_few_shot(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_fz", _make_doc_ir(5), target_lang="zh")
    tm = TranslationMemory(tmp_path / "tm.sqlite")
    # Fuzzy hits are context-gated like the exact path — the staged
    # entry must carry the draft stage's context fingerprint to be reusable.
    tm.writeback(
        [
            TMPendingEntry(
                "en",
                "zh",
                "Paragraph 3 source contents for testing.",
                "第三段测试内容以及更多上下文。",
                context_hash=compute_tm_context(PROMPT_VERSION, "general", "", "en", "zh"),
            )
        ]
    )
    provider = MockModelProvider(default_response="[TRANSLATED]")
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    try:
        asyncio.run(
            _drain_stage(
                ledger=ledger,
                actual_job_id="job_fz",
                manifest=_make_manifest(),
                router=router,
                config=_base_config(),
                all_blocks_count=5,
                concurrency_sem=asyncio.Semaphore(4),
                tm=tm,
            )
        )

        # All blocks drafted via the LLM, but the fuzzy block's prompt carried
        # the few-shot reference block.
        assert len(provider.call_history) == 5
        fuzzy_prompts = [c for c in provider.call_history if "Reference Translation" in c["prompt"]]
        assert len(fuzzy_prompts) == 1
        assert "第三段测试内容以及更多上下文。" in fuzzy_prompts[0]["prompt"]
    finally:
        ledger.close()
        tm.close()


def test_registry_profiles_supported_for_batch_tests() -> None:
    """Sanity: AUTO extraction resolves for the unknown test model."""
    registry = ModelCapabilityRegistry()
    profile = registry.resolve("batch-test-model")
    assert profile.extraction_strategy == ExtractionStrategy.AUTO


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_draft_stage_routes_all_blocks_through_llm(tmp_path: Path) -> None:
    """All blocks route uniformly through the LLM draft model (no MT tier is wired in)."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_mt", _make_doc_ir(6), target_lang="zh")
    provider = MockModelProvider(default_response="[LLM-DRAFT]")
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    try:
        asyncio.run(
            _drain_stage(
                ledger=ledger,
                actual_job_id="job_mt",
                manifest=_make_manifest(),
                router=router,
                config=_base_config(),
                all_blocks_count=6,
                concurrency_sem=asyncio.Semaphore(4),
            )
        )

        blocks = ledger.get_all_blocks("job_mt")
        assert len(blocks) == 6
        assert all(b.status == BlockStatus.DRAFTED for b in blocks)
        assert all(b.target_text == "[LLM-DRAFT]" for b in blocks)
        # All 6 calls routed uniformly through the LLM draft model.
        assert len(provider.call_history) == 6
        assert all(c["model"] == "batch-test-model" for c in provider.call_history)
    finally:
        ledger.close()


def test_draft_stage_retries_transient_failures(tmp_path: Path) -> None:
    """Draft stage retries on transient errors and delivers all blocks."""

    class TransientFailingProvider(MockModelProvider):
        def __init__(self) -> None:
            super().__init__(default_response="[LLM-DRAFT]")
            self.failed_once = False

        async def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> str:
            if not self.failed_once:
                self.failed_once = True
                raise ModelProviderError("Transient rate limit / network error")
            return await super().generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )

    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_mtfb", _make_doc_ir(6), target_lang="zh")
    provider = TransientFailingProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    try:
        asyncio.run(
            _drain_stage(
                ledger=ledger,
                actual_job_id="job_mtfb",
                manifest=_make_manifest(),
                router=router,
                config=_base_config(),
                all_blocks_count=6,
                concurrency_sem=asyncio.Semaphore(4),
            )
        )

        blocks = ledger.get_all_blocks("job_mtfb")
        assert len(blocks) == 6
        assert all(b.status == BlockStatus.DRAFTED for b in blocks)
        assert all(b.target_text == "[LLM-DRAFT]" for b in blocks)
        assert all(c["model"] == "batch-test-model" for c in provider.call_history)
    finally:
        ledger.close()


def test_draft_stage_keeps_term_blocks_on_llm(tmp_path: Path) -> None:
    """All blocks draft through the LLM with terminology preserved directly in the prompt."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_mtterm", _make_doc_ir(6), target_lang="zh")
    provider = MockModelProvider(default_response="[DRAFT]")
    router = ModelRouter(provider=provider, draft_model="batch-test-model")
    # "Paragraph 3" appears verbatim in block ch01#b003 only.
    glossary = [{"source": "Paragraph 3", "aliases": [], "translation": "第三段", "frequency": 5}]

    try:
        asyncio.run(
            _drain_stage(
                ledger=ledger,
                actual_job_id="job_mtterm",
                manifest=_make_manifest(),
                router=router,
                config=_base_config(),
                all_blocks_count=6,
                glossary_dicts=glossary,
                concurrency_sem=asyncio.Semaphore(4),
            )
        )

        llm_calls = [c for c in provider.call_history if c["model"] == "batch-test-model"]
        # Every block routed uniformly to the LLM.
        assert len(llm_calls) == 6
        # Block with glossary terms has the term table included in its prompt.
        term_prompts = [c["prompt"] for c in llm_calls if "Paragraph 3" in c["prompt"]]
        assert len(term_prompts) >= 1
    finally:
        ledger.close()


def test_draft_stage_without_mt_tier_keeps_llm_path(tmp_path: Path) -> None:
    """Without a dedicated MT tier the pipeline routes uniformly through the LLM."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_mtoff", _make_doc_ir(6), target_lang="zh")
    provider = MockModelProvider(default_response="[TRANSLATED]")
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    try:
        asyncio.run(
            _drain_stage(
                ledger=ledger,
                actual_job_id="job_mtoff",
                manifest=_make_manifest(),
                router=router,
                config=_base_config(),
                all_blocks_count=6,
                concurrency_sem=asyncio.Semaphore(4),
            )
        )

        assert len(provider.call_history) == 6
        assert all(c["model"] == "batch-test-model" for c in provider.call_history)
    finally:
        ledger.close()


# ---------------------------------------------------------------------------
# Batch idempotency, state persistence, error-file parsing
# ---------------------------------------------------------------------------


def test_ledger_batch_jobs_roundtrip(tmp_path: Path) -> None:
    """Batch_jobs table persists jobs and resumes live ones by key."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.register_batch_job("batch-1", "job1", "key-abc", status="submitted")
    ledger.update_batch_job_status("batch-1", "in_progress")
    assert ledger.find_live_batch_by_idempotency_key("key-abc") == "batch-1"
    # ``completed`` is the provider's terminal status but the batch is not yet
    # consumed: results are not fetched/persisted. It must still resume or a
    # crash in that window re-creates (and re-bills) the identical payload.
    ledger.update_batch_job_status("batch-1", "completed")
    assert ledger.find_live_batch_by_idempotency_key("key-abc") == "batch-1"
    # ``consumed`` is the true terminal: results have been taken.
    ledger.update_batch_job_status("batch-1", "consumed")
    assert ledger.find_live_batch_by_idempotency_key("key-abc") is None
    # Unknown key never resumes.
    assert ledger.find_live_batch_by_idempotency_key("key-other") is None
    ledger.close()


def test_reserve_batch_job_resumes_completed_unconsumed(tmp_path: Path) -> None:
    """A crash between completed and consumed must resume, never re-create.

    Pre-fix the resumed reservation saw no ``LIVE`` row, inserted a fresh
    sentinel and returned ("create", None) -> the same batch was submitted and
    billed a second time.
    """
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.register_batch_job("batch-c", "job1", "key-c", status="submitted")
    ledger.update_batch_job_status("batch-c", "completed")

    assert ledger.reserve_batch_job("key-c", "job2") == ("resume", "batch-c")
    ledger.close()


def test_reserve_batch_job_create_finalize_resume(tmp_path: Path) -> None:
    """Reserve->finalize->reserve: the second caller resumes, never re-creates."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    # First caller owns the create and leaves a reclaimable sentinel behind.
    assert ledger.reserve_batch_job("key-1", "job1") == ("create", None)
    assert ledger.find_live_batch_by_idempotency_key("key-1") is None  # not pollable yet
    # Concurrent caller must not double-create while the sentinel is fresh.
    assert ledger.reserve_batch_job("key-1", "job2") == ("pending", None)
    # Owner finishes submitting and promotes the sentinel to the real id.
    ledger.finalize_batch_job("key-1", "batch-real")
    assert ledger.find_live_batch_by_idempotency_key("key-1") == "batch-real"
    # A restart resumes the live batch instead of paying for a duplicate.
    assert ledger.reserve_batch_job("key-1", "job3") == ("resume", "batch-real")
    ledger.close()


def test_reserve_batch_job_adopts_stale_sentinel(tmp_path: Path) -> None:
    """A reservation whose owner died mid-create is taken over, not blocked on."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    assert ledger.reserve_batch_job("key-stale", "jobA") == ("create", None)
    # Horizon far in the future => the fresh sentinel reads as abandoned.
    assert ledger.reserve_batch_job("key-stale", "jobB", lease_seconds=-1.0) == ("create", None)
    ledger.close()


def test_draft_batch_pending_reservation_skips_create(tmp_path: Path) -> None:
    """When a concurrent worker owns the reservation, skip batch — never double-bill."""
    provider = _ResumeBatchProvider()
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    requests = [BatchDraftRequest(custom_id="b1", source_text="hello")]
    # Force the "another worker is mid-create" outcome regardless of timing.
    ledger.reserve_batch_job = lambda *a, **k: ("pending", None)  # type: ignore[method-assign]
    router = ModelRouter(provider=provider, draft_model="batch-test-model")
    with pytest.raises(BatchTranslationError, match="another"):
        asyncio.run(
            router.draft_batch(
                requests, poll_interval=0.01, poll_timeout=0.05, ledger=ledger, job_id="job1"
            )
        )
    assert provider.create_calls == 0  # fell back rather than submit a duplicate
    ledger.close()


def test_fetch_batch_results_parses_error_file() -> None:
    """Empty output + error file -> structured per-line errors."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/batches/batch-err"):
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output_file_id": "out-empty",
                    "error": {"file_id": "err-file"},
                },
            )
        if url.endswith("/files/out-empty/content"):
            return httpx.Response(200, text="")
        if url.endswith("/files/err-file/content"):
            return httpx.Response(
                200,
                text=json.dumps(
                    {
                        "custom_id": "blk-9",
                        "error": {"code": "invalid_request", "message": "context too long"},
                    }
                ),
            )
        return httpx.Response(404, json={"error": "unreachable"})

    provider = OpenAICompatibleProvider(
        api_key="k", default_model="m", transport=httpx.MockTransport(handler)
    )
    results = asyncio.run(provider.fetch_batch_results("batch-err"))
    assert results == {"blk-9": {"content": None, "error": "context too long"}}


class _ResumeBatchProvider(MockModelProvider):
    """Batch provider whose job stays in_progress until poked, then completes."""

    @property
    def supports_batch_api(self) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__(default_response="[TRANSLATED]")
        self.create_calls = 0
        self.complete = False

    async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
        self.create_calls += 1
        return f"batch-{self.create_calls}"

    async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
        if not self.complete:
            return {"status": "in_progress"}
        return {"status": "completed", "output_file_id": "out"}

    async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
        return {
            "b1": {"content": "[BATCH-TRANSLATED]", "error": None},
            "b2": {"content": "[BATCH-TRANSLATED]", "error": None},
        }


def test_draft_batch_resumes_live_batch_without_resubmit(tmp_path: Path) -> None:
    """A restart with the same block set resumes polling, no double bill."""
    provider = _ResumeBatchProvider()
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    requests = [
        BatchDraftRequest(custom_id="b1", source_text="hello"),
        BatchDraftRequest(custom_id="b2", source_text="world"),
    ]

    # "First run": submission succeeds but the job never finishes in time.
    router1 = ModelRouter(provider=provider, draft_model="batch-test-model")
    with pytest.raises(BatchTranslationError, match="timed out"):
        asyncio.run(
            router1.draft_batch(
                requests,
                poll_interval=0.01,
                poll_timeout=0.05,
                ledger=ledger,
                job_id="job1",
            )
        )
    assert provider.create_calls == 1

    # "Restart": the same requests must resume batch-1, not submit batch-2.
    provider.complete = True  # the provider-side job finished meanwhile
    router2 = ModelRouter(provider=provider, draft_model="batch-test-model")
    results = asyncio.run(
        router2.draft_batch(
            requests,
            poll_interval=0.01,
            poll_timeout=5.0,
            ledger=ledger,
            job_id="job1",
        )
    )
    assert provider.create_calls == 1  # no duplicate submission
    assert [r.text for r in results] == ["[BATCH-TRANSLATED]", "[BATCH-TRANSLATED]"]
    ledger.close()


class _CancelTrackingBatchProvider(_ResumeBatchProvider):
    """Records cancel_batch_job calls so the abandon path can be asserted."""

    def __init__(self) -> None:
        super().__init__()
        self.cancelled: list[str] = []

    async def cancel_batch_job(self, batch_id: str) -> None:
        self.cancelled.append(batch_id)


def test_draft_batch_timeout_carries_batch_id(tmp_path: Path) -> None:
    """A poll timeout must surface the submitted batch_id on the error, so the
    caller that falls back to interactive can cancel (not silently orphan) it."""
    provider = _CancelTrackingBatchProvider()
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    requests = [BatchDraftRequest(custom_id="b1", source_text="hello")]
    router = ModelRouter(provider=provider, draft_model="batch-test-model")
    with pytest.raises(BatchTranslationError) as ei:
        asyncio.run(
            router.draft_batch(
                requests,
                poll_interval=0.01,
                poll_timeout=0.05,
                ledger=ledger,
                job_id="job1",
            )
        )
    # draft_batch itself must NOT cancel — a later restart resumes the live
    # batch instead; cancel happens only when the caller abandons to interactive.
    assert provider.cancelled == []
    assert ei.value.batch_id == "batch-1"
    ledger.close()


def test_abandon_batch_cancels_and_marks_cancelled(tmp_path: Path) -> None:
    """The interactive-fallback abandonment point cancels the still-running
    batch and records it, which is what prevents the duplicate charge."""
    provider = _CancelTrackingBatchProvider()
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    router = ModelRouter(provider=provider, draft_model="batch-test-model")
    asyncio.run(router.abandon_batch("batch-1", ledger=ledger, job_id="job1"))
    assert provider.cancelled == ["batch-1"]
    # Bookkeeping must not raise even when no batch_jobs row exists to update.
    ledger.close()


def test_draft_batch_prompt_change_invalidates_idempotency(tmp_path: Path) -> None:
    """Regression: the idempotency key must cover the built prompts,
    not just model+custom_id — a changed glossary on restart must not resume
    a batch that was submitted with the old terminology."""
    provider = _ResumeBatchProvider()
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    old_requests = [BatchDraftRequest(custom_id="b1", source_text="hello")]
    new_requests = [
        BatchDraftRequest(custom_id="b1", source_text="hello", glossary_table="KV = 键值缓存")
    ]

    router1 = ModelRouter(provider=provider, draft_model="batch-test-model")
    with pytest.raises(BatchTranslationError, match="timed out"):
        asyncio.run(
            router1.draft_batch(
                old_requests,
                poll_interval=0.01,
                poll_timeout=0.05,
                ledger=ledger,
                job_id="job1",
            )
        )
    assert provider.create_calls == 1

    # Restart with the same custom_id but a different prompt payload: the
    # stale live batch must NOT be resumed.
    provider.complete = True
    router2 = ModelRouter(provider=provider, draft_model="batch-test-model")
    results = asyncio.run(
        router2.draft_batch(
            new_requests,
            poll_interval=0.01,
            poll_timeout=5.0,
            ledger=ledger,
            job_id="job1",
        )
    )
    assert provider.create_calls == 2
    assert [r.text for r in results] == ["[BATCH-TRANSLATED]"]
    ledger.close()


# ---------------------------------------------------------------------------
# Batch file reaping (privacy: uploaded source text must not outlive the job)
# ---------------------------------------------------------------------------


def _cleanup_transport(state: dict[str, Any], *, delete_status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/batches/batch-1":
            return httpx.Response(
                200,
                json={
                    "id": "batch-1",
                    "status": "completed",
                    "input_file_id": "in-1",
                    "output_file_id": "out-1",
                    "error": {"file_id": "err-1"},
                },
            )
        if path.startswith("/v1/files/") and request.method == "DELETE":
            fid = path.rsplit("/", 1)[-1]
            state.setdefault("deleted", []).append(fid)
            return httpx.Response(delete_status, text="ok")
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


def test_openai_provider_cleanup_deletes_all_batch_files() -> None:
    """input+output+error file ids are read back from metadata and each deleted.

    Reading from the batch object (not a carried return value) is the point: it
    means a job resumed after a restart still gets reaped.
    """
    state: dict[str, Any] = {}
    provider = OpenAICompatibleProvider(
        api_key="k", base_url="http://test/v1", transport=_cleanup_transport(state)
    )
    asyncio.run(provider.cleanup_batch_files("batch-1"))
    assert set(state["deleted"]) == {"in-1", "out-1", "err-1"}


def test_openai_provider_cleanup_is_best_effort_on_delete_error() -> None:
    """A gateway without DELETE (405) must not raise out of a completed run."""
    state: dict[str, Any] = {}
    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="http://test/v1",
        transport=_cleanup_transport(state, delete_status=405),
    )
    asyncio.run(provider.cleanup_batch_files("batch-1"))  # no exception
    assert len(state["deleted"]) == 3


def test_router_draft_batch_reaps_files_and_marks_consumed(tmp_path: Path) -> None:
    class ReapingBatchProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        def __init__(self) -> None:
            super().__init__()
            self.cleanups: list[str] = []

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            self._last = [r["custom_id"] for r in requests]
            return "batch-r"

        async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
            return {"status": "completed"}

        async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
            return {
                cid: {"content": "<translation>x</translation>", "error": None}
                for cid in self._last
            }

        async def cleanup_batch_files(self, batch_id: str) -> None:
            self.cleanups.append(batch_id)

    provider = ReapingBatchProvider()
    router = ModelRouter(provider=provider, draft_model="m")
    ledger = SQLiteJobLedger(tmp_path / "l.db")
    seed_job(ledger, "job-1", _make_doc_ir(1), target_lang="zh")
    asyncio.run(
        router.draft_batch(
            [BatchDraftRequest(custom_id="a", source_text="A")],
            poll_interval=0.01,
            ledger=ledger,
            job_id="job-1",
        )
    )
    assert provider.cleanups == ["batch-r"]
    with __import__("sqlite3").connect(tmp_path / "l.db") as conn:
        row = conn.execute("SELECT status FROM batch_jobs WHERE batch_id='batch-r'").fetchone()
    assert row is not None and row[0] == "consumed"


def test_router_draft_batch_keeps_files_when_disabled(tmp_path: Path) -> None:
    class BatchProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        def __init__(self) -> None:
            super().__init__()
            self.cleanups: list[str] = []

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            return "batch-k"

        async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
            return {"status": "completed"}

        async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
            return {"a": {"content": "<translation>x</translation>", "error": None}}

        async def cleanup_batch_files(self, batch_id: str) -> None:
            self.cleanups.append(batch_id)

    provider = BatchProvider()
    router = ModelRouter(provider=provider, draft_model="m")
    asyncio.run(
        router.draft_batch(
            [BatchDraftRequest(custom_id="a", source_text="A")],
            poll_interval=0.01,
            cleanup_files=False,
        )
    )
    assert provider.cleanups == []


def test_router_draft_batch_propagates_status_callback_errors() -> None:
    """The batch poll callback is the caller's interrupt channel (cancel
    check, budget cap); swallowing its exceptions let a capped run keep
    billing through the whole batch lifetime."""

    class CallbackRaisingProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            return "batch-cb"

        async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
            return {"status": "in_progress"}

    from ubt.core.exceptions import BudgetExceededError

    async def _cb(status: str, job: dict[str, Any]) -> None:
        raise BudgetExceededError("budget blown")

    router = ModelRouter(provider=CallbackRaisingProvider(), draft_model="batch-test-model")
    with pytest.raises(BudgetExceededError, match="budget blown"):
        asyncio.run(
            router.draft_batch(
                [BatchDraftRequest(custom_id="r1", source_text="Hi.")],
                poll_interval=0.01,
                poll_timeout=0.1,
                status_callback=_cb,
            )
        )


def test_draft_stage_batch_budget_stop_aborts_not_falls_back(tmp_path: Path) -> None:
    """A budget violation surfaced by the poll callback must stop the run,
    not be reinterpreted as 'batch unavailable' into an interactive
    re-draft past the cap."""

    from ubt.core.exceptions import BudgetExceededError
    from ubt.core.ir.models import BlockStatus as _BS

    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_budget", _make_doc_ir(6), target_lang="zh")
    provider = BatchSuccessProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")

    async def _billing_event(*args: Any, **kwargs: Any) -> Any:
        raise BudgetExceededError("Budget exceeded: $1 > $0.50")

    with pytest.raises(BudgetExceededError):
        asyncio.run(
            _drain_stage(
                ledger=ledger,
                actual_job_id="job_budget",
                manifest=_make_manifest(),
                router=router,
                config=_base_config(batch_enabled=True, offline_batch_enabled=True),
                all_blocks_count=6,
                concurrency_sem=asyncio.Semaphore(4),
                create_event=_billing_event,
            )
        )

    # The whole-book batch stops at the cap on the first poll that bills
    # past it: no interactive fallback re-billing, blocks left as-is.
    assert provider.call_history == []
    blocks = ledger.get_all_blocks("job_budget")
    assert all(b.status == _BS.PENDING for b in blocks)
    ledger.close()


def test_transient_poll_failure_does_not_abandon_a_live_batch() -> None:
    """One 502 on the status endpoint must not cost the user the whole batch.

    The poll loop used to let any ModelProviderError out as a
    BatchTranslationError, which the draft stage reads as "batch is dead": it
    cancels the job and re-drafts every block interactively. On a multi-hour
    whole-book batch polled every 30 s, one network blip therefore destroyed a
    committed spend and re-billed the same work at the higher interactive price.
    """

    class FlakyPollProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        def __init__(self) -> None:
            super().__init__(default_response="[INTERACTIVE]")
            self.polls = 0
            self.cancelled: list[str] = []
            self._lines: list[str] = []

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            self._lines = [str(r["custom_id"]) for r in requests]
            return "batch-live"

        async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
            self.polls += 1
            if self.polls == 1:
                raise ModelProviderError(
                    "502 bad gateway while reading the batch", details={"status_code": 502}
                )
            return {"status": "completed", "output_file_id": "out"}

        async def cancel_batch_job(self, batch_id: str) -> None:
            self.cancelled.append(batch_id)

        async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
            return {cid: {"content": "[BATCH-OK]", "error": None} for cid in self._lines}

    provider = FlakyPollProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")
    results = asyncio.run(
        router.draft_batch(
            [
                BatchDraftRequest(custom_id="b1", source_text="Hello."),
                BatchDraftRequest(custom_id="b2", source_text="World."),
            ],
            poll_interval=0.01,
            poll_timeout=5.0,
        )
    )
    assert provider.polls == 2, "the retryable poll failure did not retry"
    assert [r.text for r in results] == ["[BATCH-OK]", "[BATCH-OK]"]
    assert provider.cancelled == [], "a healthy batch was cancelled over one blip"


def test_budget_cap_cancels_the_still_billing_batch() -> None:
    """A hard cost cap has to stop the batch, not just stop the run.

    The callback is the cap's enforcement point; it used to propagate out of the
    poll loop with the batch left in_progress at the provider, so the limit the
    user set was a suggestion about the local loop rather than about spend.
    """

    class CappedBatchProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        def __init__(self) -> None:
            super().__init__(default_response="[INTERACTIVE]")
            self.cancelled: list[str] = []

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            return "batch-spend"

        async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
            return {"status": "in_progress"}

        async def cancel_batch_job(self, batch_id: str) -> None:
            self.cancelled.append(batch_id)

    async def _over_cap(status: str, job: dict[str, Any]) -> None:
        raise BudgetExceededError("cost cap reached")

    provider = CappedBatchProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")
    with pytest.raises(BudgetExceededError):
        asyncio.run(
            router.draft_batch(
                [BatchDraftRequest(custom_id="b1", source_text="Hello.")],
                poll_interval=0.01,
                poll_timeout=5.0,
                status_callback=_over_cap,
            )
        )
    assert provider.cancelled == ["batch-spend"]


def test_cancel_cancels_the_still_billing_batch() -> None:
    """A cooperative cancel must abandon the live batch, exactly like a cap.

    ``ctx.check_cancelled()`` in the poll callback raises ``JobInterruptedError``;
    it used to propagate with the batch left in_progress, so a cancelled job kept
    billing at the provider. The router only caught ``BudgetExceededError``.
    """
    from ubt.core.exceptions import JobInterruptedError

    class CappedBatchProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        def __init__(self) -> None:
            super().__init__(default_response="[INTERACTIVE]")
            self.cancelled: list[str] = []

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            return "batch-cancel"

        async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
            return {"status": "in_progress"}

        async def cancel_batch_job(self, batch_id: str) -> None:
            self.cancelled.append(batch_id)

    async def _cancelled(status: str, job: dict[str, Any]) -> None:
        raise JobInterruptedError("job cancelled")

    provider = CappedBatchProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")
    with pytest.raises(JobInterruptedError):
        asyncio.run(
            router.draft_batch(
                [BatchDraftRequest(custom_id="b1", source_text="Hello.")],
                poll_interval=0.01,
                poll_timeout=5.0,
                status_callback=_cancelled,
            )
        )
    assert provider.cancelled == ["batch-cancel"]


def test_draft_stage_redrafts_batch_lines_that_came_back_empty(tmp_path: Path) -> None:
    """An empty completion is a missing line, not a drafted one.

    The result mapping accepted ``text=""`` (only ``None`` was rejected), so a
    batch line with no content was stamped DRAFTED with a blank target and the
    interactive re-draft was skipped -- the macro path already treats "" as
    missing, so the two batch shapes disagreed about the same provider answer.
    """

    class BatchEmptyLineProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        def __init__(self) -> None:
            super().__init__(default_response="[INTERACTIVE-RESCUE]")
            self.submissions: list[list[str]] = []

        async def create_batch_job(self, requests: list[dict[str, Any]]) -> str:
            self.submissions.append([str(r["custom_id"]) for r in requests])
            return "batch-empty"

        async def get_batch_job(self, batch_id: str) -> dict[str, Any]:
            return {"status": "completed", "output_file_id": "out"}

        async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, Any]]:
            return {cid: {"content": "", "error": None} for cid in self.submissions[-1]}

    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_empty", _make_doc_ir(3), target_lang="zh")
    provider = BatchEmptyLineProvider()
    router = ModelRouter(provider=provider, draft_model="batch-test-model")
    asyncio.run(
        _drain_stage(
            ledger=ledger,
            actual_job_id="job_empty",
            manifest=_make_manifest(),
            router=router,
            config=_base_config(batch_enabled=True),
            all_blocks_count=3,
            concurrency_sem=asyncio.Semaphore(4),
        )
    )
    blocks = ledger.get_all_blocks("job_empty")
    assert provider.call_history, "empty batch lines were accepted without re-drafting"
    assert all(b.target_text and b.target_text.strip() for b in blocks), [
        b.target_text for b in blocks
    ]
    ledger.close()


@pytest.mark.fast
def test_openai_batch_results_reads_error_file_id() -> None:
    """OpenAI returns error_file_id at the top level of the Batch object.

    When a batch has 100% failure (output_file_id is None, error_file_id is set),
    fetch_batch_results must download the error file and populate errors,
    rather than crash claiming the batch is not completed.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/batches/batch-err":
            return httpx.Response(
                200,
                json={
                    "id": "batch-err",
                    "status": "completed",
                    "output_file_id": None,
                    "error_file_id": "err-file-1",
                },
            )
        if path == "/v1/files/err-file-1/content":
            line = {
                "custom_id": "req-1",
                "error": {
                    "code": "context_length_exceeded",
                    "message": "Maximum context length exceeded",
                },
            }
            return httpx.Response(200, text=json.dumps(line) + "\n")
        return httpx.Response(404, text="not found")

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="http://test/v1",
        transport=httpx.MockTransport(handler),
        sanitize_output=False,
    )
    results = asyncio.run(provider.fetch_batch_results("batch-err"))
    assert "req-1" in results
    assert results["req-1"]["content"] is None
    assert "Maximum context length exceeded" in results["req-1"]["error"]


_d2echo_CLEAN_SRC = "The channel voltage is set to the source voltage in this model."

_d2echo_CLEAN_TGT = "在本模型中，沟道电压被设置为源端电压，用于计算源端表面势。"

_d2echo_CTX_RENAMES = {
    "actual_job_id": "job_id",
    "create_event_fn": "create_event",
    "all_blocks_count": "block_count",
}

_d2echo_ECHO_SRC = "".join(
    (
        "where ψ pert is given by ψ 2 evaluated at x = T fin /2. ",
        "Eq. (3.11) is an implicit equation in β which must be solved using numerical methods, ",
        "then, once β is calculated, the surface potential and the charge in the channel ",
        "can be obtained. ",
        "Fig. 3.5 shows the surface potential obtained from Eq. (3.11) and the numerical ",
        "solution of Eq. (3.1) for different doping concentrations. ",
        "The amount of doping in the channel determines the threshold voltage of the device ",
        "as shown in Fig. 3.6, which represents the mobile charge density obtained from the ",
        "proposed compact model and the numerical solution of Eq. (3.1) for different ",
        "doping concentrations. ",
        "In the case of lightly doped DG FinFETs, the thickness of the channel determines ",
        "the amount of mobile carrier charge density in the channel in a linear manner, ",
        "as shown in Fig. 3.7.",
    )
)

_d2echo_ECHO_TGT = "".join(
    (
        "式 (3.7) 和 (2.2) 可以合并为一个方程：\n\n",
        "使用数值方法求解式 (3.11) 在紧凑建模应用中并不实际，因为其使用会增加计算时间并可能导致",
        "发散问题 [2]。因此，首先通过解析近似法获得初始猜测值，随后对式 (3.11) 进行求解。",
        "一旦计算出 β 值，即可获得表面势和沟道中的电荷。图 3.5 展示了由式 (3.11) 得到的表面势以及",
        "对式 (3.1) 进行数值求解在不同掺杂浓度下的结果。如图 3.6 所示，沟道中的掺杂量决定了器件的",
        "阈值电压，该图表示了由所提出的紧凑模型和对式 (3.1) 进行数值求解在不同掺杂浓度下得到的",
        "可移动电荷密度。在轻掺杂的 DG FinFET（双栅鳍式场效应晶体管）情况下，沟道厚度以线性方式",
        "决定了沟道中的可移动载流子电荷密度，如图 3.7 所示。",
    )
)


def _d2echo_block(bid: str, spine: int, source: str) -> IRBlock:
    return IRBlock(id=bid, flow_id=FlowID.MAIN_STORY, spine_index=spine, source_text=source)


def _d2echo_config() -> UBTConfig:
    return UBTConfig(batch_limit=30, max_concurrency=4, batch_enabled=False)


def _d2echo_doc(blocks: list[IRBlock]) -> SeedDoc:
    return SeedDoc(
        doc_id="d2_doc",
        source_path="/tmp/synthetic-duo.pdf",
        format_type="pdf",
        metadata={},
        blocks=blocks,
    )


def _d2echo_drain_draft(
    ledger: SQLiteJobLedger, job_id: str, tm: TranslationMemory, response: str
) -> None:
    router = ModelRouter(provider=MockModelProvider(default_response=response), draft_model="m")
    asyncio.run(
        _d2echo_drain_stage(
            ledger=ledger,
            actual_job_id=job_id,
            manifest=_d2echo_manifest(),
            profile_name="general",
            target_lang="zh",
            source_lang="en",
            router=router,
            code_masker=CodeMasker(),
            citation_masker=CitationMasker(),
            config=_d2echo_config(),
            all_blocks_count=1,
            glossary_dicts=[],
            abbreviation_entries=[],
            concurrency_sem=asyncio.Semaphore(4),
            create_event_fn=inert_event,
            tm=tm,
            fast_pass=FastPassFilter(),
        )
    )


async def _d2echo_drain_stage(**kwargs: Any) -> None:
    """Old-style keywords in, one StageContext out.

    The draft stage takes the run's context now; this keeps the single call site
    in this file written the way it read before, and any keyword that is not a
    context field fails in the constructor rather than being ignored.
    """
    ctx = build_stage_ctx(**{_d2echo_CTX_RENAMES.get(k, k): v for k, v in kwargs.items()})
    async for _event in run_draft_stage(ctx):
        pass


def _d2echo_manifest() -> BookManifest:
    return BookManifest(
        doc_id="d2_doc",
        title="D2",
        source_path="/tmp/synthetic-duo.pdf",
        source_lang="en",
        target_lang="zh",
        chapters=[ChapterMeta(chapter_id="pdf_main", title="chapter-3", spine_index=1)],
        metadata={},
    )


def _d2echo_seed_tm(tm: TranslationMemory, target: str) -> None:
    tm.writeback(
        [
            TMPendingEntry(
                "en",
                "zh",
                _d2echo_CLEAN_SRC,
                target,
                context_hash=compute_tm_context(PROMPT_VERSION, "general", "", "en", "zh"),
            )
        ]
    )


def test_tm_writeback_refuses_the_echo_and_keeps_the_clean_pair(tmp_path: Path) -> None:
    """Drives writeback_tm_from_ledger: terminal status is not a correctness proof."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(
        ledger,
        "job_wb",
        _d2echo_doc(
            [
                _d2echo_block("pdf_main#b_echo", 1, _d2echo_ECHO_SRC),
                _d2echo_block("pdf_main#b_ok", 2, _d2echo_CLEAN_SRC),
            ]
        ),
        target_lang="zh",
    )
    # The exact legacy state: the echo was marked MTQE_PASSED, so before this fix
    # the writeback promoted it into reusable memory.
    for bid, _src, tgt in (
        ("pdf_main#b_echo", _d2echo_ECHO_SRC, _d2echo_ECHO_TGT),
        ("pdf_main#b_ok", _d2echo_CLEAN_SRC, _d2echo_CLEAN_TGT),
    ):
        ledger.save_checkpoint(
            block_id=bid, status=BlockStatus.MTQE_PASSED, target_text=tgt, draft_text=tgt
        )

    tm = TranslationMemory(tmp_path / "tm.sqlite")
    written = writeback_tm_from_ledger(ledger, "job_wb", tm, "en", "zh")

    assert written == 1, "only the clean pair may be promoted"
    assert tm.lookup_exact("en", "zh", _d2echo_ECHO_SRC, domain=None) is None
    assert tm.lookup_exact("en", "zh", _d2echo_CLEAN_SRC, domain=None) is not None
    assert tm.entry_count() == 1
    tm.close()
    ledger.close()


def test_clean_tm_entry_is_still_served(tmp_path: Path) -> None:
    """Positive control: the TM read path really does run in this harness."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(
        ledger,
        "job_tm_ok",
        _d2echo_doc([_d2echo_block("pdf_main#b1", 1, _d2echo_CLEAN_SRC)]),
        target_lang="zh",
    )
    tm = TranslationMemory(tmp_path / "tm.sqlite")
    # Long enough to clear the length-ratio floor, distinct from the mock's
    # response so the assertion can tell which source supplied the text.
    _d2echo_seed_tm(tm, "这是翻译记忆库中已经存在的既有译文内容，用于验证读取路径确实生效。")

    _d2echo_drain_draft(ledger, "job_tm_ok", tm, response="【LLM】不应被调用")

    block = ledger.get_block("pdf_main#b1")
    assert block is not None
    assert block.target_text == "这是翻译记忆库中已经存在的既有译文内容，用于验证读取路径确实生效。"
    assert block.status is BlockStatus.MTQE_PASSED
    tm.close()
    ledger.close()


def test_poisoned_tm_entry_is_rejected_and_redrafted(tmp_path: Path) -> None:
    """The read-side trust boundary, proven by contrast with the test above.

    Same harness, same source, same clean LLM response — the only difference is
    that the stored TM target carries a citation the source never had. If the
    hit were still trusted, the block would hold the poisoned text.
    """
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(
        ledger,
        "job_tm_bad",
        _d2echo_doc([_d2echo_block("pdf_main#b1", 1, _d2echo_CLEAN_SRC)]),
        target_lang="zh",
    )
    tm = TranslationMemory(tmp_path / "tm.sqlite")
    poisoned = "在本模型中，沟道电压被设置为源端电压，用于计算源端表面势 [25]。"
    _d2echo_seed_tm(tm, poisoned)

    # The added-content gate is the one that rejects it: validate_structural_invariants
    # (which hosts it) runs before the numeric and length gates, so a reason from
    # the numeric validator could not appear first.
    rejection = FastPassFilter().evaluate(_d2echo_CLEAN_SRC, poisoned)
    assert not rejection.passed
    assert rejection.reason.startswith("Added reference(s)"), rejection.reason

    _d2echo_drain_draft(ledger, "job_tm_bad", tm, response=_d2echo_CLEAN_TGT)

    block = ledger.get_block("pdf_main#b1")
    assert block is not None
    assert "[25]" not in (block.target_text or ""), (
        "the poisoned TM entry was served — the read-side boundary is not wired"
    )
    assert block.target_text == _d2echo_CLEAN_TGT, "the hit should have fallen through to the LLM"

    # And the freshly drafted text is clean, so it may enter the TM on writeback.
    assert writeback_tm_from_ledger(ledger, "job_tm_bad", tm, "en", "zh") >= 0
    tm.close()
    ledger.close()


def test_preceding_context_carries_the_translation(tmp_path: Path) -> None:
    """The prose before a block is the prose the model is continuing.

    Both directions of the neighbor window read ``source_text``, so an English
    book being rendered into Chinese was handed English context for a paragraph
    it had already translated: register, term choice and sentence rhythm were
    re-decided per block instead of carried forward. The *following* excerpt must
    stay source-side -- that text has no translation yet.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BookManifest

    ledger = SQLiteJobLedger(tmp_path / "ctx.sqlite")
    job_id = "job_ctx"
    ledger.init_job_from_manifest(job_id, BookManifest(doc_id="d1", title="T", source_path="x"))
    blocks = [
        IRBlock(
            id="ch01#b1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            source_text="The elf closed the door quietly.",
            target_text="L'elfe ferma la porte en silence.",
        ),
        IRBlock(
            id="ch01#b2",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            source_text="Nobody heard it.",
        ),
    ]
    ledger.append_chapter(
        job_id, ChapterIR(doc_id="d1", chapter_id="ch01", title="c", spine_index=1, blocks=blocks)
    )

    preceding = ledger.get_preceding_text_tail(
        job_id=job_id, flow_id=FlowID.MAIN_STORY, before_spine_index=2
    )
    assert "L'elfe ferma la porte" in preceding
    assert "closed the door" not in preceding

    following = ledger.get_following_text_head(
        job_id=job_id, flow_id=FlowID.MAIN_STORY, after_spine_index=1
    )
    assert following == "Nobody heard it."

    # Untranslated neighbours still provide their source rather than nothing.
    untranslated = ledger.get_preceding_text_tail(
        job_id=job_id, flow_id=FlowID.MAIN_STORY, before_spine_index=3
    )
    assert "Nobody heard it." in untranslated and "L'elfe" in untranslated
    ledger.close()


def test_router_batch_flags_a_length_truncated_answer_as_an_error() -> None:
    """A ``finish_reason=length`` batch answer must not ship as a translation.

    Regression: the batch parser read ``message.content`` and ignored
    ``finish_reason``, so a max-token-truncated answer was accepted as a
    complete translation (the interactive path has a continuation loop; batch
    has none).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/files":
            return httpx.Response(200, json={"id": "file-1"})
        if path == "/v1/batches" and request.method == "POST":
            return httpx.Response(200, json={"id": "batch-1", "status": "validating"})
        if path == "/v1/batches/batch-1":
            return httpx.Response(
                200, json={"id": "batch-1", "status": "completed", "output_file_id": "out-1"}
            )
        if path == "/v1/files/out-1/content":
            line = {
                "custom_id": "b-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {"content": "半句被打断的译文"},
                            }
                        ]
                    },
                },
            }
            return httpx.Response(200, text=json.dumps(line) + "\n")
        return httpx.Response(404, text="not found")

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="http://test/v1",
        transport=httpx.MockTransport(handler),
        sanitize_output=False,
    )
    router = ModelRouter(provider=provider, draft_model="batch-test-model")
    results = asyncio.run(
        router.draft_batch(
            [BatchDraftRequest(custom_id="b-1", source_text="Hello world. " * 20)],
            poll_interval=0.01,
            cleanup_files=False,
        )
    )
    assert results and results[0].error
    assert "finish_reason=length" in results[0].error

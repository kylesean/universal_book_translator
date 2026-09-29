"""Unit tests for structured macro-chunking (pack-by-macro-block)."""

import asyncio
from pathlib import Path

import pytest

from tests.stage_ctx_factory import build_stage_ctx
from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.draft import run_draft_stage
from ubt.core.exceptions import ModelProviderError
from ubt.core.ir.models import BlockStatus, BookManifest, ChapterMeta, FlowID, IRBlock
from ubt.core.router.extractor import TranslationOutputExtractor
from ubt.core.router.prompts import build_macro_chunk_draft_prompt
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


def test_build_macro_chunk_draft_prompt_structure() -> None:
    blocks = [("b001", "Hello world."), ("b002", "This is chapter one.")]
    sys_prompt, user_prompt = build_macro_chunk_draft_prompt(
        blocks=blocks,
        glossary_table="| term | 术语 |",
        target_lang="zh",
        source_lang="en",
    )
    assert "<blocks>" in user_prompt
    assert '<block id="b001">Hello world.</block>' in user_prompt
    assert '<block id="b002">This is chapter one.</block>' in user_prompt
    assert "Structured Macro-Blocks Output" in sys_prompt


def test_build_macro_chunk_draft_prompt_escapes_source_markup() -> None:
    """Source text carrying envelope-closing markup must not break the prompt.

    Books about XML/HTML (or adversarial documents) can contain </block> or
    </blocks>; interpolated raw it closes the envelope early and every later
    block is silently hidden from the model.
    """
    blocks = [
        ("b001", "Use the </block> tag to close a block."),
        ("b002", "Then write </blocks> to finish."),
    ]
    _, user_prompt = build_macro_chunk_draft_prompt(
        blocks=blocks,
        target_lang="zh",
        source_lang="en",
    )
    # The injected markup is escaped, so it adds no structural tags.
    assert "&lt;/block&gt;" in user_prompt
    assert "&lt;/blocks&gt;" in user_prompt
    assert "Use the </block> tag" not in user_prompt
    assert "write </blocks> to finish" not in user_prompt
    # The envelope still holds exactly the two real blocks and closes once.
    envelope = user_prompt.split("<blocks>\n", 1)[1].split("</blocks>", 1)[0]
    assert envelope.count("<block ") == 2
    assert envelope.count("</block>") == 2


def test_macro_prompt_escaping_round_trips_through_extraction() -> None:
    """The extractor must undo the escaping the prompt applies.

    The model is shown ``AT&amp;T`` and mirrors it back; storing the entity
    form renders ``AT&amp;T`` literally in PDF/Typst output and can be
    double-escaped by the HTML/EPUB sanitizer.
    """
    blocks = [("b001", "AT&T and a < b & c")]
    _, user_prompt = build_macro_chunk_draft_prompt(
        blocks=blocks, target_lang="zh", source_lang="en"
    )
    assert "AT&amp;T" in user_prompt
    assert "AT&T and a < b" not in user_prompt

    # A model that echoes the escaped envelope (the usual behaviour for entities).
    model_reply = '<block id="b001">AT&amp;T 与 a &lt; b &amp; c</block>'
    extracted = TranslationOutputExtractor.extract_macro_blocks(model_reply)
    assert extracted["b001"] == "AT&T 与 a < b & c"


def test_extract_macro_blocks() -> None:
    raw_output = """
    <think>
    Thinking about translating these blocks...
    </think>
    ```xml
    <blocks>
      <block id="ch1_b1">第一段中文译文。</block>
      <block id="ch1_b2">第二段中文译文。</block>
    </blocks>
    ```
    """
    extracted = TranslationOutputExtractor.extract_macro_blocks(raw_output)
    assert len(extracted) == 2
    assert extracted["ch1_b1"] == "第一段中文译文。"
    assert extracted["ch1_b2"] == "第二段中文译文。"


def _make_doc(count: int) -> SeedDoc:
    blocks = [
        IRBlock(
            id=f"ch01#b{i:03d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            source_text=f"Paragraph {i} content to translate.",
        )
        for i in range(1, count + 1)
    ]
    return SeedDoc(
        doc_id="macro_doc",
        source_path="/tmp/macro.epub",
        format_type="epub",
        metadata={},
        blocks=blocks,
    )


def _make_manifest() -> BookManifest:
    return BookManifest(
        doc_id="macro_doc",
        title="Macro Test",
        source_path="/tmp/macro.epub",
        source_lang="en",
        target_lang="zh",
        chapters=[ChapterMeta(chapter_id="ch01", title="Chapter 1", spine_index=1)],
        metadata={},
    )


@pytest.mark.asyncio
async def test_draft_stage_with_macro_chunking(tmp_path: Path) -> None:
    """With macro_chunk_size=5, 10 blocks are translated in exactly 2 LLM requests."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_macro", _make_doc(10), target_lang="zh")

    provider = MockModelProvider(default_response="宏块译文")
    router = ModelRouter(provider=provider, draft_model="test-model")

    config = UBTConfig(
        macro_chunk_size=5,
        max_concurrency=4,
        batch_limit=30,
        enable_rolling_summary=False,
    )
    ctx = build_stage_ctx(
        tmp_path,
        job_id="job_macro",
        ledger=ledger,
        router=router,
        manifest=_make_manifest(),
        config=config,
        block_count=10,
    )

    async for _ in run_draft_stage(ctx):
        pass

    # 10 blocks packed into groups of 5 = 2 requests
    assert len(provider.call_history) == 2

    blocks = ledger.get_all_blocks("job_macro")
    assert len(blocks) == 10
    assert all(b.status == BlockStatus.DRAFTED for b in blocks)
    assert all(b.target_text == "宏块译文" for b in blocks)


@pytest.mark.asyncio
async def test_macro_chunk_partial_omission_falls_back(tmp_path: Path) -> None:
    """If the LLM omits one block in the macro response, that block falls back to single draft."""
    ledger = SQLiteJobLedger(tmp_path / "job_fallback.sqlite")
    seed_job(ledger, "job_fb", _make_doc(3), target_lang="zh")

    # Custom provider that omits b002 in the macro response
    class OmissionProvider(MockModelProvider):
        async def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> str:
            self.call_history.append({"prompt": prompt})
            if "<blocks>" in prompt:
                # Omit b002 from response
                return (
                    "<blocks>\n"
                    '<block id="ch01#b001">第一段正常。</block>\n'
                    '<block id="ch01#b003">第三段正常。</block>\n'
                    "</blocks>"
                )
            # Single block fallback for b002
            return "<translation>第二段单块降级成功。</translation>"

    provider = OmissionProvider()
    router = ModelRouter(provider=provider, draft_model="test-model")

    config = UBTConfig(macro_chunk_size=5, max_concurrency=2, enable_rolling_summary=False)
    ctx = build_stage_ctx(
        tmp_path,
        job_id="job_fb",
        ledger=ledger,
        router=router,
        manifest=_make_manifest(),
        config=config,
        block_count=3,
    )

    async for _ in run_draft_stage(ctx):
        pass

    # Call history: 1 macro-chunk call + 1 fallback call for b002 = 2 calls
    assert len(provider.call_history) == 2

    blocks = {b.id: b for b in ledger.get_all_blocks("job_fb")}
    assert blocks["ch01#b001"].target_text == "第一段正常。"
    assert blocks["ch01#b002"].target_text == "第二段单块降级成功。"
    assert blocks["ch01#b003"].target_text == "第三段正常。"
    assert all(b.status == BlockStatus.DRAFTED for b in blocks.values())


@pytest.mark.asyncio
async def test_macro_chunk_total_failure_falls_back_without_deadlock(
    tmp_path: Path,
) -> None:
    """A failing macro call must redraft blocks individually *outside* the
    concurrency semaphore. The fallback used to run while the group still
    held its permit, and ``draft_single_block`` re-acquires the same
    non-reentrant semaphore — with ``max_concurrency=1`` the stage hung
    forever after logging the fallback (regression guard)."""
    ledger = SQLiteJobLedger(tmp_path / "job_hang.sqlite")
    seed_job(ledger, "job_hang", _make_doc(3), target_lang="zh")

    class ExplodingMacroProvider(MockModelProvider):
        async def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> str:
            self.call_history.append({"prompt": prompt})
            if "<blocks>" in prompt:
                raise ModelProviderError("macro extraction exploded", details={"fail_fast": True})
            return "<translation>单块救回。</translation>"

    provider = ExplodingMacroProvider()
    router = ModelRouter(provider=provider, draft_model="test-model")

    config = UBTConfig(macro_chunk_size=3, max_concurrency=1, enable_rolling_summary=False)
    ctx = build_stage_ctx(
        tmp_path,
        job_id="job_hang",
        ledger=ledger,
        router=router,
        manifest=_make_manifest(),
        config=config,
        block_count=3,
    )

    async def _drive() -> None:
        async for _ in run_draft_stage(ctx):
            pass

    # wait_for turns the historical hang into a hard test failure.
    await asyncio.wait_for(_drive(), timeout=30)

    blocks = {b.id: b for b in ledger.get_all_blocks("job_hang")}
    assert all(b.status == BlockStatus.DRAFTED for b in blocks.values())
    assert all(b.target_text == "单块救回。" for b in blocks.values())

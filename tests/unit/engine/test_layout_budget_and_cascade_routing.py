from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ubt.adapters.pdf.native_adapter import _unit_to_ir_block
from ubt.codecs.protocol import TranslationUnit, UnitRole
from ubt.core.config import UBTConfig
from ubt.core.engine.stages.draft import (
    _build_draft_runtime,
    _DraftInputs,
    _DraftProcessor,
)
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    IRBlock,
    StyleMeta,
    make_element,
)
from ubt.core.policy.layout_budget import estimate_tight_char_budget
from ubt.core.router.dynamic_scheduler import (
    BlockComplexityClassifier,
    DynamicParetoScheduler,
    DynamicRoutingPolicy,
)
from ubt.core.router.prompts import (
    build_hybrid_draft_prompt,
    build_macro_chunk_draft_prompt,
    build_minimal_draft_prompt,
    build_rich_draft_prompt,
    draft_source_from_prompt,
)

pytestmark = pytest.mark.fast


def test_native_adapter_populates_box_dimensions_and_estimates_tight_budget() -> None:
    tight_unit = TranslationUnit(
        unit_id="u1",
        chapter_id="page_0001",
        role=UnitRole.PARAGRAPH,
        source_marked="This long caption describes a dense multi-column experimental architecture in detail.",
        source_plain="This long caption describes a dense multi-column experimental architecture in detail.",
        page_no=1,
        box_hint={
            "width_pt": 110.0,
            "height_pt": 28.0,
            "font_size_pt": 10.0,
            "line_height_pt": 12.0,
        },
        codec_meta={"bbox": [10.0, 10.0, 120.0, 38.0], "alignment": "left"},
    )
    tight_block = _unit_to_ir_block(tight_unit, 1)
    assert tight_block.style is not None
    assert tight_block.style.box_width_pt == 110.0
    assert tight_block.style.box_height_pt == 28.0
    assert tight_block.style.line_height_pt == 12.0

    tight_budget = estimate_tight_char_budget(tight_block.style, tight_block.source_text, "zh")
    assert tight_budget is not None and tight_budget > 0

    roomy_style = StyleMeta(
        font_size=10.0, box_width_pt=450.0, box_height_pt=400.0, line_height_pt=12.0
    )
    assert estimate_tight_char_budget(roomy_style, "Short paragraph sentence.", "zh") is None
    assert estimate_tight_char_budget(None, "Unconstrained reflow block.", "zh") is None


def test_draft_prompts_inject_length_budget_and_preserve_source_extraction() -> None:
    src = "Experimental setup shows substantial throughput improvement."
    for builder in (
        build_minimal_draft_prompt,
        build_hybrid_draft_prompt,
        build_rich_draft_prompt,
    ):
        sys_plain, usr_plain = builder(source_text=src)
        sys_bgt, usr_bgt = builder(source_text=src, char_budget=24)
        assert sys_plain == sys_bgt
        assert "24 characters" not in usr_plain
        assert "<= 24 characters" in usr_bgt
        assert draft_source_from_prompt(usr_bgt) == src

    _sys_m, usr_m = build_macro_chunk_draft_prompt(
        blocks=[("b1", src), ("b2", "Another block")],
        block_char_budgets=[("b1", 28)],
    )
    assert "Layout Length Budgets" in usr_m
    assert "`b1`: <= 28 characters" in usr_m


def test_dynamic_classifier_scores_inline_formula_anchors_and_char_budget() -> None:
    clf = BlockComplexityClassifier()
    plain = IRBlock(
        element=make_element(
            id="b-plain",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Standard paragraph with normal scientific terminology and moderate length.",
        )
    )
    tagged = IRBlock(
        element=make_element(
            id="b-tag",
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text="Given <f1/> and <f2/> plus ⟦MATH_MASK_0001⟧, solve for equilibrium.",
        ),
        style=StyleMeta(font_size=10.0, char_budget=32),
    )
    plain_m = clf.classify_block(plain)
    tagged_m = clf.classify_block(tagged)
    assert tagged_m.math_inline_count >= 3
    assert "Fixed-box length constraint" in tagged_m.reasons
    assert tagged_m.score > plain_m.score


@pytest.mark.asyncio
async def test_draft_processor_passes_char_budget_and_cascades_on_tag_mismatch() -> None:
    seen_calls: list[tuple[str | None, int | None]] = []

    class _FakeEngine:
        cache = None

        async def draft(self, masked_source: str, generate: Any, *, context: str = "") -> str:
            del context
            return str(await generate(masked_source))

        def resolve(self, raw: str, _masked: Any) -> SimpleNamespace:
            return SimpleNamespace(text=raw, dirty=[], clean=True)

    class _FakeRouter:
        draft_model = "budget-m"

        async def draft(
            self,
            *,
            block: IRBlock,
            model: str | None = None,
            char_budget: int | None = None,
            **_kwargs: Any,
        ) -> str:
            del block
            seen_calls.append((model, char_budget))
            if model == "budget-m":
                return "缺失标签的草稿"
            return "带完整公式 <f1/> 的精简译文"

    class _FakeMemory:
        def record_drafted_block(self, _block: IRBlock) -> None:
            return

    class _FakeFlusher:
        def __init__(self) -> None:
            self.updates: list[dict[str, Any]] = []

        async def enqueue(self, update: dict[str, Any]) -> None:
            self.updates.append(update)

    scheduler = DynamicParetoScheduler(
        policy=DynamicRoutingPolicy.PARETO,
        budget_model="budget-m",
        standard_model="flagship-m",
        flagship_model="flagship-m",
    )
    policy = SimpleNamespace(
        glossary_dicts=[],
        abbreviation_entries=[],
        global_glossary_table="",
        source_lang="en",
        target_lang="zh",
        profile_name="academic",
        domain=None,
        tm_fuzzy_threshold=0.85,
        tm_context="ctx",
        draft_max_retries=0,
        draft_retry_base_delay=0.0,
        rolling_enabled=False,
        dynamic_routing_enabled=True,
        dynamic_scheduler=scheduler,
    )
    flusher = _FakeFlusher()
    runtime = SimpleNamespace(
        fail_fast_consecutive=0,
        engine=_FakeEngine(),
        router=_FakeRouter(),
        memory_mgr=_FakeMemory(),
        flusher=flusher,
        counters={},
        concurrency_sem=asyncio.Semaphore(2),
    )
    proc = _DraftProcessor(
        runtime=runtime,  # type: ignore[arg-type]
        policy=policy,  # type: ignore[arg-type]
    )
    blk = IRBlock(
        element=make_element(
            id="b-cascade",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Consider <f1/> in steady state operation.",
        )
    )
    inp = _DraftInputs(
        masked_source="Consider <f1/> in steady state operation.",
        code_map=None,
        cite_map=None,
        math_map=None,
        glossary_table="",
        neighbor_ctx="",
        macro_ctx="",
        few_shot_reference="",
        block_type="narrative",
        char_budget=26,
    )
    await proc.draft_single_block(blk, [blk], inp)
    assert seen_calls == [("budget-m", 26), ("flagship-m", 26)]
    assert len(flusher.updates) == 1
    assert flusher.updates[0]["status"] == BlockStatus.DRAFTED
    assert "<f1/>" in flusher.updates[0]["target_text"]


@pytest.mark.asyncio
async def test_build_draft_runtime_dynamic_routing_defaults_flagship_to_repair_model(
    tmp_path: Path,
) -> None:
    cfg = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        enable_dynamic_routing=True,
        draft_model="small-draft",
        repair_model="strong-repair",
    )
    cfg.db_dir.mkdir(parents=True, exist_ok=True)
    router = SimpleNamespace(draft_model="small-draft", repair_model="strong-repair")
    ledger = SimpleNamespace(
        get_all_blocks=lambda _job: [],
        get_job_stats=lambda _job: {"total": 0},
    )
    ctx = SimpleNamespace(
        ledger=ledger,
        job_id="job-dyn",
        profile_name="general",
        target_lang="zh",
        source_lang="en",
        router=router,
        config=cfg,
        manifest=BookManifest(doc_id="job-dyn", title="t", source_path="in.md", chapters=[]),
        create_event=lambda *_a, **_k: None,
        is_mock_run=False,
    )
    services = SimpleNamespace(
        concurrency_sem=asyncio.Semaphore(2),
        tm=None,
        fast_pass=None,
        source_document=None,
    )
    assembly = await _build_draft_runtime(
        ctx,  # type: ignore[arg-type]
        services,  # type: ignore[arg-type]
        SimpleNamespace(glossary_dicts=[], abbreviation_entries=[]),  # type: ignore[arg-type]
    )
    scheduler = assembly.processor.policy.dynamic_scheduler
    assert scheduler is not None
    assert scheduler.flagship_model == "strong-repair"

"""Golden tests for the bilingual-mode advisory (policy/bilingual_advisor).

Pure-function tests with synthetic block lists — no PDF fixtures needed.
Calibration anchors: a formula/figure-dense technical handbook must
discourage inline interleave; clean prose must pass it.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.stage_ctx_factory import build_stage_ctx
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.advisory import run_difficulty_advisory_stage
from ubt.core.ir.models import BlockStatus, BlockType, BookManifest, BoundingBox, FlowID, IRBlock
from ubt.core.policy.bilingual_advisor import (
    Advisory,
    ModeScore,
    advise_layout,
    assess_difficulty,
    downgrade_mode,
    resolve_effective_mode,
    secondary_mode,
)


def _block(
    idx: int,
    block_type: BlockType,
    text: str,
    *,
    page: int = 1,
    skip: bool = False,
) -> IRBlock:
    return IRBlock(
        id=f"t#b{idx:04d}",
        spine_index=idx,
        block_type=block_type,
        flow_id=FlowID.MAIN_STORY,
        source_text=text,
        skip_translate=skip,
        status=BlockStatus.MTQE_PASSED,
        bbox=None if page <= 0 else BoundingBox(page=page, x0=0.0, y0=0.0, x1=100.0, y1=50.0),
    )


def _handbook_blocks() -> list[IRBlock]:
    """Synthetic KV-handbook shape: prose + formulas + tables + skips."""
    blocks: list[IRBlock] = []
    idx = 0
    for page in range(1, 11):
        for _ in range(8):
            idx += 1
            blocks.append(
                _block(
                    idx,
                    BlockType.NARRATIVE,
                    "Consider one self-attention layer with learned projections. " * 4,
                    page=page,
                )
            )
        idx += 1
        blocks.append(_block(idx, BlockType.FORMULA, "K V softmax", page=page, skip=True))
    idx += 1
    blocks.append(_block(idx, BlockType.TABLE, "| A | B |", page=3))
    idx += 1
    blocks.append(_block(idx, BlockType.NARRATIVE, "@techNmak", page=1, skip=True))
    return blocks


def _novel_blocks() -> list[IRBlock]:
    blocks: list[IRBlock] = []
    for idx in range(1, 101):
        blocks.append(
            _block(
                idx,
                BlockType.NARRATIVE,
                "It was a bright cold day in April, and the clocks were striking thirteen. " * 3,
                page=(idx // 10) + 1,
            )
        )
    return blocks


def test_handbook_discourages_inline() -> None:
    advisory = advise_layout(_handbook_blocks(), "inline", profile="paper", figure_pages={2, 5, 8})
    assert advisory.tier == "discourage"
    assert advisory.recommended in ("alternating", "monolingual")
    assert advisory.ranking[0].score >= advisory.ranking[-1].score
    assert any("interruption" in r or "figure" in r for r in advisory.reasons)
    assert advisory.to_dict()["requested"] == "inline"


def test_novel_passes_inline() -> None:
    advisory = advise_layout(_novel_blocks(), "inline", profile="general")
    assert advisory.tier == "ok"
    assert advisory.recommended == "inline"
    assert advisory.reasons == ()


def test_auto_enforcement_switches_on_discourage() -> None:
    advisory = advise_layout(_handbook_blocks(), "inline", profile="paper", figure_pages={2, 5, 8})
    assert advisory.tier == "discourage"
    easy = assess_difficulty(total=100, repaired=2, failed=0)
    assert not easy.hard
    assert (
        resolve_effective_mode("inline", advisory, easy, enforcement="auto") == advisory.recommended
    )


def test_advise_enforcement_never_switches() -> None:
    advisory = advise_layout(_handbook_blocks(), "inline", profile="paper", figure_pages={2, 5, 8})
    easy = assess_difficulty(total=100, repaired=2, failed=0)
    assert resolve_effective_mode("inline", advisory, easy, enforcement="advise") == "inline"


def test_difficulty_downgrades_one_step_only() -> None:
    advisory = advise_layout(_novel_blocks(), "inline", profile="general")
    assert advisory.tier == "ok"
    hard = assess_difficulty(total=100, repaired=12, failed=5)
    assert hard.hard
    assert resolve_effective_mode("inline", advisory, hard, enforcement="auto") == "alternating"
    assert downgrade_mode("alternating") == "monolingual"
    assert downgrade_mode("monolingual") == "monolingual"


def test_difficulty_threshold_boundary() -> None:
    assert not assess_difficulty(total=100, repaired=14, failed=0).hard
    assert assess_difficulty(total=100, repaired=15, failed=0).hard
    assert assess_difficulty(total=0, repaired=0, failed=0).hard is False


def test_secondary_mode_pairs() -> None:
    assert secondary_mode("inline") == "monolingual"
    assert secondary_mode("alternating") == "monolingual"
    assert secondary_mode("monolingual") == "alternating"


def test_advisory_serializes_for_report() -> None:
    advisory: Advisory = advise_layout(_novel_blocks(), "inline", profile="general")
    payload = advisory.to_dict()
    assert set(payload) == {
        "requested",
        "tier",
        "recommended",
        "ranking",
        "reasons",
        "signals",
    }
    assert payload["ranking"][0]["mode"] == "inline"


@pytest.mark.fast
async def test_advisory_rigid_engine_monolingual_in_difficulty_stage(tmp_path: Path) -> None:
    """run_difficulty_advisory_stage preserves 'monolingual' for rigid engines under auto mode."""
    config = UBTConfig(render_engine="rigid", dual_mode="auto")
    manifest = BookManifest(
        doc_id="doc1", title="Title", source_path=str(tmp_path / "input.pdf"), chapters=[]
    )
    ledger = SQLiteJobLedger(tmp_path / "test.sqlite")
    ledger.init_job_from_manifest("job_test", manifest)

    ctx = build_stage_ctx(
        tmp_path,
        job_id="job_test",
        input_path=tmp_path / "input.pdf",
        config=config,
        manifest=manifest,
        ledger=ledger,
        short_chain=False,
    )
    ctx.tier_basis = "inline"
    ctx.enforcement = "auto"
    ctx.advisory = Advisory(
        requested="inline",
        tier="ok",
        ranking=(ModeScore(mode="inline", score=1.0),),
        reasons=(),
    )
    manifest.run.effective_dual_mode = "monolingual"

    await run_difficulty_advisory_stage(ctx)

    # For rigid engine, effective_mode must remain monolingual, never inline
    assert manifest.run.effective_dual_mode == "monolingual"


@pytest.mark.fast
@pytest.mark.asyncio
async def test_forced_reflow_on_formula_dense_pdf_enables_companion_rigid_delivery(
    tmp_path: Path,
) -> None:
    """When a user forces --render-engine reflow --dual-mode inline on a formula-dense PDF
    where auto dispatch would select 'rigid' (and advisory tier is 'discourage'),
    run_mode_advisory_stage must schedule emit_secondary_engine='rigid' so export
    delivers a companion *_rigid.pdf alongside the requested reflow PDF."""
    from ubt.core.config import UBTConfig
    from ubt.core.engine.stages.advisory import run_mode_advisory_stage
    from ubt.core.ir.models import BookManifest

    blocks = [
        IRBlock(
            id=f"b{i}",
            spine_index=i,
            block_type=BlockType.FORMULA if i % 3 == 0 else BlockType.NARRATIVE,
            source_text=r"x^2 + y^2 = z^2" if i % 3 == 0 else "Short fragment",
            target_text=r"x^2 + y^2 = z^2" if i % 3 == 0 else "短片段",
            skip_translate=(i % 3 == 0),
            bbox=BoundingBox(page=1, x0=50.0, y0=100.0 + i * 20, x1=400.0, y1=115.0 + i * 20),
        )
        for i in range(1, 16)
    ]
    manifest = BookManifest(doc_id="doc1", title="Test", source_path=str(tmp_path / "paper.pdf"))
    config = UBTConfig(render_engine="reflow", dual_mode="inline")

    ctx = MagicMock()
    ctx.config = config
    ctx.manifest = manifest
    ctx.source_pdf_path = tmp_path / "paper.pdf"
    ctx.input_path = tmp_path / "paper.pdf"
    ctx.profile_name = "paper"
    ctx.job_id = "job_test"

    async def _current_blocks(force_refresh: bool = False) -> list[IRBlock]:
        return blocks

    async def _create_event(*args: object, **kwargs: object) -> object:
        return MagicMock()

    ctx.current_blocks = _current_blocks
    ctx.create_event = _create_event

    events = [ev async for ev in run_mode_advisory_stage(ctx)]
    assert len(events) == 1
    assert manifest.run.emit_secondary_engine == "rigid"

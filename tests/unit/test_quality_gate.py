"""Quality-gate stage: pass-sample wiring for FastPass passes."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.stage_ctx_factory import build_stage_ctx, drain, inert_event
from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.quality_gate import run_quality_gate_stage
from ubt.core.ir.models import BlockStatus, FlowID, IRBlock
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.comet_runner import HeuristicQERunner
from ubt.core.qe.fast_pass import FastPassFilter


class _LowPassRunner(BaseQERunner):
    """Advertises pass sampling and scores every pair below threshold."""

    pass_sample = 1.0

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        return [0.1] * len(pairs)


def _seed_doc() -> SeedDoc:
    return SeedDoc(
        doc_id="pass_sample",
        source_path="/tmp/pass_sample.epub",
        format_type="epub",
        metadata={},
        blocks=[
            IRBlock(
                id="ch01#b001",
                flow_id=FlowID.MAIN_STORY,
                spine_index=1,
                source_text="The capital of France is Paris.",
            ),
        ],
    )


def _seed(tmp_path: Path, job_id: str, filename: str) -> SQLiteJobLedger:
    ledger = SQLiteJobLedger(tmp_path / filename)
    seed_job(ledger, job_id, _seed_doc(), target_lang="zh")
    ledger.save_checkpoint(
        block_id="ch01#b001",
        target_text="法国的首都是巴黎。",
        draft_text="法国的首都是巴黎。",
        status=BlockStatus.DRAFTED,
    )
    return ledger


def _run(tmp_path: Path, job_id: str, filename: str, runner: BaseQERunner) -> BlockStatus:
    ledger = _seed(tmp_path, job_id, filename)
    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id=job_id,
        fast_pass=FastPassFilter(source_lang="en", target_lang="zh"),
        qe_runner=runner,
        create_event=inert_event,
    )
    asyncio.run(drain(run_quality_gate_stage(ctx)))
    blocks = {b.id: b for b in ledger.get_all_blocks(job_id)}
    return blocks["ch01#b001"].status


def test_sampled_clean_pass_below_threshold_is_routed_to_repair(tmp_path: Path) -> None:
    """A FastPass pass the pass-sample judge lowers must not ship as MTQE_PASSED.

    Regression: FastPass-passing blocks were written straight to
    MTQE_PASSED, so the ``pass_sample`` mechanism meant to audit clean passes
    (``TieredQERunner``) never saw them — the knob was inert.
    """
    assert _run(tmp_path, "job_ps", "ps.sqlite", _LowPassRunner()) is BlockStatus.REPAIR_PENDING


def test_without_pass_sampling_a_clean_pass_still_releases(tmp_path: Path) -> None:
    """A runner without ``pass_sample`` (the default heuristic) keeps the fast path."""
    assert _run(tmp_path, "job_ns", "ns.sqlite", HeuristicQERunner()) is BlockStatus.MTQE_PASSED


def test_glossary_hits_are_persisted(tmp_path: Path) -> None:
    """``glossary_hits`` records the enforced terms present in the target."""
    ledger = _seed(tmp_path, "job_gh", "gh.sqlite")
    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id="job_gh",
        fast_pass=FastPassFilter(source_lang="en", target_lang="zh"),
        qe_runner=HeuristicQERunner(),
        create_event=inert_event,
        glossary_dicts=[{"source": "Paris", "translation": "巴黎", "aliases": []}],
    )
    asyncio.run(drain(run_quality_gate_stage(ctx)))
    block = {b.id: b for b in ledger.get_all_blocks("job_gh")}["ch01#b001"]
    assert block.glossary_hits == ["巴黎"]

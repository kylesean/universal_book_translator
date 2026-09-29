"""Unit tests for MQM severity triage: classification, escalation, quarantine."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.stage_ctx_factory import build_stage_ctx, inert_event
from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.repair import run_repair_stage
from ubt.core.engine.stages.triage import run_triage_stage
from ubt.core.ir.models import BlockStatus, FlowID, IRBlock
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter
from ubt.core.validators.span_repair import (
    MQMSpanAnnotator,
    max_severity,
    severity_for_error_type,
)


class ControlledScoreQERunner(BaseQERunner):
    """Controlled score provider for repair verification."""

    def __init__(self, next_scores: list[float]) -> None:
        self.scores = list(next_scores)

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        result: list[float] = []
        for _ in pairs:
            result.append(self.scores.pop(0) if self.scores else 0.5)
        return result


def _doc_ir(blocks: list[IRBlock]) -> SeedDoc:
    return SeedDoc(
        doc_id="test_doc_sha256",
        source_path="/tmp/test_book.epub",
        format_type="epub",
        metadata={"title": "Test Book"},
        blocks=blocks,
    )


def _block(
    block_id: str,
    source_text: str,
    target_text: str,
    *,
    status: BlockStatus = BlockStatus.REPAIR_PENDING,
    score: float = 0.42,
    repair_rounds: int = 2,
    flags: list[str] | None = None,
    spine_index: int = 1,
) -> IRBlock:
    return IRBlock(
        id=block_id,
        flow_id=FlowID.MAIN_STORY,
        spine_index=spine_index,
        source_text=source_text,
        draft_text=target_text,
        target_text=target_text,
        status=status,
        mtqe_score=score,
        repair_rounds=repair_rounds,
        error_flags=flags or [],
    )


def _init_ledger(tmp_path: Path, blocks: list[IRBlock]) -> SQLiteJobLedger:
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger, "job_triage", _doc_ir(blocks), target_lang="zh")
    return ledger


# ---------------------------------------------------------------------------
# Severity classification helpers
# ---------------------------------------------------------------------------


def test_severity_mapping_numeric_is_critical() -> None:
    assert severity_for_error_type("numeric") == "critical"
    assert severity_for_error_type("terminology") == "major"
    assert severity_for_error_type("terminology_leak") == "major"
    assert severity_for_error_type("something_new") == "minor"
    assert severity_for_error_type("numeric_distortion") == "critical"


def test_max_severity_prefers_highest_tier() -> None:
    assert max_severity(["minor", "major"]) == "major"
    assert max_severity(["major", "critical", "minor"]) == "critical"
    assert max_severity([]) == "minor"


def test_annotator_spans_carry_severity() -> None:
    from ubt.core.validators.span_repair import MQMSpanAnnotator

    annotator = MQMSpanAnnotator()
    _, spans = annotator.annotate_draft(
        source_text="The reactor outputs 42 megawatts daily.",
        draft_text="The reactor outputs 84 megawatts daily.",
        error_flags=["numeric_mismatch"],
    )
    # The altered number (84) yields one Critical span whose expected value
    # (42) already explains the source number — the missing-number
    # loop de-duplicates against explained expectations to avoid
    # double-reporting the same fact.
    assert len(spans) == 1
    assert spans[0].error_type == "numeric"
    assert spans[0].severity == "critical"
    assert spans[0].erroneous_text == "84"
    assert spans[0].expected == "42"


def test_annotator_missing_source_number_produces_span() -> None:
    """A number present in the source but deleted from the draft
    must still yield a numeric (Critical) span, not be silently auto-passed."""
    from ubt.core.validators.span_repair import MQMSpanAnnotator

    annotator = MQMSpanAnnotator()
    _, spans = annotator.annotate_draft(
        source_text="共 37 章",
        draft_text="共章",
        error_flags=["Numeric fidelity failure: Missing numeric tokens in translation: 37"],
    )
    numeric_spans = [s for s in spans if s.error_type == "numeric"]
    assert numeric_spans, spans
    assert numeric_spans[0].severity == "critical"
    assert "37" in numeric_spans[0].erroneous_text


def test_annotator_alias_requires_word_boundary() -> None:
    """'cat' must not match inside 'category' (exporter bar parity)."""
    from ubt.core.validators.span_repair import MQMSpanAnnotator

    annotator = MQMSpanAnnotator()
    glossary = [{"source": "cat", "translation": "猫", "aliases": ["cat"]}]
    _, spans = annotator.annotate_draft(
        source_text="src",
        draft_text="category theory is fun",
        error_flags=[],
        glossary_entries=glossary,
    )
    assert [s for s in spans if s.error_type == "terminology"] == []
    # Standalone alias still annotated.
    _, spans = annotator.annotate_draft(
        source_text="src",
        draft_text="the cat sat",
        error_flags=[],
        glossary_entries=glossary,
    )
    assert [(s.start_pos, s.end_pos) for s in spans if s.error_type == "terminology"] == [(4, 7)]


def test_annotator_cjk_expansion_skips_compound() -> None:
    """CJK compound guard parity — '网络' inside '计算机网络' is not a span."""
    from ubt.core.validators.span_repair import MQMSpanAnnotator

    annotator = MQMSpanAnnotator()
    glossary = [{"source": "network", "translation": "神经网络", "aliases": ["网络"]}]
    _, spans = annotator.annotate_draft(
        source_text="src",
        draft_text="计算机网络很好",
        error_flags=[],
        glossary_entries=glossary,
    )
    assert [s for s in spans if s.error_type == "terminology"] == []


# ---------------------------------------------------------------------------
# Triage stage routing (ledger-backed)
# ---------------------------------------------------------------------------


def _triage_ctx(
    tmp_path: Path,
    ledger: SQLiteJobLedger,
    repair_loop: RepairLoop,
    **overrides: Any,
) -> StageContext:
    """The run state every triage case shares, as the stage now wants it."""
    shared: dict[str, Any] = {
        "ledger": ledger,
        "job_id": "job_triage",
        "repair_loop": repair_loop,
        "glossary_dicts": [],
        "abbreviation_entries": [],
        "target_lang": "zh",
        "source_lang": "en",
        "fast_pass": None,
        "concurrency_sem": asyncio.Semaphore(4),
        "create_event": inert_event,
    }
    shared.update(overrides)
    return build_stage_ctx(tmp_path, **shared)


@pytest.mark.asyncio
async def test_triage_critical_unresolved_becomes_blocked_human(tmp_path: Path) -> None:
    """Critical (numeric) block that fails escalation is quarantined, source-only."""
    source = "The reactor outputs 42 megawatts daily."
    bad_draft = "The reactor outputs 84 megawatts daily."
    ledger = _init_ledger(
        tmp_path,
        [
            _block(
                "ch01#b001",
                source,
                bad_draft,
                score=0.42,
                flags=["numeric_mismatch"],
            )
        ],
    )
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch01#b001",
                "status": BlockStatus.REPAIR_PENDING,
                "target_text": bad_draft,
                "mtqe_score": 0.42,
                "error_flags": ["numeric_mismatch"],
            }
        ]
    )

    # Escalated repair keeps returning the same wrong number and a low score:
    # the block must be quarantined, never shipped as machine output.
    router = ModelRouter(provider=MockModelProvider(default_response=bad_draft))
    repair_loop = RepairLoop(router=router, qe_runner=ControlledScoreQERunner([0.40]))

    async for _ in run_triage_stage(_triage_ctx(tmp_path, ledger, repair_loop)):
        pass

    rows = {b.id: b for b in ledger.get_all_blocks("job_triage")}
    blocked = rows["ch01#b001"]
    assert blocked.status == BlockStatus.BLOCKED_HUMAN
    assert blocked.mqm_severity == "critical"
    assert blocked.mqm_spans and blocked.mqm_spans[0]["error_type"] == "numeric"
    assert "mqm_critical_blocked" in blocked.error_flags
    # Quarantine placeholder carries the SOURCE text, not the machine draft.
    assert "【待人工审校" in (blocked.target_text or "")
    assert "42 megawatts" in (blocked.target_text or "")
    assert "84 megawatts" not in (blocked.target_text or "")
    ledger.close()


@pytest.mark.asyncio
async def test_triage_critical_resolved_after_escalation(tmp_path: Path) -> None:
    """Critical block fixed by the escalated round ships as REPAIRED."""
    source = "The reactor outputs 42 megawatts daily."
    bad_draft = "The reactor outputs 84 megawatts daily."
    fixed = "反应堆每天输出42兆瓦。"
    ledger = _init_ledger(
        tmp_path,
        [_block("ch01#b001", source, bad_draft, score=0.42, flags=["numeric_mismatch"])],
    )
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch01#b001",
                "status": BlockStatus.REPAIR_PENDING,
                "target_text": bad_draft,
                "mtqe_score": 0.42,
                "error_flags": ["numeric_mismatch"],
            }
        ]
    )

    router = ModelRouter(provider=MockModelProvider(default_response=fixed))
    repair_loop = RepairLoop(router=router, qe_runner=ControlledScoreQERunner([0.91]))

    async for _ in run_triage_stage(_triage_ctx(tmp_path, ledger, repair_loop)):
        pass

    rows = {b.id: b for b in ledger.get_all_blocks("job_triage")}
    fixed_block = rows["ch01#b001"]
    assert fixed_block.status == BlockStatus.REPAIRED
    assert fixed_block.target_text == fixed
    assert fixed_block.mtqe_score == 0.91
    ledger.close()


@pytest.mark.asyncio
async def test_triage_reclassifies_repair_failed_block(tmp_path: Path) -> None:
    """Regression: a repair-path FAILED block is terminal per
    ``is_finalized`` but must still reach the triage net. Structural corruption
    (math span mismatch) is now Critical: after a failed escalation it is
    quarantined rather than preserved as a shippable NEEDS_HUMAN draft."""
    flags = ["repair_structural_failure: Math span mismatch: source carries 2"]
    source = "See $x$ and $y$."
    bad_draft = "见 $x$。"
    ledger = _init_ledger(
        tmp_path,
        [
            _block(
                "ch01#b001",
                source,
                bad_draft,
                status=BlockStatus.FAILED,
                score=0.4,
                flags=flags,
            )
        ],
    )
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch01#b001",
                "status": BlockStatus.FAILED,
                "target_text": bad_draft,
                "mtqe_score": 0.4,
                "error_flags": flags,
            }
        ]
    )
    router = ModelRouter(provider=MockModelProvider(default_response="unused"))
    repair_loop = RepairLoop(router=router, qe_runner=MockQERunner(default_score=0.5))

    async for _ in run_triage_stage(_triage_ctx(tmp_path, ledger, repair_loop)):
        pass

    rows = {b.id: b for b in ledger.get_all_blocks("job_triage")}
    assert rows["ch01#b001"].status == BlockStatus.BLOCKED_HUMAN
    assert rows["ch01#b001"].mqm_severity == "critical"
    ledger.close()


@pytest.mark.asyncio
async def test_triage_major_terminology_needs_human(tmp_path: Path) -> None:
    """Terminology violation (Major) keeps the draft and routes to NEEDS_HUMAN."""
    glossary = [{"source": "Darcy", "translation": "达西", "aliases": ["德西"]}]
    draft = "德西先生走进了房间。"
    ledger = _init_ledger(
        tmp_path, [_block("ch01#b001", "Mr. Darcy entered the room.", draft, score=0.62)]
    )
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch01#b001",
                "status": BlockStatus.REPAIR_PENDING,
                "target_text": draft,
                "mtqe_score": 0.62,
            }
        ]
    )

    router = ModelRouter(provider=MockModelProvider(default_response="unused"))
    repair_loop = RepairLoop(router=router, qe_runner=MockQERunner(default_score=0.62))

    async for _ in run_triage_stage(
        _triage_ctx(tmp_path, ledger, repair_loop, glossary_dicts=glossary)
    ):
        pass

    rows = {b.id: b for b in ledger.get_all_blocks("job_triage")}
    needs_human = rows["ch01#b001"]
    assert needs_human.status == BlockStatus.NEEDS_HUMAN
    assert needs_human.mqm_severity == "major"
    assert needs_human.mqm_spans[0]["error_type"] == "terminology"
    # Draft preserved for the human post-editor.
    assert "德西" in (needs_human.target_text or "")
    assert "needs_human_review" in needs_human.error_flags
    ledger.close()


@pytest.mark.asyncio
async def test_triage_minor_autopass_and_low_score_escalation(tmp_path: Path) -> None:
    """Minor severity: score >= max(0.5, qe_threshold) auto-passes; lower goes NEEDS_HUMAN."""
    good = _block(
        "ch01#b001",
        "A plain sentence with no defects.",
        "一个没有缺陷的普通句子。",
        score=0.61,
        spine_index=1,
    )
    poor = _block(
        "ch01#b002",
        "Another entirely different sentence here.",
        "完全不同的另一句话。",
        score=0.31,
        spine_index=2,
    )
    ledger = _init_ledger(tmp_path, [good, poor])
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch01#b001",
                "status": BlockStatus.REPAIR_PENDING,
                "target_text": good.target_text,
                "mtqe_score": 0.61,
            },
            {
                "block_id": "ch01#b002",
                "status": BlockStatus.REPAIR_PENDING,
                "target_text": poor.target_text,
                "mtqe_score": 0.31,
            },
        ]
    )

    router = ModelRouter(provider=MockModelProvider(default_response="unused"))
    repair_loop = RepairLoop(router=router, qe_runner=MockQERunner(default_score=0.5))

    # Default threshold (0.75): 0.61 is below the bar -> NEEDS_HUMAN.
    async for _ in run_triage_stage(_triage_ctx(tmp_path, ledger, repair_loop)):
        pass

    rows = {b.id: b for b in ledger.get_all_blocks("job_triage")}
    assert rows["ch01#b001"].status == BlockStatus.NEEDS_HUMAN
    assert rows["ch01#b001"].mqm_severity == "minor"
    assert rows["ch01#b002"].status == BlockStatus.NEEDS_HUMAN
    assert rows["ch01#b002"].mqm_severity == "minor"

    # Relaxed threshold (0.5): the 0.5 floor still auto-passes 0.61.
    # Re-queue b001 as a fresh repair leftover (score intact, flags cleared).
    ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.REPAIR_PENDING,
        target_text=good.target_text,
        mtqe_score=0.61,
        error_flags=[],
    )
    async for _ in run_triage_stage(
        _triage_ctx(tmp_path, ledger, repair_loop, config=UBTConfig(qe_threshold=0.5))
    ):
        pass

    rows = {b.id: b for b in ledger.get_all_blocks("job_triage")}
    assert rows["ch01#b001"].status == BlockStatus.MTQE_PASSED
    assert rows["ch01#b002"].status == BlockStatus.NEEDS_HUMAN
    ledger.close()


@pytest.mark.asyncio
async def test_triage_skips_skip_translate_blocks(tmp_path: Path) -> None:
    """Code/formula blocks are never triaged even when left non-terminal."""
    block = _block("ch01#b001", "print(42)", "print(42)", flags=["numeric_mismatch"])
    block.skip_translate = True
    ledger = _init_ledger(tmp_path, [block])

    router = ModelRouter(provider=MockModelProvider(default_response="unused"))
    repair_loop = RepairLoop(router=router, qe_runner=MockQERunner(default_score=0.5))

    async for _ in run_triage_stage(_triage_ctx(tmp_path, ledger, repair_loop)):
        pass

    rows = {b.id: b for b in ledger.get_all_blocks("job_triage")}
    # Untouched by triage: no quarantine, no MQM annotation.
    assert rows["ch01#b001"].status == BlockStatus.REPAIR_PENDING
    assert rows["ch01#b001"].mqm_severity is None
    ledger.close()


@pytest.mark.asyncio
async def test_triage_emits_throttled_progress_events_not_silence(tmp_path: Path) -> None:
    """E2E: 4m56s passed between REPAIR_BATCH_COMPLETED and
    TRIAGE_COMPLETED with zero events while 310 escalated repairs ran block by
    block. The per-block loops must spend throttled heartbeats (every 20 units
    or 5s) on the generator's yield — the only progress channel the pipeline
    consumer has — without changing what triage decides.
    """
    bad_draft = "反应堆每天输出84兆瓦。"
    blocks = [
        _block(
            f"ch01#b{i:03d}",
            "The reactor outputs 42 megawatts daily.",
            bad_draft,
            score=0.42,
            flags=["numeric_mismatch"],
            spine_index=i,
        )
        for i in range(1, 26)
    ]
    ledger = _init_ledger(tmp_path, blocks)
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": b.id,
                "status": BlockStatus.REPAIR_PENDING,
                "target_text": bad_draft,
                "mtqe_score": 0.42,
                "error_flags": ["numeric_mismatch"],
            }
            for b in blocks
        ]
    )

    router = ModelRouter(provider=MockModelProvider(default_response=bad_draft))
    repair_loop = RepairLoop(router=router, qe_runner=ControlledScoreQERunner([0.40] * 64))

    ctx = _triage_ctx(tmp_path, ledger, repair_loop)
    events = [ev async for ev in run_triage_stage(ctx)]

    # A heartbeat exists between the batch event that preceded triage and the
    # final TRIAGE_COMPLETED — classification and the escalated-repair wait
    # loop both reported progress.
    mid = events[:-1]
    assert mid, "triage yielded no in-stage progress events"
    assert all(ev.event_type.value == "repair_batch_completed" for ev in mid), mid
    assert any("classified" in ev.message for ev in mid)
    assert events[-1].event_type.value == "triage_completed"
    # Throttled: 25 blocks cannot produce a per-block flood (<= 20-unit
    # cadence on each loop, plus slack for the time-based path).
    assert len(events) <= 8, [ev.message for ev in events]

    # The heartbeats changed nothing about the outcome: every critical block
    # still ends quarantined after its failed escalation.
    rows = {b.id: b for b in ledger.get_all_blocks("job_triage")}
    assert len(rows) == 25
    assert all(b.status == BlockStatus.BLOCKED_HUMAN for b in rows.values())
    ledger.close()


# ---------------------------------------------------------------------------
# Repair stage defer flag + lifecycle guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_stage_defer_unresolved_to_triage(tmp_path: Path) -> None:
    """With defer=True the repair tail leaves REPAIR_PENDING blocks for triage."""
    block = _block("ch01#b001", "Source text.", "草稿文本。", score=0.31, repair_rounds=2)
    ledger = _init_ledger(tmp_path, [block])
    ledger.save_checkpoints_batch(
        [
            {
                "block_id": "ch01#b001",
                "status": BlockStatus.REPAIR_PENDING,
                "target_text": "草稿文本。",
                "mtqe_score": 0.31,
            }
        ]
    )

    router = ModelRouter(provider=MockModelProvider(default_response="unused"))
    repair_loop = RepairLoop(router=router, qe_runner=MockQERunner(default_score=0.5))

    _RENAME = {"actual_job_id": "job_id", "create_event_fn": "create_event"}

    async def _drain(**kwargs: Any) -> None:
        kwargs.pop("max_repair_rounds")
        defer = kwargs.pop("defer_unresolved_to_triage")
        ctx = build_stage_ctx(tmp_path, **{_RENAME.get(k, k): v for k, v in kwargs.items()})
        async for _ in run_repair_stage(ctx, defer_unresolved_to_triage=defer):
            pass

    base: dict[str, Any] = {
        "ledger": ledger,
        "actual_job_id": "job_triage",
        "repair_loop": repair_loop,
        "glossary_dicts": [],
        "abbreviation_entries": [],
        "target_lang": "zh",
        "source_lang": "en",
        "fast_pass": None,
        "max_repair_rounds": 2,
        "concurrency_sem": asyncio.Semaphore(4),
        "create_event_fn": inert_event,
    }

    await _drain(**base, defer_unresolved_to_triage=True)
    rows = {b.id: b for b in ledger.get_all_blocks("job_triage")}
    assert rows["ch01#b001"].status == BlockStatus.REPAIR_PENDING

    await _drain(**base, defer_unresolved_to_triage=False)
    rows = {b.id: b for b in ledger.get_all_blocks("job_triage")}
    assert rows["ch01#b001"].status == BlockStatus.FAILED
    assert 'class="ubt-failed-draft"' in (rows["ch01#b001"].target_text or "")
    ledger.close()


def test_new_statuses_are_terminal_and_repair_ineligible(tmp_path: Path) -> None:
    """NEEDS_HUMAN / BLOCKED_HUMAN are terminal and excluded from repair eligibility."""
    assert BlockStatus.NEEDS_HUMAN.value == "needs_human"
    assert BlockStatus.BLOCKED_HUMAN.value == "blocked_human"
    needs = _block("b1", "s", "t", status=BlockStatus.NEEDS_HUMAN)
    blocked = _block("b2", "s", "t", status=BlockStatus.BLOCKED_HUMAN)
    assert needs.is_finalized
    assert blocked.is_finalized

    ledger = _init_ledger(tmp_path, [needs, blocked])
    ledger.save_checkpoints_batch(
        [
            {"block_id": "b1", "status": BlockStatus.NEEDS_HUMAN},
            {"block_id": "b2", "status": BlockStatus.BLOCKED_HUMAN},
        ]
    )
    eligible = ledger.fetch_repair_eligible_blocks("job_triage", max_rounds=2)
    assert eligible == []
    ledger.close()


def test_span_repair_accurate_numeric_candidate() -> None:
    """Verify that MQMSpanAnnotator picks the missing number as expected, not blindly src_nums[0]."""
    annotator = MQMSpanAnnotator()

    source = "In 1999, there were 500 cars produced in the factory."
    # 1999 was translated correctly, but 500 was mistranslated as 400
    draft = "在1999年，该工厂生产了400辆汽车。"
    flags = ["Numeric fidelity failure: number 500 missing from target"]

    annotated, spans = annotator.annotate_draft(
        source_text=source,
        draft_text=draft,
        error_flags=flags,
    )

    assert len(spans) == 1
    assert spans[0].error_type == "numeric"
    assert spans[0].erroneous_text == "400"
    # MUST be 500 (the missing number), NOT 1999 (src_nums[0])!
    assert spans[0].expected == "500"


def test_mqm_span_annotator_terminology_alias() -> None:
    """Verify that non-preferred terminology aliases are accurately annotated with <error_span>."""
    annotator = MQMSpanAnnotator()

    glossary = [
        {
            "source": "distributed ledger",
            "translation": "分布式账本",
            "aliases": ["分散记账簿", "分布式账目"],
        }
    ]

    source = "In this architecture, all nodes synchronize via a distributed ledger."
    draft = "在这个体系中，所有的数据节点都依托于分散记账簿进行状态同步。"
    flags = ["terminology_inconsistency: found alias '分散记账簿'"]

    annotated, spans = annotator.annotate_draft(
        source_text=source,
        draft_text=draft,
        error_flags=flags,
        glossary_entries=glossary,
    )

    assert len(spans) == 1
    span = spans[0]
    assert span.error_type == "terminology"
    assert span.expected == "分布式账本"
    assert span.erroneous_text == "分散记账簿"
    assert '<error_span id="1" type="terminology"' in annotated
    assert "分散记账簿</error_span>" in annotated


def test_mqm_span_annotator_source_leak() -> None:
    """Verify that untranslated source terms leaking into target are annotated as error spans."""
    annotator = MQMSpanAnnotator()

    glossary = [
        {
            "source": "attention mechanism",
            "translation": "注意力机制",
            "aliases": [],
        }
    ]

    source = "The core breakthrough is the attention mechanism."
    draft = "这一核心突破正是attention mechanism的应用。"
    flags = ["terminology_leak"]

    annotated, spans = annotator.annotate_draft(
        source_text=source,
        draft_text=draft,
        error_flags=flags,
        glossary_entries=glossary,
    )

    assert len(spans) == 1
    assert spans[0].error_type == "terminology_leak"
    assert spans[0].expected == "注意力机制"
    assert '<error_span id="1" type="terminology_leak"' in annotated
    assert "attention mechanism</error_span>" in annotated


def test_mqm_span_annotator_numeric_discrepancy() -> None:
    """Verify that numeric distortions are accurately isolated into error spans."""
    annotator = MQMSpanAnnotator()

    source = "A total of 1984 participants took part in the survey."
    draft = "共有1985名参与者参加了本次调查。"
    flags = ["Numeric fidelity failure: number 1984 missing from target"]

    annotated, spans = annotator.annotate_draft(
        source_text=source,
        draft_text=draft,
        error_flags=flags,
    )

    assert len(spans) == 1
    assert spans[0].error_type == "numeric"
    assert spans[0].erroneous_text == "1985"
    assert spans[0].expected == "1984"


def test_mqm_span_annotator_ignores_thousands_separator_variance() -> None:
    """'1,234' vs '1234' is one figure, not a numeric error.

    span_repair used to diff raw strings, so a locale-thousands difference was
    escalated critical while consistency.py already treated it equal — the same
    pair passed QE yet spawned a repair round. Now both go through
    canonicalize_numeric_token.
    """
    annotator = MQMSpanAnnotator()
    source = "The dataset holds 1,234 records."
    draft = "数据集包含1234条记录。"
    flags = ["Numeric fidelity failure"]
    _, spans = annotator.annotate_draft(source_text=source, draft_text=draft, error_flags=flags)
    assert spans == []


def test_triage_structural_defect_markers_include_repair_and_draft_errors() -> None:
    from ubt.core.qe.defect_taxonomy import STRUCTURAL_DEFECT_MARKERS

    assert any("Repair error" in m for m in STRUCTURAL_DEFECT_MARKERS)
    assert any("Drafting error" in m for m in STRUCTURAL_DEFECT_MARKERS)


@pytest.mark.asyncio
async def test_triage_non_prose_table_under_rigid_mode_degrades_gracefully(tmp_path: Path) -> None:
    """Non-prose tables under rigid mode (preserved verbatim via render_skip:non_prose
    in rigid PDF rendering) must degrade to NEEDS_HUMAN rather than BLOCKED_HUMAN
    on non-fatal defects, avoiding catastrophic false-blocking of export.
    """
    from ubt.core.config import UBTConfig
    from ubt.core.ir.models import BlockType
    from ubt.core.policy.layout_policy import is_rigid_non_prose_degradable

    # Policy helper check
    assert (
        is_rigid_non_prose_degradable(
            BlockType.TABLE,
            ["Repetitive loop hallucination detected"],
            render_engine="rigid",
        )
        is True
    )
    # Truly fatal defect (prompt template leak) must still be quarantined
    assert (
        is_rigid_non_prose_degradable(
            BlockType.TABLE,
            ["Prompt template XML artifacts leaked into target text"],
            render_engine="rigid",
        )
        is False
    )

    # Stage integration check: table block with non-fatal defect under rigid mode
    source_table = "| Method | Accuracy |\n|---|---|\n| Model | 0.95 |"
    table_block = IRBlock(
        id="ch01#t001",
        flow_id=FlowID.TABLE_GRID,
        block_type=BlockType.TABLE,
        spine_index=1,
        source_text=source_table,
        draft_text=source_table,
        target_text=source_table,
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=0.35,
        repair_rounds=2,
        error_flags=["Repetitive loop hallucination detected", "render_skip:non_prose"],
    )
    ledger = _init_ledger(tmp_path, [table_block])

    router = ModelRouter(provider=MockModelProvider(default_response=source_table))
    repair_loop = RepairLoop(router=router, qe_runner=ControlledScoreQERunner([0.40]))

    config = UBTConfig(render_engine="rigid")
    ctx = _triage_ctx(tmp_path, ledger, repair_loop, config=config)

    async for _ in run_triage_stage(ctx):
        pass

    rows = {b.id: b for b in ledger.get_all_blocks("job_triage")}
    table_row = rows["ch01#t001"]
    assert table_row.status == BlockStatus.NEEDS_HUMAN, (
        f"Expected NEEDS_HUMAN degradation, got: {table_row.status}"
    )
    ledger.close()

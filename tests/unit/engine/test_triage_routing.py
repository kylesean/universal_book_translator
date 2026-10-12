"""Triage's severity → status routing, and the fail-open rule for quality defects.

The triage stage is the last net before export: it decides whether a block that
survived (or failed) repair ships its machine draft, ships its source, or is
handed to a human. That decision is invisible to the render e2e (which only
drives clean blocks), so it is pinned here.

The load-bearing rule is the split introduced for RFC-2026-02:

- a **quality** verdict (numeric fidelity, math drift, glossary, omission) keeps
  the machine draft and routes to ``NEEDS_HUMAN`` — the reader gets a usable
  sentence, and the flag stays on the record;
- a **must-not-ship leak** (system prompt / template scaffolding) still
  quarantines as ``BLOCKED_HUMAN`` with the machine output stripped.

The rule exists because the arXiv 2101.03961 run quarantined 18 blocks, all of
them quality verdicts, 17 of them *correct* Chinese drafts rejected by a
false-positive numeric check. The old behaviour published English where good
Chinese existed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from ubt.core.engine.events import EventType
from ubt.core.engine.stages.triage import run_triage_stage
from ubt.core.ir.models import BlockStatus, BlockType, IRBlock, make_element
from ubt.core.qe.defect_taxonomy import (
    FLAG_MQM_CRITICAL_BLOCKED,
    FLAG_NEEDS_HUMAN_REVIEW,
)

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    *,
    status: BlockStatus = BlockStatus.REPAIR_PENDING,
    error_flags: list[str] | None = None,
    target_text: str = "一个正确的草稿",
    block_type: BlockType = BlockType.NARRATIVE,
) -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=0,
        block_type=block_type,
        source_text="A correct source sentence.",
    )
    return IRBlock(
        element=element,
        status=status,
        target_text=target_text,
        draft_text=target_text,
        error_flags=error_flags or [],
    )


@dataclass
class _Ledger:
    """Minimal ledger: records the triage batch and serves the candidates."""

    candidates: list[IRBlock]
    batches: list[list[dict[str, Any]]] = field(default_factory=list)

    def fetch_blocks_by_status(self, _job_id: str, status: Any) -> list[IRBlock]:
        return [b for b in self.candidates if b.status is status]

    def save_checkpoints_batch(self, checkpoints: Any, **_kw: Any) -> int:
        self.batches.append(list(checkpoints))
        return len(checkpoints)


@dataclass
class _RepairLoop:
    """Repair that never fixes anything, so every block reaches the verdict."""

    span_annotator: Any = None

    def __post_init__(self) -> None:
        self.span_annotator = _Annotator()

    async def repair_single_block(self, *, block: IRBlock, **_kw: Any) -> IRBlock:
        return block


class _Annotator:
    def annotate_draft(self, **_kw: Any) -> tuple[str, list[Any]]:
        return "", []


@dataclass
class _Facts:
    repair_loop: _RepairLoop


@dataclass
class _Scoring:
    repair_loop: _RepairLoop


@dataclass
class _Terminology:
    glossary_dicts: list[Any] = field(default_factory=list)
    abbreviation_entries: list[Any] = field(default_factory=list)


@dataclass
class _Services:
    repair_loop: _RepairLoop
    fast_pass: Any = None
    concurrency_sem: Any = None


class _Ctx:
    def __init__(self, ledger: _Ledger) -> None:
        self.ledger = ledger
        self.job_id = "job"
        self.target_lang = "zh"
        self.source_lang = "en"
        self.config = _Config()

    async def create_event(self, event_type: Any, *_a: Any, **kw: Any) -> Any:
        return _Event(event_type, kw.get("message", ""))

    def check_cancelled(self) -> None:
        return None


class _Config:
    qe_threshold = 0.5


@dataclass
class _Event:
    type: Any
    message: str


async def _run(ledger: _Ledger, repair: _RepairLoop) -> list[dict[str, Any]]:
    ctx = _Ctx(ledger)
    services = _Services(repair_loop=repair, concurrency_sem=asyncio.Semaphore(1))
    events: list[Any] = []
    gen: AsyncIterator[Any] = run_triage_stage(
        ctx,  # type: ignore[arg-type]
        services,  # type: ignore[arg-type]
        _Terminology(),  # type: ignore[arg-type]
        _Scoring(repair_loop=repair),  # type: ignore[arg-type]
    )
    async for event in gen:
        events.append(event)
    assert any(e.type is EventType.TRIAGE_COMPLETED for e in events)
    return ledger.batches[-1] if ledger.batches else []


def test_a_quality_defect_keeps_the_draft_and_routes_to_human() -> None:
    """A critical *quality* verdict must not replace the draft with the source."""
    block = _block(
        "b1",
        error_flags=["Numeric fidelity failure: Unauthorized numeric tokens in translation: ['2']"],
    )
    ledger = _Ledger([block])
    updates = asyncio.run(_run(ledger, _RepairLoop()))

    assert len(updates) == 1
    update = updates[0]
    assert update["status"] is BlockStatus.NEEDS_HUMAN
    # The draft survives — this is the whole point of the rule.
    assert update["target_text"] == "一个正确的草稿"
    assert FLAG_NEEDS_HUMAN_REVIEW in update["error_flags"]
    assert FLAG_MQM_CRITICAL_BLOCKED in update["error_flags"]


def test_a_must_not_ship_leak_is_still_quarantined() -> None:
    """Prompt/template leakage keeps the hard quarantine with the source shown."""
    block = _block("b2", error_flags=["Prompt scaffold: <|system|> leaked"])
    ledger = _Ledger([block])
    updates = asyncio.run(_run(ledger, _RepairLoop()))

    assert len(updates) == 1
    update = updates[0]
    assert update["status"] is BlockStatus.BLOCKED_HUMAN
    # The machine output is stripped: the placeholder wraps the *source*.
    assert "一个正确的草稿" not in update["target_text"]
    assert "A correct source sentence." in update["target_text"]


def test_a_correct_draft_after_repair_passes() -> None:
    """A block the escalated repair resolves is released, not flagged."""
    block = _block("b3", error_flags=["Numeric fidelity failure: missing ['2']"])

    class _Fixing(_RepairLoop):
        async def repair_single_block(self, *, block: IRBlock, **_kw: Any) -> IRBlock:
            fixed = _block(block.id, status=BlockStatus.REPAIRED, target_text="修好的草稿")
            return fixed

    ledger = _Ledger([block])
    updates = asyncio.run(_run(ledger, _Fixing()))

    assert len(updates) == 1
    assert updates[0]["status"] is BlockStatus.REPAIRED
    assert updates[0]["target_text"] == "修好的草稿"

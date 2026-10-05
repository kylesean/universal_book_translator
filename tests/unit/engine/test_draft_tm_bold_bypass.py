"""Exact TM hits and preserved emphasis.

A TM entry can carry the target-side emphasis runs the marker mechanism produced
(see ``ubt.core.ir.emphasis``). A hit that carries them restores the bold without
a model call; an entry that predates runs is refused for a partial-bold block so
the block is re-drafted instead of losing its bold.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ubt.core.engine.stages.draft import _DraftProcessor
from ubt.core.ir.emphasis import runs_to_json
from ubt.core.ir.models import BlockType, InlineRun, IRBlock, StyleMeta, make_element

pytestmark = pytest.mark.fast


class _TM:
    def __init__(self, hit: Any = None) -> None:
        self.hit = hit
        self.exact_calls = 0

    def lookup_exact(self, *args: Any, **kwargs: Any) -> Any:
        self.exact_calls += 1
        return self.hit

    def lookup_fuzzy(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FastPass:
    def evaluate(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(passed=True, reason="")


class _Flusher:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    async def enqueue(self, update: dict[str, Any]) -> None:
        self.updates.append(update)


def _runtime(tm: _TM, flusher: _Flusher) -> SimpleNamespace:
    def _mask(text: str) -> SimpleNamespace:
        return SimpleNamespace(
            text=text, code_map=None, cite_map=None, math_map=None, soup_map=None, email_map=None
        )

    return SimpleNamespace(
        tm=tm,
        active_fast_pass=_FastPass(),
        engine=SimpleNamespace(mask=_mask),
        ledger=SimpleNamespace(
            get_preceding_text_tail=lambda *a, **k: "",
            get_following_text_head=lambda *a, **k: "",
        ),
        memory_mgr=SimpleNamespace(
            get_l1_context=lambda *a, **k: "",
            get_macro_context_for_block=lambda *a, **k: "",
            get_l3_summary=lambda *a, **k: "",
        ),
        flusher=flusher,
        counters={"tm_exact_hits": 0},
        actual_job_id="job",
    )


def _policy() -> SimpleNamespace:
    return SimpleNamespace(
        source_lang="en",
        target_lang="zh",
        tm_context="",
        glossary_dicts=[],
        abbreviation_entries=[],
        rolling_enabled=False,
        profile_name="paper",
        domain=None,
        tm_fuzzy_threshold=0.8,
    )


def _block(source: str, style: StyleMeta | None = None) -> IRBlock:
    element = make_element(
        id="b1", spine_index=1, block_type=BlockType.NARRATIVE, source_text=source
    )
    return IRBlock(element=element, style=style)


async def _prepare(block: IRBlock, tm: _TM) -> tuple[Any, list[dict[str, Any]]]:
    flusher = _Flusher()
    processor = _DraftProcessor(runtime=_runtime(tm, flusher), policy=_policy())  # type: ignore[arg-type]
    result = await processor.prepare_draft_inputs(block, [block])
    return result, flusher.updates


def _bold_block() -> IRBlock:
    return _block(
        "图注 SoL-Pi 降低成本 50.0%。",
        StyleMeta(inline_runs=(InlineRun(text="50.0%", bold=True),)),
    )


@pytest.mark.asyncio
async def test_a_bold_block_rejects_a_hit_without_emphasis_runs() -> None:
    tm = _TM(hit=SimpleNamespace(target_text="旧译文", runs_json=""))
    result, _ = await _prepare(_bold_block(), tm)
    assert tm.exact_calls == 1
    assert result is not None  # falls through to the LLM draft


@pytest.mark.asyncio
async def test_a_bold_block_uses_a_hit_that_carries_runs() -> None:
    runs_json = runs_to_json((InlineRun(text="50.0%", bold=True),))
    tm = _TM(hit=SimpleNamespace(target_text="旧译文 50.0%", runs_json=runs_json))
    result, updates = await _prepare(_bold_block(), tm)
    assert result is None  # served from TM
    assert len(updates) == 1
    assert updates[0]["tm_hit"] is True
    assert [r.text for r in updates[0]["style"].target_runs] == ["50.0%"]


@pytest.mark.asyncio
async def test_a_plain_block_still_uses_the_exact_hit() -> None:
    tm = _TM(hit=SimpleNamespace(target_text="旧译文", runs_json=""))
    result, _ = await _prepare(_block("plain source"), tm)
    assert tm.exact_calls == 1
    assert result is None  # served from TM

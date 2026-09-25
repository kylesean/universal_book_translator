"""Build a :class:`StageContext` so a test can drive ONE pipeline stage.

Stages used to take up to 24 keyword arguments, so a unit test of the quality
gate passed five and a test of export passed most of a run's wiring by hand.
Stages take one context now, which is right for production and awkward for a
test unless something fills in the parts it does not care about — that is this
module. Every default is deliberately inert: a mock provider that answers with
fixed text, the heuristic QE runner, no TM, one slot of concurrency. A test
names only what it is about.

Unknown names are NOT swallowed: they land on ``StageContext``'s constructor and
raise there, so a typo in a test reads as a real error instead of a silently
ignored knob.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ubt.core.config import UBTConfig
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.engine.stage_context import StageContext
from ubt.core.ir.models import BookManifest, ChapterMeta
from ubt.core.policy.adaptive_policy import AdaptivePolicy, Granularity
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.comet_runner import HeuristicQERunner
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


async def inert_event(
    event_type: EventType,
    job_id: str,
    ledger: SQLiteJobLedger,
    message: str = "",
    active_block_id: str | None = None,
    artifact_path: str | None = None,
) -> TranslationProgressEvent:
    """An event with no ledger traffic in it.

    The real factory re-reads job stats on every event; a stage test asserts the
    stage's effect on blocks, not the stats query that surrounds it. Six test
    files used to carry their own copy of this; import it instead.
    """
    return TranslationProgressEvent(
        event_type=event_type,
        job_id=job_id,
        total_blocks=0,
        completed_blocks=0,
        message=message,
    )


def build_stage_ctx(tmp_path: Path | None = None, **overrides: Any) -> StageContext:
    """A context whose unset parts are inert; see the module docstring.

    ``tmp_path`` is only needed for the defaults this builds itself (a ledger,
    a config, a manifest). A test that supplies its own needs no directory.
    """
    if tmp_path is None and not overrides.get("ledger"):
        raise TypeError("build_stage_ctx needs tmp_path when it has to create the ledger")
    home = tmp_path or Path()
    job_id: str = overrides.get("job_id", "job_ctx")
    config: UBTConfig = overrides.get("config") or UBTConfig(db_dir=home)
    router: ModelRouter = overrides.get("router") or ModelRouter(
        provider=MockModelProvider(default_response="[模拟翻译]"),
        draft_model="mock-draft",
        repair_model="mock-repair",
    )
    qe_runner: BaseQERunner = overrides.get("qe_runner") or HeuristicQERunner()
    source_lang: str = overrides.get("source_lang", "en")
    target_lang: str = overrides.get("target_lang", "zh")
    defaults: dict[str, Any] = {
        "config": config,
        "router": router,
        "ledger": overrides.get("ledger") or SQLiteJobLedger(home / f"{job_id}.sqlite"),
        "manifest": overrides.get("manifest")
        or BookManifest(
            doc_id="ctx",
            title="Ctx",
            source_path=str(home / f"{job_id}.epub"),
            chapters=[ChapterMeta(chapter_id="ch01", title="One", spine_index=1)],
        ),
        "job_id": job_id,
        "input_path": overrides.get("input_path") or home / f"{job_id}.epub",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "profile_name": "general",
        "fast_pass": overrides.get("fast_pass")
        or FastPassFilter(source_lang=source_lang, target_lang=target_lang),
        "qe_runner": qe_runner,
        "repair_loop": overrides.get("repair_loop")
        or RepairLoop(router=router, qe_runner=qe_runner),
        "adaptive_policy": overrides.get("adaptive_policy")
        or AdaptivePolicy(
            granularity=Granularity.MICRO,
            render_engine="publication",
            fast_lane_bible=False,
            visual_blocking=False,
            deterministic_glossary=False,
            reason="test default",
        ),
        "concurrency_sem": overrides.get("concurrency_sem") or asyncio.Semaphore(1),
        "create_event": overrides.get("create_event") or inert_event,
        "output_path": None,
        "adapter": None,
        "tm": None,
        "short_chain": False,
        "glossary_dicts": [],
        "abbreviation_entries": [],
        "block_count": 0,
    }
    defaults.update(overrides)
    return StageContext(**defaults)


async def drain(stream: AsyncIterator[Any]) -> list[Any]:
    """Collect a stage's events.

    Stages are async generators now, so ``asyncio.run(run_x_stage(ctx))`` fails
    with "a coroutine was expected"; a test that only cares about the ledger
    state after the stage wants this instead.
    """
    return [event async for event in stream]

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

import ubt.core.engine.plan as run_plan
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.facts import RunFacts
from ubt.core.engine.plan import RunGates, run_stages
from ubt.core.engine.stage_context import EventFactory, StageContext

if TYPE_CHECKING:
    from ubt.core.config import UBTConfig
    from ubt.core.engine.blocks import BlockReader
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.services import RunServices
    from ubt.core.ir.models import BookManifest
    from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.fast

_FULL_ORDER = (
    "ingest",
    "extraction_witness",
    "mode_advisory",
    "render_preflight",
    "cost_preflight",
    "bible",
    "draft",
    "c_text",
    "quality_gate",
    "repair",
    "consistency",
    "triage",
    "difficulty_advisory",
    "export",
)
_STREAMING_ORDER = (
    "ingest",
    "extraction_witness",
    "mode_advisory",
    "render_preflight",
    "cost_preflight",
    "bible",
    "chapter_streaming",
    "consistency",
    "triage",
    "difficulty_advisory",
    "export",
)
_AWAITED = frozenset(
    {"extraction_witness", "render_preflight", "cost_preflight", "difficulty_advisory"}
)

_CTX = StageContext(
    config=cast("UBTConfig", SimpleNamespace(name="config")),
    router=cast("ModelRouter", SimpleNamespace(name="router")),
    ledger=cast("SQLiteJobLedger", SimpleNamespace(name="ledger")),
    manifest=cast("BookManifest", SimpleNamespace(name="manifest")),
    job_id="job",
    input_path=Path("book.epub"),
    source_lang="en",
    target_lang="zh",
    profile_name="general",
    create_event=cast("EventFactory", lambda *a, **k: None),
)
_FACTS = RunFacts()
_SERVICES = cast("RunServices", SimpleNamespace(name="services"))
_BLOCKS = cast("BlockReader", object())


def _event(event_type: EventType, message: str) -> TranslationProgressEvent:
    return TranslationProgressEvent(
        event_type=event_type, job_id="job", total_blocks=1, completed_blocks=0, message=message
    )


@dataclass(frozen=True, slots=True)
class _Recorder:
    calls: list[str]
    args: dict[str, tuple[object, ...]]

    @classmethod
    def start(cls) -> _Recorder:
        return cls(calls=[], args={})

    def yielding(
        self, name: str, *events: TranslationProgressEvent
    ) -> Callable[..., AsyncIterator[TranslationProgressEvent]]:
        def factory(*call_args: object) -> AsyncIterator[TranslationProgressEvent]:
            async def stage() -> AsyncIterator[TranslationProgressEvent]:
                self.calls.append(name)
                self.args[name] = call_args
                for event in events:
                    yield event

            return stage()

        return factory

    def awaiting(self, name: str) -> Callable[..., Awaitable[None]]:
        def factory(*call_args: object) -> Awaitable[None]:
            async def stage() -> None:
                self.calls.append(name)
                self.args[name] = call_args

            return stage()

        return factory


def _install(monkeypatch: pytest.MonkeyPatch, rec: _Recorder) -> None:
    for name in _FULL_ORDER:
        attr = f"run_{name}_stage"
        event_type = (
            EventType.EXPORT_COMPLETED if name == "export" else EventType.PREPROCESSING_DONE
        )
        if name in _AWAITED:
            monkeypatch.setattr(run_plan, attr, rec.awaiting(name))
        else:
            monkeypatch.setattr(run_plan, attr, rec.yielding(name, _event(event_type, name)))
    monkeypatch.setattr(
        run_plan,
        "run_chapter_streaming_pipeline",
        rec.yielding("chapter_streaming", _event(EventType.CHAPTER_COMPLETED, "chapter_streaming")),
    )


def _gates(
    *, chapter_streaming: bool = False, c_text: bool = True, consistency: bool = True
) -> RunGates:
    return RunGates(chapter_streaming=chapter_streaming, c_text=c_text, consistency=consistency)


async def _collect(
    rec: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    gates: RunGates,
    *,
    on_export_completed: Callable[[TranslationProgressEvent], Awaitable[None]] | None = None,
) -> list[TranslationProgressEvent]:
    _install(monkeypatch, rec)
    events: list[TranslationProgressEvent] = []
    async for event in run_stages(
        _CTX, gates, _FACTS, _SERVICES, blocks=_BLOCKS, on_export_completed=on_export_completed
    ):
        events.append(event)
    return events


async def test_the_plan_runs_every_stage_in_declared_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder.start()
    await _collect(rec, monkeypatch, _gates())
    assert rec.calls == list(_FULL_ORDER)


async def test_zero_token_preflights_run_before_the_billable_bible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder.start()
    await _collect(rec, monkeypatch, _gates())
    assert rec.calls.index("render_preflight") < rec.calls.index("bible")
    assert rec.calls.index("cost_preflight") < rec.calls.index("bible")


async def test_events_are_yielded_in_stage_order(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder.start()
    events = await _collect(rec, monkeypatch, _gates())
    assert [event.message for event in events] == [
        name for name in _FULL_ORDER if name not in _AWAITED
    ]


async def test_the_chapter_streaming_gate_replaces_the_draft_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder.start()
    await _collect(rec, monkeypatch, _gates(chapter_streaming=True))
    assert rec.calls == list(_STREAMING_ORDER)
    for absent in ("draft", "c_text", "quality_gate", "repair"):
        assert absent not in rec.calls


async def test_the_c_text_gate_skips_only_the_c_text_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder.start()
    await _collect(rec, monkeypatch, _gates(c_text=False))
    assert "c_text" not in rec.calls
    assert rec.calls == [name for name in _FULL_ORDER if name != "c_text"]


async def test_the_consistency_gate_skips_only_the_consistency_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder.start()
    await _collect(rec, monkeypatch, _gates(consistency=False))
    assert "consistency" not in rec.calls
    assert rec.calls == [name for name in _FULL_ORDER if name != "consistency"]


async def test_inter_stage_facts_go_to_the_stages_that_read_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder.start()
    await _collect(rec, monkeypatch, _gates())
    args = rec.args
    assert args["bible"][1] is _FACTS.terminology
    assert args["mode_advisory"][1] is _FACTS.layout
    assert args["mode_advisory"][2] is _FACTS.render
    assert args["difficulty_advisory"][1] is _FACTS.layout
    assert args["difficulty_advisory"][2] is _FACTS.render
    assert args["render_preflight"][1] is _FACTS.render
    assert args["export"][2] is _FACTS.render
    for name in ("draft", "quality_gate", "repair", "consistency", "triage"):
        assert args[name][2] is _FACTS.terminology
    assert args["export"][3] is _FACTS.terminology
    for name in ("quality_gate", "repair", "consistency", "triage"):
        assert args[name][3] is _FACTS.scoring


async def test_run_resources_are_the_same_objects_for_every_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder.start()
    await _collect(rec, monkeypatch, _gates())
    args = rec.args
    for name in _FULL_ORDER:
        assert args[name][0] is _CTX
    for name in ("draft", "c_text", "quality_gate", "repair", "consistency", "triage", "export"):
        assert args[name][1] is _SERVICES
    for name in ("extraction_witness", "cost_preflight"):
        assert args[name][-1] is _BLOCKS
    for name in ("mode_advisory", "difficulty_advisory"):
        assert args[name][3] is _BLOCKS
        assert args[name][4] is _SERVICES
    assert args["render_preflight"][2] is _BLOCKS


_ORDINARY = "ordinary export event"
_TERMINAL = "terminal export event"


async def test_the_export_hook_runs_on_the_terminal_event_before_it_is_yielded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder.start()
    _install(monkeypatch, rec)
    ordinary = _event(EventType.MODE_ADVISED, _ORDINARY)
    terminal = _event(EventType.EXPORT_COMPLETED, _TERMINAL)
    monkeypatch.setattr(run_plan, "run_export_stage", rec.yielding("export", ordinary, terminal))
    record: list[tuple[str, TranslationProgressEvent]] = []

    async def hook(event: TranslationProgressEvent) -> None:
        record.append(("hook", event))

    async for event in run_stages(
        _CTX, _gates(), _FACTS, _SERVICES, blocks=_BLOCKS, on_export_completed=hook
    ):
        record.append(("yield", event))

    assert [(kind, event.message) for kind, event in record][-3:] == [
        ("yield", _ORDINARY),
        ("hook", _TERMINAL),
        ("yield", _TERMINAL),
    ]
    assert record[-2][1] is terminal


async def test_a_consumer_that_stops_at_the_terminal_event_keeps_the_hook_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder.start()
    _install(monkeypatch, rec)
    terminal = _event(EventType.EXPORT_COMPLETED, _TERMINAL)
    monkeypatch.setattr(run_plan, "run_export_stage", rec.yielding("export", terminal))
    record: list[tuple[str, TranslationProgressEvent]] = []

    async def hook(event: TranslationProgressEvent) -> None:
        record.append(("hook", event))

    plan = run_stages(_CTX, _gates(), _FACTS, _SERVICES, blocks=_BLOCKS, on_export_completed=hook)
    async for event in plan:
        record.append(("yield", event))
        if event.event_type is EventType.EXPORT_COMPLETED:
            break
    await plan.aclose()

    assert [(kind, event.message) for kind, event in record][-2:] == [
        ("hook", _TERMINAL),
        ("yield", _TERMINAL),
    ]
    assert [kind for kind, _ in record if kind == "hook"] == ["hook"]

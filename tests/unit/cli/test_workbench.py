from __future__ import annotations

import asyncio

import pytest

from ubt.cli.workbench.controller import WorkbenchController
from ubt.cli.workbench.dashboard import render_dashboard
from ubt.cli.workbench.models import (
    JobControlStatus,
    JobViewItem,
    KeyAction,
    WorkbenchMode,
    WorkbenchState,
)
from ubt.cli.workbench.runner import InteractiveKeyReader, run_workbench

pytestmark = pytest.mark.fast


def test_workbench_models_state() -> None:
    state = WorkbenchState(mode=WorkbenchMode.SINGLE_JOB)
    assert state.mode == WorkbenchMode.SINGLE_JOB
    assert state.status == JobControlStatus.IDLE
    assert state.elapsed_s >= 0.0

    state.add_event("Event 1", max_events=3)
    state.add_event("Event 2", max_events=3)
    state.add_event("Event 3", max_events=3)
    state.add_event("Event 4", max_events=3)
    assert len(state.recent_events) == 3
    assert "Event 4" in state.recent_events[-1]


def test_workbench_controller_pause_resume() -> None:
    ctrl = WorkbenchController(concurrency=2)
    assert not ctrl.is_paused
    assert ctrl.state.status == JobControlStatus.IDLE

    ctrl.pause()
    assert ctrl.is_paused
    assert ctrl.state.status == JobControlStatus.PAUSED
    assert "PAUSED" in ctrl.state.status_message

    ctrl.resume()
    assert not ctrl.is_paused
    assert ctrl.state.status == JobControlStatus.RUNNING
    assert "RUNNING" in ctrl.state.status_message

    ctrl.toggle_pause()
    assert ctrl.is_paused
    ctrl.toggle_pause()
    assert not ctrl.is_paused


@pytest.mark.asyncio
async def test_workbench_controller_wait_if_paused() -> None:
    ctrl = WorkbenchController()
    # Unpaused: wait_if_paused returns immediately
    await ctrl.wait_if_paused()

    ctrl.pause()
    task = asyncio.create_task(ctrl.wait_if_paused())
    await asyncio.sleep(0.01)
    assert not task.done()

    ctrl.resume()
    await asyncio.sleep(0.01)
    assert task.done()


def test_workbench_controller_concurrency_bounds() -> None:
    ctrl = WorkbenchController(concurrency=2)
    assert ctrl.state.concurrency == 2

    # Increase
    for _ in range(20):
        ctrl.increase_concurrency(max_limit=16)
    assert ctrl.state.concurrency == 16

    # Decrease
    for _ in range(25):
        ctrl.decrease_concurrency(min_limit=1)
    assert ctrl.state.concurrency == 1


def test_workbench_controller_tab_cycling() -> None:
    ctrl = WorkbenchController()
    assert ctrl.state.selected_tab == 0

    ctrl.next_tab()
    assert ctrl.state.selected_tab == 1
    ctrl.next_tab()
    assert ctrl.state.selected_tab == 2
    ctrl.next_tab()
    assert ctrl.state.selected_tab == 3
    ctrl.next_tab()
    assert ctrl.state.selected_tab == 0

    ctrl.prev_tab()
    assert ctrl.state.selected_tab == 3


def test_workbench_controller_key_handling() -> None:
    ctrl = WorkbenchController()

    assert ctrl.handle_key("p") == KeyAction.TOGGLE_PAUSE
    assert ctrl.is_paused
    assert ctrl.handle_key(" ") == KeyAction.TOGGLE_PAUSE
    assert not ctrl.is_paused

    assert ctrl.handle_key("+") == KeyAction.INCREASE_CONCURRENCY
    assert ctrl.handle_key("-") == KeyAction.DECREASE_CONCURRENCY
    assert ctrl.handle_key("\t") == KeyAction.NEXT_TAB
    assert ctrl.handle_key("h") == KeyAction.PREV_TAB
    assert ctrl.handle_key("j") == KeyAction.NEXT_JOB
    assert ctrl.handle_key("k") == KeyAction.PREV_JOB
    assert ctrl.handle_key("r") == KeyAction.RETRY_FAILED
    assert ctrl.handle_key("?") == KeyAction.TOGGLE_HELP
    assert ctrl.state.show_help
    assert ctrl.handle_key("?") == KeyAction.TOGGLE_HELP
    assert not ctrl.state.show_help
    assert ctrl.handle_key("q") == KeyAction.QUIT
    assert ctrl.is_cancelled


def test_workbench_resource_sampling() -> None:
    ctrl = WorkbenchController()
    snap = ctrl.sample_resources()
    assert snap.rss_mb >= 0.0
    assert snap.cpu_time_s >= 0.0
    assert snap.concurrency_limit == 1


def test_workbench_update_from_event() -> None:
    ctrl = WorkbenchController()
    ctrl.update_from_event(
        event_type="DRAFT",
        message="Drafting block 10",
        meta={
            "pct": 45.0,
            "blocks_done": 45,
            "blocks_total": 100,
            "quarantined": 2,
            "repaired": 1,
            "prompt_tokens": 1200,
            "completion_tokens": 400,
            "cached_tokens": 800,
            "cost_usd": 0.005,
        },
    )

    assert ctrl.state.current_stage == "DRAFT"
    assert ctrl.state.progress_pct == 45.0
    assert ctrl.state.blocks_done == 45
    assert ctrl.state.quarantined_count == 2
    assert ctrl.state.repaired_count == 1
    assert ctrl.state.resource.prompt_tokens == 1200
    assert ctrl.state.resource.completion_tokens == 400
    assert ctrl.state.resource.cached_tokens == 800
    assert ctrl.state.resource.cost_usd == 0.005
    assert ctrl.state.resource.cache_hit_rate == 40.0


def test_workbench_dashboard_rendering_all_tabs() -> None:
    ctrl = WorkbenchController()
    ctrl.state.jobs = [
        JobViewItem(job_id="job_1", title="Book 1", status="running", progress_pct=50.0),
        JobViewItem(job_id="job_2", title="Book 2", status="queued", progress_pct=0.0),
    ]
    ctrl.state.quarantined_details = [
        {"block_id": "b001", "reason": "Latex unbalanced", "source": "$ x + y "}
    ]

    for tab in range(4):
        ctrl.state.selected_tab = tab
        layout = render_dashboard(ctrl.state)
        assert layout is not None

    # Help overlay
    ctrl.state.show_help = True
    layout = render_dashboard(ctrl.state)
    assert layout is not None


def test_interactive_key_reader_non_tty() -> None:
    reader = InteractiveKeyReader(on_key=lambda k: None)
    # On non-interactive test runs, fd is either None or not a tty
    loop = asyncio.new_event_loop()
    try:
        reader.start(loop)
        reader.stop(loop)
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_run_workbench_clean_completion() -> None:
    ctrl = WorkbenchController()

    async def _mock_task(c: WorkbenchController) -> None:
        await asyncio.sleep(0.05)

    await run_workbench(ctrl, task_fn=_mock_task, refresh_rate=20.0)
    assert ctrl.state.status == JobControlStatus.COMPLETED

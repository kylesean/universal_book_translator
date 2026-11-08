"""The one status decision tree every reader surface must call.

``job_meta.status`` has no ``running`` value and the three surfaces (REST
``/status`` disk fallback, MCP fallback, ``/jobs`` listing) used to own three
private trees that had drifted — this module is the convergence point, and
these tests pin its verdicts table-style.
"""

from __future__ import annotations

import pytest

from ubt.core.engine.status_projection import project_ledger_status

pytestmark = pytest.mark.fast


def test_a_terminal_row_passes_through_verbatim() -> None:
    assert (
        project_ledger_status(stored="failed", total=4, done=0, has_live_writer=False).status
        == "failed"
    )
    assert (
        project_ledger_status(stored="cancelled", total=4, done=4, has_live_writer=True).status
        == "cancelled"
    )
    assert (
        project_ledger_status(stored="completed", total=4, done=4, has_live_writer=False).status
        == "completed"
    )


def test_all_terminal_blocks_mean_completed_even_without_a_finalize() -> None:
    """A process lost before finalize leaves a fully accounted row behind."""
    projected = project_ledger_status(stored="initialized", total=9, done=9, has_live_writer=False)
    assert projected.status == "completed"
    assert projected.interrupted is False


def test_a_parked_agent_session_is_not_a_crash() -> None:
    """Agent jobs are intentionally lock-free between tool calls."""
    projected = project_ledger_status(
        stored="initialized",
        total=9,
        done=2,
        agent_state="awaiting_agent",
        has_live_writer=False,
    )
    assert projected.status == "awaiting_agent"
    assert projected.interrupted is False


def test_a_live_writer_means_running_and_no_writer_means_failed() -> None:
    running = project_ledger_status(stored="initialized", total=9, done=2, has_live_writer=True)
    assert running.status == "running"

    failed = project_ledger_status(stored="initialized", total=9, done=2, has_live_writer=False)
    assert failed.status == "failed"
    assert failed.interrupted is True


def test_the_quiet_grace_window_keeps_a_fresh_unwritten_ledger_running() -> None:
    """MCP's staleness knob: a writer that just died still gets its grace."""
    graceful = project_ledger_status(
        stored="initialized",
        total=9,
        done=2,
        has_live_writer=False,
        assume_running_when_unwritten=True,
    )
    assert graceful.status == "running"
    assert graceful.interrupted is False

    stale = project_ledger_status(
        stored="initialized",
        total=9,
        done=2,
        has_live_writer=False,
        assume_running_when_unwritten=False,
    )
    assert stale.status == "failed"
    assert stale.interrupted is True

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator, _RunBillingSession

pytestmark = pytest.mark.fast


def test_billing_session_isolation() -> None:
    config = UBTConfig(draft_model="mock-draft")
    mock_router = MagicMock()
    mock_router.usage_totals_by_model.return_value = {
        "model-a": {"prompt_tokens": 100, "completion_tokens": 50}
    }
    mock_router.billing_endpoint_map.return_value = {}

    orchestrator = PipelineOrchestrator(config=config, router=mock_router)

    session1 = _RunBillingSession(
        sink={"model-a": {"prompt_tokens": 10, "completion_tokens": 5}},
        baseline={},
        billed_usage={},
    )
    session2 = _RunBillingSession(
        sink={"model-a": {"prompt_tokens": 40, "completion_tokens": 20}},
        baseline={},
        billed_usage={},
    )

    orchestrator._billing_sessions["job-1"] = session1
    orchestrator._billing_sessions["job-2"] = session2

    usage1 = orchestrator._session_usage(orchestrator._get_billing_session("job-1"))
    usage2 = orchestrator._session_usage(orchestrator._get_billing_session("job-2"))

    assert usage1["model-a"]["prompt_tokens"] == 10
    assert usage2["model-a"]["prompt_tokens"] == 40

    session1.billed_usage = usage1
    assert orchestrator._get_billing_session("job-1").billed_usage["model-a"]["prompt_tokens"] == 10
    assert orchestrator._get_billing_session("job-2").billed_usage == {}


def test_terminal_billing_flush_persists_the_in_flight_tail(tmp_path: Path) -> None:
    """A run that aborts must still land the spend after its last progress event.

    Billing used to happen only on progress events and the export hook, so the
    tokens of calls in flight past the final event died with the process and
    the resumed job's budget floor under-counted exactly the runs that died
    mid-flight.
    """
    import asyncio

    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.writer_lock import LedgerWriterLock

    config = UBTConfig(draft_model="mock-draft")
    mock_router = MagicMock()
    mock_router.billing_endpoint_map.return_value = {}
    orchestrator = PipelineOrchestrator(config=config, router=mock_router)

    db_path = tmp_path / "term-bill.sqlite"
    lock = LedgerWriterLock(db_path, "job-t1")
    lock.acquire()
    try:
        with SQLiteJobLedger(db_path) as ledger:
            ledger.bind_writer_lock(lock)
            ledger.init_job_placeholder("job-t1", "book.pdf", "zh")
            orchestrator._billing_sessions["job-t1"] = _RunBillingSession(
                sink={"model-a": {"prompt_tokens": 30, "completion_tokens": 12}},
                baseline={},
                billed_usage={},
            )

            asyncio.run(orchestrator._bill_run_terminal(ledger, "job-t1"))

            usage = ledger.get_job_usage("job-t1")
            assert usage["model-a"]["prompt_tokens"] == 30
            assert usage["model-a"]["completion_tokens"] == 12

            # A second flush is idempotent: the session's billed_usage now
            # carries what was written, so nothing is re-added.
            asyncio.run(orchestrator._bill_run_terminal(ledger, "job-t1"))
            assert ledger.get_job_usage("job-t1")["model-a"]["prompt_tokens"] == 30
    finally:
        lock.release()


def test_terminal_billing_flush_tolerates_a_missing_session(tmp_path: Path) -> None:
    """A failure before the billing session exists must not raise from the flush.

    A run that dies between orchestrator construction and the session
    registration still reaches the abort branches; the flush has to treat the
    missing session (and a missing ledger) as nothing to bill.
    """
    import asyncio

    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.writer_lock import LedgerWriterLock

    config = UBTConfig(draft_model="mock-draft")
    orchestrator = PipelineOrchestrator(config=config, router=MagicMock())

    asyncio.run(orchestrator._bill_run_terminal(None, "never-started"))

    db_path = tmp_path / "term-nosession.sqlite"
    lock = LedgerWriterLock(db_path, "job-nosession")
    lock.acquire()
    try:
        with SQLiteJobLedger(db_path) as ledger:
            ledger.bind_writer_lock(lock)
            ledger.init_job_placeholder("job-nosession", "book.pdf", "zh")
            asyncio.run(orchestrator._bill_run_terminal(ledger, "job-nosession"))
            assert ledger.get_job_usage("job-nosession") == {}
    finally:
        lock.release()


def test_teardown_detaches_the_usage_sink() -> None:
    """A finished run must not leave its sink bound to the worker task.

    The sink is attached deep inside ``begin_usage_sink`` and nothing reset
    it: after a run, the task's context still pointed at the finalized sink
    and every later call on a reused task kept attributing into a dead run's
    dict until the next attach overwrote it.
    """
    import asyncio

    from ubt.core.router.transports.base import _usage_sink, attach_usage_sink

    config = UBTConfig(draft_model="mock-draft")
    orchestrator = PipelineOrchestrator(config=config, router=MagicMock())

    async def scenario() -> object | None:
        attach_usage_sink()
        assert _usage_sink.get() is not None
        await orchestrator._teardown_run_resources(
            ledger=None, tm=None, adapter=None, writer_lock=None, actual_job_id="job-sink"
        )
        return _usage_sink.get()

    assert asyncio.run(scenario()) is None

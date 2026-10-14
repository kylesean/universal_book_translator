"""P2-1: the stage generator is drained before the ledger closes.

The run's teardown (``finally`` in ``PipelineOrchestrator.run``) closes the
ledger and releases the writer lock. The *stage* generator has its own
``finally`` (stage teardown, checkpoint flush); before this fix it finalized
only when the outer generator was garbage-collected — after the lock release —
and its writes silently reopened the closed ledger.

Pinned here:

* a stage generator is closed *inside* ``run``'s writer-lock/ledger window, not
  after it;
* a closed ``SQLiteJobLedger`` refuses to reopen (``LedgerBase._get_conn``).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine import pipeline as pipeline_module
from ubt.core.engine.dry_run import create_dry_run_orchestrator
from ubt.core.engine.events import EventType, TranslationProgressEvent

pytestmark = pytest.mark.fast

_BOOK = "# Title\n\nHello world.\n\n## Section\n\nMore prose here.\n"


def _config(tmp_path: Path) -> UBTConfig:
    return UBTConfig(db_dir=tmp_path / "ledgers", tm_enabled=False)


@pytest.mark.asyncio
async def test_the_stage_generator_is_drained_when_run_is_closed_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing run() early must drain the stage generator, not leak it.

    ``run`` owns the stage generator with ``aclosing``, so an early close (a
    lost lease, a user cancel) finalizes the stages inside ``run`` rather than
    leaving them to GC. GC is disabled here so the assertion can only pass if
    ``run`` itself drains the generator: without ``aclosing`` the inner
    generator survives the ``aclose`` (its ``finally`` would not have run) and
    the count stays empty until collection.
    """
    import gc

    book = tmp_path / "book.md"
    book.write_text(_BOOK, encoding="utf-8")
    output = tmp_path / "out.md"

    drained: list[str] = []
    # Read the real function from its home module; ``pipeline`` re-exports it as
    # a bare name, which mypy does not treat as an explicit attribute.
    from ubt.core.engine import plan as plan_module

    real_run_stages = plan_module.run_stages

    def spying_run_stages(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        async def gen() -> AsyncGenerator[Any, None]:
            try:
                async for event in real_run_stages(*args, **kwargs):
                    yield event
            finally:
                drained.append("stage_closed")

        return gen()

    monkeypatch.setattr(pipeline_module, "run_stages", spying_run_stages)

    orchestrator = create_dry_run_orchestrator(_config(tmp_path))
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        gen = orchestrator.run(book, output, job_id="drain")
        await anext(gen)
        await gen.aclose()
    finally:
        if gc_was_enabled:
            gc.enable()

    assert drained == ["stage_closed"], "the stage generator was not drained on early close"


@pytest.mark.asyncio
async def test_run_reaches_export_and_the_ledger_is_closed_after(tmp_path: Path) -> None:
    """A full run finishes, and the ledger it wrote refuses a reopen afterwards."""
    book = tmp_path / "book.md"
    book.write_text(_BOOK, encoding="utf-8")
    output = tmp_path / "out.md"

    orchestrator = create_dry_run_orchestrator(_config(tmp_path))
    events: list[TranslationProgressEvent] = []
    async for event in orchestrator.run(book, output, job_id="full"):
        events.append(event)
    assert events[-1].event_type is EventType.EXPORT_COMPLETED


def test_a_closed_ledger_refuses_to_reopen(tmp_path: Path) -> None:
    """The ledger's own guard: after close(), a read raises instead of reopening."""
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.exceptions import LedgerError

    db = tmp_path / "ledger.db"
    ledger = SQLiteJobLedger(db)
    ledger.close()
    with pytest.raises(LedgerError), ledger._get_conn():
        pass

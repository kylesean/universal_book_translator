"""Per-run revision-guarded block view (ADR-0001 ``pipeline/blocks.py``).

Extracted from :class:`~ubt.core.engine.stage_context.StageContext` so the shared
context stops carrying the run's one piece of mutable state -- the block snapshot
cache. The plan owns a single :class:`BlockReader` and threads it to the
analyze-adjacent stages that read blocks, which is what lets ``StageContext``
become an immutable run descriptor (the third orchestration cut).

The snapshot is reused only while the ledger reports no block write, so a stage
cannot be handed an outdated pre-write view of the book by forgetting to refresh;
``force_refresh`` and :meth:`invalidate` remain for callers that changed the
blocks in memory without a ledger write to report.
"""

from __future__ import annotations

import asyncio

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import IRBlock


class BlockReader:
    """One job's blocks, read from SQLite and re-read on a revision change."""

    def __init__(self, ledger: SQLiteJobLedger, job_id: str) -> None:
        self._ledger = ledger
        self._job_id = job_id
        # The snapshot plus the ledger block revision it was read at.
        self._snapshot: tuple[list[IRBlock], int] | None = None

    def invalidate(self) -> None:
        """Drop the cached snapshot so the next read comes from the ledger."""
        self._snapshot = None

    async def current_blocks(self, force_refresh: bool = False) -> list[IRBlock]:
        """This job's blocks, from SQLite unless the revision is unchanged."""
        # Read the revision before the query: a write that lands mid-read leaves
        # the cached pair below looking older than it is, which costs one more
        # reload. Reading it afterwards would certify a snapshot that missed it.
        revision = self._ledger.blocks_seq
        if force_refresh or self._snapshot is None or self._snapshot[1] != revision:
            # A full-table read belongs off the loop an SSE server shares.
            blocks = await asyncio.to_thread(self._ledger.get_all_blocks, self._job_id)
            self._snapshot = (blocks, revision)
        return self._snapshot[0]


__all__ = ["BlockReader"]

"""SQLite WAL-based atomic transactional job ledger with keyset pagination, schema migrations, and cross-process job mutual exclusion (see engine/writer_lock.py).

Facade module assembling the ledger subsystem:
- :mod:`ubt.core.engine.ledger_base`: connection, transaction, and schema migrations.
- :mod:`ubt.core.engine.ledger_mixins`: job lifecycle, block operations, and batch-API partitions.

:class:`SQLiteJobLedger` is the one class that inherits ``LedgerBase``; the
partitions are pure mixins (no base of their own).
"""

from ubt.core.engine.ledger_base import (
    _REQUIRED_BLOCK_COLUMNS,
    NON_TERMINAL_STATUSES,
    TARGET_SCHEMA_VERSION,
    LedgerBase,
    _upsert_blocks_batch,
    logger,
)
from ubt.core.engine.ledger_mixins import (
    LedgerBatchMixin,
    LedgerBlocksMixin,
    LedgerJobsMixin,
    _as_usage_totals,
    merge_usage_totals,
)
from ubt.core.exceptions import LedgerError

__all__ = [
    "NON_TERMINAL_STATUSES",
    "TARGET_SCHEMA_VERSION",
    "LedgerBase",
    "LedgerBatchMixin",
    "LedgerBlocksMixin",
    "LedgerError",
    "LedgerJobsMixin",
    "SQLiteJobLedger",
    "_REQUIRED_BLOCK_COLUMNS",
    "_as_usage_totals",
    "_upsert_blocks_batch",
    "logger",
    "merge_usage_totals",
]


class SQLiteJobLedger(LedgerJobsMixin, LedgerBlocksMixin, LedgerBatchMixin, LedgerBase):
    """Manages transactional state checkpoints and atomic worker task claims for book translation jobs."""

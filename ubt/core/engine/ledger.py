"""SQLite WAL-based atomic transactional job ledger with keyset pagination, a
flat baseline schema, and cross-process job mutual exclusion
(see :mod:`ubt.core.engine.writer_lock`).

``SQLiteJobLedger`` is the one class that inherits :class:`LedgerBase`. The
domain partitions in :mod:`ubt.core.engine.ledger_mixins` extend ``_LedgerHost``
(``LedgerBase`` under ``TYPE_CHECKING``, plain ``object`` at runtime) rather
than the base itself, so connection and transaction scope live in exactly one
place.
"""

from ubt.core.engine.ledger_base import (
    NON_TERMINAL_STATUSES,
    LedgerBase,
)
from ubt.core.engine.ledger_mixins import (
    LedgerBatchMixin,
    LedgerBlocksMixin,
    LedgerJobsMixin,
    _as_usage_totals,
    merge_usage_totals,
)

__all__ = [
    "NON_TERMINAL_STATUSES",
    "SQLiteJobLedger",
    "_as_usage_totals",
    "merge_usage_totals",
]


class SQLiteJobLedger(LedgerJobsMixin, LedgerBlocksMixin, LedgerBatchMixin, LedgerBase):
    """Manages transactional state checkpoints and atomic worker task claims for book translation jobs."""

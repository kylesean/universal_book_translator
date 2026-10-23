"""The run's source document, indexed once and shared by every consumer.

Two stages need the book's whole source text: the draft stage teaches its
FastPass filter the document's shape (so a table label in capitals is not
demanded verbatim in the target), and the quality gate teaches its QE runner
the same thing. Both used to rebuild it from a full-table scan **per chapter**,
and the omission gate then re-derived its one document-wide answer from that raw
text **per block** — 143 ms/block, 5581 s across a 20k-block run, measured.

The text is immutable once ingest finishes (no stage ever UPDATEs
``blocks.source_text``), so it is loaded once per run here and every later
caller gets the same string and the same
:class:`~ubt.core.qe.omission.DocumentVocabulary`.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from ubt.core.qe.omission import DocumentVocabulary, document_vocabulary

if TYPE_CHECKING:
    from ubt.core.engine.ledger import SQLiteJobLedger


class SourceDocumentIndex:
    """Lazily loaded, thread-safe view of one job's source text.

    ``load`` is idempotent and cheap after the first call. It holds no lock
    while loading from a second thread: the second caller waits on the lock and
    then finds the value already built, so a chapter-per-worker pipeline still
    performs exactly one full read.

    Mutable on purpose. :class:`~ubt.core.engine.services.RunServices` is frozen
    so a stage cannot rebind a *service*, but the index is a cache whose whole
    purpose is to fill in after the service graph is built (the ledger's blocks
    only exist once ingest has run).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._text: str | None = None
        self._vocabulary: DocumentVocabulary | None = None

    @property
    def bound(self) -> bool:
        """Whether the text has been loaded (no I/O, safe from any thread)."""
        return self._text is not None

    def load(self, ledger: SQLiteJobLedger, job_id: str) -> None:
        """Read the job's source text once; later calls are no-ops.

        Uses ``fetch_source_texts`` (a bare column read, no Pydantic model per
        row): the full ``get_all_blocks`` costs 346 ms on a 20k-block book of
        which 329 ms is :class:`IRBlock` construction neither caller uses.
        """
        if self._text is not None:
            return
        with self._lock:
            if self._text is not None:
                return
            texts = ledger.fetch_source_texts(job_id)
            self._text = "\n".join(texts)
            self._vocabulary = document_vocabulary(self._text)

    def text(self) -> str:
        """The joined source text of every block, in spine order."""
        return self._text or ""

    def vocabulary(self) -> DocumentVocabulary:
        """The document's indexed ordinary-case words (empty when unbound)."""
        return self._vocabulary or DocumentVocabulary(frozenset())


__all__ = ["SourceDocumentIndex"]

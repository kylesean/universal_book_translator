"""UBT Domain Exceptions hierarchy."""

from typing import Any


class UBTError(Exception):
    """Base exception for all Universal Book Translator domain errors."""

    def __init__(
        self,
        message: str,
        doc_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.doc_id = doc_id
        self.details = details or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.doc_id:
            base += f" [doc_id={self.doc_id}]"
        if self.details:
            base += f" [details={self.details}]"
        return base


class DocumentParseError(UBTError):
    """Raised when document parsing fails or IR cannot be extracted."""


class UnsupportedDocumentFormatError(DocumentParseError):
    """Raised when an input document file format has no registered adapter."""


class OptionalDependencyError(UBTError):
    """Raised when a feature's optional extra is not installed.

    Deliberately a normal, catchable exception rather than ``SystemExit``:
    importing a module is not the same act as running a command, so a library
    caller (an agent framework probing which tools exist, a test harness) must
    be able to handle the missing extra without its own process exiting. The
    console-script entry points convert this back into a clean exit.
    """


class ModelProviderError(UBTError):
    """Raised on LLM API timeouts, 429 rate limit exhaustion, or provider failures."""


class MTQEEvaluationError(UBTError):
    """Raised when local MTQE model scoring fails."""


class IntegrityViolationError(UBTError):
    """Raised when translation bible consistency or HTML delta validation detects severe corruption."""


class JobInterruptedError(UBTError):
    """Raised when job is actively cancelled or interrupted."""


class BudgetExceededError(UBTError):
    """Raised when the run's billed cost passes its ``budget_usd`` cap.

    A hard stop by design (see ``budget_violation``): callers that convert
    failures into fallbacks must let this one through, or the cap becomes
    advisory exactly during the phases that spend the most.
    """


class LeaseLostError(UBTError):
    """Raised when another worker took this job over because its lease expired.

    Deliberately distinct from :class:`JobInterruptedError`: the losing worker
    must stop *without* writing a terminal state, because the job is still
    running under the new owner and a terminal write here would be refused
    there instead.
    """


class LedgerError(UBTError):
    """Raised on SQLite ledger transaction failure or database corruption."""


class LedgerWriterLockConflictError(LedgerError):
    """Raised when an exclusive ledger writer lock cannot be acquired because another process holds it."""


class QueueDepthExceededError(UBTError):
    """Raised when ``JobQueue.enqueue`` would push the queue past its depth cap.

    Refusing at submit time is the only place the growth can be stopped: the
    worker drains at LLM speed, so an uncapped intake fills the queue's SQLite
    file faster than anything downstream can retire it.
    """


class ServerCapacityError(UBTError):
    """Raised when the server has reached maximum concurrent running jobs."""


class OutputPathConflictError(UBTError):
    """A live job already owns this output path.

    The filesystem ``exists()`` check at submit cannot see the path until the
    owner's export writes it, so the in-memory job map rejects the second claim.
    """


class RenderBlocksNotImplementedError(NotImplementedError):
    """The base ``render_blocks`` stub signaling the adapter does not render.

    Deliberately a distinct type so that export stage fallback logic does
    not accidentally swallow unrelated ``NotImplementedError`` exceptions
    raised within an adapter's internal execution path.
    """

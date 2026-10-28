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


class InvalidBlockStatusTransitionError(UBTError, ValueError):
    """Raised when an IRBlock lifecycle transition violates the state machine rules."""

    def __init__(
        self,
        from_status: str,
        to_status: str,
        block_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        msg = f"Invalid block status transition: '{from_status}' -> '{to_status}'"
        if block_id:
            msg += f" for block {block_id!r}"
        if reason:
            msg += f" ({reason})"
        super().__init__(
            msg,
            details={
                "from_status": from_status,
                "to_status": to_status,
                "block_id": block_id,
            },
        )
        self.from_status = from_status
        self.to_status = to_status
        self.block_id = block_id


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

    @property
    def lease_expires_at(self) -> float | None:
        """The UNIX timestamp when the current holder's lease expires, if known."""
        raw = self.details.get("lease_expires_at")
        try:
            return float(raw) if raw is not None else None
        except (ValueError, TypeError):
            return None


class LedgerWriterLeaseLostError(LedgerError):
    """Raised when this process's writer lease was reclaimed by another writer.

    A lease is an *expiring* claim: if this process stalls long enough for the
    lease to lapse (a long GC pause, swap, a frozen container, an NFS hiccup),
    another process legitimately takes the job over. Continuing to write would
    then put two writers on one ledger — each paying the provider for the same
    blocks and overwriting the other's checkpoints.

    ``renew()`` used to return silently in that situation (it never checked
    whether its ``UPDATE`` matched a row), so the stalled process resumed
    writing as if it still owned the job. Raising turns the reclaim into a
    stop, which is the only safe response: the work this process already paid
    for is in the ledger, and the new owner will pick it up from there.
    """


class QueueDepthExceededError(UBTError):
    """Raised when ``JobQueue.enqueue`` would push the queue past its depth cap.

    Refusing at submit time is the only place the growth can be stopped: the
    worker drains at LLM speed, so an uncapped intake fills the queue's SQLite
    file faster than anything downstream can retire it.
    """


class ProviderUnreachableError(UBTError):
    """The fail-fast circuit tripped because the provider endpoint is dead.

    Every block failed with ``transport_unreachable`` (connection refused /
    DNS failure), so the job aborted to stop burning blocks against a dead
    endpoint. The message names the original transport error, and the class
    itself is operator-displayable: ``public_job_error`` must let it through
    so the console can show "cannot reach the translation service" instead of
    a generic "(see server logs)".
    """


class ServerCapacityError(UBTError):
    """Raised when the server has reached maximum concurrent running jobs."""


class RenderBlocksNotImplementedError(NotImplementedError):
    """The base ``render_blocks`` stub signaling the adapter does not render.

    Deliberately a distinct type so that export stage fallback logic does
    not accidentally swallow unrelated ``NotImplementedError`` exceptions
    raised within an adapter's internal execution path.
    """


#: Domain errors whose message is written to be SHOWN to the operator: each
#: names the knob to turn or the step to take (raise the budget, drop
#: --pages, drain the queue, install typst, enable OCR). Everything else may
#: echo provider bodies or host paths, so only its class name surfaces.
#:
#: ``DocumentParseError`` and its subclass belong here for the same reason: the
#: parser writes those messages *for the operator* ("every page lacks an
#: extractable text layer … enable OCR", "input document does not exist", "PDF is
#: password-protected"). Folding them into "(see server logs)" left a failed job
#: showing only ``DocumentParseError`` — the operator could not tell a missing
#: file from an encrypted PDF from a missing Typst binary.
ACTIONABLE_ERROR_TYPES: tuple[type[Exception], ...] = (
    BudgetExceededError,
    DocumentParseError,
    UnsupportedDocumentFormatError,
    IntegrityViolationError,
    QueueDepthExceededError,
    ProviderUnreachableError,
)


def public_job_error(exc: Exception, job_id: str) -> str:
    """The one message a job terminal record shows the operator.

    Shared by the queue worker and the API's embedded manager so a budget
    exhaustion says \"raise the cap and resume with the same job id\" on both
    paths instead of the worker folding it into
    ``BudgetExceededError (see server logs)``. Full detail is always logged
    by the caller; this is the display surface only.
    """
    if isinstance(exc, ACTIONABLE_ERROR_TYPES):
        return f"{type(exc).__name__}: {exc}"
    return f"{type(exc).__name__} (see server logs; job_id={job_id})"

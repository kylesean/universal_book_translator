"""Base protocols and structured results for 0-Token validation.

A ``Protocol`` rather than a base class: the gates live in sibling modules and
are selected by the repair loop by shape alone, so inheritance would couple
every new validator to one hierarchy for no benefit.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Structured result returned by all deterministic content validators."""

    is_valid: bool
    error_code: str | None = None
    message: str | None = None
    #: The only action emitted today is ``"RETRY"`` (html_delta, consistency).
    #: ``FALLBACK`` / ``DROP`` / ``ACCEPT`` are vocabulary reserved for the
    #: repair loop; no validator produces them yet.
    suggested_action: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(is_valid=True)

    @classmethod
    def failure(
        cls,
        error_code: str,
        message: str,
        suggested_action: str = "RETRY",
        details: dict[str, Any] | None = None,
    ) -> "ValidationResult":
        return cls(
            is_valid=False,
            error_code=error_code,
            message=message,
            suggested_action=suggested_action,
            details=details or {},
        )


@runtime_checkable
class ContentValidator(Protocol):
    """Protocol for validators that are stateless *per call*.

    Implementations may do setup work in ``__init__`` (the glossary validator
    sorts and scope-resolves its terms there); what the contract requires is
    that ``validate`` itself mutates no state, so a shared instance can serve
    concurrent blocks.
    """

    def validate(self, original: str, translated: str) -> ValidationResult: ...

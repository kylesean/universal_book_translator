"""Base protocols and structured results for 0-Token validation."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Structured result returned by all deterministic content validators."""

    is_valid: bool
    error_code: str | None = None
    message: str | None = None
    suggested_action: str | None = None  # e.g., "RETRY", "FALLBACK", "DROP", "ACCEPT"
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls) -> "ValidationResult":
        """Convenience constructor for passed validation."""
        return cls(is_valid=True)

    @classmethod
    def failure(
        cls,
        error_code: str,
        message: str,
        suggested_action: str = "RETRY",
        details: dict[str, Any] | None = None,
    ) -> "ValidationResult":
        """Convenience constructor for failed validation."""
        return cls(
            is_valid=False,
            error_code=error_code,
            message=message,
            suggested_action=suggested_action,
            details=details or {},
        )


@runtime_checkable
class ContentValidator(Protocol):
    """Protocol for stateless, pure-function validators."""

    def validate(self, original: str, translated: str) -> ValidationResult:
        """Validate translated content against original content."""
        ...

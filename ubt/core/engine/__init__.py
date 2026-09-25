"""Core execution engine, event contracts, and SQLite state ledger."""

from typing import TYPE_CHECKING, Any

from ubt.core.engine.events import (
    EventType,
    TranslationProgressEvent,
)
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.reporter import (
    QualityReport,
    build_quality_report,
    save_quality_report,
)

if TYPE_CHECKING:
    from ubt.core.engine.pipeline import PipelineOrchestrator
    from ubt.core.engine.repair_loop import RepairLoop

__all__ = [
    "EventType",
    "PipelineOrchestrator",
    "QualityReport",
    "RepairLoop",
    "SQLiteJobLedger",
    "TranslationProgressEvent",
    "build_quality_report",
    "save_quality_report",
]


def __getattr__(name: str) -> Any:
    if name == "PipelineOrchestrator":
        from ubt.core.engine.pipeline import PipelineOrchestrator

        return PipelineOrchestrator
    if name == "RepairLoop":
        from ubt.core.engine.repair_loop import RepairLoop

        return RepairLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # ``__getattr__`` alone hides the lazy names from ``dir()``, pydoc and IDE
    # completion; advertise them explicitly.
    return sorted(set(globals()) | set(__all__))

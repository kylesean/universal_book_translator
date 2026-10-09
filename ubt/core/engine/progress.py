"""One projection of the engine's telemetry, shared by every delivery surface:
the durable queue worker, the API's in-memory record, its ``/status``
branches, and the MCP server's job table. Duplicated per-surface
enumerations of the same field names drift the moment one side adds a
metric, so all surfaces derive their snapshot here.

The persisted key names are an on-disk contract (``job_queue.progress_json``
rows survive upgrades).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.job_options import artifact_and_report_paths

if TYPE_CHECKING:
    from ubt.core.engine.ledger import SQLiteJobLedger

# Fields that only exist once a render has named them, as opposed to the
# counters, which every row reports from the first event.
ARTIFACT_KEYS = ("output_file", "report_file", "visual_report_file")

#: Pipeline stage vocabulary, coarse enough to be one stepper in the console and
#: stable enough to be an on-disk contract (the value lands in ``progress_json``).
PipelineStage = Literal[
    "extract",
    "segment",
    "tm",
    "translate",
    "qe",
    "repair",
    "render",
    "verify",
    "package",
]

#: Event -> stage. The engine's event vocabulary is finer than the console's
#: stepper (C-track, chapter streaming and triage all report as their nearest
#: step), so the console can highlight a real pipeline position instead of
#: guessing one from ``progress_percent`` thresholds — a guess that marked
#: "Typst" as active for a job still in repair, because both sit past 75%.
_STAGE_BY_EVENT: dict[EventType, PipelineStage] = {
    EventType.JOB_STARTED: "extract",
    EventType.PREPROCESSING_DONE: "segment",
    EventType.BIBLE_EXTRACTED: "tm",
    EventType.MODE_ADVISED: "segment",
    EventType.DRAFT_BATCH_COMPLETED: "translate",
    EventType.CTEXT_COMPLETED: "translate",
    EventType.CHAPTER_COMPLETED: "translate",
    EventType.MTQE_EVALUATED: "qe",
    EventType.REPAIR_BATCH_COMPLETED: "repair",
    EventType.TRIAGE_COMPLETED: "verify",
    EventType.EXPORT_COMPLETED: "package",
}

_PIPELINE_STAGES: frozenset[str] = frozenset(_STAGE_BY_EVENT.values())


def _processed_blocks(completed: int, failed: int, needs_human: int, blocked_human: int) -> int:
    """Blocks that reached a *final disposition* — the percent numerator.

    The quality report's ``summary.completed_blocks`` (mtqe_passed + repaired)
    alone is too narrow for a progress bar: a successful run whose blocks were
    all routed to the human queue or quarantined reports 0 completed, and a bar
    reading only completed blocks would fall from 100% at draft time back to 0%
    at export. Counting
    every terminal status keeps the numerator honest *and* non-decreasing:
    within one run the pipeline only moves blocks into terminal states (repair
    selection skips finalized blocks) or between them (triage/render flips),
    so this value never regresses mid-run. Drafted/repair-pending blocks are
    in-flight, not done, and are deliberately excluded.
    """
    return completed + failed + needs_human + blocked_human


def _progress_percent(processed: int, total: int) -> float:
    """0-100 completion percent; 0.0 while the total is still unknown."""
    if total <= 0:
        return 0.0
    return round(min(1.0, processed / total) * 100, 1)


class ProgressSnapshot(BaseModel):
    """Block counters, live QE metrics and the artifacts a finished job left."""

    model_config = ConfigDict(frozen=True)

    total_blocks: int = 0
    completed_blocks: int = 0
    repaired_blocks: int = 0
    failed_blocks: int = 0
    needs_human_blocks: int = 0
    blocked_human_blocks: int = 0
    # Blocks that reached ANY final disposition (completed + failed +
    # needs_human + blocked_human): the numerator of ``progress_percent``.
    # ``completed_blocks`` keeps its narrow ledger meaning — it is the same
    # number the quality report's ``summary.completed_blocks`` reads — while
    # this counter answers "how much of the book has been fully processed",
    # which is what a progress bar must show.
    processed_blocks: int = 0
    # 0-100 completion percent derived from ``processed_blocks``. Monotone
    # non-decreasing within a run: drafted work is never credited (the old
    # numerator counted drafts, so the bar hit 100% during the draft stage and
    # fell back once drafts were finalized elsewhere).
    progress_percent: float = 0.0
    current_avg_qe: float = 0.0
    bottom_15_avg_qe: float = 0.0
    # None means "no usage reported yet", never a fabricated zero: a polling
    # client reads 0.0 as "confirmed no spend".
    estimated_cost_usd: float | None = None
    #: The pipeline step the last event reported, or ``None`` before the first
    #: event. Derived from the engine's own event type (``_STAGE_BY_EVENT``),
    #: not from ``progress_percent``: the console highlights a real stage.
    stage: PipelineStage | None = None
    #: The last event's human-readable message. A progress *log* line, kept for
    #: the console's stream drawer. Path-bearing messages (export naming its
    #: artifact) are dropped here — the wire form must not leak host layout; the
    #: artifact keys above carry the basename instead.
    message: str | None = None
    output_file: str | None = None
    report_file: str | None = None
    visual_report_file: str | None = None

    @classmethod
    def from_event(cls, event: TranslationProgressEvent) -> ProgressSnapshot:
        """Fold one engine event, resolving an export event's artifact triple."""
        files: dict[str, str] = {}
        if event.event_type is EventType.EXPORT_COMPLETED:
            artifact = event.artifact_path or event.message
            # A malformed export event (no artifact location) must not fail the
            # job: the translation is done, only the pointer is missing.
            if artifact:
                output_path, quality_report, visual_report = artifact_and_report_paths(artifact)
                files["output_file"] = str(output_path)
                if quality_report is not None:
                    files["report_file"] = str(quality_report)
                if visual_report is not None:
                    files["visual_report_file"] = str(visual_report)
        processed = _processed_blocks(
            event.completed_blocks,
            event.failed_blocks,
            event.needs_human_blocks,
            event.blocked_human_blocks,
        )
        # EXPORT_COMPLETED's message names the output path; the console gets the
        # basename through ``output_file`` instead, so the raw text is dropped
        # rather than forwarded to the wire.
        message = (
            None if event.event_type is EventType.EXPORT_COMPLETED else (event.message or None)
        )
        return cls(
            total_blocks=event.total_blocks,
            completed_blocks=event.completed_blocks,
            repaired_blocks=event.repaired_blocks,
            failed_blocks=event.failed_blocks,
            needs_human_blocks=event.needs_human_blocks,
            blocked_human_blocks=event.blocked_human_blocks,
            processed_blocks=processed,
            progress_percent=_progress_percent(processed, event.total_blocks),
            current_avg_qe=event.current_avg_qe,
            bottom_15_avg_qe=event.bottom_15_avg_qe,
            estimated_cost_usd=event.estimated_cost_usd,
            stage=_STAGE_BY_EVENT.get(event.event_type),
            message=message,
            **files,
        )

    @classmethod
    def from_ledger(
        cls,
        stats: Mapping[str, Any],
        metadata: Callable[[str], Any] | None = None,
    ) -> ProgressSnapshot:
        """Rebuild from a persisted job ledger's own (differently named) columns.

        ``metadata`` reads one ``job_metadata`` key; absent means the store
        predates metadata persistence and every optional field stays unknown.
        """

        def meta(key: str) -> Any:
            return None if metadata is None else metadata(key)

        failed = int(stats.get("failed", 0))
        needs_human = int(stats.get("needs_human", 0))
        blocked_human = int(stats.get("blocked_human", 0))
        completed = int(stats.get("completed", 0))
        total = int(stats.get("total", 0))
        processed = _processed_blocks(completed, failed, needs_human, blocked_human)
        stage_raw = meta("stage")
        stage = stage_raw if stage_raw in _PIPELINE_STAGES else None
        return cls(
            total_blocks=total,
            completed_blocks=completed,
            repaired_blocks=int(stats.get("repaired", 0)),
            failed_blocks=failed,
            needs_human_blocks=needs_human,
            blocked_human_blocks=blocked_human,
            processed_blocks=processed,
            progress_percent=_progress_percent(processed, total),
            current_avg_qe=float(stats.get("avg_qe_score", 0.0)),
            bottom_15_avg_qe=float(stats.get("bottom_15_avg_qe", 0.0)),
            estimated_cost_usd=_as_float(meta("estimated_cost_usd")),
            stage=stage,
            message=_as_str(meta("message")),
            output_file=_as_str(meta("output_file")),
            report_file=_as_str(meta("report_file")),
            visual_report_file=_as_str(meta("visual_report_file")),
        )

    def to_payload(self) -> dict[str, Any]:
        """Queue-row form: counters always present, artifact paths only when known.

        The nine counters plus the ``processed_blocks``/``progress_percent``
        projection keep their keys even when a value is unknown
        (``estimated_cost_usd: null`` before the first usage report), because
        that is the shape rows have always had and the queue-mode SSE frame
        forwards this dict to clients verbatim. The three artifact paths are
        omitted until an export names them. Rows written before
        ``processed_blocks``/``progress_percent`` existed stay readable: both
        default to 0 and are recomputed by ``from_ledger`` on the next read.
        """
        payload = self.model_dump()
        for key in ARTIFACT_KEYS:
            if payload[key] is None:
                del payload[key]
        return payload


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)


def persist_progress_metadata(
    event: TranslationProgressEvent,
    ledger_or_path: SQLiteJobLedger | Path,
    job_id: str,
) -> None:
    """Persist a finished run's artifact triple and cost into its ledger.

    The one fold both completion hooks share (the REST manager's in-lock hook
    and the queue worker's): ``ProgressSnapshot.from_event`` is the same
    projection the SSE record shows, so what is persisted cannot drift from
    what subscribers were told. A missing ledger file is a no-op — the run
    never got one.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger

    progress = ProgressSnapshot.from_event(event)
    persisted_keys = (*ARTIFACT_KEYS, "estimated_cost_usd", "stage", "message")
    if isinstance(ledger_or_path, SQLiteJobLedger):
        for metadata_key in persisted_keys:
            value = getattr(progress, metadata_key)
            if value is not None:
                ledger_or_path.set_job_metadata_value(job_id, metadata_key, value)
        return

    ledger_path = Path(ledger_or_path)
    if not ledger_path.exists():
        return
    with SQLiteJobLedger(ledger_path) as ldg:
        for metadata_key in persisted_keys:
            value = getattr(progress, metadata_key)
            if value is not None:
                ldg.set_job_metadata_value(job_id, metadata_key, value)


__all__ = ["ARTIFACT_KEYS", "ProgressSnapshot", "persist_progress_metadata"]

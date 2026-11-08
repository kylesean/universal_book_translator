"""Standard asynchronous progress and lifecycle event contracts."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    """Lifecycle event types emitted by the translation engine."""

    JOB_STARTED = "job_started"
    PREPROCESSING_DONE = "preprocessing_done"
    BIBLE_EXTRACTED = "bible_extracted"
    DRAFT_BATCH_COMPLETED = "draft_batch_completed"
    MTQE_EVALUATED = "mtqe_evaluated"
    REPAIR_STARTED = "repair_started"
    REPAIR_PROGRESS = "repair_progress"
    REPAIR_BATCH_COMPLETED = "repair_batch_completed"
    RENDER_STARTED = "render_started"
    TRIAGE_COMPLETED = "triage_completed"
    CTEXT_COMPLETED = "ctext_completed"
    CHAPTER_COMPLETED = "chapter_completed"
    MODE_ADVISED = "mode_advised"
    COMPANIONS_EXPORTED = "companions_exported"
    EXPORT_COMPLETED = "export_completed"
    PIPELINE_FAILED = "pipeline_failed"


class TranslationProgressEvent(BaseModel):
    """The progress event payload the engine emits outward."""

    model_config = ConfigDict(frozen=True)

    event_type: EventType
    job_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    total_blocks: int
    #: Blocks excluded from ``total_blocks`` because the skip rule marked them
    #: ``mtqe_passed`` without translating them — images, bibliography entries,
    #: bylines, short headings: whatever the policy keeps verbatim. Reported so a
    #: panel can say "99 translatable, 49 preserved" instead of silently showing a
    #: smaller book than the source.
    skipped_blocks: int = 0
    completed_blocks: int
    drafted_blocks: int = 0
    #: Drafted blocks queued for targeted repair. The event query counts them
    #: with ``drafted_blocks`` so a stage transition cannot report a progress
    #: regression.
    repair_pending_blocks: int = 0
    repaired_blocks: int = 0
    failed_blocks: int = 0
    needs_human_blocks: int = 0
    blocked_human_blocks: int = 0

    current_avg_qe: float = 0.0
    bottom_15_avg_qe: float = 0.0
    #: False means QE never scored anything (disabled or heuristic-only), so
    #: ``current_avg_qe`` is a placeholder 0.0 and not a real measurement — the
    #: same fact the ledger's ``qe_measured`` column carries and the CLI already
    #: reads. Without it, an API consumer cannot tell "QE off" from "scored 0".
    qe_measured: bool = False
    # Estimated token spend in USD; None = no usage reported yet (never a
    # fabricated 0.0, which a polling client reads as "confirmed zero spend").
    estimated_cost_usd: float | None = None
    # Fraction of prompt tokens served from the provider cache
    # (verifies the static-prefix TCO lever; 0.0 = unmeasured)
    cache_hit_rate: float = 0.0

    message: str = ""
    active_block_id: str | None = None
    # Machine-readable artifact location (EXPORT_COMPLETED). Keeps ``message``
    # free to be human-readable without breaking file-path consumers.
    artifact_path: str | None = None

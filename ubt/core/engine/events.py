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
    REPAIR_BATCH_COMPLETED = "repair_batch_completed"
    TRIAGE_COMPLETED = "triage_completed"
    CTEXT_COMPLETED = "ctext_completed"
    CHAPTER_COMPLETED = "chapter_completed"
    MODE_ADVISED = "mode_advised"
    EXPORT_COMPLETED = "export_completed"
    PIPELINE_FAILED = "pipeline_failed"


class TranslationProgressEvent(BaseModel):
    """Standard progress event payload emitted outward by Level 0 Core engine."""

    model_config = ConfigDict(frozen=True)

    event_type: EventType
    job_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Block counters
    total_blocks: int
    completed_blocks: int
    drafted_blocks: int = 0
    repaired_blocks: int = 0
    failed_blocks: int = 0
    needs_human_blocks: int = 0
    blocked_human_blocks: int = 0

    # Real-time metrics
    current_avg_qe: float = 0.0  # Current book-level average MTQE score
    bottom_15_avg_qe: float = 0.0  # Lowest 15% blocks average score
    # Estimated token spend in USD; None = no usage reported yet (never a
    # fabricated 0.0, which a polling client reads as "confirmed zero spend").
    estimated_cost_usd: float | None = None
    # Fraction of prompt tokens served from the provider cache
    # (verifies the static-prefix TCO lever; 0.0 = unmeasured)
    cache_hit_rate: float = 0.0

    # Detail info
    message: str = ""
    active_block_id: str | None = None
    # Machine-readable artifact location (EXPORT_COMPLETED). Keeps ``message``
    # free to be human-readable without breaking file-path consumers.
    artifact_path: str | None = None

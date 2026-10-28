"""Data models for document differential semantic AST comparison and incremental patching."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ubt.core.ir.models import BlockStatus, BlockType


class PatchBlockStatus(StrEnum):
    """Classification of an IRBlock in differential comparison."""

    UNCHANGED = "unchanged"  # Exactly identical content, 100% reusable
    MODIFIED = "modified"  # Semantically aligned but text altered (similarity >= threshold)
    ADDED = "added"  # Newly inserted block in revised edition
    DELETED = "deleted"  # Removed block from base edition


class BlockDiffItem(BaseModel):
    """Fine-grained semantic diff record for a single block."""

    model_config = ConfigDict(extra="ignore")

    status: PatchBlockStatus
    base_block_id: str | None = None
    revised_block_id: str | None = None
    spine_index: int = 0
    block_type: BlockType = BlockType.NARRATIVE
    base_source_text: str | None = None
    revised_source_text: str | None = None
    similarity: float = 0.0
    base_target_text: str | None = None
    base_status: BlockStatus | None = None
    diff_reason: str = ""


class PatchSummary(BaseModel):
    """Aggregate statistics and itemized diff for a document pair."""

    model_config = ConfigDict(extra="ignore")

    base_job_id: str | None = None
    base_source: str = ""
    revised_source: str = ""
    total_base_blocks: int = 0
    total_revised_blocks: int = 0
    unchanged_count: int = 0
    modified_count: int = 0
    added_count: int = 0
    deleted_count: int = 0
    reuse_ratio: float = 0.0
    estimated_tokens_saved: int = 0
    estimated_cost_saved_usd: float = 0.0
    diff_items: list[BlockDiffItem] = Field(default_factory=list)

"""Incremental document patching and semantic differencing subsystem."""

from ubt.core.patch.differ import (
    DocumentSemanticDiffer,
    estimate_block_tokens,
    normalize_source_text,
)
from ubt.core.patch.engine import BookPatchEngine, load_document_blocks
from ubt.core.patch.models import BlockDiffItem, PatchBlockStatus, PatchSummary

__all__ = [
    "BlockDiffItem",
    "BookPatchEngine",
    "DocumentSemanticDiffer",
    "PatchBlockStatus",
    "PatchSummary",
    "estimate_block_tokens",
    "load_document_blocks",
    "normalize_source_text",
]

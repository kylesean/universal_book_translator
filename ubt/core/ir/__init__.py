"""Unified Semantic Flow-Isolated Intermediate Representation (IR)."""

from ubt.core.ir.document_tree import Section, build_section_tree, section_texts
from ubt.core.ir.models import (
    BlockProvenance,
    BlockStatus,
    BlockType,
    BookManifest,
    BoundingBox,
    ChapterIR,
    ChapterMeta,
    FlowID,
    IRBlock,
    StyleMeta,
)
from ubt.core.ir.serializer import compute_file_sha256
from ubt.core.ir.state_machine import (
    can_transition,
    get_allowed_predecessors,
    get_valid_transitions,
    validate_status_transition,
)

__all__ = [
    "BlockProvenance",
    "BlockStatus",
    "BlockType",
    "BookManifest",
    "BoundingBox",
    "ChapterIR",
    "ChapterMeta",
    "FlowID",
    "IRBlock",
    "Section",
    "StyleMeta",
    "build_section_tree",
    "can_transition",
    "compute_file_sha256",
    "get_allowed_predecessors",
    "get_valid_transitions",
    "section_texts",
    "validate_status_transition",
]

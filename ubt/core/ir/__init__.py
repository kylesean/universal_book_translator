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
    "compute_file_sha256",
    "section_texts",
]

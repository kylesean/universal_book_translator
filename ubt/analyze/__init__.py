"""``ubt.analyze`` -- turning sources into the typed Document AST.

Phase 1 seeds this layer with a *bootstrap* bridge: the pipeline already knows
how to get ``IRBlock``s out of every format, so the bridge projects those into a
:class:`~ubt.model.ast.Document` and back. That lets the rest of the kernel
migrate to the typed model before a native per-format reader exists, and the
round-trip is the proof that the model loses no document structure.

A native reader (PDF/DOCX/EPUB -> Document directly, without the IRBlock
detour) lands later in Phase 1; it will join this package beside the bridge.
"""

from __future__ import annotations

from ubt.analyze.bridge import (
    block_from_element,
    blocks_from_document,
    document_from_blocks,
    element_from_block,
)
from ubt.analyze.reader_pdf import read_pdf

__all__ = [
    "block_from_element",
    "blocks_from_document",
    "document_from_blocks",
    "element_from_block",
    "read_pdf",
]

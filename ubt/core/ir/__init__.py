"""The IR layer: block models, lifecycle rules, fingerprints, section grouping.

Import surface the pipeline stages use. A block's *structure* is not defined
here — it is the typed ``element`` from ``ubt.model.ast`` — so this package
carries execution state, the transitions between them, and the views (sections)
derived from blocks, never a second document model.
"""

from ubt.core.ir.ast_protector import (
    DecodedPayload,
    DominantStyleTagger,
    SlotIntegrityValidator,
    SlotReport,
    TaggedPayload,
    TagSelfHealingDecoder,
)
from ubt.core.ir.document_tree import Section, build_section_tree
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
    "DecodedPayload",
    "DominantStyleTagger",
    "FlowID",
    "IRBlock",
    "Section",
    "SlotIntegrityValidator",
    "SlotReport",
    "StyleMeta",
    "TaggedPayload",
    "TagSelfHealingDecoder",
    "build_section_tree",
    "can_transition",
    "compute_file_sha256",
    "get_allowed_predecessors",
    "get_valid_transitions",
    "validate_status_transition",
]

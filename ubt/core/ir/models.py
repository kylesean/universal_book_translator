"""Semantic Flow-Isolated IR Data Models (Pydantic v2)."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ubt.core.ir.run_metadata import RunMetadata


class FlowID(StrEnum):
    """Semantic flow types: determines physical boundaries for sliding context windows."""

    MAIN_STORY = "main_story"  # Main narrative/dialogue body stream
    SIDEBAR_ASIDE = "sidebar_aside"  # Sidebar, textbook callout boxes, feature cards
    FOOTNOTE = "footnote"  # Footnotes and endnotes
    TABLE_GRID = "table_grid"  # Table cell text grid
    CAPTION = "caption"  # Figure and table captions / callouts


class BlockType(StrEnum):
    """Semantic type of IRBlock."""

    NARRATIVE = "narrative"  # Descriptive narrative paragraph
    DIALOGUE = "dialogue"  # Character dialogue
    HEADING = "heading"  # Section headings / chapter titles
    CODE = "code"  # Source code / terminal listings (triggers token masking)
    FORMULA = "formula"  # Math formulas (LaTeX protection)
    IMAGE = "image"  # Illustration or picture block
    TABLE = "table"  # Composite table
    LIST_ITEM = "list_item"  # Bulleted or numbered list item


class BlockStatus(StrEnum):
    """Lifecycle status of IRBlock."""

    PENDING = "pending"  # Waiting to be drafted
    DRAFTED = "drafted"  # Drafted by tier-1 cheaper model
    MTQE_PASSED = "mtqe_passed"  # Quality passed MTQE, released directly
    REPAIR_PENDING = "repair_pending"  # Below threshold, queued for targeted repair
    REPAIRED = "repaired"  # Repaired by tier-2 flagship model
    FAILED = "failed"  # Failed after retries / circuit-break
    NEEDS_HUMAN = "needs_human"  # MQM Major severity: shippable but queued for human PE
    BLOCKED_HUMAN = "blocked_human"  # MQM Critical unresolved: MUST NOT ship machine output


# (Code review): single source of truth for the lifecycle's terminal
# states. is_finalized, the ledger's NON_TERMINAL_STATUSES and the repair
# eligibility SQL all derive from this frozenset, so adding a status can never
# drift between the three former parallel definitions.
TERMINAL_STATUSES: frozenset[BlockStatus] = frozenset(
    {
        BlockStatus.MTQE_PASSED,
        BlockStatus.REPAIRED,
        BlockStatus.FAILED,
        BlockStatus.NEEDS_HUMAN,
        BlockStatus.BLOCKED_HUMAN,
    }
)


MQM_SEVERITY_LEVELS: tuple[str, ...] = ("critical", "major", "minor")


class LayoutRole(StrEnum):
    """Page-layout role of a block (document.v1 ``layout_role`` layer)."""

    BODY = "body"
    TITLE = "title"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    CAPTION = "caption"
    FOOTNOTE = "footnote"


class SemanticRole(StrEnum):
    """Semantic role of a block (document.v1 ``semantic_role`` layer)."""

    MAIN_TEXT = "main_text"
    ABSTRACT = "abstract"
    REFERENCE = "reference"
    METADATA = "metadata"
    AFFILIATION = "affiliation"
    UNKNOWN = "unknown"


class StructureRole(StrEnum):
    """Structural role of a block (document.v1 ``structure_role`` layer)."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    CODE = "code"


# Layout roles that never enter translation (policy derivation fallback).
_NON_TRANSLATABLE_LAYOUT_ROLES = frozenset(
    {
        LayoutRole.HEADER,
        LayoutRole.FOOTER,
        LayoutRole.PAGE_NUMBER,
    }
)
# Semantic roles that never enter translation (policy derivation fallback).
_NON_TRANSLATABLE_SEMANTIC_ROLES = frozenset(
    {
        SemanticRole.REFERENCE,
        SemanticRole.METADATA,
        SemanticRole.AFFILIATION,
    }
)
# Block types that keep origin unless an explicit verdict says otherwise.
_NON_TRANSLATABLE_BLOCK_TYPES = frozenset(
    {
        BlockType.FORMULA,
        BlockType.CODE,
        BlockType.IMAGE,
        BlockType.TABLE,
    }
)


class BoundingBox(BaseModel):
    """Bounding box for fixed layout formats (e.g. PDF)."""

    model_config = ConfigDict(frozen=True)

    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class StyleMeta(BaseModel):
    """Typography and layout metadata."""

    model_config = ConfigDict(extra="ignore")

    font_name: str | None = None
    font_size: float | None = None
    color_hex: str | None = None
    alignment: str | None = None
    line_height: float | None = None


class IRBlock(BaseModel):
    """Atomic content block in Universal Book Translator IR."""

    model_config = ConfigDict(extra="ignore")

    id: str  # Globally unique block identifier (e.g., ch03#p012 or doc_flow_index)
    flow_id: FlowID = FlowID.MAIN_STORY  # Flow boundary partition key
    spine_index: int  # Strict document physical reading order index
    block_type: BlockType = BlockType.NARRATIVE
    bbox: BoundingBox | None = None
    style: StyleMeta | None = None

    # Content payloads
    source_text: str  # Original source text
    target_text: str | None = None  # Final translated text
    draft_text: str | None = None  # First draft backup

    # Status and quality metrics
    status: BlockStatus = BlockStatus.PENDING
    skip_translate: bool = False  # Code or formulas skipped from translation
    glossary_hits: list[str] = Field(default_factory=list)  # Matched glossary terms
    mtqe_score: float | None = None  # Quality metric score (0.00 ~ 1.00)
    repair_rounds: int = 0  # Completed targeted repair iterations (max 2)
    error_flags: list[str] = Field(default_factory=list)  # e.g., ["html_attr_mismatch"]

    # MQM severity triage
    mqm_severity: str | None = None  # "critical" | "major" | "minor" (None = not triaged)
    mqm_spans: list[dict[str, Any]] = Field(default_factory=list)  # Serialized MQMErrorSpan dicts

    # Document-v1 seven-layer contract extension (all optional). Layer mapping:
    #   geometry → bbox, content → source_text/block_type,
    #   layout_role/semantic_role/structure_role → below,
    #   policy → policy_translate/policy_reason (+ skip_translate fallback),
    #   provenance → provenance dict (parser/provider/adapter that produced it).
    layout_role: LayoutRole | None = None
    semantic_role: SemanticRole | None = None
    structure_role: StructureRole | None = None
    policy_translate: bool | None = None  # None = undecided, fall back to skip_translate
    policy_reason: str | None = None  # Required when policy_translate is False
    provenance: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_finalized(self) -> bool:
        """Returns True if the block has reached a terminal status."""
        return self.status in TERMINAL_STATUSES

    def effective_should_translate(self) -> bool:
        """Single translation-gate entry point.

        Explicit policy verdict wins; undecided blocks fall back to checking
        the ``skip_translate`` flag.
        """
        if self.policy_translate is not None:
            return self.policy_translate
        return not self.skip_translate

    def derive_roles(self) -> None:
        """Fill unset role layers from explicit FlowID/BlockType (no guessing).

        Only writes layers that are still None — an explicitly set role
        always wins over derivation.
        """
        if self.layout_role is None:
            self.layout_role = _layout_role_from_flow(self.flow_id)
        if self.structure_role is None:
            self.structure_role = _structure_role_from_block_type(self.block_type)
        if self.semantic_role is None:
            self.semantic_role = SemanticRole.MAIN_TEXT

    def derive_policy(self, reason: str = "derived from roles") -> bool:
        """Derive policy_translate from roles when undecided.

        Returns the effective verdict. Explicit verdicts are never
        overwritten; non-text block types (formula/code/image/table) keep
        origin unless a verdict says otherwise.
        """
        if self.policy_translate is not None:
            return self.policy_translate
        verdict = True
        if (
            self.layout_role in _NON_TRANSLATABLE_LAYOUT_ROLES
            or self.semantic_role in _NON_TRANSLATABLE_SEMANTIC_ROLES
            or self.block_type in _NON_TRANSLATABLE_BLOCK_TYPES
        ):
            verdict = False
        self.policy_translate = verdict
        if not verdict:
            self.policy_reason = reason
        return verdict

    def validate_contract(self) -> list[str]:
        """Check document-v1 invariants; returns violation messages (empty = ok)."""
        violations: list[str] = []
        if self.bbox is not None:
            vals = (self.bbox.x0, self.bbox.y0, self.bbox.x1, self.bbox.y1)
            if any(not (v == v and v not in (float("inf"), float("-inf"))) for v in vals):
                violations.append(f"block {self.id}: bbox has non-finite coordinates")
            elif self.bbox.x1 <= self.bbox.x0 or self.bbox.y1 <= self.bbox.y0:
                violations.append(f"block {self.id}: bbox has non-positive area")
        if self.policy_translate is False and not (self.policy_reason or "").strip():
            violations.append(f"block {self.id}: policy_translate=False requires policy_reason")
        return violations


def _layout_role_from_flow(flow_id: FlowID) -> LayoutRole:
    if flow_id == FlowID.CAPTION:
        return LayoutRole.CAPTION
    if flow_id == FlowID.FOOTNOTE:
        return LayoutRole.FOOTNOTE
    return LayoutRole.BODY


def _structure_role_from_block_type(block_type: BlockType) -> StructureRole:
    mapping = {
        BlockType.NARRATIVE: StructureRole.PARAGRAPH,
        BlockType.DIALOGUE: StructureRole.PARAGRAPH,
        BlockType.HEADING: StructureRole.HEADING,
        BlockType.CODE: StructureRole.CODE,
        BlockType.FORMULA: StructureRole.FORMULA,
        BlockType.IMAGE: StructureRole.FIGURE,
        BlockType.TABLE: StructureRole.TABLE,
        BlockType.LIST_ITEM: StructureRole.LIST_ITEM,
    }
    return mapping[block_type]


class DocumentIR(BaseModel):
    """Full in-memory intermediate representation for a document."""

    doc_id: str  # SHA-256 fingerprint of the source document
    source_path: str  # Path string of the original source file
    format_type: str  # epub, pdf, md, txt
    metadata: dict[str, Any] = Field(default_factory=dict)
    blocks: list[IRBlock] = Field(default_factory=list)

    def get_blocks_by_flow(self, flow_id: FlowID) -> list[IRBlock]:
        """Filter blocks belonging strictly to a given semantic flow."""
        return [b for b in self.blocks if b.flow_id == flow_id]

    @property
    def total_blocks(self) -> int:
        return len(self.blocks)


class ChapterMeta(BaseModel):
    """Lightweight metadata for a document chapter or spine item."""

    model_config = ConfigDict(frozen=True)

    chapter_id: str
    title: str
    spine_index: int
    source_file: str | None = None


class BookManifest(BaseModel):
    """Top-level book manifest for $O(1)$ memory consumption and chapter indexing."""

    doc_id: str
    title: str
    source_path: str
    source_lang: str = "en"
    target_lang: str = "zh"
    chapters: list[ChapterMeta] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="what the source document and the rendered artifact look like "
        "(author, opf_path, page_kinds, typst_version, ...). Run decisions do NOT "
        "go here — they are typed fields on :attr:`run`.",
    )
    run: RunMetadata = Field(default_factory=RunMetadata)

    @property
    def total_chapters(self) -> int:
        return len(self.chapters)


class ChapterIR(BaseModel):
    """Streamed partition chunk of a document for constant memory execution."""

    doc_id: str
    chapter_id: str
    title: str
    spine_index: int
    blocks: list[IRBlock] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)  # e.g. page_kinds

    def get_blocks_by_flow(self, flow_id: FlowID) -> list[IRBlock]:
        """Filter blocks belonging strictly to a given semantic flow."""
        return [b for b in self.blocks if b.flow_id == flow_id]

    @property
    def total_blocks(self) -> int:
        return len(self.blocks)

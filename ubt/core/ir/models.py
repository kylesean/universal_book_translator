"""Semantic Flow-Isolated IR Data Models (Pydantic v2).

``IRBlock`` is the pipeline's mutable working record. Its *structure* is the
typed :class:`~ubt.model.ast.Element` it carries (single source of truth: one attribute, one
origin); everything structural -- ``block_type``, ``flow_id``, ``region``,
``source_text``, ``bbox``, ``spine_index`` -- is derived from that element, so
the two can no longer disagree. What is left on the block is execution state
(status, translation, scores) plus the adapter's typography.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from ubt.core.ir.run_metadata import RunMetadata
from ubt.model.ast import (
    Caption,
    CodeBlock,
    Confidence,
    Dialogue,
    Element,
    ElementKind,
    ElementT,
    Figure,
    FlowKind,
    Formula,
    Heading,
    ListItem,
    Paragraph,
    RegionKind,
    Table,
    TextElement,
)
from ubt.model.span import Span


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


# Single source of truth for the lifecycle's terminal
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


class BoundingBox(BaseModel):
    """Bounding box for fixed layout formats (e.g. PDF)."""

    model_config = ConfigDict(frozen=True)

    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class StyleMeta(BaseModel):
    """Typography and layout metadata (the adapter's finding, not structure)."""

    model_config = ConfigDict(extra="ignore")

    font_name: str | None = None
    font_size: float | None = None
    color_hex: str | None = None
    alignment: str | None = None
    line_height: float | None = None


# --------------------------------------------------------------------------- #
# Element <-> block-type / flow vocabulary
# --------------------------------------------------------------------------- #
_FLOW_TO_KIND: dict[FlowID, FlowKind] = {
    FlowID.MAIN_STORY: FlowKind.MAIN,
    FlowID.SIDEBAR_ASIDE: FlowKind.SIDEBAR,
    FlowID.FOOTNOTE: FlowKind.FOOTNOTE,
    FlowID.TABLE_GRID: FlowKind.TABLE_GRID,
    FlowID.CAPTION: FlowKind.CAPTION,
}
_KIND_TO_FLOW: dict[FlowKind, FlowID] = {kind: flow for flow, kind in _FLOW_TO_KIND.items()}

#: Element kind -> block type. Not a bijection: a caption is a ``NARRATIVE``
#: block type in the caption region, so ``CAPTION`` and ``PARAGRAPH`` both
#: project to it.
_KIND_TO_BLOCK_TYPE: dict[ElementKind, BlockType] = {
    ElementKind.HEADING: BlockType.HEADING,
    ElementKind.PARAGRAPH: BlockType.NARRATIVE,
    ElementKind.DIALOGUE: BlockType.DIALOGUE,
    ElementKind.LIST_ITEM: BlockType.LIST_ITEM,
    ElementKind.CAPTION: BlockType.NARRATIVE,
    ElementKind.CODE_BLOCK: BlockType.CODE,
    ElementKind.FORMULA: BlockType.FORMULA,
    ElementKind.TABLE: BlockType.TABLE,
    ElementKind.FIGURE: BlockType.IMAGE,
}


def _element_block_type(element: Element) -> BlockType:
    return _KIND_TO_BLOCK_TYPE[element.kind]


def _element_source(element: Element) -> str:
    if isinstance(element, TextElement):
        return element.text
    if isinstance(element, Formula):
        return element.source
    if isinstance(element, Table):
        return element.markup
    if isinstance(element, Figure):
        return element.asset_id
    return ""


def _with_element_source(element: ElementT, text: str) -> ElementT:
    """Replace an element's carried text (the field differs per class)."""
    if isinstance(element, TextElement):
        return _replace_element(element, text=text)
    if isinstance(element, Formula):
        return _replace_element(element, source=text)
    if isinstance(element, Table):
        return _replace_element(element, markup=text)
    if isinstance(element, Figure):
        return _replace_element(element, asset_id=text)
    return element


def _replace_element(element: ElementT, **changes: Any) -> ElementT:
    """``dataclasses.replace`` with the union type preserved for mypy."""
    return dataclasses.replace(element, **changes)


def _region_from_flow(flow_id: FlowID) -> RegionKind:
    if flow_id == FlowID.CAPTION:
        return RegionKind.CAPTION
    if flow_id == FlowID.FOOTNOTE:
        return RegionKind.FOOTNOTE
    return RegionKind.BODY


def _span_of(bbox: BoundingBox | None, chars: tuple[int, int] | None = None) -> Span:
    if bbox is None:
        return Span(chars=chars)
    return Span(page=bbox.page, bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1), chars=chars)


def _bbox_of(span: Span) -> BoundingBox | None:
    if span.bbox is None:
        return None
    x0, y0, x1, y1 = span.bbox
    return BoundingBox(page=span.page, x0=x0, y0=y0, x1=x1, y1=y1)


def make_element(
    *,
    id: str,
    spine_index: int,
    block_type: BlockType,
    flow_id: FlowID = FlowID.MAIN_STORY,
    region: RegionKind | None = None,
    source_text: str = "",
    bbox: BoundingBox | None = None,
    skip_translate: bool = False,
    confidence: Confidence = Confidence.INFERRED,
    level: int = 1,
    marker: str = "",
    chars: tuple[int, int] | None = None,
) -> ElementT:
    """Build the typed element a block is made of (the structure, once).

    The one place that maps the extractor's flat finding (block type + flow +
    region + text + geometry) onto the closed element union. A ``None`` region
    falls back to the explicit flow (no guessing beyond what the flow states).
    """
    common: dict[str, Any] = {
        "id": id,
        "spine_index": spine_index,
        "span": _span_of(bbox, chars),
        "confidence": confidence,
        "flow": _FLOW_TO_KIND.get(flow_id, FlowKind.MAIN),
        "region": region if region is not None else _region_from_flow(flow_id),
        "skip_translate": skip_translate,
    }
    if block_type is BlockType.HEADING:
        return Heading(text=source_text, level=level, **common)
    if block_type is BlockType.DIALOGUE:
        return Dialogue(text=source_text, **common)
    if block_type is BlockType.LIST_ITEM:
        return ListItem(text=source_text, marker=marker, **common)
    if block_type is BlockType.CODE:
        return CodeBlock(text=source_text, **common)
    if block_type is BlockType.FORMULA:
        return Formula(source=source_text, **common)
    if block_type is BlockType.TABLE:
        return Table(markup=source_text, **common)
    if block_type is BlockType.IMAGE:
        return Figure(asset_id=source_text, **common)
    # A narrative in the caption region is a caption; the rest is a paragraph.
    if common["region"] is RegionKind.CAPTION:
        return Caption(text=source_text, **common)
    return Paragraph(text=source_text, **common)


class IRBlock(BaseModel):
    """Atomic content block in Universal Book Translator IR.

    ``element`` is the block's structure (single source of truth); the remaining
    fields are execution state and the adapter's typography. Structural reads
    go through the properties below, so a caller cannot set ``block_type`` or
    ``region`` to a value disagreeing with the element.
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    #: The typed structure: element class = type, flow, region, span, text.
    element: Annotated[ElementT, Field(discriminator="kind")]
    style: StyleMeta | None = None

    # Content payloads (state)
    target_text: str | None = None  # Final translated text
    draft_text: str | None = None  # First draft backup

    # Status and quality metrics
    status: BlockStatus = BlockStatus.PENDING
    #: Exact translation-memory hit: ``mtqe_score`` is a pass stamp, not a
    #: measurement, so score aggregates exclude it by provenance (see
    #: ``ubt.core.qe.score_policy``) rather than by a magic score value.
    tm_hit: bool = False
    glossary_hits: list[str] = Field(default_factory=list)  # Matched glossary terms
    mtqe_score: float | None = None  # Quality metric score (0.00 ~ 1.00)
    repair_rounds: int = 0  # Completed targeted repair iterations (max 2)
    error_flags: list[str] = Field(default_factory=list)  # e.g., ["html_attr_mismatch"]

    # MQM severity triage
    mqm_severity: str | None = None  # "critical" | "major" | "minor" (None = not triaged)
    mqm_spans: list[dict[str, Any]] = Field(default_factory=list)  # Serialized MQMErrorSpan dicts

    # Policy verdict (single source of truth: the pipeline's decision, not the extractor's)
    policy_translate: bool | None = None  # None = undecided, fall back to skip_translate
    policy_reason: str | None = None  # Required when policy_translate is False
    provenance: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Structural projections of ``element`` (read-only; one source)
    # ------------------------------------------------------------------ #
    @property
    def id(self) -> str:
        return self.element.id

    @property
    def spine_index(self) -> int:
        return self.element.spine_index

    @property
    def block_type(self) -> BlockType:
        return _element_block_type(self.element)

    @property
    def flow_id(self) -> FlowID:
        return _KIND_TO_FLOW.get(self.element.flow, FlowID.MAIN_STORY)

    @property
    def region(self) -> RegionKind:
        return self.element.region

    @property
    def source_text(self) -> str:
        return _element_source(self.element)

    @property
    def bbox(self) -> BoundingBox | None:
        return _bbox_of(self.element.span)

    @property
    def skip_translate(self) -> bool:
        return self.element.skip_translate

    @skip_translate.setter
    def skip_translate(self, value: bool) -> None:
        if self.element.skip_translate != value:
            self.element = _replace_element(self.element, skip_translate=value)

    @property
    def is_finalized(self) -> bool:
        """Returns True if the block has reached a terminal status."""
        return self.status in TERMINAL_STATUSES

    # ------------------------------------------------------------------ #
    # Structural writes rebuild the element (still one source)
    # ------------------------------------------------------------------ #
    def set_id(self, block_id: str) -> None:
        self.element = _replace_element(self.element, id=block_id)

    def set_spine_index(self, spine_index: int) -> None:
        self.element = _replace_element(self.element, spine_index=spine_index)

    def set_source_text(self, text: str) -> None:
        self.element = _with_element_source(self.element, text)

    def with_source_text(self, text: str) -> IRBlock:
        """A copy whose carried source text is replaced (rebuilds the element).

        ``model_copy(update={"source_text": ...})`` cannot be used: it writes the
        key into ``__dict__`` and the structural property ignores it.
        """
        copy = self.model_copy()
        copy.set_source_text(text)
        return copy

    def set_bbox(self, bbox: BoundingBox | None) -> None:
        chars = self.element.span.chars
        self.element = _replace_element(self.element, span=_span_of(bbox, chars))

    def validate_contract(self) -> list[str]:
        """Check document-v1 invariants; returns violation messages (empty = ok)."""
        violations: list[str] = []
        bbox = self.bbox
        if bbox is not None:
            vals = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
            if any(not (v == v and v not in (float("inf"), float("-inf"))) for v in vals):
                violations.append(f"block {self.id}: bbox has non-finite coordinates")
            elif bbox.x1 <= bbox.x0 or bbox.y1 <= bbox.y0:
                violations.append(f"block {self.id}: bbox has non-positive area")
        if self.policy_translate is False and not (self.policy_reason or "").strip():
            violations.append(f"block {self.id}: policy_translate=False requires policy_reason")
        return violations


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

    @property
    def total_blocks(self) -> int:
        return len(self.blocks)

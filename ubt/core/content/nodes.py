"""Content-graph node model: text vs non-text, and their dispositions.

The two axioms the whole delivery contract rests on:

- **Non-text (Axiom A):** a figure/formula/table is either *losslessly
  reconstructed* or *preserved whole as an immutable object*. A partially
  corrupted reconstruction is never acceptable -- prefer preserving a table as
  one image over shattering it into single-character cells.
- **Text (Axiom B):** body text is either *translated* or *explicitly kept in
  the source language*. It is never dropped because it did not fit.

The node types here make those two dispositions first-class and checkable.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class NodeKind(StrEnum):
    """Whether a content node carries translatable text or an immutable asset."""

    TEXT = "text"
    ASSET = "asset"


class TextDisposition(StrEnum):
    """What happened to one translatable text node.

    ``PENDING`` and ``SKIPPED`` are contract violations at delivery: a node that
    has not been decided on, or was dropped without explanation, means the
    delivery silently omitted content. ``VERBATIM`` is a *deliberate* keep (a
    code listing, a proper noun, chrome). ``SOURCE_KEPT`` is an *explicit but
    unintended* keep: the node reached delivery in the source language because a
    translation could not be placed or was quarantined. It is not silent (it is
    recorded), so it is a warning; it becomes a hard error once renderers are
    ledger-bound.
    """

    PENDING = "pending"  # not yet decided (pre-translation)
    TRANSLATED = "translated"  # target text delivered
    VERBATIM = "verbatim"  # deliberately kept in the source language
    SOURCE_KEPT = "source_kept"  # shipped source because it could not be translated/placed
    SKIPPED = "skipped"  # dropped with no explanation -- a violation


class AssetKind(StrEnum):
    """Semantic class of a non-text node."""

    FIGURE = "figure"
    FORMULA = "formula"
    TABLE = "table"
    DIAGRAM = "diagram"


class AssetRepresentation(StrEnum):
    """How the asset is physically carried into the output.

    ``LATEX`` / ``STRUCTURED_TABLE`` / ``SVG`` are *interpreted* forms: the
    pipeline looked inside and rebuilt it. ``OPAQUE_CROP`` is the safe form: the
    source region is copied whole and never interpreted, so its internals cannot
    be corrupted.
    """

    LATEX = "latex"
    STRUCTURED_TABLE = "structured_table"
    SVG = "svg"
    RASTER = "raster"
    OPAQUE_CROP = "opaque_crop"


class AssetIntegrity(StrEnum):
    """Whether an asset survived the delivery intact.

    ``RECONSTRUCTED`` demands ``verified=True`` (a round-trip check passed);
    without that proof the asset must be downgraded to ``PRESERVED_OPAQUE`` or
    reported ``MISSING``. ``DROPPED`` is an *intentional* removal of decorative
    chrome (a banner, a cover ornament), recorded with a reason so it is never
    confused with a loss. ``MISSING`` is a hard violation.
    """

    RECONSTRUCTED = "reconstructed"
    PRESERVED_OPAQUE = "preserved_opaque"
    DROPPED = "dropped"
    MISSING = "missing"


class SourceRegion(BaseModel):
    """Where the node came from in the source document (1-based page)."""

    model_config = ConfigDict(frozen=True)

    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class TextNode(BaseModel):
    """One translatable (or deliberately kept) text span."""

    model_config = ConfigDict(frozen=True)

    kind: Literal[NodeKind.TEXT] = NodeKind.TEXT
    id: str
    order: int
    source_text: str
    target_text: str | None = None
    disposition: TextDisposition = TextDisposition.PENDING
    reason: str = ""
    source_region: SourceRegion | None = None
    # Provenance back to the IR block, so a violation can name the origin.
    block_type: str = ""
    flow_id: str = ""
    region: str = ""


class AssetDescriptor(BaseModel):
    """Everything the contract must know about one non-text node."""

    model_config = ConfigDict(frozen=True)

    asset_kind: AssetKind
    representation: AssetRepresentation
    integrity: AssetIntegrity
    source_region: SourceRegion | None = None
    #: Round-trip / structural verification passed: only then may integrity be
    #: RECONSTRUCTED without a warning.
    verified: bool = False
    #: Structural check found a corrupt reconstruction (e.g. a shattered table).
    corrupt: bool = False
    #: Content hash of the preserved region, for ledger reconciliation.
    digest: str = ""
    detail: str = ""


class AssetNode(BaseModel):
    """One immutable non-text node (figure / formula / table / diagram)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal[NodeKind.ASSET] = NodeKind.ASSET
    id: str
    order: int
    descriptor: AssetDescriptor
    block_type: str = ""
    flow_id: str = ""


ContentNode = Annotated[TextNode | AssetNode, Field(discriminator="kind")]

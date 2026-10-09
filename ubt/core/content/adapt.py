"""Boundary adapter: ``IRBlock`` (pipeline working form) -> content graph.

This is the only place the contract layer knows about the pipeline's internal
representation. Every delivery gets a graph built here, so the reconciliation
gate applies to *all* render paths.

Asset policy (Axiom A): a non-text block is only RECONSTRUCTED alongside the
round-trip check that verifies it. The unified LayerCompositor (overlay engine)
composes onto the source page, so non-text nodes are PRESERVED_OPAQUE by
construction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ubt.core.content.graph import ContentGraph
from ubt.core.content.nodes import (
    AssetDescriptor,
    AssetIntegrity,
    AssetKind,
    AssetNode,
    AssetRepresentation,
    ContentNode,
    SourceRegion,
    TextDisposition,
    TextNode,
)
from ubt.core.ir.models import TERMINAL_STATUSES, BlockType, IRBlock
from ubt.core.qe.defect_taxonomy import INTENTIONAL_PRESERVED_SKIP_PREFIXES

_ASSET_KIND: dict[BlockType, AssetKind] = {
    BlockType.FORMULA: AssetKind.FORMULA,
    BlockType.TABLE: AssetKind.TABLE,
    BlockType.IMAGE: AssetKind.FIGURE,
}

_SKIP_PREFIXES = ("render_skip:", "inplace_skip:")
#: Skip reasons naming decorative chrome, not a meaning-bearing asset. Reused
#: from the artifact-parity vocabulary so the two gates cannot drift.
_DECORATIVE_SKIP_REASONS = frozenset({"decorative_banner", "cover_asset_missing"})


def _region(block: IRBlock) -> SourceRegion | None:
    bbox = getattr(block, "bbox", None)
    if bbox is None or bbox.page <= 0 or bbox.x1 <= bbox.x0 or bbox.y1 <= bbox.y0:
        return None
    return SourceRegion(page=bbox.page, x0=bbox.x0, y0=bbox.y0, x1=bbox.x1, y1=bbox.y1)


def _skip_flags(block: IRBlock) -> list[str]:
    flags = getattr(block, "error_flags", None) or []
    return [f for f in flags if isinstance(f, str) and f.startswith(_SKIP_PREFIXES)]


def _skip_reason(flag: str) -> str:
    for prefix in _SKIP_PREFIXES:
        if flag.startswith(prefix):
            return flag[len(prefix) :].split("(", 1)[0].strip()
    return flag


def _all_intentional(flags: Sequence[str]) -> bool:
    return bool(flags) and all(f.startswith(INTENTIONAL_PRESERVED_SKIP_PREFIXES) for f in flags)


def kept_in_source(block: IRBlock) -> bool:
    """Whether the delivery keeps this *text* block in the source.

    True for a deliberate keep (``skip_translate``, an intentional render-skip
    such as page chrome) *and* for a fail-closed render-skip (a translation that
    could not be placed). In both cases the reader gets the source, so there is
    no placed translation -- and none for a backend to reflow. This is exactly
    the contract's ``TRANSLATED`` condition, stated once beside the skip-flag
    vocabulary it reads, so the two cannot drift.
    """
    return block.skip_translate or bool(_skip_flags(block))


def _text_node(block: IRBlock, order: int, region: SourceRegion | None) -> TextNode:
    flags = _skip_flags(block)
    if flags and not _all_intentional(flags):
        # A translation existed but the renderer left source visible: not silent
        # (it is recorded), so a warning.
        disposition, reason = TextDisposition.SOURCE_KEPT, f"render:{_skip_reason(flags[0])}"
    elif flags:  # intentional preserved element (chrome/non-prose/policy/footer)
        disposition, reason = TextDisposition.VERBATIM, f"preserved:{_skip_reason(flags[0])}"
    elif block.skip_translate:
        disposition, reason = TextDisposition.VERBATIM, "skip_translate"
    elif (block.target_text or "").strip():
        disposition, reason = TextDisposition.TRANSLATED, ""
    elif block.status in TERMINAL_STATUSES:
        # Finished without a target: source shipped. Explicit (the quality
        # report / PE queue already flag it), so recorded, not silent.
        disposition, reason = TextDisposition.SOURCE_KEPT, f"source_kept:{block.status.value}"
    else:
        disposition, reason = TextDisposition.PENDING, ""
    return TextNode(
        id=block.id,
        order=order,
        source_text=block.source_text or "",
        target_text=block.target_text,
        disposition=disposition,
        reason=reason,
        source_region=region,
        block_type=str(block.block_type),
        flow_id=str(block.flow_id),
        region=str(block.region) if block.region else "",
    )


def _representation(
    block: IRBlock, asset_kind: AssetKind, integrity: AssetIntegrity
) -> AssetRepresentation:
    if asset_kind is AssetKind.FIGURE:
        src = (block.target_text or block.source_text or "").lower()
        return AssetRepresentation.SVG if src.endswith(".svg") else AssetRepresentation.RASTER
    if integrity is not AssetIntegrity.RECONSTRUCTED:
        return AssetRepresentation.OPAQUE_CROP
    if asset_kind is AssetKind.FORMULA:
        return AssetRepresentation.LATEX
    return AssetRepresentation.STRUCTURED_TABLE


def _digest(block: IRBlock, region: SourceRegion | None) -> str:
    payload = f"{block.source_text or ''}\x1f{region!r}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _asset_node(
    block: IRBlock,
    order: int,
    region: SourceRegion | None,
) -> AssetNode:
    asset_kind = _ASSET_KIND[block.block_type]
    flags = _skip_flags(block)
    reasons = {_skip_reason(f) for f in flags}
    # The overlay engine keeps the source page as the canvas, so every non-text
    # node survives whole: no asset is ever reconstructed or re-verified.
    if reasons & _DECORATIVE_SKIP_REASONS:
        integrity = AssetIntegrity.DROPPED
        detail = "intentional:" + ",".join(sorted(reasons & _DECORATIVE_SKIP_REASONS))
    elif flags and not _all_intentional(flags):
        integrity = AssetIntegrity.MISSING
        detail = f"render:{_skip_reason(flags[0])}"
    else:
        integrity = AssetIntegrity.PRESERVED_OPAQUE
        detail = ""
    verified = False
    corrupt = False
    return AssetNode(
        id=block.id,
        order=order,
        descriptor=AssetDescriptor(
            asset_kind=asset_kind,
            representation=_representation(block, asset_kind, integrity),
            integrity=integrity,
            source_region=region,
            verified=verified,
            corrupt=corrupt,
            digest=_digest(block, region),
            detail=detail,
        ),
        block_type=str(block.block_type),
        flow_id=str(block.flow_id),
    )


def graph_from_blocks(
    blocks: Sequence[IRBlock],
    *,
    doc_id: str = "",
    title: str = "",
    source_path: str = "",
) -> ContentGraph:
    """Build the delivery contract's content graph from the delivered blocks.

    The overlay engine keeps the source page as the canvas, so non-text nodes are
    always preserved whole.
    """
    nodes: list[ContentNode] = []
    for order, block in enumerate(blocks):
        region = _region(block)
        if block.block_type in _ASSET_KIND:
            nodes.append(_asset_node(block, order, region))
        else:
            nodes.append(_text_node(block, order, region))
    return ContentGraph(doc_id=doc_id, title=title, source_path=source_path, nodes=tuple(nodes))


__all__ = ["graph_from_blocks", "kept_in_source"]

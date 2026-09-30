"""Boundary adapter: ``IRBlock`` (pipeline working form) -> content graph.

This is the only place the contract layer knows about the pipeline's internal
representation. Consumers migrate to the graph piecemeal; until they do, every
delivery still gets a graph built here, so the reconciliation gate applies to
*all* render paths from day one.

Asset policy (Axiom A): a non-text block is only RECONSTRUCTED once a round-trip
check exists to verify it. Phase 0 has no such check, so a rigid (source-canvas)
render is PRESERVED_OPAQUE by construction and a reflow placement is
RECONSTRUCTED-but-unverified -- recorded as a *warning* by
:func:`ubt.core.content.contract.reconcile`, promoted to an error in Phase 1.
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


def _text_node(block: IRBlock, order: int, region: SourceRegion | None) -> TextNode:
    flags = _skip_flags(block)
    if flags and not _all_intentional(flags):
        # A translation existed but the renderer left source visible: not silent
        # (it is recorded), so a warning in Phase 0.
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
        layout_role=str(block.layout_role) if block.layout_role else "",
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
    engine: str,
    substituted_ids: frozenset[str],
) -> AssetNode:
    asset_kind = _ASSET_KIND[block.block_type]
    flags = _skip_flags(block)
    reasons = {_skip_reason(f) for f in flags}
    verified = False
    corrupt = False
    if reasons & _DECORATIVE_SKIP_REASONS:
        integrity = AssetIntegrity.DROPPED
        detail = "intentional:" + ",".join(sorted(reasons & _DECORATIVE_SKIP_REASONS))
    elif flags and not _all_intentional(flags):
        integrity = AssetIntegrity.MISSING
        detail = f"render:{_skip_reason(flags[0])}"
    elif engine != "rigid" and block.id in substituted_ids:
        # The formula witness failed and the renderer swapped in the source
        # graphic: lossless by construction, so the asset is preserved opaque.
        integrity = AssetIntegrity.PRESERVED_OPAQUE
        detail = "witness_substituted"
        verified = True
    elif engine == "rigid":
        # The source canvas is kept, so every non-text node survives whole.
        integrity = AssetIntegrity.PRESERVED_OPAQUE
        detail = ""
    else:
        # Reflow reconstruction: level-1 structural verification (Phase 1a).
        # A PASS is verified; a FAIL is corruption (Axiom A) and awaits the
        # opaque source-crop fallback (Phase 1b); a SKIP stays unverified.
        if asset_kind is AssetKind.FIGURE:
            # A figure is placed as its original graphic, not rebuilt.
            integrity = AssetIntegrity.PRESERVED_OPAQUE
            verified = True
            detail = "placed"
        else:
            # Structural verification now goes through the unified verifier
            # seam (ADR-0001). The import is function-local because
            # ``ubt.verify`` imports this package's ``asset_verify`` submodule:
            # a module-level edge here would close a cycle when ``ubt.verify``
            # is the first package imported. The behaviour is unchanged -- the
            # seam wraps the very same check, proven by scripts/shadow_verify.py.
            from ubt.verify.verifier import StructuralAsset, StructuralAssetVerifier

            proof = StructuralAssetVerifier().verify(
                StructuralAsset(asset_kind, block.target_text or block.source_text or "")
            )
            integrity = AssetIntegrity.RECONSTRUCTED
            if proof.verified:
                verified = True
                detail = proof.detail
            elif proof.failed:
                corrupt = True
                detail = f"structural:{proof.detail}"
            else:
                detail = proof.detail
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
    engine: str = "publication",
    doc_id: str = "",
    title: str = "",
    source_path: str = "",
    witness_findings: Sequence[str] = (),
    table_fallbacks: Sequence[str] = (),
) -> ContentGraph:
    """Build the delivery contract's content graph from the delivered blocks.

    ``engine`` selects the asset-preservation policy: ``rigid`` preserves every
    non-text node whole, ``publication``/``reflow`` reconstructs it and runs the
    structural verification. ``witness_findings`` / ``table_fallbacks`` are the
    ``"<block_id>: <detail>"`` lines for assets the renderer swapped for their
    source graphic; those are honoured as preserved-opaque, not reconstructions.
    """
    substituted_ids = frozenset(
        f.split(":", 1)[0].strip() for f in (*witness_findings, *table_fallbacks) if f and ":" in f
    )
    nodes: list[ContentNode] = []
    for order, block in enumerate(blocks):
        region = _region(block)
        if block.block_type in _ASSET_KIND:
            nodes.append(_asset_node(block, order, region, engine, substituted_ids))
        else:
            nodes.append(_text_node(block, order, region))
    return ContentGraph(doc_id=doc_id, title=title, source_path=source_path, nodes=tuple(nodes))


__all__ = ["graph_from_blocks"]

"""Boundary adapter: ``IRBlock`` (pipeline working form) -> content graph.

This is the only place the contract layer knows about the pipeline's internal
representation. Every delivery gets a graph built here, so the reconciliation
gate applies to *all* render paths.

Asset policy (Axiom A): a non-text block is only RECONSTRUCTED alongside the
round-trip check that verifies it. The renderer places figures as the
extracted image and rebuilds tables from their markup, so a figure stays
PRESERVED_OPAQUE while a rebuilt table is a reconstruction the contract has to
be able to verify.
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
    ReconstructionRecord,
    SourceRegion,
    TextDisposition,
    TextNode,
)
from ubt.core.ir.models import TERMINAL_STATUSES, BlockStatus, BlockType, IRBlock

_ASSET_KIND: dict[BlockType, AssetKind] = {
    BlockType.FORMULA: AssetKind.FORMULA,
    BlockType.TABLE: AssetKind.TABLE,
    BlockType.IMAGE: AssetKind.FIGURE,
}

_SKIP_PREFIXES = ("render_skip:",)
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


def kept_in_source(block: IRBlock) -> bool:
    """Whether the delivery keeps this *text* block in the source.

    True for a deliberate keep (``skip_translate``, an intentional render-skip
    such as page chrome), for a fail-closed render-skip (a translation that
    could not be placed), and for a quarantined block (``BLOCKED_HUMAN``). In
    every case the reader gets the source, so there is no placed translation --
    and none for a backend to reflow. This is exactly the contract's
    ``TRANSLATED`` condition, stated once beside the skip-flag vocabulary it
    reads, so the two cannot drift.

    ``BLOCKED_HUMAN`` must be named here, not left to the skip flags: its
    ``target_text`` is a ``<mark>`` wrapper around the *source*, not a
    translation, and it carries no skip flag -- so a truthiness test on the
    target would call it delivered and ship the raw marker into the HTML/EPUB
    views (and into the artifact's expected-text map).
    """
    return (
        block.skip_translate
        or bool(_skip_flags(block))
        or block.status is BlockStatus.BLOCKED_HUMAN
    )


def _text_node(block: IRBlock, order: int, region: SourceRegion | None) -> TextNode:
    flags = _skip_flags(block)
    if flags:
        # A translation existed but the renderer left source visible: not silent
        # (it is recorded), so a warning. Every ``render_skip:`` is fail-closed
        # now — the intentional-keep families had no producer and were removed.
        disposition, reason = TextDisposition.SOURCE_KEPT, f"render:{_skip_reason(flags[0])}"
    elif block.skip_translate:
        # Carry the specific policy reason (``skip:bibliography entry …``) into
        # the contract instead of the opaque ``skip_translate``: the audit then
        # says *why* the block shipped in the source language, not merely that it
        # did. The reason is the one ``ingest`` recorded, so the two cannot drift.
        reason = block.policy_reason or "skip_translate"
        disposition = TextDisposition.VERBATIM
    elif block.status is BlockStatus.BLOCKED_HUMAN:
        # Quarantined for human post-editing. The *contract* counts this as
        # source-kept, and it must be counted here rather than falling through
        # to the target-truthiness test below: the non-empty ``<mark>`` wrapper
        # would otherwise be counted as delivered -- the defect that let 20
        # untranslated blocks report ``violations: []`` and
        # ``source_kept_text: 0`` in arXiv 2610.02185.
        #
        # Note the renderers do NOT agree with this disposition, on purpose:
        # the PDF/HTML/EPUB views typeset the rejected *draft* (fail-open, see
        # ``document_renderer._block_text`` and
        # ``delivered_text.is_placeholder``) rather than the source, so the
        # reader gets the best translation available while the audit still says
        # "not verified". ``kept_in_source`` answers "does the delivery have a
        # verified translation?", which is a different question from "what text
        # does the renderer place?".
        disposition, reason = (
            TextDisposition.SOURCE_KEPT,
            f"quarantined:{block.status.value}",
        )
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
    reconstruction: ReconstructionRecord | None = None,
) -> AssetNode:
    asset_kind = _ASSET_KIND[block.block_type]
    flags = _skip_flags(block)
    reasons = {_skip_reason(f) for f in flags}
    verified = False
    corrupt = False
    # A non-text node that was neither dropped on purpose nor lost to a render
    # skip survived whole: the renderer placed the extracted asset (a figure) or
    # rebuilt it from its own markup (a table).
    if reasons & _DECORATIVE_SKIP_REASONS:
        integrity = AssetIntegrity.DROPPED
        detail = "intentional:" + ",".join(sorted(reasons & _DECORATIVE_SKIP_REASONS))
    elif flags:
        integrity = AssetIntegrity.MISSING
        detail = f"render:{_skip_reason(flags[0])}"
    elif reconstruction is not None:
        # The renderer reported an *interpreted* rebuild of this asset (a table
        # rebuilt from the translated grid, a formula transpiled to Typst math).
        # That is a RECONSTRUCTED asset, not an opaque copy — and only a
        # shape-verified rebuild may claim it without a warning. A corrupt one
        # (a grid that dropped/gained a column or row) is an ERROR at the
        # contract. This is the branch the RECONSTRUCTED integrity and both
        # ASSET_*_RECONSTRUCTION violation kinds exist for; before it, every
        # asset was recorded PRESERVED_OPAQUE and the whole gate was dead.
        integrity = AssetIntegrity.RECONSTRUCTED
        verified = reconstruction.verified
        corrupt = reconstruction.corrupt
        detail = reconstruction.reason
    else:
        integrity = AssetIntegrity.PRESERVED_OPAQUE
        detail = ""
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
    reconstructions: Sequence[ReconstructionRecord] | None = None,
) -> ContentGraph:
    """Build the delivery contract's content graph from the delivered blocks.

    Every non-text node is accounted for: preserved when the renderer placed it,
    dropped only for a recorded decorative reason, missing when a render skip
    says so.

    ``reconstructions`` is the renderer's per-asset verdict ledger (a table
    rebuilt from the translated grid, verified or corrupt). When a block has a
    record it is recorded ``RECONSTRUCTED`` instead of ``PRESERVED_OPAQUE`` — the
    difference between "we proved we rebuilt it" and "we only know it exists".
    Absent (the ledger re-derivation path, which has no render side channel), the
    graph is the existence-only account it always was.
    """
    by_id = {rec.block_id: rec for rec in (reconstructions or ())}
    nodes: list[ContentNode] = []
    for order, block in enumerate(blocks):
        region = _region(block)
        if block.block_type in _ASSET_KIND:
            nodes.append(_asset_node(block, order, region, by_id.get(block.id)))
        else:
            nodes.append(_text_node(block, order, region))
    return ContentGraph(doc_id=doc_id, title=title, source_path=source_path, nodes=tuple(nodes))


__all__ = ["graph_from_blocks", "kept_in_source"]

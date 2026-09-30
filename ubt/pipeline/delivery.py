"""Wiring a delivered run into the attestation core (ADR-0001 Phase 3).

During the migration the existing renderer still draws, so the backends are fed
the run's own decisions -- which engine ran, and which elements the delivery
deliberately kept -- and reproduce them as attestations, verifying each. That is
the ADR's "wrap" shape: the pipeline's decisions drive the backend, and the
backend answers for fidelity and the account.

When a backend draws for real, this glue is what disappears: the decisions come
from :func:`~ubt.pipeline.steps.realize` itself instead of from the renderer.
"""

from __future__ import annotations

from collections.abc import Sequence

from ubt.analyze.bridge import document_from_blocks
from ubt.core.content.adapt import kept_in_source
from ubt.core.ir.models import BlockType, IRBlock
from ubt.model.ast import Document

#: Block types that are immutable assets, not translatable text. A rigid run
#: places them opaque instead of reconstructing them, so they are never fed a
#: "target" to reconstruct from.
_ASSET_TYPES = (BlockType.FORMULA, BlockType.TABLE, BlockType.IMAGE)


def delivery_document(blocks: Sequence[IRBlock], *, doc_id: str = "") -> Document:
    """The typed :class:`Document` the delivery's blocks describe."""
    return document_from_blocks(list(blocks), doc_id=doc_id)


def delivery_translations(blocks: Sequence[IRBlock], *, engine: str) -> dict[str, str]:
    """The ``element id -> placed target`` map the run's backend is built with.

    Only a translation the renderer actually *placed* is in the map: a block kept
    in the source (:func:`ubt.core.content.adapt.kept_in_source`) has none, so the
    backend places it opaque and the attestation matches the artifact. A rigid run
    places every asset opaque rather than reconstructing it, so an asset's markup
    is not a realization either -- the engine is the whole-document choice the
    per-element backends replace, and it is exactly what this map carries.
    """
    rigid = engine == "rigid"
    placed: dict[str, str] = {}
    for block in blocks:
        target = block.target_text or ""
        if not target.strip():
            continue
        if block.block_type in _ASSET_TYPES:
            if rigid:
                continue
        elif kept_in_source(block):
            continue
        placed[block.id] = target
    return placed


__all__ = ["delivery_document", "delivery_translations"]

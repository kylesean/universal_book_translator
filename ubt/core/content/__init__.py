"""Canonical content graph: the delivery contract's single source of truth.

This package is deliberately independent of the pipeline's working
representation (:mod:`ubt.core.ir.models`). The pipeline keeps ``IRBlock``
internally; :mod:`ubt.core.content.adapt` bridges the two at the boundary so
consumers migrate incrementally. The graph answers one question only:

    What content must a delivery account for, no matter how it is rendered?

Three invariants follow from it (see :mod:`ubt.core.content.contract`):

1. every translatable text node is either translated or explicitly kept;
2. every non-text node is either losslessly reconstructed or preserved whole;
3. any inability is reported, never silent.
"""

from ubt.core.content.adapt import graph_from_blocks
from ubt.core.content.contract import (
    ReconciliationReport,
    Violation,
    ViolationKind,
)
from ubt.core.content.graph import ContentGraph
from ubt.core.content.ledger import AssetLedger, ContentLedger, build_ledgers
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

__all__ = [
    "AssetDescriptor",
    "AssetIntegrity",
    "AssetKind",
    "AssetLedger",
    "AssetNode",
    "AssetRepresentation",
    "ContentGraph",
    "ContentLedger",
    "ContentNode",
    "ReconciliationReport",
    "SourceRegion",
    "TextDisposition",
    "TextNode",
    "Violation",
    "ViolationKind",
    "build_ledgers",
    "graph_from_blocks",
]

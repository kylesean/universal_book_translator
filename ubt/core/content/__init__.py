"""Canonical content graph: the classifier behind the delivery contract.

The delivery contract's primary source is the run's per-element attestations
(:func:`ubt.core.content.project.contract_from_attestations`); this graph only
classifies the delivered blocks into the contract's vocabulary and supplies the
detail the AST does not model -- it is the projection's *input*, not a second
verdict.

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
from ubt.core.content.project import contract_from_attestations

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
    "contract_from_attestations",
    "graph_from_blocks",
]

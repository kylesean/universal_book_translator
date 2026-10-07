"""Regression: the content-graph asset policy for the single overlay engine.

Every PDF composes onto the source page through the unified LayerCompositor, so
a non-text block must be reported PRESERVED_OPAQUE rather than RECONSTRUCTED.
"""

from __future__ import annotations

import pytest

from ubt.core.content.adapt import graph_from_blocks
from ubt.core.content.nodes import AssetIntegrity, AssetNode
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


def _formula_block(block_id: str = "f1") -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=0,
        block_type=BlockType.FORMULA,
        source_text="E = mc^2",
        bbox=BoundingBox(page=1, x0=0.0, y0=0.0, x1=10.0, y1=10.0),
        skip_translate=False,
    )
    return IRBlock(element=element)


def test_non_text_assets_are_preserved_opaque() -> None:
    graph = graph_from_blocks([_formula_block()])
    (node,) = graph.nodes
    assert isinstance(node, AssetNode)
    assert node.descriptor.integrity is AssetIntegrity.PRESERVED_OPAQUE

"""Regression: the content-graph asset policy matches the unified render route.

Every canonical PDF engine (``rigid`` / ``composite`` / ``publication``, and the
``reflow`` alias) composes onto the source page through the unified
LayerCompositor, so a non-text block must be reported PRESERVED_OPAQUE rather
than RECONSTRUCTED. This pins the contract ``graph_from_blocks`` documents: only
a non-source-canvas engine name falls back to the reconstruction path.
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


@pytest.mark.parametrize("engine", ["rigid", "composite", "publication", "reflow"])
def test_source_canvas_engines_preserve_assets_opaque(engine: str) -> None:
    graph = graph_from_blocks([_formula_block()], engine=engine)
    (node,) = graph.nodes
    assert isinstance(node, AssetNode)
    assert node.descriptor.integrity is AssetIntegrity.PRESERVED_OPAQUE


def test_publication_is_not_reconstructed() -> None:
    """The pre-unified docstring claimed publication reconstructs; the contract
    must not regress to that, since the runtime composes every PDF route onto the
    source canvas."""
    graph = graph_from_blocks([_formula_block()], engine="publication")
    (node,) = graph.nodes
    assert isinstance(node, AssetNode)
    assert node.descriptor.integrity is not AssetIntegrity.RECONSTRUCTED

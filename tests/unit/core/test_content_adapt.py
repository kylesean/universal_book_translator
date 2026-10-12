"""Regression: the content-graph asset policy for the single publish renderer.

The document is rebuilt from the IR by Typst, so a non-text block the renderer
placed or rebuilt is reported PRESERVED_OPAQUE rather than RECONSTRUCTED — the
latter demands a verified round trip, which nothing claims.
"""

from __future__ import annotations

import pytest

from ubt.core.content.adapt import graph_from_blocks, kept_in_source
from ubt.core.content.nodes import AssetIntegrity, AssetNode, TextDisposition
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element

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


def _blocked_human_block(block_id: str = "q1") -> IRBlock:
    """A quarantined block: its target is a <mark> around the *source*."""
    block = IRBlock(
        element=make_element(
            id=block_id,
            spine_index=0,
            block_type=BlockType.NARRATIVE,
            source_text="The reader sees this English.",
            bbox=BoundingBox(page=1, x0=0.0, y0=0.0, x1=100.0, y1=10.0),
        )
    )
    block.target_text = (
        '<mark class="ubt-blocked-human" title="MQM Critical unresolved">'
        "【待人工审校 | Human review required】The reader sees this English.</mark>"
    )
    block.status = BlockStatus.BLOCKED_HUMAN
    return block


def test_a_quarantined_block_is_source_kept_not_delivered() -> None:
    """A BLOCKED_HUMAN target is a marker around the source, not a translation.

    The truthiness test on ``target_text`` used to run first and call the
    non-empty marker DELIVERED, so arXiv 2610.02185 shipped 20 untranslated
    blocks while its contract reported ``source_kept_text: 0`` and
    ``violations: []``.
    """
    graph = graph_from_blocks([_blocked_human_block()])
    (node,) = graph.text_nodes
    assert node.disposition is TextDisposition.SOURCE_KEPT
    assert node.reason.startswith("quarantined:")
    assert not kept_in_source(_formula_block())  # sanity: predicate is not always True
    assert kept_in_source(_blocked_human_block())


def test_a_real_translation_is_still_delivered() -> None:
    """Guard against over-correcting: an ordinary translated block is unchanged."""
    block = _blocked_human_block("ok1")
    block.status = BlockStatus.MTQE_PASSED
    block.target_text = "读者看到的是这段中文。"
    graph = graph_from_blocks([block])
    (node,) = graph.text_nodes
    assert node.disposition is TextDisposition.TRANSLATED
    assert not kept_in_source(block)

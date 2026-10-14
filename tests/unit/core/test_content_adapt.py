"""Regression: the content-graph asset policy for the single publish renderer.

The document is rebuilt from the IR by Typst. A non-text block the renderer
*placed* but did not interpret (a figure) is reported PRESERVED_OPAQUE — the
``RECONSTRUCTED`` verdict demands a rebuild the renderer proved, which only a
table rebuilt from its translated grid (or a transpiled formula) can claim.
"""

from __future__ import annotations

import pytest

from ubt.core.content.adapt import graph_from_blocks, kept_in_source
from ubt.core.content.nodes import (
    AssetIntegrity,
    AssetNode,
    ReconstructionRecord,
    TextDisposition,
)
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


def test_a_verified_reconstruction_record_is_recorded_reconstructed() -> None:
    """A renderer verdict of "rebuilt and shape-verified" flips the integrity.

    Before the reconstruction channel existed, every asset was recorded
    PRESERVED_OPAQUE (verified/corrupt hardcoded False), so the contract's
    RECONSTRUCTED branch and both ``ASSET_*_RECONSTRUCTION`` kinds were
    unreachable — the whole "did we actually rebuild it" gate was dead.
    """
    graph = graph_from_blocks(
        [_formula_block("t1")],
        reconstructions=[ReconstructionRecord(block_id="t1", verified=True, corrupt=False)],
    )
    (node,) = graph.nodes
    assert isinstance(node, AssetNode)
    assert node.descriptor.integrity is AssetIntegrity.RECONSTRUCTED
    assert node.descriptor.verified is True
    assert node.descriptor.corrupt is False


def test_a_corrupt_reconstruction_record_marks_the_asset_corrupt() -> None:
    graph = graph_from_blocks(
        [_formula_block("t2")],
        reconstructions=[
            ReconstructionRecord(
                block_id="t2", verified=False, corrupt=True, reason="column_count_mismatch:3->2"
            )
        ],
    )
    (node,) = graph.nodes
    assert isinstance(node, AssetNode)
    assert node.descriptor.integrity is AssetIntegrity.RECONSTRUCTED
    assert node.descriptor.corrupt is True
    assert node.descriptor.detail == "column_count_mismatch:3->2"


def test_a_fallback_record_is_unverified_not_corrupt() -> None:
    """A table delivered as its *source* grid is a fallback, not a corruption."""
    graph = graph_from_blocks(
        [_formula_block("t3")],
        reconstructions=[ReconstructionRecord(block_id="t3", verified=False, corrupt=False)],
    )
    (node,) = graph.nodes
    assert isinstance(node, AssetNode)
    assert node.descriptor.integrity is AssetIntegrity.RECONSTRUCTED
    assert node.descriptor.verified is False
    assert node.descriptor.corrupt is False


def test_a_missing_asset_record_is_ignored() -> None:
    """A record for a block that is not an asset (or is absent) changes nothing."""
    graph = graph_from_blocks(
        [_formula_block("t4")],
        reconstructions=[ReconstructionRecord(block_id="other", verified=True, corrupt=False)],
    )
    (node,) = graph.nodes
    assert isinstance(node, AssetNode)
    assert node.descriptor.integrity is AssetIntegrity.PRESERVED_OPAQUE


def test_a_corrupt_reconstruction_fails_the_contract_end_to_end() -> None:
    """The reconstruction channel is reachable from the adapt path to the verdict.

    This is the acceptance the roadmap names: a deliberately-shattered rebuild
    must surface as ``ASSET_CORRUPT_RECONSTRUCTION`` — not the ``PRESERVED_OPAQUE``
    the existence-only graph used to assert for every asset.
    """
    from ubt.core.content.contract import ViolationKind, reconcile

    graph = graph_from_blocks(
        [_formula_block("t5")],
        reconstructions=[
            ReconstructionRecord(
                block_id="t5", verified=False, corrupt=True, reason="row_count_mismatch:4->3"
            )
        ],
    )
    report = reconcile(graph)
    assert report.reconstructed_assets == 1
    assert not report.passed
    assert ViolationKind.ASSET_CORRUPT_RECONSTRUCTION in {v.kind for v in report.errors}


def test_a_verified_reconstruction_passes_the_contract_end_to_end() -> None:
    from ubt.core.content.contract import reconcile

    graph = graph_from_blocks(
        [_formula_block("t6")],
        reconstructions=[ReconstructionRecord(block_id="t6", verified=True, corrupt=False)],
    )
    report = reconcile(graph)
    assert report.reconstructed_assets == 1
    assert report.passed


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


def test_a_skip_translate_block_carries_its_policy_reason_into_the_contract() -> None:
    """The contract says *why* a block shipped in the source, not just that it did.

    ``ingest`` records the classifier's reason as ``policy_reason``
    (``skip:bibliography entry …``); the contract used to flatten every such
    block to the opaque ``skip_translate``, so the audit could not tell a kept
    bibliography entry from a watermark.
    """
    block = IRBlock(
        element=make_element(
            id="bib1",
            spine_index=0,
            block_type=BlockType.NARRATIVE,
            source_text="Smith et al. IEEE, 2019",
            bbox=BoundingBox(page=1, x0=0.0, y0=0.0, x1=100.0, y1=10.0),
            skip_translate=True,
        ),
        policy_translate=False,
        policy_reason="skip:bibliography entry (kept verbatim for retrievability)",
    )
    graph = graph_from_blocks([block])
    (node,) = graph.text_nodes
    assert node.disposition is TextDisposition.VERBATIM
    assert node.reason == "skip:bibliography entry (kept verbatim for retrievability)"

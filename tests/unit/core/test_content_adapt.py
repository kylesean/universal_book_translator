from __future__ import annotations

import pytest

from ubt.core.content.adapt import graph_from_blocks, kept_in_source
from ubt.core.content.nodes import (
    AssetIntegrity,
    AssetNode,
    ReconstructionRecord,
    TextDisposition,
    TextNode,
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


def test_a_table_preserved_in_the_source_layer_is_preserved_opaque() -> None:
    block = IRBlock(
        element=make_element(
            id="t1",
            spine_index=0,
            block_type=BlockType.TABLE,
            source_text="| A | B |",
            bbox=BoundingBox(page=1, x0=0.0, y0=0.0, x1=100.0, y1=10.0),
        )
    )
    block.error_flags = ["render_skip:table_preserved_source"]
    graph = graph_from_blocks([block])
    (node,) = graph.nodes
    assert isinstance(node, AssetNode)
    assert node.descriptor.integrity is AssetIntegrity.PRESERVED_OPAQUE


def test_a_verified_reconstruction_record_is_recorded_reconstructed() -> None:
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
    graph = graph_from_blocks(
        [_formula_block("t4")],
        reconstructions=[ReconstructionRecord(block_id="other", verified=True, corrupt=False)],
    )
    (node,) = graph.nodes
    assert isinstance(node, AssetNode)
    assert node.descriptor.integrity is AssetIntegrity.PRESERVED_OPAQUE


def test_a_corrupt_reconstruction_fails_the_contract_end_to_end() -> None:
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
    graph = graph_from_blocks([_blocked_human_block()])
    (node,) = graph.text_nodes
    assert node.disposition is TextDisposition.SOURCE_KEPT
    assert node.reason.startswith("quarantined:")
    assert not kept_in_source(_formula_block())
    assert kept_in_source(_blocked_human_block())


def test_a_real_translation_is_still_delivered() -> None:
    block = _blocked_human_block("ok1")
    block.status = BlockStatus.MTQE_PASSED
    block.target_text = "读者看到的是这段中文。"
    graph = graph_from_blocks([block])
    (node,) = graph.text_nodes
    assert node.disposition is TextDisposition.TRANSLATED
    assert not kept_in_source(block)


def test_a_skip_translate_block_carries_its_policy_reason_into_the_contract() -> None:
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


def _text_node(block: IRBlock) -> TextNode:
    node = graph_from_blocks([block], doc_id="d").nodes[0]
    assert isinstance(node, TextNode)
    return node


def _prose(block_id: str, *, status: BlockStatus, flags: list[str], skip: bool = False) -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="Some source prose here.",
    )
    block = IRBlock(element=element)
    block.target_text = "某段译文。"
    block.status = status
    block.error_flags = list(flags)
    block.skip_translate = skip
    return block


def test_a_needs_human_block_is_delivered_not_missing() -> None:
    block = _prose("b1", status=BlockStatus.NEEDS_HUMAN, flags=[])
    assert kept_in_source(block) is False
    assert _text_node(block).disposition is (TextDisposition.TRANSLATED)

    noted = _prose("b2", status=BlockStatus.NEEDS_HUMAN, flags=["non_prose_degraded: note"])
    assert kept_in_source(noted) is False
    assert _text_node(noted).disposition is (TextDisposition.TRANSLATED)


def test_a_real_render_skip_is_still_source_kept() -> None:
    block = _prose("b1", status=BlockStatus.REPAIRED, flags=["render_skip:table_unparseable"])
    assert kept_in_source(block) is True
    node = _text_node(block)
    assert node.disposition is TextDisposition.SOURCE_KEPT
    assert node.reason and node.reason.startswith("render:")


def test_a_verbatim_block_is_the_only_preserved_disposition() -> None:
    block = _prose("b1", status=BlockStatus.MTQE_PASSED, flags=[], skip=True)
    assert _text_node(block).disposition is (TextDisposition.VERBATIM)

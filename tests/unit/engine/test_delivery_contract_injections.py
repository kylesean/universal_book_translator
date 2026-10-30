"""G1 delivery-contract injection suite.

One case per failure the PRD says must block export: deleted block, duplicated
block, overlap, out-of-bounds, sentinel leak, page-count mismatch, missing
asset. These are pure evidence tests (no compiler/render), so a red case names
the exact invariant that stopped protecting the artifact.

The 26-page replay (``output/adaptive_delivery/g0/26p_replay_20261030.md``)
uses the same evaluator; this suite is the deterministic matrix that must stay
green when a detector changes.
"""

from __future__ import annotations

import pytest

from ubt.core.engine.delivery_contract import (
    ContractBlock,
    ContractEvidence,
    ContractName,
    ContractReport,
    ContractTextBox,
    evaluate_delivery_contract,
)

pytestmark = pytest.mark.fast


def _block(block_id: str, text: str, *, page: int = 1) -> ContractBlock:
    return ContractBlock(block_id=block_id, text=text, page=page)


def _box(
    page: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    text: str = "",
    font_size_pt: float | None = 9.0,
) -> ContractTextBox:
    return ContractTextBox(
        page=page,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        font_size_pt=font_size_pt,
        text=text,
    )


def _evidence(
    *,
    blocks: tuple[ContractBlock, ...],
    page_texts: tuple[str, ...],
    page_count: int | None = None,
    boxes: tuple[ContractTextBox, ...] = (),
    expected_page_count: int | None = None,
    contract: ContractName = "readable",
    missing_assets: tuple[str, ...] = (),
) -> ContractEvidence:
    pages = page_count if page_count is not None else len(page_texts)
    return ContractEvidence(
        contract=contract,
        page_count=pages,
        page_texts=page_texts,
        page_bounds=dict.fromkeys(range(1, pages + 1), (0.0, 0.0, 540.0, 666.0)),
        text_boxes=boxes,
        blocks=blocks,
        expected_page_count=expected_page_count,
        missing_assets=missing_assets,
    )


def _codes(report: ContractReport) -> set[str]:
    return {finding.code for finding in report.findings}


def test_injected_deleted_block_blocks_the_artifact() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "第一段完整译文。"), _block("b2", "第二段完整译文。")),
            page_texts=("第一段完整译文。",),
        )
    )
    assert not report.passed
    assert "text_missing" in _codes(report)


def test_injected_duplicated_block_blocks_the_artifact() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "跨页段落必须只出现一次，不能在两页重复。"),),
            page_texts=(
                "跨页段落必须只出现一次，不能在两页重复。",
                "跨页段落必须只出现一次，不能在两页重复。",
            ),
        )
    )
    assert not report.passed
    assert "text_duplicated" in _codes(report)


def test_injected_overlap_blocks_the_artifact() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "正文完整译文。"),),
            page_texts=("正文完整译文。",),
            boxes=(
                _box(1, 120.0, 500.0, 420.0, 540.0, text="正文完整译文。"),
                _box(1, 120.0, 500.0, 420.0, 540.0, text="另一段完整译文。"),
            ),
        )
    )
    assert not report.passed
    assert "text_overlap" in _codes(report)


def test_injected_out_of_bounds_blocks_the_artifact() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=("完整译文。",),
            boxes=(_box(1, 120.0, -20.0, 420.0, 10.0, text="完整译文。"),),
        )
    )
    assert not report.passed
    assert "text_out_of_bounds" in _codes(report)


def test_injected_sentinel_blocks_the_artifact() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=("完整译文。⟦SOUP_MASK_0001−2a3⟧",),
        )
    )
    assert not report.passed
    assert "placeholder_leak" in _codes(report)


def test_injected_page_count_mismatch_blocks_faithful() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=("完整译文。",),
            page_count=2,
            expected_page_count=1,
            contract="faithful",
        )
    )
    assert not report.passed
    assert "page_count_mismatch" in _codes(report)


def test_injected_missing_asset_blocks_the_artifact() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=("完整译文。",),
            missing_assets=("fig_0002",),
        )
    )
    assert not report.passed
    assert "asset_missing" in _codes(report)


def test_injected_clipped_block_blocks_the_artifact() -> None:
    expected = "在鳍片表面任意位置处的表面电势是通过在鳍片表面评估两个分量之和得到的完整句子。"
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", expected),),
            page_texts=(expected[:18],),
        )
    )
    assert not report.passed
    assert "clipped_overflow" in _codes(report)

from __future__ import annotations

import pytest

from ubt.core.engine.delivery_contract import (
    ContractBlock,
    ContractEvidence,
    ContractName,
    ContractReport,
    ContractTextBox,
    OverflowRecord,
    StyleSample,
    evaluate_delivery_contract,
)

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    text: str,
    *,
    page: int = 1,
    kind: str = "body",
    allow_cross_page: bool = True,
) -> ContractBlock:
    return ContractBlock(
        block_id=block_id,
        text=text,
        page=page,
        kind=kind,
        allow_cross_page=allow_cross_page,
    )


def _box(
    page: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    block_id: str | None = None,
    font_size_pt: float | None = 9.0,
    text: str = "",
) -> ContractTextBox:
    return ContractTextBox(
        page=page,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        block_id=block_id,
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
    font_issues: tuple[str, ...] = (),
    overflow_records: tuple[OverflowRecord, ...] = (),
    style_samples: tuple[StyleSample, ...] = (),
    declared_page_expansion: bool = False,
    exhibit_texts: tuple[str, ...] = (),
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
        font_issues=font_issues,
        overflow_records=overflow_records,
        style_samples=style_samples,
        declared_page_expansion=declared_page_expansion,
        exhibit_texts=exhibit_texts,
    )


def _codes(report: ContractReport) -> set[str]:
    return {finding.code for finding in report.findings}


def test_a_complete_artifact_passes() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(
                _block("b1", "第一段完整译文。"),
                _block("b2", "第二段完整译文。", page=2),
            ),
            page_texts=("第一段完整译文。", "第二段完整译文。"),
            expected_page_count=2,
            contract="faithful",
            boxes=(
                _box(1, 120.0, 500.0, 420.0, 540.0, block_id="b1"),
                _box(2, 120.0, 500.0, 420.0, 540.0, block_id="b2"),
            ),
            style_samples=(StyleSample(page=1, size_pt=9.0), StyleSample(page=2, size_pt=9.0)),
        )
    )
    assert report.passed, report.summary_line()
    assert not report.findings


def test_a_missing_block_is_critical() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "第一段完整译文。"), _block("b2", "第二段完整译文。")),
            page_texts=("第一段完整译文。",),
        )
    )
    assert not report.passed
    assert "text_missing" in _codes(report)


def test_a_duplicated_block_is_critical() -> None:
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


def test_partial_text_is_clipped_overflow_not_a_pass() -> None:
    expected = "在鳍片表面任意位置处的表面电势是通过在鳍片表面评估两个分量之和得到的完整句子。"
    clipped = expected[:18]
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", expected),),
            page_texts=(clipped,),
        )
    )
    assert not report.passed
    assert "clipped_overflow" in _codes(report)


def test_an_overflow_record_with_full_text_is_major_not_critical() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文仍然在页面上。"),),
            page_texts=("完整译文仍然在页面上。",),
            overflow_records=(
                OverflowRecord(block_id="b1", page=1, deficit_pt=3.0, reason="clip_guard"),
            ),
        )
    )
    assert report.passed, "no critical finding: the text is present"
    assert not report.deliverable, "a major finding must block a strict delivery"
    assert "overflow_recorded" in _codes(report)


def test_large_text_overlap_is_critical_but_formula_fragments_are_not() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "正文完整译文。"), _block("b2", "标题完整译文。")),
            page_texts=("正文完整译文。标题完整译文。",),
            boxes=(
                _box(1, 120.0, 500.0, 420.0, 540.0, block_id="b2"),
                _box(1, 120.0, 495.0, 420.0, 535.0, block_id="b1"),
                _box(1, 300.0, 200.0, 312.0, 204.0, font_size_pt=None),
                _box(1, 300.0, 201.0, 316.0, 205.0, font_size_pt=None),
            ),
        )
    )
    assert "text_overlap" in _codes(report)
    overlap_pages = [f for f in report.findings if f.code == "text_overlap"]
    assert {f.page for f in overlap_pages} == {1}
    assert all("formula" not in (f.message or "").lower() for f in overlap_pages)


def test_nested_formula_aggregate_is_not_an_overlap() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(),
            page_texts=("ψB = Vtm ln(Nch/ni)",),
            boxes=(
                _box(1, 100.0, 200.0, 300.0, 260.0, text="ψB = Vtm ln(Nch/ni)"),
                _box(1, 150.0, 220.0, 200.0, 240.0, text="ψB"),
            ),
        )
    )
    assert "text_overlap" not in _codes(report)


def test_formula_subfragment_outside_aggregate_is_not_an_overlap() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(),
            page_texts=("ln cos 2 4",),
            boxes=(
                _box(1, 223.2, 120.2, 247.6, 126.4, text="ln cos\n2\n4"),
                _box(1, 233.0, 122.2, 236.0, 138.3, text="2\n4"),
            ),
        )
    )
    assert "text_overlap" not in _codes(report)


def test_equal_sized_duplicate_boxes_are_still_an_overlap() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(),
            page_texts=("重复文本",),
            boxes=(
                _box(1, 100.0, 200.0, 300.0, 260.0, text="重复文本"),
                _box(1, 100.0, 200.0, 300.0, 260.0, text="重复文本"),
            ),
        )
    )
    assert "text_overlap" in _codes(report)


def test_a_text_box_outside_the_page_is_critical() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=("完整译文。",),
            boxes=(_box(1, 120.0, -20.0, 420.0, 10.0, block_id="b1"),),
        )
    )
    assert not report.passed
    assert "text_out_of_bounds" in _codes(report)


def test_faithful_page_count_mismatch_is_critical() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=("完整译文。",),
            page_count=26,
            expected_page_count=25,
            contract="faithful",
        )
    )
    assert not report.passed
    assert "page_count_mismatch" in _codes(report)


def test_a_missing_asset_is_critical() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=("完整译文。",),
            missing_assets=("fig_0002", "formula_0094"),
        )
    )
    assert not report.passed
    assert "asset_missing" in _codes(report)


def test_a_placeholder_leak_is_critical() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=("完整译文。⟦SOUP_MASK_0001−2a3⟧",),
        )
    )
    assert not report.passed
    assert "placeholder_leak" in _codes(report)


def test_every_placeholder_occurrence_is_reported() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=(
                "正文一。⟦SOUP_MASK_0001-f2d⟧ 之后继续。",
                "正文二。⟦MATH_MASK_0002-91a⟧ 之后继续。",
                "正文三。[SENTINEL_0003] 结尾。",
            ),
        )
    )

    finding = next(f for f in report.findings if f.code == "placeholder_leak")
    assert finding.severity == "critical"
    assert "3 placeholder/sentinel or stray markup site(s)" in finding.message
    assert len(finding.occurrences) == 3
    assert "SOUP_MASK_0001-f2d" in finding.occurrences[0]
    assert "SENTINEL_0003" in finding.occurrences[2]
    payload = finding.to_payload()
    assert payload["occurrence_count"] == 3
    assert isinstance(payload["occurrences"], list)
    assert len(payload["occurrences"]) == 3


def test_the_reader_facing_missing_marker_is_not_a_leak() -> None:
    """``⟨missing⟩`` is the shipped stand-in for an unrestorable span.

    The renderer writes it deliberately (``strip_pipeline_sentinels``) and keeps
    ``sentinel_leaked`` on the block, so the artifact must not be refused for it:
    job ``web_mvyp5gnb_a8278e24df34`` reached 100% and then failed export on
    exactly one such marker, and every resume reproduced the same BLOCKED
    verdict. The excerpt below is the measured one from that contract sidecar.
    """
    excerpt = "向的空间依赖性，Nch是沟道掺杂浓度，且⟨missing⟩=𝑉tmln(𝑁ch𝑁ni)。请注意，式("
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", excerpt),),
            page_texts=(excerpt,),
        )
    )

    assert "placeholder_leak" not in _codes(report)
    assert report.passed
    assert report.deliverable
    assert next(c for c in report.checks if c["name"] == "placeholders")["status"] == "pass"


def test_a_marker_does_not_mask_the_machinery_beside_it() -> None:
    """A raw token still blocks even when a sanctioned marker shares the page."""
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=("且⟨missing⟩=𝑉tmln，后文 ⟦SOUP_MASK_0001-f2d⟧。",),
        )
    )

    finding = next(f for f in report.findings if f.code == "placeholder_leak")
    assert len(finding.occurrences) == 1
    assert "SOUP_MASK_0001-f2d" in finding.occurrences[0]
    assert not report.passed


def test_a_stray_markup_tag_in_the_artifact_is_critical() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=(
                "RustyHermit-MPK [59]</span>、CubicleOS [56]</span> 利用 Intel MPK 进行隔离。",
            ),
        )
    )
    assert not report.passed
    finding = next(f for f in report.findings if f.code == "placeholder_leak")
    assert finding.severity == "critical"
    assert "2 placeholder/sentinel or stray markup site(s)" in finding.message
    assert "</span>" in finding.occurrences[0]


def test_a_clean_artifact_carries_no_occurrence_list() -> None:
    report = evaluate_delivery_contract(
        _evidence(blocks=(_block("b1", "完整译文。"),), page_texts=("完整译文。",))
    )

    assert report.passed
    assert all(f.occurrences == () for f in report.findings)
    assert all("occurrences" not in f.to_payload() for f in report.findings)


def test_formula_subscript_cluster_is_not_an_overlap() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "正文。"),),
            page_texts=("正文。",),
            boxes=(
                _box(1, 186.2, 126.6, 193.7, 134.1, text="𝑉t", font_size_pt=7.5),
                _box(1, 190.8, 124.1, 199.5, 128.8, text="𝑉tm", font_size_pt=4.66),
            ),
        )
    )

    assert report.passed
    assert "text_overlap" not in _codes(report)


def test_same_size_duplicate_of_the_same_text_is_still_an_overlap() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "正文。"),),
            page_texts=("正文。",),
            boxes=(
                _box(1, 100.0, 500.0, 300.0, 512.0, text="同一行文字", font_size_pt=9.0),
                _box(1, 101.0, 500.5, 300.5, 512.5, text="同一行文字", font_size_pt=9.0),
            ),
        )
    )

    assert "text_overlap" in _codes(report)


def test_faithful_block_on_the_wrong_page_is_critical() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "本页译文。", page=1, allow_cross_page=False),),
            page_texts=("第 1 页", "本页译文。"),
            expected_page_count=2,
            contract="faithful",
        )
    )
    assert not report.passed
    assert "text_wrong_page" in _codes(report)


def test_readable_block_may_move_pages() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "重排后的译文。", page=2, allow_cross_page=True),),
            page_texts=("第 1 页 重排后的译文。", "第 2 页"),
            expected_page_count=2,
            contract="readable",
        )
    )
    assert report.passed, report.summary_line()


def test_style_drift_is_major() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=("完整译文。",),
            style_samples=(
                StyleSample(page=1, size_pt=7.5),
                StyleSample(page=2, size_pt=11.5),
            ),
        )
    )
    assert report.passed, "style drift is major, not critical"
    assert not report.deliverable
    assert "style_drift" in _codes(report)


def test_unmeasured_style_is_not_clean() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=("完整译文。",),
            contract="faithful",
        )
    )
    assert "style_unmeasured" in _codes(report)
    assert not report.deliverable, "unmeasured is not clean"


def test_report_serializes_to_a_stable_payload() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=("完整译文。",),
            style_samples=(StyleSample(page=1, size_pt=9.0),),
        )
    )
    payload = report.to_payload()
    assert payload["contract"] == "readable"
    assert payload["passed"] is True
    assert payload["findings"] == []
    assert payload["checks"]


def test_mixed_cjk_latin_prose_run_clipping_and_duplication_are_caught() -> None:
    full_text = (
        "其中 $V_{ch}$ 是沟道的准费米电势，它仅具有 $y$ 方向的空间依赖性。"
        "请注意，方程 (3.1) 是一个一维的泊松方程，其他两个维度被忽略是由于沟道长度较长（GCA 条件）以及 DG 结构。"
    )
    clipped_rep = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b0050", full_text, page=1, allow_cross_page=False),),
            page_texts=(
                "其中 𝑉ch 是沟道的准费米电势，它仅具有 𝑦 方向的空间依赖性。",
                "第2页：请注意方程 (3.1) 一维泊松方程其他两个维度被忽略是由于沟道长度较长 GCA 条件以及 DG 结构但顺序打乱。",
            ),
            expected_page_count=2,
            contract="faithful",
            style_samples=(StyleSample(page=1, size_pt=9.0),),
        )
    )
    assert "clipped_overflow" in _codes(clipped_rep)

    dup_rep = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b0014", full_text, page=1, allow_cross_page=False),),
            page_texts=(
                "其中 𝑉ch 是沟道的准费米电势，它仅具有 𝑦",
                "其中 𝑉ch 是沟道的准费米电势，它仅具有 𝑦 方向的空间依赖性。请注意，方程 (3.1) 是一个一维的泊松方程，其他两个维度被忽略是由于沟道长度较长（GCA 条件）以及 DG 结构。",
            ),
            expected_page_count=2,
            contract="faithful",
            style_samples=(StyleSample(page=1, size_pt=9.0),),
        )
    )
    assert "text_duplicated" in _codes(dup_rep)


def test_cross_page_prose_run_with_footer_page_number_and_zwsp_is_accepted() -> None:
    block_text = "求解漂移-扩散方程与圆柱形全环绕栅极 Cy-GAA 的核心模型基于对掺杂双栅极鳍式场效应晶体管的泊松方程解。"
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b0007", block_text, page=1, allow_cross_page=True),),
            page_texts=(
                "求解漂移-\u200b扩散方程与圆柱形全环绕栅极 Cy\ufffeGAA 的核心模型基于对掺杂双栅极鳍式场1",
                "效应晶体管的泊松方程解。2",
            ),
            contract="readable",
        )
    )
    assert report.passed, report.summary_line()
    assert not report.findings


def test_smart_quotes_and_typographic_apostrophes_are_accepted() -> None:
    block_text = "In SOSP'23, Kimi-K2.5's Agent Swarm used OpenAI's \"Code Interpreter\" framework."
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b0042", block_text, page=1, allow_cross_page=True),),
            page_texts=(
                "In SOSP’23, Kimi-K2.5′s Agent Swarm used OpenAI’s “Code Interpreter” framework.1",
            ),
            contract="readable",
        )
    )
    assert report.passed, report.summary_line()
    assert not report.findings


def test_page_level_overflow_record_is_not_dropped() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "正文内容。"),),
            page_texts=("正文内容。",),
            overflow_records=(
                OverflowRecord(
                    block_id="page:1",
                    page=1,
                    deficit_pt=1.2,
                    reason="page_lock_overflow",
                ),
            ),
            style_samples=(StyleSample(page=1, size_pt=9.0),),
        )
    )
    assert "overflow_recorded" in _codes(report)
    assert not report.deliverable


def test_large_box_edge_overlap_is_not_exempted_as_aggregate() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "正文。"),),
            page_texts=("正文。",),
            boxes=(
                _box(
                    1,
                    100.0,
                    100.0,
                    400.0,
                    150.0,
                    text="这里讨论了 FIG 3.1 的核心结构与参数设置。",
                    font_size_pt=9.5,
                ),
                _box(
                    1,
                    120.0,
                    145.0,
                    170.0,
                    155.0,
                    text="FIG 3.1",
                    font_size_pt=8.0,
                ),
            ),
        )
    )
    assert "text_overlap" in _codes(report)


def test_a_declared_flow_expansion_passes_with_more_pages() -> None:
    common_blocks = (_block("b1", "第一段完整译文。"), _block("b2", "第二段完整译文。"))
    common_texts = ("第一段完整译文。", "第二段完整译文。", "延续页上的后续内容。")

    declared = evaluate_delivery_contract(
        _evidence(
            blocks=common_blocks,
            page_texts=common_texts,
            expected_page_count=2,
            contract="faithful",
            declared_page_expansion=True,
        )
    )
    assert declared.passed
    assert "page_count_mismatch" not in _codes(declared)
    assert any(
        c["name"] == "page_count" and "flow-expanded" in str(c.get("status", ""))
        for c in declared.checks
    )

    undeclared = evaluate_delivery_contract(
        _evidence(
            blocks=common_blocks,
            page_texts=common_texts,
            expected_page_count=2,
            contract="faithful",
        )
    )
    assert not undeclared.passed
    assert "page_count_mismatch" in _codes(undeclared)

    shrunk = evaluate_delivery_contract(
        _evidence(
            blocks=common_blocks,
            page_texts=("第一段完整译文。",),
            page_count=1,
            expected_page_count=2,
            contract="faithful",
            declared_page_expansion=True,
        )
    )
    assert not shrunk.passed
    assert "page_count_mismatch" in _codes(shrunk)


def test_shared_phrases_or_labels_across_blocks_do_not_falsely_trigger_text_duplicated() -> None:
    # b1 is a short label/list item, b2 is a larger table/paragraph that contains the same label
    b1 = _block("b1", "ARC-Challenge HellaSwag", page=12)
    b2 = _block("b2", "| 模型 | ARC-Challenge HellaSwag | 分数 |\n| Ouro | 75.2 |", page=32)

    # In the rendered document, each page legitimately renders its respective block
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(b1, b2),
            page_texts=(
                "• ARC-Challenge HellaSwag",
                "| 模型 | ARC-Challenge HellaSwag | 分数 |\n| Ouro | 75.2 |",
            ),
            page_count=2,
            contract="readable",
        )
    )
    assert report.passed, report.summary_line()
    assert "text_duplicated" not in _codes(report)

    # Formula-split block whose short prose run is shared with another block
    b3 = _block("b3", "LoopCD-Hidden 将 $h_1$ 和 $h_R$ 合并并求值一次。", page=4)
    b4 = _block("b4", "在评估中，LoopCD-Hidden 将准确率大幅提升。", page=1)
    report_formula = evaluate_delivery_contract(
        _evidence(
            blocks=(b3, b4),
            page_texts=(
                "在评估中，LoopCD-Hidden 将准确率大幅提升。",
                "LoopCD-Hidden 将 ℎ1 和 ℎ𝑅 合并并求值一次。",
            ),
            page_count=2,
            contract="readable",
        )
    )
    assert report_formula.passed, report_formula.summary_line()
    assert "text_duplicated" not in _codes(report_formula)


def test_cross_page_run_with_kerning_spaces_passes() -> None:
    # A multi-page block whose prose run crosses the page boundary
    text = "对于LoopCD-Hidden，参考状态的选择会考虑模型的初始化（附录E.1）。Parcae1.3B具有确定性前奏，因此其对比在R=8时生效。"
    b = ContractBlock(
        block_id="b_cross",
        text=text,
        page=1,
        kind="narrative",
        allow_cross_page=True,
    )
    # Page 1 ends with first half; Page 2 starts with second half with Typst-style spacing around Latin/punctuation
    p1 = "对于LoopCD-Hidden，参考状态的选择会考虑模型的初始化"
    p2 = "（附录 E.1） 。Parcae1.3B 具有确定性前奏，因此其对比在 R = 8 时生效。"
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(b,),
            page_texts=(p1, p2),
            page_count=2,
            expected_page_count=2,
            contract="faithful",
        )
    )
    assert report.passed, report.summary_line()
    assert "clipped_overflow" not in _codes(report)


def test_block_text_shared_with_multipage_exhibit_not_flagged_duplicated() -> None:
    # A block whose text is a benchmark name or header that recurs in a table header spanning pages
    prose_block = ContractBlock(
        block_id="b0092",
        text="ARC-Challenge HellaSwag",
        page=1,
        kind="list_item",
        allow_cross_page=True,
    )
    table_markup = "| ARC-Challenge HellaSwag | Model |\n|---|---|\n| 80.5 | Ouro |"
    # Page 1 has the prose block.
    # Pages 2 and 3 repeat the table header across pages (repeat: true).
    p1 = "Benchmark names: ARC-Challenge HellaSwag on page 1."
    p2 = "Table A11 (Part 1)\nARC-Challenge HellaSwag | Model\n80.5 | Ouro"
    p3 = "Table A11 (Part 2)\nARC-Challenge HellaSwag | Model\n75.2 | Huginn"

    report = evaluate_delivery_contract(
        _evidence(
            blocks=(prose_block,),
            exhibit_texts=(table_markup,),
            page_texts=(p1, p2, p3),
            page_count=3,
            expected_page_count=3,
            contract="readable",
        )
    )
    assert report.passed, report.summary_line()
    assert "text_duplicated" not in _codes(report)

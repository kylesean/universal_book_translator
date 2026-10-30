"""The delivery contract: eight invariants, one verdict, no self-grading.

The current export path writes a quality report, an attestation sidecar and a
visual report, and then ships even when they disagree with each other. The
26-page BSIM-CMG job is the measured example: 151/155 text realizations, 43
missing assets, seven ``needs_human`` rows, pages that visibly overlap — and a
report that says ``passed: true`` with ``needs_human_blocks: 0``.

This module is the single verdict those separate reports were missing. It reads
*evidence* (page text, artifact text boxes, page bounds, asset/font findings,
render overflow records) and returns one report. The tests below are pure data:
they pin each invariant without a compiler, so a red test names the exact
contract that failed rather than a page number.

Two rules are deliberate and are asserted here:

* **Unmeasured is not clean.** A check that could not run must say so; silence
  must never be reported as a pass. The old visual gate's ``passed: true`` on a
  page it never sampled is the failure this rule exists to prevent.
* **A clipped block is not a smaller block.** ``clip: true`` stops ink from
  painting outside its box, which is correct, but the text that did not fit is
  still lost. ``clipped_overflow`` is therefore critical, not cosmetic.
"""

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
    """The p2/p3 defect: one paragraph appended to both pages."""
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
    """``clip: true`` hides the ink; it must not hide the missing text."""
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
    """A recorded overflow that did not actually cut text is still not clean."""
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
    """p2's heading-on-body overlap is large; formula superscripts are not."""
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "正文完整译文。"), _block("b2", "标题完整译文。")),
            page_texts=("正文完整译文。标题完整译文。",),
            boxes=(
                # A heading box and a body box genuinely on top of each other.
                _box(1, 120.0, 500.0, 420.0, 540.0, block_id="b2"),
                _box(1, 120.0, 495.0, 420.0, 535.0, block_id="b1"),
                # Formula debris: tiny fragments that legitimately stack.
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
    """PDFium returns the whole formula rect and the glyph rects inside it.

    The 26-page replay produced 47 ``text_overlap`` findings from these
    aggregate/child pairs. A parent whose text is a superset of the child is
    one formula, not two blocks on top of each other.
    """
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
    """A formula subscript can extend past the aggregate rect's bottom edge."""
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
    """The aggregate exception must not hide a real duplicate text layer."""
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
    """AD-022: one message hid two more leaks behind the first one.

    The 26-page replay needed three separate repairs, and after each one the
    report showed the *same shape* of message — indistinguishable from a defect
    that cannot be fixed. The finding now counts them and lists each excerpt.
    """
    report = evaluate_delivery_contract(
        _evidence(
            blocks=(_block("b1", "完整译文。"),),
            page_texts=(
                "正文一。⟦SOUP_MASK_0001-f2d⟧ 之后继续。",
                "正文二。⟦MATH_MASK_0002-91a⟧ 之后继续。",
                "正文三。⟨missing⟩ 结尾。",
            ),
        )
    )

    finding = next(f for f in report.findings if f.code == "placeholder_leak")
    assert finding.severity == "critical"
    assert "3 placeholder/sentinel site(s)" in finding.message
    assert len(finding.occurrences) == 3
    assert "SOUP_MASK_0001-f2d" in finding.occurrences[0]
    assert "⟨missing⟩" in finding.occurrences[2]
    payload = finding.to_payload()
    assert payload["occurrence_count"] == 3
    assert isinstance(payload["occurrences"], list)
    assert len(payload["occurrences"]) == 3


def test_a_clean_artifact_carries_no_occurrence_list() -> None:
    report = evaluate_delivery_contract(
        _evidence(blocks=(_block("b1", "完整译文。"),), page_texts=("完整译文。",))
    )

    assert report.passed
    assert all(f.occurrences == () for f in report.findings)
    assert all("occurrences" not in f.to_payload() for f in report.findings)


def test_formula_subscript_cluster_is_not_an_overlap() -> None:
    """The readable replay's shape: ``𝑉t`` 7.5pt under ``𝑉tm`` 4.66pt.

    pdfium's rect for the base glyph carries the shorter text and the
    subscript's rect the longer one, so the *containment* points from the larger
    area to the smaller. Same glyph cluster at two sizes, printed once — not two
    blocks colliding. A same-size duplicate of the same text must still be
    flagged, because that one is a real double layer.
    """
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
    """A contract that could not measure style must say so, not pass silently."""
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

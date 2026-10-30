"""The contract's group-aware style check (G4 Stage 3A).

Two questions have to stay separate, and these tests are what keep them apart:

* **drift** — a member page whose *measured* size misses the declared group size;
* **exception** — a page the group declared up front it could not hold.

Merging them would let either one hide the other: a book that shrinks every
page uniformly would have zero drift, and a book that declares every page an
exception would have zero exceptions-worth of drift. The 26-page replay is why
the second one matters — its 2.29pt spread was real, page-level shrinking, and
"it was all legitimate" is exactly the excuse the group record exists to refuse.
"""

from __future__ import annotations

import pytest

from ubt.core.engine.delivery_contract import (
    ContractBlock,
    ContractEvidence,
    ContractTextBox,
    StyleSample,
    evaluate_delivery_contract,
)
from ubt.core.layout.style_group import StyleGroup, StyleGroupException, StyleGroupPlan

pytestmark = pytest.mark.fast


def _plan(
    size_pt: float,
    *,
    pages: tuple[int, ...],
    exceptions: tuple[StyleGroupException, ...] = (),
) -> StyleGroupPlan:
    return StyleGroupPlan(
        group=StyleGroup(
            group_id="body:document",
            role="body",
            size_pt=size_pt,
            pitch=1.09,
            tracking_em=-0.030,
            width_borrow_pt=4.0,
            source_size_pt=8.9,
            source_pitch=1.35,
            pages=pages,
            floor_pt=7.5,
            ceiling_pt=9.79,
            rung="leading_-0.26",
        ),
        exceptions=exceptions,
    )


def _evidence(
    *,
    sizes: dict[int, float],
    plan: StyleGroupPlan | None,
    pages: int,
    blocks_per_page: int = 1,
    blocks: tuple[ContractBlock, ...] | None = None,
) -> ContractEvidence:
    corpus: dict[int, list[str]] = {
        page: [f"第{page}页正文第{index}段。" for index in range(1, blocks_per_page + 1)]
        for page in range(1, pages + 1)
    }
    page_texts = tuple("".join(corpus[page]) for page in range(1, pages + 1))
    boxes = tuple(
        ContractTextBox(
            page=page,
            x0=40.0,
            y0=40.0,
            x1=500.0,
            y1=100.0,
            font_size_pt=size,
        )
        for page, size in sizes.items()
    )
    if blocks is None:
        blocks = tuple(
            ContractBlock(
                block_id=f"b{page}-{index}",
                text=text,
                page=page,
                kind="body",
                allow_cross_page=False,
            )
            for page in range(1, pages + 1)
            for index, text in enumerate(corpus[page], start=1)
        )
    return ContractEvidence(
        contract="faithful",
        page_count=pages,
        page_texts=page_texts,
        page_bounds=dict.fromkeys(range(1, pages + 1), (0.0, 0.0, 540.0, 666.0)),
        text_boxes=boxes,
        blocks=blocks,
        expected_page_count=pages,
        style_samples=tuple(
            StyleSample(page=page, size_pt=size, group_id="body:document")
            for page, size in sizes.items()
        ),
        style_group=plan,
    )


def test_a_group_that_holds_its_declared_size_is_clean() -> None:
    report = evaluate_delivery_contract(
        _evidence(
            sizes=dict.fromkeys(range(1, 11), 8.4),
            plan=_plan(8.4, pages=tuple(range(1, 11))),
            pages=10,
        )
    )
    assert "style_drift" not in {finding.code for finding in report.findings}
    assert "style_group_exception" not in {finding.code for finding in report.findings}
    assert report.deliverable


def test_a_member_page_that_misses_the_declared_size_is_drift() -> None:
    """Deviation is measured against the group's declaration, not against peers."""
    sizes = dict.fromkeys(range(1, 11), 8.4)
    sizes[7] = 9.6
    report = evaluate_delivery_contract(
        _evidence(sizes=sizes, plan=_plan(8.4, pages=tuple(range(1, 11))), pages=10)
    )
    drift = [finding for finding in report.findings if finding.code == "style_drift"]
    assert drift
    assert drift[0].page == 7
    assert drift[0].severity == "major"
    assert "p7 9.60pt" in drift[0].occurrences


def test_a_declared_exception_is_reported_and_blocks_no_major_delivery() -> None:
    sizes = dict.fromkeys(range(1, 11), 8.4)
    sizes[9] = 7.5
    plan = _plan(
        8.4,
        pages=tuple(range(1, 11)),
        exceptions=(
            StyleGroupException(page=9, block_ids=("b9-1",), deficit_pt=9.6, rung="leading_-0.26"),
        ),
    )
    report = evaluate_delivery_contract(
        _evidence(sizes=sizes, plan=plan, pages=10, blocks_per_page=10)
    )
    codes = {finding.code for finding in report.findings}
    assert "style_group_exception" in codes
    # The exception page is not also counted as drift: it was declared, and
    # double-reporting the same page as two faults teaches reviewers to ignore
    # the second one.
    assert "style_drift" not in codes
    assert not report.deliverable
    assert report.passed  # major, not critical: explicit accept is possible


def test_too_many_exceptions_escalate_to_critical() -> None:
    """Past the budget the group is not a group, and that must not ship."""
    pages = 10
    sizes = dict.fromkeys(range(1, pages + 1), 8.4)
    for page in (8, 9, 10):
        sizes[page] = 7.5
    plan = _plan(
        8.4,
        pages=tuple(range(1, pages + 1)),
        exceptions=tuple(
            StyleGroupException(page=page, block_ids=(f"b{page}-1",), deficit_pt=2.0, rung="plain")
            for page in (8, 9, 10)
        ),
    )
    report = evaluate_delivery_contract(
        _evidence(sizes=sizes, plan=plan, pages=pages, blocks_per_page=10)
    )
    flood = [
        finding for finding in report.findings if finding.code == "style_group_exception_flood"
    ]
    assert flood
    assert flood[0].severity == "critical"
    assert not report.passed


def test_a_page_cannot_excuse_itself_by_being_drawn_at_another_size() -> None:
    """Membership comes from the plan; the artifact only supplies the number."""
    sizes = dict.fromkeys(range(1, 11), 8.4)
    sizes[6] = 9.2
    report = evaluate_delivery_contract(
        _evidence(sizes=sizes, plan=_plan(8.4, pages=tuple(range(1, 11))), pages=10)
    )
    assert "style_drift" in {finding.code for finding in report.findings}


def test_without_a_group_record_the_document_wide_check_still_runs() -> None:
    """Artifacts that predate the group must not lose their style audit."""
    sizes = dict.fromkeys(range(1, 11), 8.4)
    sizes[5] = 9.6
    report = evaluate_delivery_contract(_evidence(sizes=sizes, plan=None, pages=10))
    codes = {finding.code for finding in report.findings}
    assert "style_drift" in codes
    assert "style_group_exception" not in codes
    assert not any(check["name"] == "style_group" for check in report.checks)

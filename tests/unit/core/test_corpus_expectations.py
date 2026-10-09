"""The corpus expectation gate: a case's ``expect`` thresholds decide pass/fail.

``evaluate_expectations`` turns a case's thresholds into failures. The subtle one
is ``max_errors``: it is documented as the number of ERROR-severity violations a
case *allows* (default 0), and it must actually be honoured -- a case that raises
it is how the corpus scopes a known, documented artifact (a mock-echo fragment
that stays source) without hiding a genuinely lost translation elsewhere.
"""

from __future__ import annotations

import pytest

from ubt.core.content.contract import ReconciliationReport, reconcile
from ubt.core.content.graph import ContentGraph
from ubt.core.content.nodes import TextDisposition, TextNode
from ubt.core.content.verify import evaluate_expectations

pytestmark = pytest.mark.fast


def _report(*nodes: TextNode) -> ReconciliationReport:
    return reconcile(ContentGraph(doc_id="d", title="t", nodes=tuple(nodes)))


def _text(node_id: str, disposition: TextDisposition, reason: str = "") -> TextNode:
    return TextNode(
        id=node_id, order=0, source_text="source", disposition=disposition, reason=reason
    )


def test_the_default_error_floor_is_zero() -> None:
    # A space failure with no ``expect`` at all still fails: 0 is the default.
    report = _report(_text("t1", TextDisposition.SOURCE_KEPT, "does not fit"))
    assert report.errors
    assert evaluate_expectations(report, {}) == ["1 error(s) > allowed 0"]


def test_a_raised_max_errors_admits_exactly_that_many() -> None:
    report = _report(_text("t1", TextDisposition.SOURCE_KEPT, "does not fit"))
    assert evaluate_expectations(report, {"max_errors": 1}) == []
    assert evaluate_expectations(report, {"max_errors": 0}) == ["1 error(s) > allowed 0"]


def test_a_raised_max_errors_still_catches_an_extra_error() -> None:
    # The floor is scoped, not a blanket exemption: a second lost node trips it.
    report = _report(
        _text("t1", TextDisposition.SOURCE_KEPT, "does not fit"),
        _text("t2", TextDisposition.SOURCE_KEPT, "overflow"),
    )
    assert evaluate_expectations(report, {"max_errors": 1}) == ["2 error(s) > allowed 1"]


def test_max_errors_does_not_mask_an_unrelated_threshold() -> None:
    # Allowing an error does not allow a missing asset: thresholds are independent.
    report = _report(_text("t1", TextDisposition.SOURCE_KEPT, "does not fit"))
    failures = evaluate_expectations(report, {"max_errors": 1, "min_delivered_ratio": 0.9})
    assert failures == ["delivered ratio 0.000 < required 0.900"]

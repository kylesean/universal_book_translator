"""Reconciliation: does a delivery account for every content node?

This is the gate the architecture exists to provide, and it reads the content
graph and its ledgers -- not the render path. The rules are the two axioms made
checkable:

- **Axiom B (text):** a text node is accounted for when it is TRANSLATED or
  deliberately VERBATIM. PENDING was never decided and SKIPPED was dropped --
  both are errors. SOURCE_KEPT is an explicit keep: a warning, *unless* the
  reason is a space failure (the translation existed but could not be placed),
  which is an error.
- **Axiom A (assets):** an asset is accounted for when PRESERVED_OPAQUE, a
  *verified* RECONSTRUCTED, or an intentional DROPPED. MISSING is an error; an
  unverified reconstruction is a warning and a corrupt one an error.

``passed`` is the single machine-checkable verdict: no ERROR-severity violation.
"""

from __future__ import annotations

import pytest

from ubt.core.content.contract import (
    Severity,
    ViolationKind,
    _is_space_failure,
    reconcile,
)
from ubt.core.content.graph import ContentGraph
from ubt.core.content.nodes import (
    AssetDescriptor,
    AssetIntegrity,
    AssetKind,
    AssetNode,
    AssetRepresentation,
    TextDisposition,
    TextNode,
)

pytestmark = pytest.mark.fast


def _text(node_id: str, disposition: TextDisposition, reason: str = "") -> TextNode:
    return TextNode(
        id=node_id, order=0, source_text="source", disposition=disposition, reason=reason
    )


def _asset(
    node_id: str,
    integrity: AssetIntegrity,
    *,
    verified: bool = False,
    corrupt: bool = False,
    detail: str = "",
) -> AssetNode:
    return AssetNode(
        id=node_id,
        order=0,
        descriptor=AssetDescriptor(
            asset_kind=AssetKind.TABLE,
            representation=AssetRepresentation.STRUCTURED_TABLE,
            integrity=integrity,
            verified=verified,
            corrupt=corrupt,
            detail=detail,
        ),
    )


def _graph(*nodes: TextNode | AssetNode) -> ContentGraph:
    return ContentGraph(doc_id="d", title="t", nodes=nodes)


# --------------------------------------------------------------------------- #
# _is_space_failure: the source-kept severity switch.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "reason", ["spill", "SPILL", "overflow", "no_fit", "text does not fit the box"]
)
def test_space_failure_reasons_are_recognized_case_insensitively(reason: str) -> None:
    assert _is_space_failure(reason) is True


@pytest.mark.parametrize("reason", ["unrenderable math", "quarantined block", ""])
def test_other_source_kept_reasons_are_not_space_failures(reason: str) -> None:
    assert _is_space_failure(reason) is False


# --------------------------------------------------------------------------- #
# Text dispositions.
# --------------------------------------------------------------------------- #


def test_translated_text_is_delivered_without_a_violation() -> None:
    report = reconcile(_graph(_text("t1", TextDisposition.TRANSLATED)))
    assert report.delivered_text == 1
    assert report.violations == ()
    assert report.passed


def test_verbatim_text_is_delivered_without_a_violation() -> None:
    report = reconcile(_graph(_text("t1", TextDisposition.VERBATIM)))
    assert report.verbatim_text == 1
    assert report.violations == ()
    assert report.passed


def test_source_kept_with_a_normal_reason_is_a_warning() -> None:
    report = reconcile(_graph(_text("t1", TextDisposition.SOURCE_KEPT, "unrenderable math")))
    assert report.source_kept_text == 1
    (violation,) = report.violations
    assert violation.kind is ViolationKind.TEXT_SOURCE_KEPT
    assert violation.severity is Severity.WARNING
    assert report.passed


def test_source_kept_because_it_could_not_be_placed_is_an_error() -> None:
    report = reconcile(_graph(_text("t1", TextDisposition.SOURCE_KEPT, "does not fit")))
    (violation,) = report.violations
    assert violation.severity is Severity.ERROR
    assert not report.passed


def test_skipped_text_is_an_undelivered_error() -> None:
    report = reconcile(_graph(_text("t1", TextDisposition.SKIPPED)))
    assert report.skipped_text == 1
    (violation,) = report.violations
    assert violation.kind is ViolationKind.TEXT_UNDELIVERED
    assert violation.severity is Severity.ERROR


def test_pending_text_is_an_unaccounted_error() -> None:
    report = reconcile(_graph(_text("t1", TextDisposition.PENDING)))
    assert report.pending_text == 1
    (violation,) = report.violations
    assert violation.kind is ViolationKind.TEXT_UNACCOUNTED
    assert violation.severity is Severity.ERROR


# --------------------------------------------------------------------------- #
# Asset integrity.
# --------------------------------------------------------------------------- #


def test_preserved_asset_is_accounted_without_a_violation() -> None:
    report = reconcile(_graph(_asset("a1", AssetIntegrity.PRESERVED_OPAQUE)))
    assert report.preserved_assets == 1
    assert report.violations == ()
    assert report.passed


def test_intentionally_dropped_asset_is_accounted_without_a_violation() -> None:
    report = reconcile(_graph(_asset("a1", AssetIntegrity.DROPPED)))
    assert report.dropped_assets == 1
    assert report.violations == ()
    assert report.passed


def test_missing_asset_is_an_error() -> None:
    report = reconcile(_graph(_asset("a1", AssetIntegrity.MISSING)))
    assert report.missing_assets == 1
    (violation,) = report.violations
    assert violation.kind is ViolationKind.ASSET_MISSING
    assert violation.severity is Severity.ERROR


def test_a_verified_reconstruction_is_accounted_without_a_violation() -> None:
    report = reconcile(_graph(_asset("a1", AssetIntegrity.RECONSTRUCTED, verified=True)))
    assert report.reconstructed_assets == 1
    assert report.violations == ()
    assert report.passed


def test_an_unverified_reconstruction_is_a_warning() -> None:
    report = reconcile(_graph(_asset("a1", AssetIntegrity.RECONSTRUCTED)))
    (violation,) = report.violations
    assert violation.kind is ViolationKind.ASSET_UNVERIFIED_RECONSTRUCTION
    assert violation.severity is Severity.WARNING
    assert report.passed


def test_a_corrupt_reconstruction_is_an_error() -> None:
    report = reconcile(_graph(_asset("a1", AssetIntegrity.RECONSTRUCTED, corrupt=True)))
    (violation,) = report.violations
    assert violation.kind is ViolationKind.ASSET_CORRUPT_RECONSTRUCTION
    assert violation.severity is Severity.ERROR


# --------------------------------------------------------------------------- #
# The report: totals, verdict, summary.
# --------------------------------------------------------------------------- #


def test_an_empty_graph_passes_with_zero_counts() -> None:
    report = reconcile(ContentGraph())
    assert report.total_text == 0
    assert report.total_assets == 0
    assert report.violations == ()
    assert report.passed


def test_totals_count_every_disposition_and_integrity() -> None:
    report = reconcile(
        _graph(
            _text("t1", TextDisposition.TRANSLATED),
            _text("t2", TextDisposition.VERBATIM),
            _text("t3", TextDisposition.SOURCE_KEPT, "unrenderable math"),
            _text("t4", TextDisposition.SKIPPED),
            _text("t5", TextDisposition.PENDING),
            _asset("a1", AssetIntegrity.PRESERVED_OPAQUE),
            _asset("a2", AssetIntegrity.RECONSTRUCTED, verified=True),
            _asset("a3", AssetIntegrity.MISSING),
        )
    )
    assert (report.total_text, report.delivered_text, report.verbatim_text) == (5, 1, 1)
    assert (report.source_kept_text, report.skipped_text, report.pending_text) == (1, 1, 1)
    assert (report.total_assets, report.reconstructed_assets) == (3, 1)
    assert (report.preserved_assets, report.missing_assets) == (1, 1)


def test_errors_and_warnings_split_by_severity() -> None:
    report = reconcile(
        _graph(
            _text("t1", TextDisposition.SOURCE_KEPT, "unrenderable math"),  # warning
            _text("t2", TextDisposition.SKIPPED),  # error
        )
    )
    assert [v.node_id for v in report.errors] == ["t2"]
    assert [v.node_id for v in report.warnings] == ["t1"]
    assert not report.passed


def test_summary_line_states_the_verdict_and_the_counts() -> None:
    report = reconcile(_graph(_text("t1", TextDisposition.TRANSLATED)))
    assert report.summary_line().startswith("[PASS] text 1/1 delivered")
    assert report.summary_line().endswith("0 error(s), 0 warning(s)")


def test_a_violation_carries_the_node_reason_or_a_default_detail() -> None:
    with_reason = reconcile(_graph(_text("t1", TextDisposition.SOURCE_KEPT, "unrenderable math")))
    assert with_reason.violations[0].detail == "unrenderable math"
    without_reason = reconcile(_graph(_text("t1", TextDisposition.SKIPPED)))
    assert without_reason.violations[0].detail

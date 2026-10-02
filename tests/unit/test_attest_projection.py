"""``project_contract()``: the delivery contract as a projection of attestations.

``reconcile()`` balanced two ledgers *after* a delivery was built;
``project_contract`` is the switch: the attestations are now the account, and
the graph only classifies. The contracts pinned here:

- the delivered-text count comes from the attestations
  (``RECONSTRUCTED_ADAPTED``), not from the graph's ``TRANSLATED`` rows;
- a graph ``TRANSLATED`` node whose attestation is not a verified
  reconstruction -- or that has no attestation at all -- is **demoted** to
  source-kept with a warning, so the text account still balances;
- a deliberate ``VERBATIM`` keep and an explicit ``SOURCE_KEPT`` are never
  demoted (the demotion filter only touches ``TRANSLATED``);
- an element with no lossless realization is a ``TEXT_UNDELIVERED`` error: the
  construction-time loss the graph cannot see (its block still carries a
  target);
- the asset account is rebalanced by **remainder**: the attestations decide how
  many assets were reconstructed, everything neither reconstructed, dropped nor
  missing is preserved -- so the books balance even where the graph and the
  attestations disagree about which blocks are assets;
- the graph's own violations survive the projection (base + demoted + lost).
"""

from __future__ import annotations

import pytest

from ubt.core.content.contract import Severity, ViolationKind
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
from ubt.model.fidelity import Attestation, Fidelity, Proof, ProofKind
from ubt.pipeline.attest import AttestationReport, project_contract

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
) -> AssetNode:
    return AssetNode(
        id=node_id,
        order=0,
        descriptor=AssetDescriptor(
            asset_kind=AssetKind.FORMULA,
            representation=AssetRepresentation.LATEX,
            integrity=integrity,
            verified=verified,
        ),
    )


def _graph(*nodes: TextNode | AssetNode) -> ContentGraph:
    return ContentGraph(doc_id="doc", title="t", nodes=nodes)


def _attestation(element_id: str, fidelity: Fidelity) -> Attestation:
    return Attestation(
        element_id=element_id, fidelity=fidelity, proof=Proof.ok(ProofKind.PREDICATE)
    )


def _report(
    *,
    text: tuple[tuple[str, int], ...] = (),
    assets: tuple[tuple[str, int], ...] = (),
    violations: tuple[str, ...] = (),
    attestations: tuple[Attestation, ...] = (),
) -> AttestationReport:
    return AttestationReport(
        total=len(attestations) + len(violations),
        text=text,
        assets=assets,
        violations=violations,
        attestations=attestations,
    )


# --------------------------------------------------------------------------- #
# The delivered-text account is the attestations, not the graph.
# --------------------------------------------------------------------------- #


def test_a_verified_translation_is_delivered_without_a_violation() -> None:
    graph = _graph(_text("t1", TextDisposition.TRANSLATED))
    report = _report(
        text=(("RECONSTRUCTED_ADAPTED", 1),),
        attestations=(_attestation("t1", Fidelity.RECONSTRUCTED_ADAPTED),),
    )
    projected = project_contract(report, graph)
    assert projected.delivered_text == 1
    assert projected.violations == ()
    assert projected.passed


def test_an_unverified_translation_is_demoted_to_source_kept_with_a_warning() -> None:
    # The block still carries a target, but realize() did not reconstruct and
    # verify it -- the reader gets the source, recorded as an explicit keep.
    graph = _graph(_text("t1", TextDisposition.TRANSLATED))
    report = _report(
        text=(("PRESERVED_OPAQUE", 1),),
        attestations=(_attestation("t1", Fidelity.PRESERVED_OPAQUE),),
    )
    projected = project_contract(report, graph)
    (violation,) = projected.violations
    assert violation.kind is ViolationKind.TEXT_SOURCE_KEPT
    assert violation.severity is Severity.WARNING
    assert violation.node_id == "t1"
    assert violation.detail == "translation did not verify; source kept"
    assert projected.delivered_text == 0
    assert projected.source_kept_text == 1
    assert projected.passed


def test_a_translated_node_with_no_attestation_is_demoted_too() -> None:
    # An attestation absence is the same as a non-reconstructed one: the
    # account must not let a graph row claim a delivery the attestations lost.
    graph = _graph(_text("t1", TextDisposition.TRANSLATED), _text("t2", TextDisposition.TRANSLATED))
    report = _report(
        text=(("RECONSTRUCTED_ADAPTED", 1),),
        attestations=(_attestation("t1", Fidelity.RECONSTRUCTED_ADAPTED),),
    )
    projected = project_contract(report, graph)
    assert projected.delivered_text == 1
    assert projected.source_kept_text == 1
    assert [v.node_id for v in projected.warnings] == ["t2"]


def test_verbatim_and_explicit_source_keeps_are_never_demoted() -> None:
    # The demotion filter only touches TRANSLATED rows: a deliberate keep is
    # honest, and an explicit source-keep already has its graph-recorded reason.
    graph = _graph(
        _text("t1", TextDisposition.VERBATIM),
        _text("t2", TextDisposition.SOURCE_KEPT, "unrenderable math"),
    )
    report = _report(
        text=(("PRESERVED_OPAQUE", 2),),
        attestations=(
            _attestation("t1", Fidelity.PRESERVED_OPAQUE),
            _attestation("t2", Fidelity.PRESERVED_OPAQUE),
        ),
    )
    projected = project_contract(report, graph)
    (violation,) = projected.violations
    assert violation.node_id == "t2"
    assert violation.detail == "unrenderable math"  # the graph's reason, unchanged
    assert projected.verbatim_text == 1
    assert projected.source_kept_text == 1  # not 2 -- nothing was demoted
    assert projected.passed


# --------------------------------------------------------------------------- #
# The construction-time loss the graph cannot see.
# --------------------------------------------------------------------------- #


def test_an_element_without_a_lossless_realization_is_an_undelivered_error() -> None:
    graph = _graph(_text("t1", TextDisposition.TRANSLATED))
    report = _report(violations=("t1",))
    projected = project_contract(report, graph)
    errors = projected.errors
    assert [v.node_id for v in errors] == ["t1"]
    assert errors[0].kind is ViolationKind.TEXT_UNDELIVERED
    assert errors[0].severity is Severity.ERROR
    assert errors[0].detail == "no lossless realization (lossless realization axiom)"
    assert not projected.passed


def test_graph_violations_survive_the_projection() -> None:
    # The graph still classifies: a node nobody decided on stays an error, and
    # the report keeps base + demoted + lost in that order.
    graph = _graph(_text("t1", TextDisposition.PENDING), _text("t2", TextDisposition.TRANSLATED))
    report = _report(
        text=(("PRESERVED_OPAQUE", 1),),
        attestations=(_attestation("t2", Fidelity.PRESERVED_OPAQUE),),
    )
    projected = project_contract(report, graph)
    assert [v.kind for v in projected.violations] == [
        ViolationKind.TEXT_UNACCOUNTED,
        ViolationKind.TEXT_SOURCE_KEPT,
    ]
    assert not projected.passed


# --------------------------------------------------------------------------- #
# The asset account: attestations decide, remainder keeps the books balanced.
# --------------------------------------------------------------------------- #


def test_a_verified_asset_reconstruction_is_counted_from_the_attestations() -> None:
    graph = _graph(_asset("a1", AssetIntegrity.RECONSTRUCTED, verified=True))
    report = _report(
        assets=(("RECONSTRUCTED_VERIFIED", 1),),
        attestations=(_attestation("a1", Fidelity.RECONSTRUCTED_VERIFIED),),
    )
    projected = project_contract(report, graph)
    assert projected.reconstructed_assets == 1
    assert projected.preserved_assets == 0
    assert projected.violations == ()


def test_a_graph_reconstruction_without_an_attestation_becomes_preserved() -> None:
    # The attestations decide how many assets were *reconstructed*; what is
    # left over after dropped and missing is preserved -- not lost.
    graph = _graph(_asset("a1", AssetIntegrity.RECONSTRUCTED, verified=True))
    report = _report()
    projected = project_contract(report, graph)
    assert projected.reconstructed_assets == 0
    assert projected.preserved_assets == 1
    assert projected.total_assets == 1


def test_reconstruction_counts_above_the_graph_are_clamped() -> None:
    # The graph is the contract's body count: an attestation histogram that
    # claims more reconstructions than the graph has assets cannot inflate it.
    graph = _graph(_asset("a1", AssetIntegrity.PRESERVED_OPAQUE))
    report = _report(
        assets=(("RECONSTRUCTED_VERIFIED", 2),),
        attestations=(
            _attestation("a1", Fidelity.RECONSTRUCTED_VERIFIED),
            _attestation("a2", Fidelity.RECONSTRUCTED_VERIFIED),
        ),
    )
    projected = project_contract(report, graph)
    assert projected.total_assets == 1
    assert projected.reconstructed_assets == 1
    assert projected.preserved_assets == 0


def test_dropped_and_missing_assets_stay_out_of_the_preserved_remainder() -> None:
    # The remainder is what *neither* reconstructed, dropped nor missing: an
    # intentional drop and a loss are never resurrected as preserved.
    graph = _graph(
        _asset("a1", AssetIntegrity.RECONSTRUCTED, verified=True),
        _asset("a2", AssetIntegrity.DROPPED),
        _asset("a3", AssetIntegrity.MISSING),
        _asset("a4", AssetIntegrity.PRESERVED_OPAQUE),
    )
    report = _report(
        assets=(("RECONSTRUCTED_VERIFIED", 1),),
        attestations=(_attestation("a1", Fidelity.RECONSTRUCTED_VERIFIED),),
    )
    projected = project_contract(report, graph)
    assert projected.total_assets == 4
    assert projected.reconstructed_assets == 1
    assert projected.dropped_assets == 1
    assert projected.missing_assets == 1
    assert projected.preserved_assets == 1


def test_the_books_balance_even_when_graph_and_attestations_disagree() -> None:
    # Two graph assets, no attested reconstructions and one graph-verified
    # reconstruction whose attestation was lost: reconstructed=0 (clamped to
    # the attestation count), and the remainder -- including the graph's
    # verified row -- is preserved, so reconstructed + preserved == total.
    graph = _graph(
        _asset("a1", AssetIntegrity.RECONSTRUCTED, verified=True),
        _asset("a2", AssetIntegrity.PRESERVED_OPAQUE),
    )
    report = _report()
    projected = project_contract(report, graph)
    assert projected.total_assets == 2
    assert projected.reconstructed_assets == 0
    assert projected.preserved_assets == 2
    assert projected.reconstructed_assets + projected.preserved_assets == projected.total_assets


def test_the_projection_keeps_the_graph_identity_and_zero_default() -> None:
    projected = project_contract(_report(), ContentGraph(doc_id="doc", title="Book"))
    assert projected.doc_id == "doc"
    assert projected.title == "Book"
    assert projected.schema_version == 1
    assert projected.total_text == 0
    assert projected.passed

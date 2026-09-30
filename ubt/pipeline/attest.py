"""Per-element attestation: realize() over a whole document (ADR-0001 Phase 3).

``reconcile()`` discovers violations *after* a delivery is built, by balancing
two ledgers. This is the construction-time twin the ADR replaces it with: every
element is lowered through :func:`~ubt.pipeline.steps.realize`, and the
attestations *are* the account -- 100% coverage, and a non-decorative element
with no lossless realization is recorded as a violation instead of hidden.

:class:`AttestationReport` mirrors the contract's vocabulary (text delivered vs
kept; assets reconstructed vs preserved) so the two can be compared element for
element during the migration shadow. Text and asset histograms stay apart because
the contract counts them apart.

:func:`project_contract` is the switch itself: the delivery contract is a
*projection* of the attestations (the account of realizations and the verdict on
loss), with the content graph supplying only the detail the AST deliberately does
not model (why a kept node was kept) -- not a second, independent audit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ubt.core.content.contract import (
    ReconciliationReport,
    Severity,
    Violation,
    ViolationKind,
    reconcile,
)
from ubt.core.content.graph import ContentGraph
from ubt.core.content.ledger import build_ledgers
from ubt.core.content.nodes import TextDisposition
from ubt.model.ast import Document
from ubt.model.fidelity import Attestation, Fidelity
from ubt.pipeline.steps import IntegrityViolation, realize
from ubt.render.capability import Backend
from ubt.verify.verifier import Verifiers


@dataclass(frozen=True, slots=True)
class AttestationReport:
    """What ``realize()`` accounted for across one document."""

    total: int
    text: tuple[tuple[str, int], ...]  # fidelity name -> count, over text elements
    assets: tuple[tuple[str, int], ...]  # fidelity name -> count, over asset elements
    violations: tuple[str, ...] = ()  # elements with no lossless realization
    attestations: tuple[Attestation, ...] = ()  # the per-element realizations

    @property
    def passed(self) -> bool:
        return not self.violations

    def text_count(self, fidelity: Fidelity) -> int:
        return dict(self.text).get(fidelity.name, 0)

    def asset_count(self, fidelity: Fidelity) -> int:
        return dict(self.assets).get(fidelity.name, 0)

    def summary_line(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        text = ", ".join(f"{name}={count}" for name, count in self.text)
        assets = ", ".join(f"{name}={count}" for name, count in self.assets)
        return (
            f"[{verdict}] {self.total} element(s): text {{{text}}}, assets {{{assets}}} | "
            f"{len(self.violations)} violation(s)"
        )


def attest_document(
    document: Document, backend: Backend, verifiers: Verifiers
) -> AttestationReport:
    """Lower every element and account for it; never raises on a lost element.

    A non-decorative element with no lossless realization is a violation -- the
    construction-time twin of ``reconcile()``'s ``TEXT_UNDELIVERED`` /
    ``ASSET_MISSING``.
    """
    text: Counter[str] = Counter()
    assets: Counter[str] = Counter()
    violations: list[str] = []
    attestations: list[Attestation] = []
    for element in document.elements:
        try:
            attestation = realize(element, backend, verifiers, document.source)
        except IntegrityViolation:
            violations.append(element.id)
            continue
        (text if element.is_text else assets)[attestation.fidelity.name] += 1
        attestations.append(attestation)
    return AttestationReport(
        total=len(document.elements),
        text=tuple(sorted(text.items())),
        assets=tuple(sorted(assets.items())),
        violations=tuple(violations),
        attestations=tuple(attestations),
    )


def project_contract(report: AttestationReport, graph: ContentGraph) -> ReconciliationReport:
    """Project the per-element attestations onto the delivery contract.

    ``reconcile()`` balanced two ledgers after a delivery was already built; the
    attestations *are* the account now, so the export builds its contract from
    them. The graph still *classifies* -- it is the bridge from ``IRBlock`` to the
    contract's vocabulary -- and supplies the detail the AST deliberately does not
    model: the reason a kept text node was kept (verbatim vs source-kept) and the
    named violations with their detail. It is no longer a second, independent
    verdict on whether a realization exists.

    The realizations the attestations *verify* replace the graph's count of them:
    a text node is delivered when ``realize()`` reconstructed and verified it, and
    an asset is reconstructed when it did -- a block that merely carries a target
    no longer counts. The asset account is then balanced by remainder, so the
    projection stays internally consistent even where the graph and the
    attestations disagree about which blocks are assets. A translation that did
    not verify is demoted to source-kept (the reader gets the source) so the text
    account balances too. An element with no lossless realization is an ERROR
    here, the construction-time loss ``realize()`` refuses to hide and the graph
    cannot see (the block still carries a target).
    """
    base = reconcile(graph)
    content, _ = build_ledgers(graph)
    attested = {attestation.element_id: attestation.fidelity for attestation in report.attestations}
    demoted = tuple(
        Violation(
            kind=ViolationKind.TEXT_SOURCE_KEPT,
            severity=Severity.WARNING,
            node_id=entry.node_id,
            detail="translation did not verify; source kept",
        )
        for entry in content.entries.values()
        if entry.disposition is TextDisposition.TRANSLATED
        and attested.get(entry.node_id) is not Fidelity.RECONSTRUCTED_ADAPTED
    )
    lost = tuple(
        Violation(
            kind=ViolationKind.TEXT_UNDELIVERED,
            severity=Severity.ERROR,
            node_id=element_id,
            detail="no lossless realization (ADR-0001 Axiom B)",
        )
        for element_id in report.violations
    )
    # The attestations decide how many assets were *reconstructed*; what the graph
    # counts as an asset and was neither reconstructed, dropped nor missing is
    # preserved. Deriving the remainder (rather than copying the graph's
    # ``preserved_assets``) keeps the account balanced even when the two disagree
    # about which blocks are assets -- a divergence the sum would otherwise hide.
    reconstructed = min(report.asset_count(Fidelity.RECONSTRUCTED_VERIFIED), base.total_assets)
    preserved = max(
        0, base.total_assets - reconstructed - base.dropped_assets - base.missing_assets
    )
    return base.model_copy(
        update={
            "delivered_text": report.text_count(Fidelity.RECONSTRUCTED_ADAPTED),
            "source_kept_text": base.source_kept_text + len(demoted),
            "reconstructed_assets": reconstructed,
            "preserved_assets": preserved,
            "violations": base.violations + demoted + lost,
        }
    )


__all__ = ["AttestationReport", "attest_document", "project_contract"]

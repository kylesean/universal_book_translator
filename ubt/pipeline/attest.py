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
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ubt.model.ast import Document
from ubt.model.fidelity import Fidelity
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
    for element in document.elements:
        try:
            attestation = realize(element, backend, verifiers, document.source)
        except IntegrityViolation:
            violations.append(element.id)
            continue
        (text if element.is_text else assets)[attestation.fidelity.name] += 1
    return AttestationReport(
        total=len(document.elements),
        text=tuple(sorted(text.items())),
        assets=tuple(sorted(assets.items())),
        violations=tuple(violations),
    )


__all__ = ["AttestationReport", "attest_document"]

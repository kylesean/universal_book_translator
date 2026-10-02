"""``attest_document()``: whole-document accounting, built on ``realize()``.

Where ``realize()`` lowers *one* element, ``attest_document`` walks a document
and turns the realizations into the account the delivery contract is projected
from. The contract pinned here is the construction-time replacement for
``reconcile()``'s after-the-fact ledger balancing:

- an element with no lossless realization is **recorded as a violation**, never
  raised, so one bad element cannot abort the account;
- text and asset histograms stay apart (the contract counts them apart);
- a decorative drop and a failed-verification floor are *accounted*, not
  violations -- Axiom B says the reader still gets something;
- the histograms are deterministic (sorted by fidelity name) and the
  attestations preserve document order.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import pytest

from ubt.model.ast import (
    Document,
    Element,
    ElementT,
    Formula,
    Heading,
    Paragraph,
    Region,
    RegionKind,
)
from ubt.model.fidelity import Fidelity, Proof, ProofKind
from ubt.model.span import CanonicalSource, Span
from ubt.pipeline.attest import attest_document
from ubt.render.capability import Capabilities, Produced
from ubt.verify.verifier import Verifiers

pytestmark = pytest.mark.fast

_SOURCE = CanonicalSource(doc_id="doc", path="doc.pdf")


class _FakeBackend:
    """A backend whose capabilities and productions are declared per test."""

    name: ClassVar[str] = "fake"

    def __init__(
        self,
        supported: Iterable[tuple[type[Element], Fidelity]] = (),
        produced: dict[tuple[type[Element], Fidelity], Produced] | None = None,
    ) -> None:
        self._capabilities = Capabilities(frozenset(supported))
        self._produced = produced or {}

    def capabilities(self) -> Capabilities:
        return self._capabilities

    def produce(
        self, element: Element, fidelity: Fidelity, source: CanonicalSource
    ) -> Produced | None:
        return self._produced.get((type(element), fidelity))


class _FixedVerifier:
    kind: ClassVar[ProofKind] = ProofKind.STRUCTURAL

    def __init__(self, result: Proof) -> None:
        self._result = result

    def verify(self, subject: object) -> Proof:
        return self._result


def _verifiers(result: Proof) -> Verifiers:
    stub = _FixedVerifier(result)
    return Verifiers(structural=stub, formula=stub, table=stub, text=stub)  # type: ignore[arg-type]


def _document(*elements: ElementT) -> Document:
    region = Region(id="r", kind=RegionKind.BODY, elements=elements)
    return Document(source=_SOURCE, regions=(region,))


def _paragraph(element_id: str, **overrides: object) -> Paragraph:
    return Paragraph(id=element_id, spine_index=0, span=Span(page=1), **overrides)  # type: ignore[arg-type]


_OK = _verifiers(Proof.ok(ProofKind.PREDICATE))


# --------------------------------------------------------------------------- #
# Shape of the account: buckets, ordering, accessors.
# --------------------------------------------------------------------------- #


def test_empty_document_has_no_account_and_passes() -> None:
    report = attest_document(_document(), _FakeBackend(), _OK)
    assert report.total == 0
    assert report.text == ()
    assert report.assets == ()
    assert report.violations == ()
    assert report.attestations == ()
    assert report.passed


def test_reconstructed_text_lands_in_the_text_histogram() -> None:
    backend = _FakeBackend(
        supported={(Paragraph, Fidelity.RECONSTRUCTED_ADAPTED)},
        produced={(Paragraph, Fidelity.RECONSTRUCTED_ADAPTED): Produced("translated")},
    )
    report = attest_document(_document(_paragraph("p1")), backend, _OK)
    assert report.text == (("RECONSTRUCTED_ADAPTED", 1),)
    assert report.assets == ()


def test_preserved_asset_lands_in_the_asset_histogram() -> None:
    formula = Formula(id="f1", spine_index=0, span=Span(page=1), source="a^2")
    backend = _FakeBackend(
        supported={(Formula, Fidelity.PRESERVED_OPAQUE)},
        produced={(Formula, Fidelity.PRESERVED_OPAQUE): Produced("a^2")},
    )
    report = attest_document(_document(formula), backend, _OK)
    assert report.assets == (("PRESERVED_OPAQUE", 1),)
    assert report.text == ()


def test_text_and_asset_histograms_stay_separate_and_sorted() -> None:
    # Paragraph -> RECONSTRUCTED_ADAPTED (text); Heading -> PRESERVED_OPAQUE
    # (also text). Both in one bucket, sorted by fidelity name.
    backend = _FakeBackend(
        supported={
            (Paragraph, Fidelity.RECONSTRUCTED_ADAPTED),
            (Heading, Fidelity.PRESERVED_OPAQUE),
        },
        produced={
            (Paragraph, Fidelity.RECONSTRUCTED_ADAPTED): Produced("translated"),
            (Heading, Fidelity.PRESERVED_OPAQUE): Produced("heading source"),
        },
    )
    paragraph = _paragraph("p1")
    heading = Heading(id="h1", spine_index=1, span=Span(page=1), text="Title")
    report = attest_document(_document(paragraph, heading), backend, _OK)
    assert report.text == (("PRESERVED_OPAQUE", 1), ("RECONSTRUCTED_ADAPTED", 1))


def test_counts_default_to_zero_for_an_absent_fidelity() -> None:
    report = attest_document(_document(), _FakeBackend(), _OK)
    assert report.text_count(Fidelity.RECONSTRUCTED_ADAPTED) == 0
    assert report.asset_count(Fidelity.PRESERVED_OPAQUE) == 0


def test_attestations_preserve_document_order() -> None:
    backend = _FakeBackend(
        supported={
            (Paragraph, Fidelity.RECONSTRUCTED_ADAPTED),
            (Formula, Fidelity.PRESERVED_OPAQUE),
        },
        produced={
            (Paragraph, Fidelity.RECONSTRUCTED_ADAPTED): Produced("translated"),
            (Formula, Fidelity.PRESERVED_OPAQUE): Produced("a^2"),
        },
    )
    elements = (
        _paragraph("p1"),
        Formula(id="f1", spine_index=1, span=Span(page=1), source="a^2"),
        _paragraph("p2"),
    )
    report = attest_document(_document(*elements), backend, _OK)
    assert [a.element_id for a in report.attestations] == ["p1", "f1", "p2"]


# --------------------------------------------------------------------------- #
# The lossless-realization axiom, at document scale.
# --------------------------------------------------------------------------- #


def test_element_without_lossless_realization_is_a_violation_not_a_raise() -> None:
    formula = Formula(id="f1", spine_index=1, span=Span(page=1), source="a^2")
    # The paragraph realizes; the backend supports no rung for the formula.
    backend = _FakeBackend(
        supported={(Paragraph, Fidelity.PRESERVED_OPAQUE)},
        produced={(Paragraph, Fidelity.PRESERVED_OPAQUE): Produced("source")},
    )
    report = attest_document(_document(_paragraph("p1"), formula), backend, _OK)
    assert report.violations == ("f1",)
    assert not report.passed
    # The delivered paragraph is still accounted for...
    assert report.text == (("PRESERVED_OPAQUE", 1),)
    # ... while the failed element is excluded from every bucket and attestation.
    assert report.assets == ()
    assert [a.element_id for a in report.attestations] == ["p1"]
    # ... but it still counts toward the total the report is accountable for.
    assert report.total == 2


def test_decorative_element_is_dropped_and_counted_not_a_violation() -> None:
    report = attest_document(_document(_paragraph("p1", decorative=True)), _FakeBackend(), _OK)
    assert report.text == (("DROPPED", 1),)
    assert report.violations == ()
    assert report.passed


def test_failed_verification_falls_to_the_floor_and_is_not_a_violation() -> None:
    # Axiom B at document scale: an unverified reconstruction is demoted to the
    # opaque floor (source kept), which is delivered -- not a lost element.
    backend = _FakeBackend(
        supported={
            (Paragraph, Fidelity.RECONSTRUCTED_ADAPTED),
            (Paragraph, Fidelity.PRESERVED_OPAQUE),
        },
        produced={
            (Paragraph, Fidelity.RECONSTRUCTED_ADAPTED): Produced("translated"),
            (Paragraph, Fidelity.PRESERVED_OPAQUE): Produced("source"),
        },
    )
    failing = _verifiers(Proof.fail(ProofKind.PREDICATE, "length ratio"))
    report = attest_document(_document(_paragraph("p1")), backend, failing)
    assert report.text == (("PRESERVED_OPAQUE", 1),)
    assert report.violations == ()
    assert report.passed


# --------------------------------------------------------------------------- #
# The verdict line.
# --------------------------------------------------------------------------- #


def test_summary_line_reports_pass() -> None:
    report = attest_document(_document(), _FakeBackend(), _OK)
    assert report.summary_line() == "[PASS] 0 element(s): text {}, assets {} | 0 violation(s)"


def test_summary_line_reports_fail_with_the_violation_count() -> None:
    report = attest_document(_document(_paragraph("p1")), _FakeBackend(), _OK)
    line = report.summary_line()
    assert line.startswith("[FAIL] 1 element(s):")
    assert line.endswith("| 1 violation(s)")

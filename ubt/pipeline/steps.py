"""Pure pipeline steps (document compiler architecture). :func:`realize` is the whole execution core.

For one element, walk its fidelity descent; the first rung the backend declares
it can produce *and* whose result verifies is that element's realization. The
two axioms of the architecture fall out of the loop instead of out of a later
audit:

- **Axiom A (assets).** A non-text element is either reconstructed (with a
  proof) or preserved opaque -- there is no third possibility, because the
  descent only offers those two and ``DROPPED`` raises for a non-decorative
  element.
- **Axiom B (text).** A text element is either reconstructed (translated and
  typeset) or preserved opaque (source kept) -- never silently dropped.

So ``reconcile()``'s after-the-fact "did we lose anything?" becomes a
construction-time error: an element with no lossless realization raises right
here, where it happened.
"""

from __future__ import annotations

from ubt.model.ast import Element
from ubt.model.fidelity import FIDELITY_DESCENT, Attestation, Fidelity, Proof, ProofKind
from ubt.model.span import CanonicalSource
from ubt.render.capability import Backend
from ubt.verify.element import verify_element
from ubt.verify.verifier import Verifiers


class IntegrityViolation(Exception):
    """An element has no lossless realization, so content would be lost."""


def realize(
    element: Element,
    backend: Backend,
    verifiers: Verifiers,
    source: CanonicalSource,
) -> Attestation:
    """Lower one element to the highest fidelity this backend can verify.

    ``source`` is the document's :class:`~ubt.model.span.CanonicalSource`, the
    common reference every verifier compares against. ``PRESERVED_OPAQUE`` is
    lossless by construction (an opaque source slice), so it is attested without
    running a reconstruction predicate; every other rung must verify.
    """
    capabilities = backend.capabilities()
    for fidelity in FIDELITY_DESCENT[type(element)]:
        if not capabilities.supports(type(element), fidelity):
            continue
        produced = backend.produce(element, fidelity, source)
        if produced is None:
            continue
        proof = (
            Proof.preserved(produced.note or "preserved opaque")
            if fidelity is Fidelity.PRESERVED_OPAQUE
            else verify_element(element, verifiers, reconstructed=produced.payload)
        )
        if proof.verified:
            return Attestation(element.id, fidelity, proof, produced.note)
    if not element.decorative:
        raise IntegrityViolation(
            f"element {element.id} ({element.kind}) has no lossless realization: "
            f"backend {backend.name!r} cannot produce any of "
            f"{tuple(rung.name for rung in FIDELITY_DESCENT[type(element)])}"
        )
    return Attestation(
        element.id, Fidelity.DROPPED, Proof.fail(ProofKind.NONE, "decorative"), "decorative"
    )


__all__ = ["IntegrityViolation", "realize"]

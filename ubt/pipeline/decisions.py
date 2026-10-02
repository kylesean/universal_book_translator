"""The realization decision source (pre-render decision plan).

:func:`ubt.pipeline.steps.realize` is the execution core for one element; this
module names its document-level projection as the run's **decision plan**: per
element, the fidelity the run commits to deliver. The production renderer
implements the same decisions -- a translatable element is reflowed, a
skipped/asset element is kept -- and the delivery contract is projected from
this plan (:func:`ubt.pipeline.attest.project_contract`). The plan is the single
source; the renderer and the contract both answer to it.

:func:`divergences` is the shadow-period check: it compares the plan against the
fidelities a lowering actually placed. A lowering may keep the source (descend),
but it must never *claim* less than, or silently fall below, what the plan
committed to -- a divergence is surfaced, not hidden.
"""

from __future__ import annotations

from collections.abc import Mapping

from ubt.model.ast import Document
from ubt.model.fidelity import Fidelity
from ubt.pipeline.attest import AttestationReport, attest_document
from ubt.render.capability import Backend
from ubt.verify.verifier import Verifiers


def plan_realization(
    document: Document, backend: Backend, verifiers: Verifiers
) -> AttestationReport:
    """The per-element realization plan: what this run commits to deliver."""
    return attest_document(document, backend, verifiers)


def plan_fidelities(plan: AttestationReport) -> dict[str, Fidelity]:
    """The plan as ``element id -> committed fidelity``, the renderer's input.

    This is the value the production renderer reads (pre-render decision plan): it no
    longer decides per element whether to reflow or preserve -- it looks the
    decision up here. A missing id means the plan does not cover it and the
    renderer falls back to its own judgment.
    """
    return {attestation.element_id: attestation.fidelity for attestation in plan.attestations}


def divergences(
    plan: AttestationReport, realized: Mapping[str, Fidelity]
) -> tuple[tuple[str, Fidelity, Fidelity], ...]:
    """``(element_id, planned, realized)`` where the artifact fell below the plan.

    An element the lowering could not draw at the planned rung (it kept the
    source instead) is a divergence: legitimate, but it must be recorded rather
    than passed off as the planned fidelity.
    """
    found: list[tuple[str, Fidelity, Fidelity]] = []
    for attestation in plan.attestations:
        actual = realized.get(attestation.element_id)
        if actual is not None and actual < attestation.fidelity:
            found.append((attestation.element_id, attestation.fidelity, actual))
    return tuple(found)


__all__ = ["divergences", "plan_fidelities", "plan_realization"]

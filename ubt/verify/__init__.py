"""``ubt.verify`` -- the unified verification seam (ADR-0001 Phase 0).

Public surface:

- :class:`Proof` / :class:`ProofOutcome` / :class:`ProofKind` -- the one
  vocabulary every check reports in.
- :class:`Verifier` and the four concrete verifiers -- one protocol in front of
  the existing structural / pixel / predicate checks.
- :func:`build_verifiers` -- the run's verifier set, bound to one language pair.
- :class:`ShadowRun` / :class:`AgreementReport` -- the differential harness that
  proves the seam changes no behaviour.

This package is the seed of the ADR's ``ubt.verify`` layer. It deliberately
depends on the existing checks rather than reimplementing them: Phase 0 wraps,
it does not replace. The render path keeps calling the old functions until
Phase 3 lowers elements through :func:`ubt.verify.verifier.Verifier`.
"""

from __future__ import annotations

from ubt.verify.proof import Proof, ProofKind, ProofOutcome
from ubt.verify.shadow import AgreementReport, Mismatch, ShadowRun, expected_outcome
from ubt.verify.verifier import (
    FormulaWitnessVerifier,
    RasterFormula,
    RasterTable,
    StructuralAsset,
    StructuralAssetVerifier,
    TableWitnessVerifier,
    TextPair,
    TranslationFastPassVerifier,
    Verifier,
    Verifiers,
    build_verifiers,
)

__all__ = [
    "AgreementReport",
    "FormulaWitnessVerifier",
    "Mismatch",
    "Proof",
    "ProofKind",
    "ProofOutcome",
    "RasterFormula",
    "RasterTable",
    "ShadowRun",
    "StructuralAsset",
    "StructuralAssetVerifier",
    "TableWitnessVerifier",
    "TextPair",
    "TranslationFastPassVerifier",
    "Verifier",
    "Verifiers",
    "build_verifiers",
    "expected_outcome",
]

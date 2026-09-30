"""The verifier seam: one protocol in front of every existing check.

Phase 0 does not add a checker -- it adds a *seam*. The four checks UBT already
runs are wrapped behind a single ``Verifier`` protocol that returns :class:`Proof`:

- :class:`StructuralAssetVerifier` wraps ``asset_verify.verify_asset_structure``
  (level-1 grammar/shape check on delivered formula/table text).
- :class:`FormulaWitnessVerifier` wraps ``formula_witness.witness_formula``
  (level-2 raster structural comparison against the source crop).
- :class:`TableWitnessVerifier` wraps ``table_witness.witness_table``.
- :class:`TranslationFastPassVerifier` wraps ``FastPassFilter.evaluate``.

Each wrapper is a *pure mapping* from the legacy result to :class:`Proof`.
Behaviour is unchanged on purpose -- the shadow harness
(:mod:`ubt.verify.shadow`) exists to prove it -- so the render path can keep
calling the old functions until Phase 3 switches it to the seam.

The subjects below are frozen dataclasses, so a verifier call is explicit about
everything it sees: no hidden reads of the ledger, the renderer or the adapter.
The PDF-only witnesses are imported lazily, so importing ``ubt.verify`` does not
drag pdfium into a process that never renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeVar

from ubt.core.content.asset_verify import (
    StructuralVerdict,
    VerifyResult,
    verify_asset_structure,
)
from ubt.core.content.nodes import AssetKind
from ubt.verify.proof import Proof, ProofKind

if TYPE_CHECKING:
    from pathlib import Path

    from ubt.adapters.pdf.formula_witness import WitnessResult
    from ubt.core.ir.models import IRBlock
    from ubt.core.qe.fast_pass import FastPassDecision, FastPassFilter

_SubjectT = TypeVar("_SubjectT", contravariant=True)


class Verifier(Protocol[_SubjectT]):
    """A check that turns a subject into a :class:`Proof`.

    Implementations are cheap adapters over existing logic; they never perform
    IO of their own beyond what the wrapped check does, and they never mutate
    the subject.
    """

    name: ClassVar[str]
    kind: ClassVar[ProofKind]

    def verify(self, subject: _SubjectT) -> Proof: ...


# --------------------------------------------------------------------------- #
# Subjects: exactly what each check needs, so a call is self-describing.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StructuralAsset:
    """Level-1 structural check inputs (no pixels, no IO)."""

    asset_kind: AssetKind
    text: str


@dataclass(frozen=True, slots=True)
class RasterFormula:
    """Level-2 formula witness inputs."""

    emitted: str
    block: IRBlock
    source_pdf: Path
    typst_binary: str
    dpi: int | None = None


@dataclass(frozen=True, slots=True)
class RasterTable:
    """Level-2 table witness inputs."""

    markup: str
    block: IRBlock
    source_pdf: Path
    typst_binary: str
    dpi: int | None = None


@dataclass(frozen=True, slots=True)
class TextPair:
    """Predicate check inputs for one translated block."""

    source_text: str
    target_text: str
    block_type: object = None
    skip_translate: bool = False


# --------------------------------------------------------------------------- #
# Verifiers: pure mappings from the legacy result to Proof.
# --------------------------------------------------------------------------- #


class StructuralAssetVerifier(Verifier[StructuralAsset]):
    name: ClassVar[str] = "structural_asset"
    kind: ClassVar[ProofKind] = ProofKind.STRUCTURAL

    def verify(self, subject: StructuralAsset) -> Proof:
        return _from_structural(verify_asset_structure(subject.asset_kind, subject.text))


class FormulaWitnessVerifier(Verifier[RasterFormula]):
    name: ClassVar[str] = "formula_witness"
    kind: ClassVar[ProofKind] = ProofKind.PIXEL

    def verify(self, subject: RasterFormula) -> Proof:
        from ubt.adapters.pdf.formula_witness import WITNESS_DPI, witness_formula

        result = witness_formula(
            subject.emitted,
            subject.block,
            subject.source_pdf,
            subject.typst_binary,
            dpi=subject.dpi or WITNESS_DPI,
        )
        return _from_witness(result)


class TableWitnessVerifier(Verifier[RasterTable]):
    name: ClassVar[str] = "table_witness"
    kind: ClassVar[ProofKind] = ProofKind.PIXEL

    def verify(self, subject: RasterTable) -> Proof:
        from ubt.adapters.pdf.formula_witness import WITNESS_DPI
        from ubt.adapters.pdf.table_witness import witness_table

        result = witness_table(
            subject.markup,
            subject.block,
            subject.source_pdf,
            subject.typst_binary,
            dpi=subject.dpi or WITNESS_DPI,
        )
        return _from_witness(result)


class TranslationFastPassVerifier(Verifier[TextPair]):
    name: ClassVar[str] = "translation_fast_pass"
    kind: ClassVar[ProofKind] = ProofKind.PREDICATE

    def __init__(self, fast_pass: FastPassFilter) -> None:
        self._fast_pass = fast_pass

    def verify(self, subject: TextPair) -> Proof:
        decision = self._fast_pass.evaluate(
            subject.source_text,
            subject.target_text,
            block_type=subject.block_type,
            skip_translate=subject.skip_translate,
        )
        return _from_fast_pass(decision)


@dataclass(frozen=True, slots=True)
class Verifiers:
    """The run's verifier set, built once and handed to the pipeline."""

    structural: StructuralAssetVerifier
    formula: FormulaWitnessVerifier
    table: TableWitnessVerifier
    text: TranslationFastPassVerifier


def build_verifiers(fast_pass: FastPassFilter) -> Verifiers:
    """Construct the verifier set bound to one run's language policy."""
    return Verifiers(
        structural=StructuralAssetVerifier(),
        formula=FormulaWitnessVerifier(),
        table=TableWitnessVerifier(),
        text=TranslationFastPassVerifier(fast_pass),
    )


# --------------------------------------------------------------------------- #
# Mappings: the single place a legacy verdict becomes a Proof.
# --------------------------------------------------------------------------- #


def _from_structural(result: VerifyResult) -> Proof:
    if result.verdict is StructuralVerdict.PASS:
        return Proof.ok(ProofKind.STRUCTURAL, result.detail)
    if result.verdict is StructuralVerdict.FAIL:
        return Proof.fail(ProofKind.STRUCTURAL, result.detail)
    return Proof.unknown(ProofKind.STRUCTURAL, result.detail)


def _from_witness(result: WitnessResult) -> Proof:
    findings = tuple(result.findings)
    if result.status == "pass":
        return Proof.ok(ProofKind.PIXEL)
    if result.status == "fail":
        return Proof.fail(ProofKind.PIXEL, findings=findings)
    return Proof.unknown(ProofKind.PIXEL, findings[0] if findings else "", findings=findings)


def _from_fast_pass(decision: FastPassDecision) -> Proof:
    if decision.passed:
        return Proof.ok(ProofKind.PREDICATE, decision.reason)
    return Proof.fail(ProofKind.PREDICATE, decision.reason)


__all__ = [
    "FormulaWitnessVerifier",
    "RasterFormula",
    "RasterTable",
    "StructuralAsset",
    "StructuralAssetVerifier",
    "TableWitnessVerifier",
    "TextPair",
    "TranslationFastPassVerifier",
    "Verifier",
    "Verifiers",
    "build_verifiers",
]

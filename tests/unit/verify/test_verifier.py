"""The verifier seam: legacy verdicts mapped to :class:`Proof`.

The seam adds no checker -- it wraps the four existing checks behind one
``Verifier`` protocol so the render path can switch to the seam later. The
contract that matters is the mapping: each wrapper is a *pure* function from
the legacy result to ``Proof``, and it must preserve the three-valued outcome
(verified / failed / unverifiable) rather than collapse it to a boolean. The
PDF-only witnesses are not exercised (they shell out to typst/pdfium); only the
two verifiers whose wrapped check is pure are driven end to end.
"""

from __future__ import annotations

import dataclasses

import pytest

from ubt.adapters.pdf.formula_witness import WitnessResult
from ubt.core.content.asset_verify import StructuralVerdict, VerifyResult
from ubt.core.content.nodes import AssetKind
from ubt.core.qe.fast_pass import FastPassDecision, FastPassFilter
from ubt.model.fidelity import ProofKind, ProofOutcome
from ubt.verify.verifier import (
    FormulaWitnessVerifier,
    StructuralAsset,
    StructuralAssetVerifier,
    TableWitnessVerifier,
    TextPair,
    TranslationFastPassVerifier,
    Verifiers,
    _from_fast_pass,
    _from_structural,
    _from_witness,
    build_verifiers,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# _from_structural
# --------------------------------------------------------------------------- #


def test_structural_pass_maps_to_verified() -> None:
    proof = _from_structural(VerifyResult(StructuralVerdict.PASS, "detail"))
    assert proof.outcome is ProofOutcome.VERIFIED
    assert proof.kind is ProofKind.STRUCTURAL
    assert proof.detail == "detail"


def test_structural_fail_maps_to_failed() -> None:
    proof = _from_structural(VerifyResult(StructuralVerdict.FAIL, "detail"))
    assert proof.outcome is ProofOutcome.FAILED


def test_structural_skip_maps_to_unverifiable_not_failed() -> None:
    # "Nothing structural to judge" is not a corruption signal.
    proof = _from_structural(VerifyResult(StructuralVerdict.SKIP, "detail"))
    assert proof.outcome is ProofOutcome.UNVERIFIABLE


# --------------------------------------------------------------------------- #
# _from_witness
# --------------------------------------------------------------------------- #


def test_witness_pass_maps_to_verified_without_findings() -> None:
    proof = _from_witness(WitnessResult("pass", []))
    assert proof.outcome is ProofOutcome.VERIFIED
    assert proof.kind is ProofKind.PIXEL
    assert proof.findings == ()


def test_witness_fail_carries_every_finding() -> None:
    proof = _from_witness(WitnessResult("fail", ["a", "b"]))
    assert proof.outcome is ProofOutcome.FAILED
    assert proof.findings == ("a", "b")


def test_witness_unwitnessable_maps_to_unverifiable_with_first_finding_as_detail() -> None:
    proof = _from_witness(WitnessResult("unwitnessable", ["x", "y"]))
    assert proof.outcome is ProofOutcome.UNVERIFIABLE
    assert proof.detail == "x"
    assert proof.findings == ("x", "y")


def test_witness_unwitnessable_without_findings_has_an_empty_detail() -> None:
    proof = _from_witness(WitnessResult("unwitnessable", []))
    assert proof.outcome is ProofOutcome.UNVERIFIABLE
    assert proof.detail == ""


# --------------------------------------------------------------------------- #
# _from_fast_pass
# --------------------------------------------------------------------------- #


def test_fast_pass_accept_maps_to_verified_with_the_reason() -> None:
    proof = _from_fast_pass(
        FastPassDecision(passed=True, reason="clean", target_ratio=1.0, length_ratio=1.0)
    )
    assert proof.outcome is ProofOutcome.VERIFIED
    assert proof.kind is ProofKind.PREDICATE
    assert proof.detail == "clean"


def test_fast_pass_reject_maps_to_failed_with_the_reason() -> None:
    proof = _from_fast_pass(
        FastPassDecision(passed=False, reason="echo", target_ratio=1.0, length_ratio=1.0)
    )
    assert proof.outcome is ProofOutcome.FAILED
    assert proof.detail == "echo"


# --------------------------------------------------------------------------- #
# build_verifiers / Verifiers
# --------------------------------------------------------------------------- #


def test_build_verifiers_wires_the_four_verifiers() -> None:
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))
    assert isinstance(verifiers, Verifiers)
    assert isinstance(verifiers.structural, StructuralAssetVerifier)
    assert isinstance(verifiers.formula, FormulaWitnessVerifier)
    assert isinstance(verifiers.table, TableWitnessVerifier)
    assert isinstance(verifiers.text, TranslationFastPassVerifier)


def test_each_verifier_declares_its_name_and_proof_kind() -> None:
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))
    assert (verifiers.structural.name, verifiers.structural.kind) == (
        "structural_asset",
        ProofKind.STRUCTURAL,
    )
    assert (verifiers.formula.name, verifiers.formula.kind) == ("formula_witness", ProofKind.PIXEL)
    assert (verifiers.table.name, verifiers.table.kind) == ("table_witness", ProofKind.PIXEL)
    assert (verifiers.text.name, verifiers.text.kind) == (
        "translation_fast_pass",
        ProofKind.PREDICATE,
    )


# --------------------------------------------------------------------------- #
# StructuralAssetVerifier end to end (wrapped check is pure)
# --------------------------------------------------------------------------- #

_GOOD_TABLE = "| Name | Value |\n| --- | --- |\n| Alpha | 100 |"
_SHATTERED_TABLE = "| a | b |\n| - | - |\n| 1 | 2 |"


def test_structural_verifier_accepts_a_well_formed_table() -> None:
    proof = StructuralAssetVerifier().verify(StructuralAsset(AssetKind.TABLE, _GOOD_TABLE))
    assert proof.outcome is ProofOutcome.VERIFIED


def test_structural_verifier_flags_a_shattered_table() -> None:
    proof = StructuralAssetVerifier().verify(StructuralAsset(AssetKind.TABLE, _SHATTERED_TABLE))
    assert proof.outcome is ProofOutcome.FAILED


def test_structural_verifier_is_unverifiable_on_an_unrecognized_grammar() -> None:
    proof = StructuralAssetVerifier().verify(StructuralAsset(AssetKind.TABLE, "a & b = c"))
    assert proof.outcome is ProofOutcome.UNVERIFIABLE


def test_structural_verifier_accepts_formula_text() -> None:
    proof = StructuralAssetVerifier().verify(StructuralAsset(AssetKind.FORMULA, "x^2 + y^2"))
    assert proof.outcome is ProofOutcome.VERIFIED


# --------------------------------------------------------------------------- #
# TranslationFastPassVerifier end to end (wrapped check is pure)
# --------------------------------------------------------------------------- #

_ECHO = "The quick brown fox jumps over the lazy dog."
_CJK = "\u5feb\u901f\u7684\u68d5\u8272\u72d0\u72f8\u8df3\u8fc7\u4e86\u61d2\u72d7\u3002"


def test_fast_pass_verifier_accepts_a_real_translation() -> None:
    verifier = TranslationFastPassVerifier(FastPassFilter(source_lang="en", target_lang="zh"))
    proof = verifier.verify(TextPair(_ECHO, _CJK))
    assert proof.outcome is ProofOutcome.VERIFIED
    assert proof.kind is ProofKind.PREDICATE


def test_fast_pass_verifier_rejects_an_echo() -> None:
    verifier = TranslationFastPassVerifier(FastPassFilter(source_lang="en", target_lang="zh"))
    proof = verifier.verify(TextPair(_ECHO, _ECHO))
    assert proof.outcome is ProofOutcome.FAILED


# --------------------------------------------------------------------------- #
# Subjects are frozen: a verifier call is explicit about everything it sees.
# --------------------------------------------------------------------------- #


def test_subjects_are_frozen() -> None:
    subject = StructuralAsset(AssetKind.TABLE, "x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        subject.text = "y"  # type: ignore[misc]


def test_the_verifier_set_is_frozen() -> None:
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        verifiers.text = verifiers.structural  # type: ignore[misc, assignment]

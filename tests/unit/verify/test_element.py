"""Per-element verification dispatch: what an element is decides how it is judged.

The Phase-1 criterion is totality -- ``verify_element`` returns a ``Proof`` for
*every* concrete element class, so no element falls through un-judged. Two
ordering facts are load-bearing: ``CodeBlock`` is a ``TextElement`` subclass but
must be judged as a preserved listing, not as text pending translation; and a
text element with no target is ``UNVERIFIABLE`` ("not translated yet"), never a
crash and never a silent pass.
"""

from __future__ import annotations

import dataclasses

import pytest

from ubt.core.qe.fast_pass import FastPassFilter
from ubt.model.ast import (
    Caption,
    CodeBlock,
    Dialogue,
    Element,
    Figure,
    Formula,
    Heading,
    ListItem,
    Paragraph,
    Table,
)
from ubt.model.fidelity import Proof, ProofKind, ProofOutcome
from ubt.model.span import Span
from ubt.verify.element import verify_element
from ubt.verify.verifier import build_verifiers

pytestmark = pytest.mark.fast

_SPAN = Span(page=1)
_ECHO = "The quick brown fox jumps over the lazy dog."
_CJK = "\u5feb\u901f\u7684\u68d5\u8272\u72d0\u72f8\u8df3\u8fc7\u4e86\u61d2\u72d7\u3002"
_GOOD_TABLE = "| Name | Value |\n| --- | --- |\n| Alpha | 100 |"
_SHATTERED_TABLE = "| a | b |\n| - | - |\n| 1 | 2 |"


def _verify(element: Element, reconstructed: str | None = None) -> Proof:
    return verify_element(
        element,
        build_verifiers(FastPassFilter(source_lang="en", target_lang="zh")),
        reconstructed=reconstructed,
    )


# --------------------------------------------------------------------------- #
# Lossless-by-construction elements
# --------------------------------------------------------------------------- #


def test_a_figure_is_preserved() -> None:
    proof = _verify(Figure(id="g", spine_index=0, span=_SPAN, asset_id="a"))
    assert (proof.outcome, proof.kind) == (ProofOutcome.VERIFIED, ProofKind.PRESERVED)


def test_a_code_block_is_preserved_not_treated_as_text() -> None:
    # CodeBlock is a TextElement subclass; the dispatch order must win.
    proof = _verify(CodeBlock(id="c", spine_index=0, span=_SPAN, text="print(1)"))
    assert (proof.outcome, proof.kind) == (ProofOutcome.VERIFIED, ProofKind.PRESERVED)


def test_a_code_block_ignores_a_reconstruction() -> None:
    proof = _verify(CodeBlock(id="c", spine_index=0, span=_SPAN, text="print(1)"), "different")
    assert proof.kind is ProofKind.PRESERVED


# --------------------------------------------------------------------------- #
# Formula / Table: structural reconstruction check
# --------------------------------------------------------------------------- #


def test_a_formula_without_a_reconstruction_is_checked_on_its_source() -> None:
    good = _verify(Formula(id="f", spine_index=0, span=_SPAN, source="x^2 + y"))
    bad = _verify(Formula(id="f", spine_index=0, span=_SPAN, source="{unbalanced"))
    assert (good.outcome, good.kind) == (ProofOutcome.VERIFIED, ProofKind.STRUCTURAL)
    assert bad.outcome is ProofOutcome.FAILED


def test_a_formula_is_checked_on_the_reconstruction_when_present() -> None:
    # The source is fine but the delivered markup is broken: the reconstruction
    # is what gets judged, not the source.
    proof = _verify(Formula(id="f", spine_index=0, span=_SPAN, source="x^2 + y"), "{unbalanced")
    assert proof.outcome is ProofOutcome.FAILED


def test_a_table_without_a_reconstruction_is_checked_on_its_markup() -> None:
    proof = _verify(Table(id="t", spine_index=0, span=_SPAN, markup=_GOOD_TABLE))
    assert (proof.outcome, proof.kind) == (ProofOutcome.VERIFIED, ProofKind.STRUCTURAL)


def test_a_table_fails_on_a_shattered_reconstruction() -> None:
    proof = _verify(Table(id="t", spine_index=0, span=_SPAN, markup=_GOOD_TABLE), _SHATTERED_TABLE)
    assert proof.outcome is ProofOutcome.FAILED


# --------------------------------------------------------------------------- #
# Text: predicate check once a target exists
# --------------------------------------------------------------------------- #


def test_text_without_a_target_is_unverifiable_not_a_crash() -> None:
    proof = _verify(Heading(id="h", spine_index=0, span=_SPAN, text="Title", level=1))
    assert (proof.outcome, proof.kind) == (ProofOutcome.UNVERIFIABLE, ProofKind.PREDICATE)
    assert proof.detail == "not translated yet"


def test_text_with_a_real_target_passes_the_predicate() -> None:
    proof = _verify(Paragraph(id="p", spine_index=0, span=_SPAN, text=_ECHO), _CJK)
    assert (proof.outcome, proof.kind) == (ProofOutcome.VERIFIED, ProofKind.PREDICATE)


def test_text_that_echoes_its_source_fails_the_predicate() -> None:
    proof = _verify(Paragraph(id="p", spine_index=0, span=_SPAN, text=_ECHO), _ECHO)
    assert proof.outcome is ProofOutcome.FAILED


# --------------------------------------------------------------------------- #
# Totality: every concrete element class is judged.
# --------------------------------------------------------------------------- #


def test_every_concrete_element_class_is_judged() -> None:
    elements: list[Element] = [
        Figure(id="g", spine_index=0, span=_SPAN, asset_id="a"),
        CodeBlock(id="c", spine_index=0, span=_SPAN, text="x"),
        Formula(id="f", spine_index=0, span=_SPAN, source="x"),
        Table(id="t", spine_index=0, span=_SPAN, markup=_GOOD_TABLE),
        Heading(id="h", spine_index=0, span=_SPAN, text="x", level=1),
        Paragraph(id="p", spine_index=0, span=_SPAN, text="x"),
        Dialogue(id="d", spine_index=0, span=_SPAN, text="x"),
        ListItem(id="l", spine_index=0, span=_SPAN, text="x"),
        Caption(id="cap", spine_index=0, span=_SPAN, text="x"),
    ]
    for element in elements:
        proof = _verify(element)
        assert proof.outcome in set(ProofOutcome), type(element).__name__


def test_an_unknown_element_kind_gets_an_explicit_no_path_proof() -> None:
    # A bare Element is none of the known kinds; it must still be judged.
    proof = _verify(Element(id="o", spine_index=0, span=_SPAN))
    assert (proof.outcome, proof.kind) == (ProofOutcome.UNVERIFIABLE, ProofKind.NONE)
    assert proof.detail == "no verification path"


def test_a_custom_element_subclass_also_reaches_the_no_path_proof() -> None:
    @dataclasses.dataclass(frozen=True)
    class _Opaque(Element):
        pass

    proof = _verify(_Opaque(id="o", spine_index=0, span=_SPAN))
    assert proof.kind is ProofKind.NONE

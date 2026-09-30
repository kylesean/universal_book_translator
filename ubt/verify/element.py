"""Dispatch an AST element to its verification path (ADR-0001 Phase 1).

The point of the typed AST is that *what an element is* decides *how it is
checked*. This module is that mapping, stated once:

- **Figure**, **CodeBlock** -- lossless by construction (an asset placed whole,
  a listing kept verbatim); the proof is ``PRESERVED``.
- **Formula**, **Table** -- the structural reconstruction check, run on the
  reconstruction when one exists, or on the source markup before render.
- **Text** (heading/paragraph/dialogue/list-item/caption) -- the translation
  predicate once a target exists; before translation the defined disposition is
  ``UNVERIFIABLE`` ("not translated yet"), never a crash or a silent skip.

``verify_element`` is *total*: it returns a :class:`Proof` for every concrete
element class, so no element can fall through un-judged. That totality is the
Phase-1 criterion "the verifier can judge every element", checked over the
corpus by ``scripts/shadow_ast.py``.
"""

from __future__ import annotations

from ubt.core.content.nodes import AssetKind
from ubt.model.ast import (
    CodeBlock,
    Element,
    Figure,
    Formula,
    Table,
    TextElement,
)
from ubt.model.fidelity import Proof, ProofKind
from ubt.verify.verifier import StructuralAsset, TextPair, Verifiers


def verify_element(
    element: Element,
    verifiers: Verifiers,
    *,
    reconstructed: str | None = None,
) -> Proof:
    """Judge one AST element; returns a :class:`Proof` for every element kind.

    ``reconstructed`` is the delivered markup (a reconstructed formula/table, or
    a translated text target). When absent, assets are checked on their source
    markup and text is reported ``UNVERIFIABLE`` pending translation.
    """
    if isinstance(element, Figure):
        return Proof.preserved("figure placed")
    if isinstance(element, CodeBlock):
        return Proof.preserved("code kept verbatim")
    if isinstance(element, Formula):
        text = element.source if reconstructed is None else reconstructed
        return verifiers.structural.verify(StructuralAsset(AssetKind.FORMULA, text))
    if isinstance(element, Table):
        text = element.markup if reconstructed is None else reconstructed
        return verifiers.structural.verify(StructuralAsset(AssetKind.TABLE, text))
    if isinstance(element, TextElement):
        if reconstructed is None:
            return Proof.unknown(ProofKind.PREDICATE, "not translated yet")
        return verifiers.text.verify(TextPair(element.text, reconstructed))
    return Proof.unknown(ProofKind.NONE, "no verification path")


__all__ = ["verify_element"]

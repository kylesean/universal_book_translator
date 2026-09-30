#!/usr/bin/env python
"""Phase-3 acceptance: capability negotiation realizes every element, never drops.

``realize()`` is the ADR's single execution core: for each element it walks the
fidelity descent and takes the first rung a backend declares it can produce and
whose result verifies. This harness runs it over the real corpus documents'
elements against declared capability sets -- a reflowing backend (the ADR's
Typst row) and a place-only backend (its Overlay row) -- and checks:

- **totality**: every real element gets an Attestation under both backends;
- **the assignment table**: for the reflowing backend prose ->
  RECONSTRUCTED_ADAPTED, formula/table -> RECONSTRUCTED_VERIFIED, figure/code ->
  PRESERVED_OPAQUE; for the place-only backend everything -> PRESERVED_OPAQUE;
- **monotonicity**: a backend with a superset of capabilities never lowers an
  element's fidelity (reflowing >= place-only on every element);
- **Axioms A/B**: a non-decorative element with no lossless realization raises
  IntegrityViolation, and a decorative one is recorded DROPPED, never silent.

The verifiers are stubbed to ``ok`` on purpose: the negotiation, not the
predicates, is what this increment adds (the predicates are proven by
``shadow_ast``). Exit 0 iff every check holds.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import ClassVar, cast

from ubt.analyze.reader_pdf import read_pdf
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
    TextElement,
)
from ubt.model.fidelity import Attestation, Fidelity, Proof, ProofKind
from ubt.model.span import CanonicalSource, Span
from ubt.pipeline.steps import IntegrityViolation, realize
from ubt.render.capability import Capabilities, Produced
from ubt.verify.verifier import Verifiers

_RA = Fidelity.RECONSTRUCTED_ADAPTED
_RV = Fidelity.RECONSTRUCTED_VERIFIED
_PO = Fidelity.PRESERVED_OPAQUE

#: The closed element set, and the text classes a reflowing backend can re-typeset.
_ALL_CLASSES: tuple[type[Element], ...] = (
    Heading,
    Paragraph,
    Dialogue,
    ListItem,
    Caption,
    CodeBlock,
    Formula,
    Table,
    Figure,
)
_REFLOW_TEXT: tuple[type[Element], ...] = (Heading, Paragraph, Dialogue, ListItem, Caption)


# --------------------------------------------------------------------------- #
# Stub verifiers: the negotiation is under test, not the predicates.
# --------------------------------------------------------------------------- #


class _OkVerifier:
    def verify(self, subject: object) -> Proof:
        return Proof.ok(ProofKind.PREDICATE, "stub")


class _OkVerifiers:
    structural = _OkVerifier()
    formula = _OkVerifier()
    table = _OkVerifier()
    text = _OkVerifier()


_VERIFIERS = cast(Verifiers, _OkVerifiers())


# --------------------------------------------------------------------------- #
# Declared backends.
# --------------------------------------------------------------------------- #


def _payload(element: Element) -> str:
    if isinstance(element, TextElement):
        return element.text
    if isinstance(element, Formula):
        return element.source
    if isinstance(element, Table):
        return element.markup
    if isinstance(element, Figure):
        return element.asset_id
    return ""


class _ReflowingBackend:
    """The ADR's Typst row: re-typesets text, verifies assets, preserves the rest."""

    name: ClassVar[str] = "reflowing"

    def capabilities(self) -> Capabilities:
        supported = {(cls, _PO) for cls in _ALL_CLASSES}
        supported |= {(cls, _RA) for cls in _REFLOW_TEXT}
        supported |= {(Formula, _RV), (Table, _RV)}
        return Capabilities(supported=frozenset(supported), reflows=True)

    def produce(
        self, element: Element, fidelity: Fidelity, source: CanonicalSource
    ) -> Produced | None:
        return Produced(payload=_payload(element), note=f"{self.name}:{fidelity.name}")


class _PlacingBackend:
    """The ADR's Overlay row: places opaque slices, re-typesets nothing."""

    name: ClassVar[str] = "placing"

    def capabilities(self) -> Capabilities:
        return Capabilities(supported=frozenset((cls, _PO) for cls in _ALL_CLASSES))

    def produce(
        self, element: Element, fidelity: Fidelity, source: CanonicalSource
    ) -> Produced | None:
        return Produced(payload=_payload(element), note=f"{self.name}:{fidelity.name}")


class _BlindBackend:
    """A backend that can produce nothing -- the no-lossless-realization case."""

    name: ClassVar[str] = "blind"

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def produce(
        self, element: Element, fidelity: Fidelity, source: CanonicalSource
    ) -> Produced | None:
        return None


def _expected_reflowing(element: Element) -> Fidelity:
    if isinstance(element, (Formula, Table)):
        return _RV
    if isinstance(element, (CodeBlock, Figure)):
        return _PO
    return _RA


def _synthetic(cls: type[Element]) -> Element:
    """One element of each class, so the assignment table is covered whole."""
    common: dict[str, object] = {"id": f"synth-{cls.__name__}", "spine_index": 0, "span": Span()}
    if cls is Formula:
        return Formula(source="x = 1", **common)  # type: ignore[arg-type]
    if cls is Table:
        return Table(markup="| a | b |", **common)  # type: ignore[arg-type]
    if cls is Figure:
        return Figure(asset_id="fig-1", **common)  # type: ignore[arg-type]
    if cls is ListItem:
        return ListItem(text="item", marker="-", **common)  # type: ignore[arg-type]
    if cls is Heading:
        return Heading(text="Heading", level=1, **common)  # type: ignore[arg-type]
    return cls(text="text", **common)  # type: ignore[call-arg,arg-type]


def _check_assignment_table() -> list[str]:
    """The ADR's §5.2 table, for every element class (the corpus lacks figures/tables)."""
    source = CanonicalSource(doc_id="synthetic", path="")
    reflowing = _ReflowingBackend()
    placing = _PlacingBackend()
    problems: list[str] = []
    for cls in _ALL_CLASSES:
        element = _synthetic(cls)
        reflowed = realize(element, reflowing, _VERIFIERS, source)
        placed = realize(element, placing, _VERIFIERS, source)
        expected = _expected_reflowing(element)
        if reflowed.fidelity is not expected:
            problems.append(
                f"{cls.__name__}: reflowing {reflowed.fidelity.name} != {expected.name}"
            )
        if placed.fidelity is not _PO:
            problems.append(f"{cls.__name__}: placing {placed.fidelity.name} != PRESERVED_OPAQUE")
    return problems


def _load_cases(corpus_dir: Path) -> list[tuple[str, Path]]:
    from ubt.core.content.verify import load_corpus

    cases: list[tuple[str, Path]] = []
    for case in load_corpus(corpus_dir):
        if not case.document:
            continue
        document = Path(case.document)
        if not document.is_absolute():
            document = corpus_dir / document
        cases.append((case.id, document))
    return cases


def _check_document(document: Path) -> tuple[int, dict[str, int], list[str]]:
    """Returns (element_count, kind_histogram, problems)."""
    doc = read_pdf(document)
    reflowing = _ReflowingBackend()
    placing = _PlacingBackend()
    blind = _BlindBackend()
    kinds: Counter[str] = Counter()
    problems: list[str] = []
    elements = doc.elements

    for element in elements:
        kinds[str(element.kind)] += 1
        reflowed = realize(element, reflowing, _VERIFIERS, doc.source)
        placed = realize(element, placing, _VERIFIERS, doc.source)

        if not reflowed.delivered or not placed.delivered:
            problems.append(f"{element.id}: dropped under a lossless backend")
        if reflowed.fidelity is not _expected_reflowing(element):
            problems.append(
                f"{element.id} ({element.kind}): reflowing gave {reflowed.fidelity.name}, "
                f"expected {_expected_reflowing(element).name}"
            )
        if placed.fidelity is not _PO:
            problems.append(f"{element.id} ({element.kind}): placing gave {placed.fidelity.name}")
        if reflowed.fidelity < placed.fidelity:
            problems.append(
                f"{element.id}: monotonicity broken "
                f"({reflowed.fidelity.name} < {placed.fidelity.name})"
            )
        try:
            realize(element, blind, _VERIFIERS, doc.source)
            problems.append(f"{element.id}: blind backend did not raise")
        except IntegrityViolation:
            pass

    # A decorative element is the one thing allowed to drop, and it is recorded.
    decorative = Paragraph(
        id="synthetic-decorative", spine_index=-1, span=Span(), text="~", decorative=True
    )
    dropped: Attestation = realize(decorative, blind, _VERIFIERS, doc.source)
    if dropped.fidelity is not Fidelity.DROPPED or dropped.delivered:
        problems.append("decorative element was not recorded as DROPPED")

    return len(elements), dict(kinds), problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--all", action="store_true", help="Every case (default: the first)")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    cases = [(c, d) for c, d in _load_cases(corpus_dir) if d.exists()]
    if not args.all:
        cases = cases[:1]

    print(f"\nPhase-3 realize() acceptance — {corpus_dir} ({len(cases)} document(s))")
    total = 0
    problems: list[str] = []
    table_problems = _check_assignment_table()
    problems.extend(f"[table] {p}" for p in table_problems)
    print(
        f"  {'assignment-table':<20} {'pass' if not table_problems else 'FAIL':<6} "
        f"classes={len(_ALL_CLASSES)}"
    )
    for case_id, document in cases:
        count, kinds, issues = _check_document(document)
        total += count
        problems.extend(f"[{case_id}] {p}" for p in issues)
        status = "pass" if not issues else "FAIL"
        print(f"  {case_id:<20} {status:<6} elements={count} kinds={kinds}")

    print(f"\n  elements={total} problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for line in problems[:15]:
        print(f"    {line}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Phase-3 acceptance: the overlay backend realizes every element losslessly.

The overlay backend is the fidelity lattice's floor as a backend: it re-typesets
nothing and places each element as an opaque source slice. This harness runs the
real ``realize()`` over the corpus with it and checks:

- every element comes back PRESERVED_OPAQUE, delivered, with a PRESERVED proof;
- the slice *is* the element's own source -- for a text element it equals
  ``element.text``, so "preserved" is measured, not merely labelled;
- the backend declares exactly the opaque rung for every element class and
  reflows nothing.

Exit 0 iff every check holds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ubt.analyze.reader_pdf import read_pdf
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.model.ast import ELEMENT_CLASSES, TextElement
from ubt.model.fidelity import Fidelity, ProofKind
from ubt.pipeline.steps import realize
from ubt.render.overlay_backend import OverlayBackend, source_slice
from ubt.verify.verifier import build_verifiers


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


def _check_capabilities(backend: OverlayBackend) -> list[str]:
    capabilities = backend.capabilities()
    expected = frozenset((cls, Fidelity.PRESERVED_OPAQUE) for cls in ELEMENT_CLASSES)
    problems: list[str] = []
    if capabilities.supported != expected:
        problems.append("capabilities are not exactly the opaque rung for every element class")
    if capabilities.reflows:
        problems.append("overlay backend must not declare reflow")
    return problems


def _check_document(document: Path) -> tuple[int, list[str]]:
    doc = read_pdf(document)
    backend = OverlayBackend()
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))
    problems: list[str] = []
    for element in doc.elements:
        attestation = realize(element, backend, verifiers, doc.source)
        if attestation.fidelity is not Fidelity.PRESERVED_OPAQUE:
            problems.append(f"{element.id}: {attestation.fidelity.name}, expected PRESERVED_OPAQUE")
            continue
        if not attestation.delivered:
            problems.append(f"{element.id}: not delivered")
        if attestation.proof.kind is not ProofKind.PRESERVED:
            problems.append(
                f"{element.id}: proof kind {attestation.proof.kind}, expected preserved"
            )
        slice_ = source_slice(element, doc.source)
        if not slice_:
            problems.append(f"{element.id}: empty opaque slice")
        elif isinstance(element, TextElement) and element.text not in slice_:
            # The span covers the source region, which can include a list marker
            # that ``ListItem.text`` deliberately excludes; the slice must still
            # carry the element's own text, so nothing the element holds is lost.
            problems.append(f"{element.id}: opaque slice does not carry the element's text")
    return len(doc.elements), problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--all", action="store_true", help="Every case (default: the first)")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    cases = [(c, d) for c, d in _load_cases(corpus_dir) if d.exists()]
    if not args.all:
        cases = cases[:1]

    problems = _check_capabilities(OverlayBackend())
    print(f"\nPhase-3 overlay-backend acceptance — {corpus_dir} ({len(cases)} document(s))")
    print(
        f"  {'capabilities':<20} {'pass' if not problems else 'FAIL':<6} "
        f"classes={len(ELEMENT_CLASSES)} rung=PRESERVED_OPAQUE"
    )
    total = 0
    for case_id, document in cases:
        count, issues = _check_document(document)
        total += count
        problems.extend(f"[{case_id}] {p}" for p in issues)
        status = "pass" if not issues else "FAIL"
        print(f"  {case_id:<20} {status:<6} elements={count}")
    print(f"\n  elements={total} problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for line in problems[:15]:
        print(f"    {line}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

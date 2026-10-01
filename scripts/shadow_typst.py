#!/usr/bin/env python
"""Phase-3 acceptance: the Typst backend reflows text elements and verifies them.

The reflowing backend declares ``RECONSTRUCTED_ADAPTED`` for the text classes it
can re-typeset. This harness runs the real ``realize()`` over the corpus with it
and checks:

- a reflowable text element with a translation comes back RECONSTRUCTED_ADAPTED,
  its fragment produced and verified;
- one with *no* translation -- no translation, or a ``skip_translate`` element
  the reader typed as page furniture / a listing -- descends to
  PRESERVED_OPAQUE, because there is nothing to reflow and the lattice keeps
  the source rather than losing it;
- Formula and Table are reconstructed (RECONSTRUCTED_VERIFIED) from the
  delivered markup when their structural witness passes, and descend to opaque
  otherwise; every other element class is placed opaque;
- the capability set is exactly the opaque rung for all classes plus
  RECONSTRUCTED_ADAPTED for the reflowable text classes and
  RECONSTRUCTED_VERIFIED for Formula and Table.

Translations are the dry-run rehearsal's (``[模拟翻译] <source>``), so the run
needs no API key. Exit 0 iff every check holds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ubt.analyze.reader_pdf import read_pdf
from ubt.core.qe.fast_pass import REHEARSAL_MARKER, FastPassFilter
from ubt.model.ast import ELEMENT_CLASSES, Formula, Table
from ubt.model.fidelity import Fidelity
from ubt.pipeline.steps import realize
from ubt.render.typst_backend import REFLOW_CLASSES, TypstBackend
from ubt.verify.verifier import Verifiers, build_verifiers


def _verifiers() -> Verifiers:
    return build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))


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


def _check_capabilities() -> list[str]:
    capabilities = TypstBackend().capabilities()
    expected = {(cls, Fidelity.PRESERVED_OPAQUE) for cls in ELEMENT_CLASSES}
    expected |= {(cls, Fidelity.RECONSTRUCTED_ADAPTED) for cls in REFLOW_CLASSES}
    expected |= {
        (Formula, Fidelity.RECONSTRUCTED_VERIFIED),
        (Table, Fidelity.RECONSTRUCTED_VERIFIED),
    }
    problems: list[str] = []
    if capabilities.supported != frozenset(expected):
        problems.append("capabilities are not exactly opaque-for-all + adapted-for-reflow")
    if not capabilities.reflows:
        problems.append("typst backend must declare reflow")
    return problems


def _check_document(document: Path) -> tuple[int, int, list[str]]:
    doc = read_pdf(document)
    verifiers = _verifiers()
    # A ``skip_translate`` element is one the reader typed as page furniture or
    # a kept listing: the backend deliberately places it opaque, so it must not
    # appear in the translation map (giving it one would reflow what the reader
    # chose to keep verbatim -- the gap the attestation shadow first caught).
    translations = {
        element.id: f"{REHEARSAL_MARKER} {element.text}"
        for element in doc.elements
        if isinstance(element, REFLOW_CLASSES) and not element.skip_translate
    }
    backend = TypstBackend(translations)
    untranslated = TypstBackend()
    problems: list[str] = []
    reflowed = 0

    for element in doc.elements:
        attestation = realize(element, backend, verifiers, doc.source)
        if isinstance(element, REFLOW_CLASSES) and not element.skip_translate:
            if attestation.fidelity is not Fidelity.RECONSTRUCTED_ADAPTED:
                problems.append(
                    f"{element.id}: {element.kind} {attestation.fidelity.name}, "
                    f"expected RECONSTRUCTED_ADAPTED"
                )
            else:
                reflowed += 1
                if not attestation.proof.verified:
                    problems.append(
                        f"{element.id}: fragment not verified ({attestation.proof.detail})"
                    )
            produced = backend.produce(element, Fidelity.RECONSTRUCTED_ADAPTED, doc.source)
            if produced is None or not produced.fragment.strip():
                problems.append(f"{element.id}: no Typst fragment produced")
            bare = realize(element, untranslated, verifiers, doc.source)
            if bare.fidelity is not Fidelity.PRESERVED_OPAQUE:
                problems.append(
                    f"{element.id}: untranslated {element.kind} {bare.fidelity.name}, "
                    f"expected PRESERVED_OPAQUE"
                )
        elif isinstance(element, (Formula, Table)):
            # The backend reconstructs formulas and tables; a verified
            # reconstruction is the design, and a failed structural check falls
            # through to the opaque slice -- both are acceptable outcomes.
            if attestation.fidelity not in (
                Fidelity.RECONSTRUCTED_VERIFIED,
                Fidelity.PRESERVED_OPAQUE,
            ):
                problems.append(
                    f"{element.id} ({element.kind}): {attestation.fidelity.name}, "
                    f"expected RECONSTRUCTED_VERIFIED or PRESERVED_OPAQUE"
                )
        elif attestation.fidelity is not Fidelity.PRESERVED_OPAQUE:
            # skip_translate reflow classes (page furniture, kept listings) and
            # every class the backend does not reflow are placed opaque.
            problems.append(
                f"{element.id} ({element.kind}): {attestation.fidelity.name}, "
                f"expected PRESERVED_OPAQUE"
            )
    return len(doc.elements), reflowed, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--all", action="store_true", help="Every case (default: the first)")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    cases = [(c, d) for c, d in _load_cases(corpus_dir) if d.exists()]
    if not args.all:
        cases = cases[:1]

    problems = _check_capabilities()
    print(f"\nPhase-3 typst-backend acceptance — {corpus_dir} ({len(cases)} document(s))")
    print(
        f"  {'capabilities':<20} {'pass' if not problems else 'FAIL':<6} "
        f"classes={len(ELEMENT_CLASSES)} reflow=text"
    )
    total = 0
    for case_id, document in cases:
        count, reflowed, issues = _check_document(document)
        total += count
        problems.extend(f"[{case_id}] {p}" for p in issues)
        status = "pass" if not issues else "FAIL"
        print(f"  {case_id:<20} {status:<6} elements={count} reflowed={reflowed}")
    print(f"\n  elements={total} problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for line in problems[:15]:
        print(f"    {line}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

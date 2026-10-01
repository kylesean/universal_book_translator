#!/usr/bin/env python
"""Phase-3 acceptance: the realization decision plan and its shadow comparison.

The decision plan is ``realize()`` projected over a document. This checks that

- the plan is total for the corpus: every element decided, zero violations;
- a lowering that *can* honor the plan does: the HTML view places every element
  at its planned fidelity, so ``divergences`` is empty;
- a lowering that cannot (the PDF overlay compositor, which only keeps source)
  is *detected*: ``divergences`` equals its ``descended_ids`` -- the shadow
  surfaces the descent instead of passing it off as the planned fidelity.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from ubt.analyze.reader_pdf import read_pdf
from ubt.core.qe.fast_pass import REHEARSAL_MARKER, FastPassFilter
from ubt.layout.theme import resolve_theme
from ubt.pipeline.decisions import divergences, plan_realization
from ubt.render.html_view import compose_html
from ubt.render.outputs import compose
from ubt.render.typst_backend import REFLOW_CLASSES, TypstBackend
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
        if document.exists():
            cases.append((case.id, document))
    return cases


def _check(case_id: str, document: Path, tmp: Path) -> tuple[int, list[str]]:
    doc = read_pdf(document)
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))
    translations = {
        element.id: f"{REHEARSAL_MARKER} {element.text}"
        for element in doc.elements
        if isinstance(element, REFLOW_CLASSES) and not element.skip_translate
    }
    backend = TypstBackend(translations, theme=resolve_theme("en", "zh"))
    plan = plan_realization(doc, backend, verifiers)

    problems: list[str] = []
    if plan.violations:
        problems.append(f"{len(plan.violations)} element(s) have no lossless realization")
    if plan.total != len(doc.elements):
        problems.append(f"plan covers {plan.total}/{len(doc.elements)} elements")

    # A lowering that can honor the plan: the HTML view.
    html_composition = compose_html(doc, plan.attestations, translations, tmp / f"{case_id}.html")
    html_realized = {p.element_id: p.placed_as for p in html_composition.placements}
    html_divergences = divergences(plan, html_realized)
    if html_divergences:
        problems.append(f"HTML view diverged on {len(html_divergences)} element(s)")

    # A lowering that cannot (keeps source): the overlay compositor descends, and
    # the shadow must report exactly that many divergences.
    pdf_composition = compose(doc, plan.attestations, document, tmp / f"{case_id}.pdf")
    pdf_realized = {p.element_id: p.placed_as for p in pdf_composition.placements}
    pdf_divergences = divergences(plan, pdf_realized)
    if len(pdf_divergences) != len(pdf_composition.descended_ids):
        problems.append(
            f"PDF descent shadow {len(pdf_divergences)} != descended {len(pdf_composition.descended_ids)}"
        )

    print(
        f"  {case_id:<20} {'pass' if not problems else 'FAIL':<5} elements={plan.total} "
        f"html_divergences={len(html_divergences)} pdf_descended={len(pdf_composition.descended_ids)}"
    )
    return plan.total, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--all", action="store_true", help="Every case (default: the first)")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    cases = _load_cases(corpus_dir)
    if not args.all:
        cases = cases[:1]

    problems: list[str] = []
    total = 0
    with tempfile.TemporaryDirectory(prefix="ubt-decisions-") as tmp_str:
        tmp = Path(tmp_str)
        print(f"\nRealization decision plan — {corpus_dir} ({len(cases)} document(s))")
        for case_id, document in cases:
            count, issues = _check(case_id, document, tmp)
            total += count
            problems.extend(f"[{case_id}] {issue}" for issue in issues)

    print(f"\n  elements={total} problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for line in problems[:15]:
        print(f"    {line}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

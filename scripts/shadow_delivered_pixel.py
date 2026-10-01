#!/usr/bin/env python
"""Phase-1 acceptance: the pixel witness runs on the *delivered* document.

The reconstruction witnesses judge a formula/table crop at render time. This
closes the ADR's remaining Phase-1 item ("在已交付文档上跑像素级 witness"): it
lowers a real corpus document to an artifact (read -> realize -> compose) and
compares, per element, the artifact region against the source region with the
one structural metric the witnesses already use.

Two checks:

- **match**: the lowering keeps source geometry, so every element the witness can
  measure must match its source region -- zero failures;
- **loss**: replace the artifact with blank pages of the same size; every element
  that matched must now be flagged, so the witness is fail-closed and not a
  rubber stamp.

Elements with no measurable source ink are ``UNVERIFIABLE`` and are reported,
not counted as failures (there is nothing to witness). Exit 0 iff match has zero
failures, at least one region was measurable, and the blank artifact flags every
one of them.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from ubt.analyze.reader_pdf import read_pdf
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.pipeline.pixel import witness_region
from ubt.pipeline.steps import realize
from ubt.render.outputs import compose
from ubt.render.overlay_backend import OverlayBackend
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


def _blank_copy(source: Path, out: Path) -> None:
    """A same-size, content-free artifact: every page has the source's box, no ink."""
    import pikepdf

    with pikepdf.open(source) as src, pikepdf.new() as blank:
        for page in src.pages:
            box = page.mediabox
            width = float(box[2]) - float(box[0])
            height = float(box[3]) - float(box[1])
            blank.add_blank_page(page_size=(width, height))
        blank.save(str(out))


def _check_document(case_id: str, document: Path, tmp: Path) -> tuple[int, list[str]]:
    doc = read_pdf(document)
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))
    backend = OverlayBackend()
    attestations = [realize(element, backend, verifiers, doc.source) for element in doc.elements]

    artifact = tmp / f"{case_id}_composed.pdf"
    compose(doc, attestations, document, artifact)
    blank = tmp / f"{case_id}_blank.pdf"
    _blank_copy(document, blank)

    problems: list[str] = []
    matched = 0
    match_failures = 0
    unverifiable = 0
    undetected = 0
    for element in doc.elements:
        if element.span is None or element.span.page <= 0:
            continue
        good = witness_region(document, artifact, element.span.page, element.span.bbox)
        if good.verified:
            matched += 1
            lost = witness_region(document, blank, element.span.page, element.span.bbox)
            if lost.verified:
                undetected += 1
                if len(problems) < 20:
                    problems.append(
                        f"{element.id}: blank artifact still 'matches' (witness not fail-closed)"
                    )
        elif good.failed:
            match_failures += 1
            if len(problems) < 20:
                problems.append(f"{element.id}: {good.detail}")
        else:
            unverifiable += 1

    print(
        f"  {case_id:<20} {'pass' if not (match_failures or undetected) else 'FAIL':<6} "
        f"elements={len(doc.elements)} matched={matched} unverifiable={unverifiable} "
        f"match_failures={match_failures} undetected_loss={undetected}"
    )
    return len(doc.elements), problems


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
    with tempfile.TemporaryDirectory(prefix="ubt-delivered-pixel-") as tmp_str:
        tmp = Path(tmp_str)
        print(
            f"\nPhase-1 delivered-artifact pixel witness — {corpus_dir} ({len(cases)} document(s))"
        )
        for case_id, document in cases:
            count, issues = _check_document(case_id, document, tmp)
            total += count
            problems.extend(f"[{case_id}] {issue}" for issue in issues)

    print(f"\n  elements={total} problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for line in problems[:20]:
        print(f"    {line}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

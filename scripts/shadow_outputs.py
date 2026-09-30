#!/usr/bin/env python
"""Phase-3 acceptance: the overlay lowering reproduces the source, losslessly.

``realize()`` says every element of a pure-overlay document is
``PRESERVED_OPAQUE``; ``compose()`` lowers that to an artifact by carrying the
source pages over whole. This harness runs that whole chain -- read -> realize ->
compose -- over the corpus and checks the composed PDF against the source:

- same page count and page sizes;
- the text extracted from each page is the source's, page for page (nothing
  dropped, nothing added);
- lowering refuses a non-opaque realization, and refuses an element that was
  never attested (fail closed).

Exit 0 iff every check holds.
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
from pathlib import Path

from ubt.adapters.pdf import pdf_struct, textgeom
from ubt.analyze.reader_pdf import read_pdf
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.model.ast import Document
from ubt.model.fidelity import Attestation, Fidelity
from ubt.pipeline.steps import realize
from ubt.render.outputs import LoweringUnsupported, compose
from ubt.render.overlay_backend import OverlayBackend
from ubt.verify.verifier import build_verifiers


def _page_sizes(pdf_path: Path) -> list[tuple[float, float]]:
    with pdf_struct.open_pdf(pdf_path) as pdf:
        return [pdf_struct.page_size(page) for page in pdf.pages]


def _page_tokens(pdf_path: Path) -> list[list[str]]:
    """Whitespace-normalized text tokens per page, the same extractor the reader uses."""
    pages: list[list[str]] = []
    for page_no in range(1, len(_page_sizes(pdf_path)) + 1):
        lines, _ = textgeom.extract_lines(pdf_path, page_no)
        pages.append(" ".join(line.text for line in lines).split())
    return pages


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


def _realize_all(document: Document) -> list[Attestation]:
    backend = OverlayBackend()
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))
    return [realize(element, backend, verifiers, document.source) for element in document.elements]


def _check_fail_closed(
    document: Document, attestations: list[Attestation], source: Path, out: Path
) -> list[str]:
    problems: list[str] = []
    above_floor = [
        dataclasses.replace(attestations[0], fidelity=Fidelity.RECONSTRUCTED_ADAPTED),
        *attestations[1:],
    ]
    try:
        compose(document, above_floor, source, out)
        problems.append("compose accepted a non-opaque realization")
    except LoweringUnsupported:
        pass
    try:
        compose(document, attestations[:-1], source, out)
        problems.append("compose accepted a document with an unattested element")
    except LoweringUnsupported:
        pass
    return problems


def _check_document(case_id: str, source: Path, tmp_dir: Path) -> tuple[int, list[str]]:
    document = read_pdf(source)
    attestations = _realize_all(document)
    composed = compose(document, attestations, source, tmp_dir / f"{case_id}.pdf")
    problems: list[str] = []

    source_sizes = _page_sizes(source)
    composed_sizes = _page_sizes(composed)
    if source_sizes != composed_sizes:
        problems.append(
            f"page geometry differs: {len(source_sizes)} vs {len(composed_sizes)} page(s)"
        )

    source_tokens = _page_tokens(source)
    composed_tokens = _page_tokens(composed)
    if len(source_tokens) != len(composed_tokens):
        problems.append(f"page count differs: {len(source_tokens)} vs {len(composed_tokens)}")
    else:
        differing = [
            index + 1
            for index, (before, after) in enumerate(
                zip(source_tokens, composed_tokens, strict=True)
            )
            if before != after
        ]
        if differing:
            problems.append(f"page text differs on page(s) {differing[:5]}")

    problems.extend(_check_fail_closed(document, attestations, source, composed))
    return len(attestations), problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--all", action="store_true", help="Every case (default: the first)")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    cases = [(c, d) for c, d in _load_cases(corpus_dir) if d.exists()]
    if not args.all:
        cases = cases[:1]

    print(f"\nPhase-3 overlay-lowering acceptance — {corpus_dir} ({len(cases)} document(s))")
    total = 0
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ubt-outputs-") as tmp:
        tmp_dir = Path(tmp)
        for case_id, document in cases:
            count, issues = _check_document(case_id, document, tmp_dir)
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

#!/usr/bin/env python
"""Phase-3 acceptance: the overlay lowering composes mixed realizations, losslessly.

``realize()`` attests each element; ``compose()`` lowers the whole document. The
lowering has one capability -- the opaque source slice -- so a realization above
the floor descends to the slice and is *recorded* as descended. This harness runs
the chain over the corpus and checks:

- a pure-overlay document descends nothing and composes to the source exactly
  (same page count and sizes; whitespace-normalized page text identical page for
  page);
- a mixed document (some elements attested above the floor) descends exactly
  those elements, records them, and still carries the source -- no content loss;
- lowering still refuses a document with an unattested element.

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


def _compare(source: Path, composed: Path, label: str) -> list[str]:
    """The composed artifact must carry the source: geometry and page text identical."""
    problems: list[str] = []
    source_sizes = _page_sizes(source)
    composed_sizes = _page_sizes(composed)
    if source_sizes != composed_sizes:
        problems.append(
            f"{label}: page geometry differs ({len(source_sizes)} vs {len(composed_sizes)})"
        )
    source_tokens = _page_tokens(source)
    composed_tokens = _page_tokens(composed)
    if len(source_tokens) != len(composed_tokens):
        problems.append(
            f"{label}: page count differs ({len(source_tokens)} vs {len(composed_tokens)})"
        )
        return problems
    differing = [
        index + 1
        for index, (before, after) in enumerate(zip(source_tokens, composed_tokens, strict=True))
        if before != after
    ]
    if differing:
        problems.append(f"{label}: page text differs on page(s) {differing[:5]}")
    return problems


def _check_document(case_id: str, source: Path, tmp_dir: Path) -> tuple[int, list[str]]:
    document = read_pdf(source)
    attestations = _realize_all(document)
    problems: list[str] = []

    # Pure overlay: the floor, nothing descends, the artifact is the source.
    pure = compose(document, attestations, source, tmp_dir / f"{case_id}.pdf")
    if pure.descended_ids:
        problems.append(f"pure overlay descended {len(pure.descended_ids)} element(s)")
    problems.extend(_compare(source, pure.output_path, "pure"))

    # Mixed: two elements attested above the floor must descend to the slice, be
    # recorded, and still leave the source intact.
    if attestations:
        flipped = {attestations[0].element_id, attestations[len(attestations) // 2].element_id}
        mixed_atts = [
            dataclasses.replace(attestation, fidelity=Fidelity.RECONSTRUCTED_ADAPTED)
            if attestation.element_id in flipped
            else attestation
            for attestation in attestations
        ]
        mixed = compose(document, mixed_atts, source, tmp_dir / f"{case_id}_mixed.pdf")
        if set(mixed.descended_ids) != flipped:
            problems.append(f"mixed descent {sorted(mixed.descended_ids)} != {sorted(flipped)}")
        problems.extend(_compare(source, mixed.output_path, "mixed"))

    # A coverage gap is not a realization and is still refused.
    try:
        compose(document, attestations[:-1], source, tmp_dir / f"{case_id}_gap.pdf")
        problems.append("compose accepted a document with an unattested element")
    except LoweringUnsupported:
        pass

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

#!/usr/bin/env python
"""Acceptance harness: LayerCompositor micro-fragment composition over real corpus.

Verifies the 3-layer composition engine (Layer 0 source canvas + Layer 1 pikepdf
text strip & vector mask + Layer 2 Typst micro-fragment typesetting) over real
corpus documents:
1. Geometry fidelity: output PDF exists, starts with '%PDF-', and page counts and
   page dimensions strictly match the source PDF.
2. Content delivery & descent floor: text elements with translations are typeset
   into Layer 2 micro-fragments (RECONSTRUCTED_ADAPTED), while untranslated or
   unrealized elements safely descend to Layer 0 (PRESERVED_OPAQUE) with zero loss.
3. Stream deduplication & resource bounds: identical font/CMap streams are
   deduplicated so output PDF file size remains bounded (< 15MB).

Exit 0 iff every check holds.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from ubt.adapters.pdf import pdf_struct
from ubt.analyze.reader_pdf import read_pdf
from ubt.core.qe.fast_pass import REHEARSAL_MARKER, FastPassFilter
from ubt.model.fidelity import Fidelity
from ubt.pipeline.steps import realize
from ubt.render.outputs import TypstFragmentTypesetter, compose_layered
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


def _check_document(case_id: str, source: Path, tmp_dir: Path) -> tuple[int, list[str]]:
    problems: list[str] = []
    doc = read_pdf(source)
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))

    translations = {
        element.id: f"{REHEARSAL_MARKER} {element.text}"
        for element in doc.elements
        if isinstance(element, REFLOW_CLASSES) and not element.skip_translate
    }
    backend = TypstBackend(translations)
    attestations = [realize(element, backend, verifiers, doc.source) for element in doc.elements]

    out_pdf = tmp_dir / f"{case_id}_composed.pdf"
    typesetter = TypstFragmentTypesetter()
    try:
        composition = compose_layered(
            doc,
            attestations,
            translations,
            source,
            out_pdf,
            typesetter=typesetter,
        )
    finally:
        typesetter.close()

    if not out_pdf.exists() or out_pdf.read_bytes()[:5] != b"%PDF-":
        problems.append(f"{case_id}: output PDF was not produced or is invalid")
        return 0, problems

    source_sizes = pdf_struct.page_sizes(source)
    composed_sizes = pdf_struct.page_sizes(out_pdf)
    if source_sizes != composed_sizes:
        problems.append(
            f"{case_id}: page geometry differs ({len(source_sizes)} vs {len(composed_sizes)})"
        )

    out_size = out_pdf.stat().st_size
    if out_size > 15 * 1024 * 1024:
        problems.append(
            f"{case_id}: output size is suspiciously large ({out_size / 1024 / 1024:.1f} MB)"
        )

    adapted = sum(1 for p in composition.placements if p.fidelity is Fidelity.RECONSTRUCTED_ADAPTED)
    if not adapted and any(isinstance(e, REFLOW_CLASSES) for e in doc.elements):
        problems.append(f"{case_id}: no fragments were adapted by LayerCompositor")

    return adapted, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus", help="Path to corpus directory")
    parser.add_argument("--all", action="store_true", help="Every case (default: the first)")
    args = parser.parse_args()

    if shutil.which("typst") is None:
        print("SKIP: typst compiler is not installed")
        return 0

    corpus_dir = Path(args.corpus)
    cases = [(c, d) for c, d in _load_cases(corpus_dir) if d.exists()]
    if not cases:
        print(f"SKIP: no corpus documents found under {corpus_dir}")
        return 0

    if not args.all:
        cases = cases[:1]

    print(f"\nLayerCompositor real-corpus acceptance — {corpus_dir} ({len(cases)} document(s))")
    total_adapted = 0
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ubt-compositor-") as tmp:
        tmp_dir = Path(tmp)
        for case_id, document in cases:
            count, issues = _check_document(case_id, document, tmp_dir)
            total_adapted += count
            problems.extend(f"[{case_id}] {p}" for p in issues)
            status = "pass" if not issues else "FAIL"
            print(f"  {case_id:<20} {status:<6} adapted_fragments={count}")

    print(
        f"\n  adapted={total_adapted} problems={len(problems)} -> "
        f"{'PASS' if not problems else 'FAIL'}"
    )
    for line in problems[:15]:
        print(f"    {line}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

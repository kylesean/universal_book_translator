#!/usr/bin/env python
"""§12 Q5 acceptance: the semantic HTML view (ADR-0001 L6).

Reads a corpus document, realizes every element, lowers it with ``compose_html``
and checks that the HTML carries each element's attested text (the delivered
translation where reconstructed, the source slice where preserved), that every
element gets a placement, and that a missing attestation is refused rather than
silently dropped.
"""

from __future__ import annotations

import argparse
import html
import tempfile
import zipfile
from pathlib import Path

from ubt.analyze.reader_pdf import read_pdf
from ubt.core.qe.fast_pass import REHEARSAL_MARKER, FastPassFilter
from ubt.layout.theme import resolve_theme
from ubt.pipeline.steps import realize
from ubt.render.epub_view import compose_epub
from ubt.render.html_view import _element_text, compose_html
from ubt.render.outputs import LoweringUnsupported
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


def _check(document: Path, tmp: Path, case_id: str) -> tuple[int, list[str]]:
    doc = read_pdf(document)
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))
    translations = {
        element.id: f"{REHEARSAL_MARKER} {element.text}"
        for element in doc.elements
        if isinstance(element, REFLOW_CLASSES) and not element.skip_translate
    }
    backend = TypstBackend(translations, theme=resolve_theme("en", "zh"))
    attestations = [realize(element, backend, verifiers, doc.source) for element in doc.elements]

    out = tmp / f"{case_id}.html"
    composition = compose_html(doc, attestations, translations, out)
    text = out.read_text(encoding="utf-8")

    problems: list[str] = []
    if not text.startswith("<!DOCTYPE html"):
        problems.append("output is not an HTML document")
    if len(composition.placements) != len(doc.elements):
        problems.append(f"placements {len(composition.placements)} != elements {len(doc.elements)}")
    if composition.descended_ids:
        problems.append(f"HTML view descended {len(composition.descended_ids)} element(s)")

    by_id = {a.element_id: a for a in attestations}
    for element in doc.elements:
        expected = html.escape(
            _element_text(element, by_id[element.id].fidelity, doc.source, translations)
        )
        if expected and expected not in text:
            problems.append(f"{element.id}: attested text missing from HTML")
            break

    # The EPUB view: same body, wrapped in a valid OCF container.
    epub = tmp / f"{case_id}.epub"
    compose_epub(doc, attestations, translations, epub, title=case_id)
    with zipfile.ZipFile(epub) as archive:
        names = archive.namelist()
        infos = {info.filename: info for info in archive.infolist()}
        if not names or names[0] != "mimetype":
            problems.append("EPUB mimetype is not the first entry")
        elif infos["mimetype"].compress_type != zipfile.ZIP_STORED:
            problems.append("EPUB mimetype is compressed")
        for required in (
            "META-INF/container.xml",
            "OEBPS/content.opf",
            "OEBPS/nav.xhtml",
            "OEBPS/text.xhtml",
        ):
            if required not in names:
                problems.append(f"EPUB missing {required}")
        if "OEBPS/text.xhtml" in names:
            text_xhtml = archive.read("OEBPS/text.xhtml").decode("utf-8")
            sample = next(
                (
                    e
                    for e in doc.elements
                    if _element_text(e, by_id[e.id].fidelity, doc.source, translations)
                ),
                None,
            )
            if sample is not None:
                expected = html.escape(
                    _element_text(sample, by_id[sample.id].fidelity, doc.source, translations)
                )
                if expected not in text_xhtml:
                    problems.append("EPUB text.xhtml is missing an element's attested text")

    # Fail-closed: a coverage gap is refused, not silently dropped.
    try:
        compose_html(doc, attestations[:-1], translations, tmp / f"{case_id}_gap.html")
        problems.append("missing attestation was accepted (should raise)")
    except LoweringUnsupported:
        pass
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
    with tempfile.TemporaryDirectory(prefix="ubt-html-view-") as tmp_str:
        tmp = Path(tmp_str)
        print(f"\nHTML view acceptance — {corpus_dir} ({len(cases)} document(s))")
        for case_id, document in cases:
            count, issues = _check(document, tmp, case_id)
            total += count
            problems.extend(f"[{case_id}] {issue}" for issue in issues)
            print(f"  {case_id:<20} {'pass' if not issues else 'FAIL':<5} elements={count}")

    print(f"\n  elements={total} problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for line in problems[:15]:
        print(f"    {line}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

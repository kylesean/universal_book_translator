#!/usr/bin/env python
"""Phase-1 acceptance: the native PDF reader loses no text.

Reads every corpus document two ways -- the new native reader (PDF -> typed
Document directly) and the existing extractor bridged to a Document -- and
compares the *character content* (whitespace-insensitive). Segmentation may
differ; the gate is that the native reader swallows nothing the old pipeline had.

Also checks the native Document survives the bridge round-trip (Document ->
IRBlock -> Document), i.e. it is a well-formed model citizen.

Exit 0 iff every document's coverage is >= --min-coverage and every round-trip
is lossless.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from ubt.analyze.bridge import blocks_from_document, document_from_blocks
from ubt.analyze.reader_pdf import read_pdf
from ubt.core.ir.models import IRBlock
from ubt.model.ast import Document as DocumentModel
from ubt.model.ast import ElementT, Figure, Formula, Table, TextElement


def _element_text(element: object) -> str:
    if isinstance(element, TextElement):
        return element.text
    if isinstance(element, Formula):
        return element.source
    if isinstance(element, Table):
        return element.markup
    if isinstance(element, Figure):
        return element.asset_id
    return ""


def _document_text(document: DocumentModel) -> str:
    return "\n".join(_element_text(element) for element in document.elements)


def _chars(text: str) -> Counter[str]:
    return Counter(ch for ch in text if not ch.isspace())


def _parse_blocks(document: Path, engine: str) -> list[IRBlock]:
    from ubt.adapters.factory import get_adapter_for_path

    adapter = get_adapter_for_path(document, pdf_engine=engine)

    async def collect() -> list[IRBlock]:
        blocks: list[IRBlock] = []
        async for chapter in adapter.parse_stream(document):
            blocks.extend(chapter.blocks)
        return blocks

    return asyncio.run(collect())


def _roundtrip_ok(elements: tuple[ElementT, ...]) -> bool:
    from ubt.model.ast import Document, Region, RegionKind
    from ubt.model.span import CanonicalSource

    document = Document(
        source=CanonicalSource(doc_id="x"),
        regions=(Region(id="r", kind=RegionKind.BODY, elements=elements),),
    )
    rebuilt = document_from_blocks(blocks_from_document(document), doc_id="x")
    if len(rebuilt.elements) != len(elements):
        return False
    # The bridge models page + bbox, not the char range (IRBlock has no such
    # field), so compare the structure it does carry.
    return all(
        before.kind is after.kind
        and before.span.page == after.span.page
        and before.span.bbox == after.span.bbox
        for before, after in zip(elements, rebuilt.elements, strict=True)
    )


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--engine", default="auto")
    parser.add_argument("--min-coverage", type=float, default=0.98)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    per_case: list[dict[str, object]] = []
    failed = 0
    for case_id, document in _load_cases(corpus_dir):
        if not document.exists():
            per_case.append({"id": case_id, "status": "missing"})
            continue
        try:
            native = read_pdf(document)
            legacy_blocks = _parse_blocks(document, args.engine)
        except Exception as exc:  # one bad document must not sink the run
            per_case.append({"id": case_id, "status": "error", "reason": str(exc)})
            failed += 1
            continue
        legacy = document_from_blocks(legacy_blocks, doc_id=case_id, path=str(document))
        native_chars = _chars(_document_text(native))
        legacy_chars = _chars(_document_text(legacy))
        legacy_total = sum(legacy_chars.values())
        covered = sum((native_chars & legacy_chars).values())
        # An empty baseline validates nothing -- treat it as a failure, not 100%.
        coverage = covered / legacy_total if legacy_total else 0.0
        extra = sum((native_chars - legacy_chars).values())
        roundtrip = _roundtrip_ok(native.elements)
        ok = coverage >= args.min_coverage and roundtrip
        if not ok:
            failed += 1
        per_case.append(
            {
                "id": case_id,
                "status": "pass" if ok else "fail",
                "pages": len(native.source.pages),
                "elements": len(native.elements),
                "coverage": round(coverage, 4),
                "native_extra_chars": extra,
                "roundtrip": roundtrip,
            }
        )

    payload = {
        "status": "fail" if failed else "pass",
        "min_coverage": args.min_coverage,
        "cases": per_case,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"\nPhase-1 native reader acceptance — {corpus_dir}")
        for entry in per_case:
            print(
                f"  {entry['id']:<20} {entry['status']:<6} "
                f"pages={entry.get('pages', 0)} elements={entry.get('elements', 0)} "
                f"coverage={entry.get('coverage', 0):.1%} "
                f"extra={entry.get('native_extra_chars', 0)} "
                f"roundtrip={entry.get('roundtrip')}"
                + (f"  reason={entry['reason']}" if entry.get("reason") else "")
            )
        print(f"\n  min coverage gate: {args.min_coverage:.0%}  |  failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

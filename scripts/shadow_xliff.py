#!/usr/bin/env python
"""Phase-2 acceptance: the XLIFF view is a lossless bijection over segments.

Builds a ``Segment`` per text block of every corpus document (real source text,
real protected spans via PlaceholderEngine), serializes them to XLIFF 2.1,
parses the document back, and asserts every modeled field survives:
id, source, target, state, and the (token, kind, original) of every placeholder.
Languages and the ``original`` file name must survive too.

This is the ADR's Phase-2 interop gate: the kernel's units can leave and return
through the industry format without loss. Exit 0 iff every segment round-trips.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.math_masker import MathMasker
from ubt.core.cleaners.soup_math import SoupMathMasker
from ubt.model.segment import Segment, SegmentState
from ubt.segment.placeholders import PlaceholderEngine
from ubt.segment.xliff import from_xliff, to_xliff, xml_safe

_STATES = tuple(SegmentState)


def _engine() -> PlaceholderEngine:
    return PlaceholderEngine(
        code=CodeMasker(),
        math=MathMasker(),
        soup=SoupMathMasker(),
        citation=CitationMasker(),
    )


def _fingerprint(segment: Segment) -> dict[str, object]:
    return {
        "source": segment.source,
        "target": segment.target,
        "state": segment.state.value,
        "placeholders": sorted(
            (placeholder.token, placeholder.kind, placeholder.original)
            for placeholder in segment.placeholders
        ),
    }


def _parse_blocks(document: Path, engine: str) -> list[str]:
    from ubt.adapters.factory import get_adapter_for_path

    adapter = get_adapter_for_path(document, pdf_engine=engine)

    async def collect() -> list[str]:
        texts: list[str] = []
        async for chapter in adapter.parse_stream(document):
            texts.extend(block.source_text for block in chapter.blocks if block.source_text)
        return texts

    return asyncio.run(collect())


def _segments(texts: list[str], engine: PlaceholderEngine) -> list[Segment]:
    segments: list[Segment] = []
    for index, text in enumerate(texts):
        masked = engine.mask(xml_safe(text))
        segments.append(
            Segment(
                id=f"seg_{index:05d}",
                source=masked.text,
                placeholders=masked.placeholders,
                target=masked.text if index % 2 == 0 else None,
                state=_STATES[index % len(_STATES)],
            )
        )
    return segments


def _extra_texts(globs: list[str], root: Path) -> list[str]:
    texts: list[str] = []
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                try:
                    texts.append(path.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue
    return texts


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


def _roundtrip(
    label: str, texts: list[str], engine: PlaceholderEngine
) -> tuple[int, int, list[str]]:
    """Returns (ok_fields, total_fields, problem_details)."""
    segments = _segments(texts, engine)
    xml = to_xliff(segments, src_lang="en", trg_lang="zh", original=label)
    parsed = from_xliff(xml)
    details: list[str] = []
    ok = 0
    total = 1  # the header (languages + original)
    if (parsed.src_lang, parsed.trg_lang, parsed.original) == ("en", "zh", label):
        ok += 1
    else:
        details.append(
            f"{label}: header mismatch {parsed.src_lang}/{parsed.trg_lang}/{parsed.original}"
        )
    by_id = {segment.id: segment for segment in parsed.segments}
    for segment in segments:
        total += 1
        after = by_id.get(segment.id)
        if after is None:
            details.append(f"{segment.id}: dropped")
        elif _fingerprint(segment) != _fingerprint(after):
            details.append(f"{segment.id}: field mismatch")
        else:
            ok += 1
    surplus = abs(len(parsed.segments) - len(segments))
    if surplus:
        total += surplus
        details.append(f"{label}: segment count {len(parsed.segments)} != {len(segments)}")
    return ok, total, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--engine", default="auto")
    parser.add_argument("--extra", action="append", default=[], help="Extra repo glob(s) of text")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    engine = _engine()
    corpus_dir = Path(args.corpus)
    per_case: list[dict[str, object]] = []
    total = 0
    total_ok = 0
    all_details: list[str] = []

    groups: list[tuple[str, list[str]]] = []
    for case_id, document in _load_cases(corpus_dir):
        if not document.exists():
            per_case.append({"id": case_id, "status": "missing"})
            continue
        try:
            groups.append((case_id, _parse_blocks(document, args.engine)))
        except Exception as exc:  # one bad document must not sink the run
            per_case.append({"id": case_id, "status": "error", "reason": str(exc)})
    extras = _extra_texts(args.extra, Path.cwd())
    if extras:
        groups.append(("repo-extra", extras))

    for label, texts in groups:
        ok, tot, details = _roundtrip(label, texts, engine)
        total += tot
        total_ok += ok
        all_details.extend(details)
        per_case.append(
            {"id": label, "status": "pass" if not details else "fail", "segments": tot - 1}
        )

    passed = total > 0 and total_ok == total
    payload = {
        "status": "pass" if passed else "fail",
        "segments": total,
        "round_tripped": total_ok,
        "cases": per_case,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"\nPhase-2 XLIFF round-trip acceptance — {corpus_dir}")
        for entry in per_case:
            print(
                f"  {entry['id']:<20} {entry['status']:<6} segments={entry.get('segments', 0)}"
                + (f"  reason={entry['reason']}" if entry.get("reason") else "")
            )
        print(f"\n  TOTAL: {total_ok}/{total} segment-fields round-tripped losslessly")
        for line in all_details[:20]:
            print(f"    {line}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

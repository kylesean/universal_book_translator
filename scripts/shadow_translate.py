#!/usr/bin/env python
"""Phase-2 acceptance: the translation engine is lossless when honest and
fail-closed when not.

Reads real corpus documents with the native reader and runs the engine twice:

- **echo** (a faithful provider that returns the masked source unchanged): every
  segment must come back TRANSLATED with an empty flag set and a target equal to
  the original element text -- masking + restore is lossless.
- **vandal** (drops every protected span): every segment that HAD a placeholder
  must come back BLOCKED with a flag; segments with nothing to protect stay
  TRANSLATED. No corrupted span may ever be marked translated.

Parse-only, no LLM: fast. Exit 0 iff both passes hold.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from ubt.analyze.reader_pdf import read_pdf
from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.math_masker import MathMasker
from ubt.core.cleaners.soup_math import SoupMathMasker
from ubt.model.ast import TextElement
from ubt.model.segment import SegmentState
from ubt.segment.placeholders import PlaceholderEngine
from ubt.segment.xliff import xml_safe
from ubt.translate.engine import TranslationEngine

_TOKEN_RE = re.compile(r"\u27e6[^\u27e7]*\u27e7")


def _engine() -> TranslationEngine:
    placeholders = PlaceholderEngine(
        code=CodeMasker(),
        math=MathMasker(),
        soup=SoupMathMasker(),
        citation=CitationMasker(),
    )
    return TranslationEngine(placeholders=placeholders, model="echo-test")


async def _echo(text: str) -> str:
    return text


async def _vandal(text: str) -> str:
    return _TOKEN_RE.sub("", text)


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


def _run(document: Path) -> tuple[int, int, list[str]]:
    """Returns (segments_checked, segments_with_placeholders, problems)."""
    doc = read_pdf(document)
    engine = _engine()
    originals = {el.id: el.text for el in doc.elements if isinstance(el, TextElement)}
    problems: list[str] = []

    echoed = asyncio.run(engine.translate_document(doc, _echo))
    for segment in echoed:
        if segment.state is not SegmentState.TRANSLATED:
            problems.append(f"echo {segment.id}: not translated ({segment.qa.flags})")
        elif segment.target != xml_safe(originals.get(segment.id, "")):
            problems.append(f"echo {segment.id}: target != source")

    vandalised = asyncio.run(engine.translate_document(doc, _vandal))
    for segment in vandalised:
        had_placeholder = bool(segment.placeholders)
        if had_placeholder and segment.state is not SegmentState.BLOCKED:
            problems.append(f"vandal {segment.id}: corrupt span NOT blocked")
        elif had_placeholder and not segment.qa.flags:
            problems.append(f"vandal {segment.id}: blocked without a flag")
        elif not had_placeholder and segment.state is not SegmentState.TRANSLATED:
            problems.append(f"vandal {segment.id}: no-placeholder segment blocked")

    placeholders = sum(1 for segment in echoed if segment.placeholders)
    return len(echoed), placeholders, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--all", action="store_true", help="All cases (default: the first)")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    cases = [c for c in _load_cases(corpus_dir) if c[1].exists()]
    if not args.all:
        cases = cases[:1]

    total_segments = 0
    total_placeholders = 0
    problems: list[str] = []
    print(f"\nPhase-2 translation-engine acceptance — {corpus_dir} ({len(cases)} document(s))")
    for case_id, document in cases:
        segments, placeholders, issues = _run(document)
        total_segments += segments
        total_placeholders += placeholders
        problems.extend(f"[{case_id}] {p}" for p in issues)
        status = "pass" if not issues else "FAIL"
        print(
            f"  {case_id:<20} {status:<6} segments={segments} "
            f"placeholders={placeholders} issues={len(issues)}"
        )

    passed = total_segments > 0 and not problems
    print(
        f"\n  segments={total_segments} placeholders={total_placeholders} "
        f"problems={len(problems)} -> {'PASS' if passed else 'FAIL'}"
    )
    for line in problems[:15]:
        print(f"    {line}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

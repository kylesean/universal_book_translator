#!/usr/bin/env python
"""Phase-1 acceptance: the native Markdown reader (ADR-0001 §12 Q5).

Checks, over every Markdown/text file in the repo and corpus:

- every element's ``Span.chars`` slices exactly its own text out of
  ``CanonicalSource.text`` (the reader's core promise: a verifiable span);
- the ``Document`` round-trips through the bridge losslessly (element count and
  text preserved), so the typed model can replace the ``IRBlock`` detour;
- the reader loses no text relative to the existing Markdown adapter (compared
  as a whitespace-normalized token multiset, since the reader splits headings
  and list items the adapter keeps inside paragraphs).

Exit 0 iff every file passes all three.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path

from ubt.analyze.bridge import blocks_from_document, document_from_blocks
from ubt.analyze.normalize import normalize_text
from ubt.analyze.reader_md import read_md
from ubt.model.ast import Document, Formula, Table, TextElement


def _element_text(element: object) -> str:
    if isinstance(element, TextElement):
        return element.text
    if isinstance(element, Formula):
        return element.source
    if isinstance(element, Table):
        return element.markup
    return ""


def _tokens(text: str) -> Counter[str]:
    return Counter(text.split())


def _reader_tokens(document: Document) -> Counter[str]:
    """The reader's tokens from canonical source text."""
    return _tokens(document.source.text)


def _adapter_text(path: Path) -> str:
    from ubt.adapters.markdown.adapter import MarkdownAdapter

    adapter = MarkdownAdapter()

    async def collect() -> str:
        parts: list[str] = []
        async for chapter in adapter.parse_stream(path):
            parts.extend(block.source_text or "" for block in chapter.blocks)
        return "\n".join(parts)

    return asyncio.run(collect())


def _check(path: Path) -> list[str]:
    problems: list[str] = []
    document = read_md(path)

    # 1. Span exactness.
    for element in document.elements:
        chars = element.span.chars
        if chars is None:
            problems.append(f"{element.id}: missing Span.chars")
            continue
        expected = _element_text(element)
        got = document.source.text[chars[0] : chars[1]]
        if got != expected:
            problems.append(f"{element.id}: span slice {got[:30]!r} != {expected[:30]!r}")

    # 2. Bridge round-trip.
    round_tripped = document_from_blocks(
        blocks_from_document(document), doc_id=document.source.doc_id
    )
    if len(round_tripped.elements) != len(document.elements):
        problems.append(
            f"round-trip element count {len(round_tripped.elements)} != {len(document.elements)}"
        )
    if Counter(_tokens(document.source.text)) != Counter(
        _tokens("\n".join(_element_text(e) for e in round_tripped.elements))
    ):
        problems.append("round-trip changed the canonical text")

    # 3. No text lost vs the adapter (normalized: the reader canonicalizes, the
    #    adapter keeps raw source bytes, so both sides must be canonicalized).
    if _reader_tokens(document) != Counter(_tokens(normalize_text(_adapter_text(path)))):
        problems.append("reader text differs from the adapter's (possible loss)")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        action="append",
        default=["docs/**/*.md", "corpus/**/*.md", "*.md"],
        help="Glob(s) of Markdown files to check",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.glob:
        paths.extend(sorted(Path.cwd().glob(pattern)))
    paths = sorted({p for p in paths if p.is_file()})

    print(f"\nPhase-1 native Markdown reader — {len(paths)} file(s)")
    total_elements = 0
    failed = 0
    for path in paths:
        problems = _check(path)
        document = read_md(path)
        total_elements += len(document.elements)
        status = "pass" if not problems else "FAIL"
        if problems:
            failed += 1
        print(f"  {status:<5} elements={len(document.elements):<5} {path}")
        for problem in problems[:5]:
            print(f"        {problem}")

    print(
        f"\n  files={len(paths)} elements={total_elements} failed={failed} -> {'PASS' if not failed else 'FAIL'}"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

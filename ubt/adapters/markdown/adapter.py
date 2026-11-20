"""Markdown and flat text document adapter.

The parse is the native Markdown reader's (``ubt.analyze.reader_md``); this
adapter only partitions its elements into chapters (at level-1 headings) for
the read-only audit surfaces. Translation and delivery are owned by
:class:`ubt.codecs.markdown.MarkdownCodec`. Classification lives in exactly one
place, so a line cannot be a heading to the reader and prose to the adapter.
"""

import asyncio
import dataclasses
from collections.abc import AsyncIterator
from pathlib import Path

from ubt.adapters.base import BaseDocumentAdapter
from ubt.analyze.assemble import element_text
from ubt.analyze.reader_md import read_md
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BookManifest,
    ChapterIR,
    ChapterMeta,
    IRBlock,
)
from ubt.core.ir.serializer import compute_file_sha256_cached
from ubt.model.ast import ElementT, Heading


def _split_chapters(
    elements: tuple[ElementT, ...],
    *,
    plain_text: bool,
    stem: str,
    source_file: str,
) -> tuple[str, list[ChapterMeta], list[list[ElementT]]]:
    """Partition the reader's elements into chapters at level-1 headings.

    A level-1 heading starts a chapter and is its first element; content before
    the first one becomes a "Preface" chapter. With no level-1 heading (or in
    plain-text mode) the whole document is one chapter titled after the file.
    """
    if plain_text:
        return (
            stem,
            [ChapterMeta(chapter_id="ch_001", title=stem, spine_index=1, source_file=source_file)],
            [list(elements)],
        )
    h1 = [i for i, el in enumerate(elements) if isinstance(el, Heading) and el.level == 1]
    if not h1:
        return (
            stem,
            [ChapterMeta(chapter_id="ch_001", title=stem, spine_index=1, source_file=source_file)],
            [list(elements)],
        )
    has_pre = h1[0] > 0
    chapters: list[ChapterMeta] = []
    groups: list[list[ElementT]] = []
    if has_pre:
        chapters.append(
            ChapterMeta(
                chapter_id="ch_000", title="Preface", spine_index=0, source_file=source_file
            )
        )
        groups.append(list(elements[: h1[0]]))
    title = stem if has_pre else element_text(elements[h1[0]])
    for n, start in enumerate(h1):
        end = h1[n + 1] if n + 1 < len(h1) else len(elements)
        chapters.append(
            ChapterMeta(
                chapter_id=f"ch_{n + 1:03d}",
                title=element_text(elements[start]),
                spine_index=n + 1,
                source_file=source_file,
            )
        )
        groups.append(list(elements[start:end]))
    return title, chapters, groups


class MarkdownInspectionAdapter(BaseDocumentAdapter):
    """Adapter for Markdown (.md) and flat text (.txt) documents."""

    output_suffixes = frozenset({".md", ".markdown", ".txt"})

    def __init__(self, plain_text: bool = False) -> None:
        # ``.txt`` is routed here too: logs and configs use ``#`` for
        # comments and ``` for nothing, so interpreting them as Markdown
        # shreds a line starting with ``# `` into a fake chapter. In plain-text
        # mode the reader treats the file as blank-line-separated paragraphs
        # and never reads Markdown syntax.
        self._plain_text = plain_text

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Extract the top-level manifest; the reader decides the structure."""
        if not input_path.exists():
            raise DocumentParseError(f"Input file not found: {input_path}")
        doc_id = await asyncio.to_thread(compute_file_sha256_cached, input_path)
        document = await asyncio.to_thread(
            read_md, input_path, doc_id=doc_id, plain_text=self._plain_text
        )
        title, chapters, _ = _split_chapters(
            document.elements,
            plain_text=self._plain_text,
            stem=input_path.stem,
            source_file=input_path.name,
        )
        return BookManifest(
            doc_id=doc_id,
            title=title,
            source_path=str(input_path),
            chapters=chapters,
        )

    async def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]:
        """Stream the reader's elements, partitioned into chapters."""
        manifest = await self.extract_manifest(input_path)
        document = await asyncio.to_thread(
            read_md, input_path, doc_id=manifest.doc_id, plain_text=self._plain_text
        )
        _, _, groups = _split_chapters(
            document.elements,
            plain_text=self._plain_text,
            stem=input_path.stem,
            source_file=input_path.name,
        )
        global_spine = 1
        for chapter, elements in zip(manifest.chapters, groups, strict=True):
            blocks: list[IRBlock] = []
            for block_idx, element in enumerate(elements, start=1):
                blocks.append(
                    IRBlock(
                        element=dataclasses.replace(
                            element,
                            id=f"{chapter.chapter_id}#b{block_idx:04d}",
                            spine_index=global_spine,
                        )
                    )
                )
                global_spine += 1
            if blocks:
                yield ChapterIR(
                    doc_id=manifest.doc_id,
                    chapter_id=chapter.chapter_id,
                    title=chapter.title,
                    spine_index=chapter.spine_index,
                    blocks=blocks,
                )

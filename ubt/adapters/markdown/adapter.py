"""Markdown and flat text document adapter.

The parse is the native Markdown reader's (``ubt.analyze.reader_md``); this
adapter only partitions its elements into chapters (at level-1 headings) and
renders the bilingual/monolingual output. Classification lives in exactly one
place, so a line cannot be a heading to the reader and prose to the adapter.
"""

import asyncio
import dataclasses
import os
import re
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ubt.adapters.base import BaseDocumentAdapter
from ubt.adapters.unresolved import UNRESOLVED_STATUSES, failure_note_markdown
from ubt.analyze.assemble import element_text
from ubt.analyze.reader_md import read_md
from ubt.core.cleaners.html_sanitizer import sanitize_inline_html, strip_html_mark_tags
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    IRBlock,
)
from ubt.core.ir.serializer import compute_file_sha256_cached
from ubt.model.ast import ElementT, Heading

# Blocks that left the pipeline without an approved translation must stay
# visible in the rendered document (the <mark> wrapper is what flags them).
# The status set and notes are shared with every other adapter via
# ``ubt.adapters.unresolved`` so the formats cannot drift apart.
_UNRESOLVED_STATUSES = UNRESOLVED_STATUSES


def _with_heading_marker(block: IRBlock, target: str) -> str:
    """Put the heading's ``#`` run back in target-only output.

    The marker lives in the source line, so the monolingual branch handed back
    bare text for every heading: a chapter book exported as Markdown with zero
    structure — no outline, no TOC, every section a paragraph. Other adapters
    keep the paragraph *type* (a ``<h2>`` stays an ``<h2>``), so this only
    restores what the format itself stores inline.
    """
    if block.block_type is not BlockType.HEADING or target.lstrip().startswith("#"):
        return target
    level = getattr(block.element, "level", None)
    if isinstance(level, int) and 1 <= level <= 6:
        return f"{'#' * level} {target}"
    marker = re.match(r"(#{1,6})\s+", block.source_text or "")
    return f"{marker.group(1)} {target}" if marker else f"# {target}"


def _with_list_marker(block: IRBlock, target: str) -> str:
    """Prepend list marker (e.g. '-' or '1.') to list item if missing."""
    if block.block_type is not BlockType.LIST_ITEM:
        return target
    marker = getattr(block.element, "marker", "")
    if not marker:
        src_match = re.match(r"^(\s*[-*+]\s+|\s*\d+\.\s+)", block.source_text or "")
        if src_match:
            marker = src_match.group(1).rstrip()
    if marker:
        clean_marker = marker.rstrip()
        stripped_target = target.lstrip()
        if not re.match(r"^([-*+]\s+|\d+\.\s+)", stripped_target):
            return f"{clean_marker} {stripped_target}"
    return target


def _format_markdown_block(block: IRBlock, target: str) -> str:
    """Format heading or list item markers onto translated markdown block."""
    if block.block_type is BlockType.HEADING:
        return _with_heading_marker(block, target)
    if block.block_type is BlockType.LIST_ITEM:
        return _with_list_marker(block, target)
    return target


def _format_failure_note(status: BlockStatus) -> str:
    return failure_note_markdown(status)


def _format_blockquoted(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines())


def _sanitize_markdown_content(text: str) -> str:
    """Sanitize inline HTML tags without escaping Markdown or ``$…$`` LaTeX.

    Deliberately NOT :func:`sanitize_html_fragment`: that allowlist escapes every
    ``<``/``&`` in the text, mangling Markdown and LaTeX (``List<T>`` becomes
    ``List&lt;T&gt;``). :func:`sanitize_inline_html` applies the same allowlist to
    the *tags* only: a surviving tag keeps only its allowlisted attributes (with
    URL-scheme gating) while the surrounding text is byte-for-byte unchanged.
    """
    return sanitize_inline_html(strip_html_mark_tags(text))


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


class MarkdownAdapter(BaseDocumentAdapter):
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

    async def render_blocks(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Render bilingual Markdown document interleaving source and translated paragraphs.

        The render is synchronous string building + a file write; run it off the
        event loop so a concurrent job/task is not stalled.
        """
        return await asyncio.to_thread(
            self._render_blocks_sync,
            manifest,
            blocks,
            target_lang,
            output_path,
            bilingual_mode,
            **kwargs,
        )

    def _render_blocks_sync(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Render bilingual Markdown document interleaving source and translated paragraphs."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rendered_sections: list[str] = []
        is_monolingual = bilingual_mode in ("target", "monolingual")

        for b in blocks:
            # If block skipped translation (e.g. code/formula), preserve directly
            if b.skip_translate:
                rendered_sections.append(b.source_text)
                continue
            unresolved = b.status in _UNRESOLVED_STATUSES
            if unresolved and not b.target_text:
                # No draft at all (crash before draft, or blocked): keep the
                # source and say why instead of silently shipping raw source.
                rendered_sections.append(f"{b.source_text}\n\n{_format_failure_note(b.status)}")
                continue
            if not b.target_text:
                rendered_sections.append(b.source_text)
                continue
            clean_target = _sanitize_markdown_content(b.target_text)
            if unresolved:
                note = _format_failure_note(b.status)
                draft_block = _format_blockquoted(clean_target)
                if is_monolingual:
                    rendered_sections.append(f"{note}\n\n{draft_block}")
                else:
                    rendered_sections.append(f"{b.source_text}\n\n{note}\n\n{draft_block}")
            elif is_monolingual:
                rendered_sections.append(_format_markdown_block(b, clean_target))
            else:
                # Interleave source and target bilingual paragraphs (default: alternating)
                # (LLM output is untrusted — strip dangerous HTML and error marks).
                target_formatted = _format_markdown_block(b, clean_target)
                rendered_sections.append(f"{b.source_text}\n\n{target_formatted}")

        full_content = "\n\n".join(rendered_sections) + "\n"

        # Atomic POSIX write
        temp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                dir=output_path.parent,
                delete=False,
                encoding="utf-8",
            ) as tf:
                temp_name = tf.name
                tf.write(full_content)
                tf.flush()
                os.fsync(tf.fileno())
            Path(temp_name).replace(output_path)
        except Exception:
            # A failed write/fsync/replace must not leave an orphan temp file in
            # the output directory.
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)
            raise
        return output_path

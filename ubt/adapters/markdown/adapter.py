"""Markdown and flat text document adapter."""

import asyncio
import os
import re
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ubt.adapters.base import BaseDocumentAdapter, decode_markup
from ubt.adapters.unresolved import UNRESOLVED_STATUSES, failure_note_markdown
from ubt.analyze.structure import is_display_math, is_markdown_table, markdown_heading
from ubt.core.cleaners.html_sanitizer import sanitize_inline_html, strip_html_mark_tags
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    FlowID,
    IRBlock,
    make_element,
)
from ubt.core.ir.serializer import compute_file_sha256_cached

# Project Gutenberg structural markers: transcribing these verbatim is the
# only correct rendering, and LLM translation of them is wasted spend (they
# also trip script-density repair loops). Skip them at parse time, 0-token.
_GUTENBERG_MARKER = re.compile(r"^\[(Illustration|Footnote)\b", re.IGNORECASE)

# Blocks that left the pipeline without an approved translation must stay
# visible in the rendered document (the <mark> wrapper is what flags them).
# The status set and notes are shared with every other adapter via
# ``ubt.adapters.unresolved`` so the formats cannot drift apart.
_UNRESOLVED_STATUSES = UNRESOLVED_STATUSES


def _read_text(path: Path) -> str:
    """Read a Markdown/text file, sniffing BOM / declared encoding before UTF-8.

    A hard ``utf-8`` decode with ``errors="replace"`` turned every GBK/Shift-JIS
    byte into ``\\ufffd`` before anything could look at the source. Going through
    the shared :func:`decode_markup` keeps a BOM (UTF-8/UTF-16) and any declared
    encoding authoritative, falling back to a replacement decode only last.
    """
    return decode_markup(path.read_bytes())


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
    marker = re.match(r"(#{1,6})\s+", block.source_text or "")
    return f"{marker.group(1)} {target}" if marker else target


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


class MarkdownAdapter(BaseDocumentAdapter):
    """Adapter for Markdown (.md) and flat text (.txt) documents."""

    output_suffixes = frozenset({".md", ".markdown", ".txt"})

    def __init__(self, plain_text: bool = False) -> None:
        # ``.txt`` is routed here too: logs and configs use ``#`` for
        # comments and ``` for nothing, so interpreting them as Markdown
        # shreds a line starting with ``# `` into a fake chapter. In plain-text
        # mode the adapter treats the file as blank-line-separated paragraphs
        # and never reads Markdown syntax.
        self._plain_text = plain_text

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Extract top-level document manifest by scanning headers."""
        if not input_path.exists():
            raise DocumentParseError(f"Input file not found: {input_path}")

        # Off-loop: sha computation and file read run in thread pool to prevent blocking the event loop on large sources.
        doc_id = await asyncio.to_thread(compute_file_sha256_cached, input_path)
        content = await asyncio.to_thread(_read_text, input_path)
        lines = content.split("\n")

        # Scan for markdown headers (# Heading) as chapter boundaries
        chapters: list[ChapterMeta] = []
        chapter_idx = 1
        book_title = input_path.stem

        has_pre_heading_content = False
        in_code_block = False
        in_math_block = False
        scan_lines = lines if not self._plain_text else []
        for line in scan_lines:
            line_s = line.strip()
            if line_s.startswith("```"):
                in_code_block = not in_code_block
                if not chapters and line_s:
                    has_pre_heading_content = True
                continue

            if in_code_block:
                if not chapters:
                    has_pre_heading_content = True
                continue

            # Mirror parse_stream's $$ tracking: a "# " line inside a fenced
            # math block is math content (a LaTeX-style comment), not a chapter
            # heading; without this the two passes disagree and every later
            # chapter id shifts by one.
            if line_s.startswith("$$"):
                if in_math_block:
                    in_math_block = False
                elif not (line_s.endswith("$$") and len(line_s) > 2):
                    in_math_block = True
                continue

            if in_math_block:
                if not chapters:
                    has_pre_heading_content = True
                continue

            if line_s.startswith("# "):
                title = line_s[2:].strip()
                if not chapters and not has_pre_heading_content:
                    book_title = title  # First h1 as book title
                chapters.append(
                    ChapterMeta(
                        chapter_id=f"ch_{chapter_idx:03d}",
                        title=title,
                        spine_index=chapter_idx,
                        source_file=input_path.name,
                    )
                )
                chapter_idx += 1
            elif not chapters and line_s:
                has_pre_heading_content = True

        if has_pre_heading_content and chapters:
            chapters.insert(
                0,
                ChapterMeta(
                    chapter_id="ch_000",
                    title="Preface",
                    spine_index=0,
                    source_file=input_path.name,
                ),
            )

        if not chapters:
            chapters.append(
                ChapterMeta(
                    chapter_id="ch_001",
                    title=book_title,
                    spine_index=1,
                    source_file=input_path.name,
                )
            )

        return BookManifest(
            doc_id=doc_id,
            title=book_title,
            source_path=str(input_path),
            chapters=chapters,
        )

    async def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]:
        """Stream Markdown document partitioned by chapters."""
        manifest = await self.extract_manifest(input_path)
        content = await asyncio.to_thread(_read_text, input_path)
        # The only blocking IO (the file read) is offloaded above. The remaining
        # line loop is pure-Python with a small per-line cost, and it stays on
        # the loop deliberately: moving it into a worker thread would have to
        # buffer every chapter, breaking this generator's incremental
        # (bounded-memory) contract for very large books.
        lines = content.split("\n")

        current_chapter_idx = 0
        current_chapter = manifest.chapters[current_chapter_idx]
        current_blocks: list[IRBlock] = []
        global_spine = 1
        block_idx = 1

        # Paragraph accumulator
        in_code_block = False
        in_math_block = False
        code_lines: list[str] = []
        math_lines: list[str] = []
        para_lines: list[str] = []

        def flush_paragraph() -> None:
            nonlocal block_idx, global_spine
            if not para_lines:
                return
            text = "\n".join(para_lines).strip()
            para_lines.clear()
            if not text:
                return

            if is_display_math(text):
                block_type = BlockType.FORMULA
                flow_id = FlowID.MAIN_STORY
                skip_translate = True
            elif not self._plain_text and markdown_heading(text) is not None:
                block_type = BlockType.HEADING
                flow_id = FlowID.MAIN_STORY
                skip_translate = False
            elif not self._plain_text and is_markdown_table(text):
                block_type = BlockType.TABLE
                flow_id = FlowID.TABLE_GRID
                skip_translate = False
            else:
                block_type = BlockType.NARRATIVE
                flow_id = FlowID.MAIN_STORY
                skip_translate = bool(_GUTENBERG_MARKER.match(text))

            current_blocks.append(
                IRBlock(
                    element=make_element(
                        id=f"{current_chapter.chapter_id}#b{block_idx:04d}",
                        flow_id=flow_id,
                        spine_index=global_spine,
                        block_type=block_type,
                        source_text=text,
                        skip_translate=skip_translate,
                    )
                )
            )
            block_idx += 1
            global_spine += 1

        def flush_code() -> None:
            nonlocal block_idx, global_spine
            if not code_lines:
                return
            text = "\n".join(code_lines).strip()
            code_lines.clear()
            if not text:
                return

            current_blocks.append(
                IRBlock(
                    element=make_element(
                        id=f"{current_chapter.chapter_id}#b{block_idx:04d}",
                        flow_id=FlowID.MAIN_STORY,
                        spine_index=global_spine,
                        block_type=BlockType.CODE,
                        source_text=text,
                        skip_translate=True,
                    )
                )
            )
            block_idx += 1
            global_spine += 1

        def flush_math() -> None:
            nonlocal block_idx, global_spine
            if not math_lines:
                return
            text = "\n".join(math_lines).strip()
            math_lines.clear()
            if not text:
                return

            current_blocks.append(
                IRBlock(
                    element=make_element(
                        id=f"{current_chapter.chapter_id}#b{block_idx:04d}",
                        flow_id=FlowID.MAIN_STORY,
                        spine_index=global_spine,
                        block_type=BlockType.FORMULA,
                        source_text=text,
                        skip_translate=True,
                    )
                )
            )
            block_idx += 1
            global_spine += 1

        for line in lines:
            line_s = line.strip()

            if self._plain_text:
                # Blank lines separate paragraphs; nothing is Markdown syntax.
                if not line_s:
                    flush_paragraph()
                else:
                    para_lines.append(line)
                continue

            # Handle fenced code block boundary — before the math fence, so a
            # ``$$`` line inside a fence stays code content (mirrors the order
            # extract_manifest scans the same document).
            if line_s.startswith("```"):
                if in_code_block:
                    code_lines.append(line)
                    in_code_block = False
                    flush_code()
                else:
                    flush_paragraph()
                    in_code_block = True
                    code_lines.append(line)
                continue

            if in_code_block:
                code_lines.append(line)
                continue

            # Handle fenced math block boundary ($$)
            if line_s.startswith("$$"):
                if in_math_block:
                    math_lines.append(line)
                    in_math_block = False
                    flush_math()
                elif line_s.endswith("$$") and len(line_s) > 2:
                    flush_paragraph()
                    math_lines.append(line)
                    flush_math()
                else:
                    flush_paragraph()
                    in_math_block = True
                    math_lines.append(line)
                continue

            if in_math_block:
                math_lines.append(line)
                continue

            # Check if this line is an ATX heading
            is_heading = not self._plain_text and markdown_heading(line_s) is not None
            if is_heading:
                flush_paragraph()
                if line_s.startswith("# ") and current_blocks:
                    yield ChapterIR(
                        doc_id=manifest.doc_id,
                        chapter_id=current_chapter.chapter_id,
                        title=current_chapter.title,
                        spine_index=current_chapter.spine_index,
                        blocks=list(current_blocks),
                    )
                    current_blocks.clear()
                    current_chapter_idx += 1
                    if current_chapter_idx < len(manifest.chapters):
                        current_chapter = manifest.chapters[current_chapter_idx]
                    block_idx = 1
                para_lines.append(line)
                flush_paragraph()
                continue

            if not line_s:
                flush_paragraph()
            else:
                para_lines.append(line)

        flush_paragraph()
        if in_code_block:
            flush_code()
        if in_math_block:
            flush_math()

        if current_blocks:
            yield ChapterIR(
                doc_id=manifest.doc_id,
                chapter_id=current_chapter.chapter_id,
                title=current_chapter.title,
                spine_index=current_chapter.spine_index,
                blocks=list(current_blocks),
            )

    async def render_blocks(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        render_engine: str | None = None,
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
            render_engine,
            **kwargs,
        )

    def _render_blocks_sync(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        render_engine: str | None = None,
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
                rendered_sections.append(_with_heading_marker(b, clean_target))
            else:
                # Interleave source and target bilingual paragraphs (default: alternating)
                # (LLM output is untrusted — strip dangerous HTML and error marks).
                target_formatted = (
                    _with_heading_marker(b, clean_target)
                    if b.block_type is BlockType.HEADING
                    else clean_target
                )
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

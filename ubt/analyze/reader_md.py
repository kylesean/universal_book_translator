"""Native Markdown reader: Markdown/text -> typed ``Document`` (ADR-0001 Phase 1, §12 Q5).

The second native reader, and the first for a non-paginated format. Unlike the
bootstrap bridge (adapter -> ``IRBlock`` -> Document), this reader decides the
structure itself from the one shared rule set
(:mod:`ubt.analyze.structure`) and gives every element a real ``Span.chars``
into a :class:`~ubt.model.span.CanonicalSource` ``text``, so the typed document
can be sliced, exported and witnessed without an ``IRBlock`` detour.

Per ADR §8.7 ("add one reader at a time") this is the only non-PDF reader added:
Markdown is the one format whose source *is* a text stream, so a genuine
offset-stable canonical text is cheap and faithful. DOCX/EPUB/HTML wait until
their readers can also supply real spans (page/anchor/char offsets) and the AST
can carry what they uniquely have (table grids, figure assets, provenance).

Deliberately synchronous, like :func:`ubt.analyze.reader_pdf.read_pdf`: the only
IO is one file read, and the line loop is pure.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ubt.analyze._identity import file_digest
from ubt.analyze.assemble import assemble
from ubt.analyze.structure import (
    is_display_math,
    is_gutenberg_marker,
    is_markdown_table,
    markdown_heading,
    markdown_list_item,
)
from ubt.model.ast import (
    CodeBlock,
    Confidence,
    Document,
    Formula,
    Heading,
    ListItem,
    Paragraph,
    Table,
)
from ubt.model.span import Span

if TYPE_CHECKING:
    from ubt.model.ast import ElementT


def _decode(path: Path) -> str:
    """Decode a text source (BOM / declared encoding first), as the adapter does."""
    from ubt.adapters.base import decode_markup

    return decode_markup(path.read_bytes())


def read_md(path: str | Path, *, doc_id: str | None = None, plain_text: bool = False) -> Document:
    """Read a Markdown/plain-text file into a typed :class:`Document`.

    ATX headings keep their real level, list items keep their marker, fenced
    code and ``$$ … $$`` display math are typed and kept verbatim, and GFM
    tables become a :class:`Table`. Every element carries a character range into
    the canonical text (``page`` stays 0: the format has no pages).

    ``plain_text`` reads the file as blank-line-separated paragraphs and never
    interprets Markdown syntax (``.txt`` logs and configs use ``#`` for
    comments; interpreting it as Markdown shreds a line into a fake chapter).
    """
    md_path = Path(path)
    elements: list[ElementT] = []

    def emit(element: ElementT) -> None:
        elements.append(element)

    def _id() -> str:
        return f"md#{len(elements):05d}"

    def _spine() -> int:
        return len(elements)

    def _paragraph_text() -> str:
        return "\n".join(paragraph).strip()

    paragraph: list[str] = []
    code: list[str] = []
    math: list[str] = []
    in_code = False
    in_math = False

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = _paragraph_text()
        paragraph.clear()
        if not text:
            return
        if is_display_math(text):
            emit(
                Formula(
                    id=_id(),
                    spine_index=_spine(),
                    span=Span(),
                    source=text,
                    skip_translate=True,
                    confidence=Confidence.VERIFIED,
                )
            )
        elif is_markdown_table(text):
            emit(
                Table(
                    id=_id(),
                    spine_index=_spine(),
                    span=Span(),
                    markup=text,
                    confidence=Confidence.VERIFIED,
                )
            )
        else:
            emit(
                Paragraph(
                    id=_id(),
                    spine_index=_spine(),
                    span=Span(),
                    text=text,
                    skip_translate=is_gutenberg_marker(text),
                )
            )

    def flush_code() -> None:
        if not code:
            return
        text = "\n".join(code).strip()
        code.clear()
        if not text:
            return
        emit(
            CodeBlock(
                id=_id(),
                spine_index=_spine(),
                span=Span(),
                text=text,
                skip_translate=True,
                confidence=Confidence.VERIFIED,
            )
        )

    def flush_math() -> None:
        if not math:
            return
        text = "\n".join(math).strip()
        math.clear()
        if not text:
            return
        emit(
            Formula(
                id=_id(),
                spine_index=_spine(),
                span=Span(),
                source=text,
                skip_translate=True,
                confidence=Confidence.VERIFIED,
            )
        )

    for line in _decode(md_path).split("\n"):
        stripped = line.strip()

        if plain_text:
            # Blank lines separate paragraphs; nothing is Markdown syntax.
            if not stripped:
                flush_paragraph()
            else:
                paragraph.append(line)
            continue

        # Fenced code first, so a `$$` line inside a fence stays code content.
        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                code.append(line)
                in_code = False
                flush_code()
            else:
                in_code = True
                code.append(line)
            continue
        if in_code:
            code.append(line)
            continue

        if stripped.startswith("$$"):
            flush_paragraph()
            if in_math:
                math.append(line)
                in_math = False
                flush_math()
            elif stripped.endswith("$$") and len(stripped) > 2:
                math.append(line)
                flush_math()
            else:
                in_math = True
                math.append(line)
            continue
        if in_math:
            math.append(line)
            continue

        heading = markdown_heading(stripped)
        if heading is not None:
            flush_paragraph()
            level, text = heading
            emit(
                Heading(
                    id=_id(),
                    spine_index=_spine(),
                    span=Span(),
                    text=text,
                    level=level,
                    confidence=Confidence.VERIFIED,
                )
            )
            continue

        item = markdown_list_item(line)
        if item is not None:
            flush_paragraph()
            marker, text = item
            emit(
                ListItem(
                    id=_id(),
                    spine_index=_spine(),
                    span=Span(),
                    text=text,
                    marker=marker,
                    confidence=Confidence.VERIFIED,
                )
            )
            continue

        if not stripped:
            flush_paragraph()
        else:
            paragraph.append(line)

    flush_paragraph()
    if in_code:
        flush_code()
    if in_math:
        flush_math()

    return assemble(elements, doc_id=doc_id or file_digest(md_path), path=str(md_path))


__all__ = ["read_md"]

"""Native DOCX reader: DOCX -> typed ``Document`` (native multi-format reader seam).

A DOCX is a zip whose ``word/document.xml`` is a flat body of paragraphs and
tables. The reader walks that body in order: a paragraph's style decides heading
level, a numbering property makes it a list item, a table becomes its grid
(Markdown pipe markup), and every embedded image becomes a :class:`Figure` whose
``asset_id`` is the image's *package part name* (``/word/media/…``) -- real
provenance, not a relative href.

``Span.chars`` indexes the canonical text built from the paragraph texts; DOCX has
no page model, so ``Span.page`` stays ``0``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docx import Document as DocxDocument
from docx.document import Document as _DocxDocument
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.table import _Cell as _DocxCell
from docx.text.paragraph import Paragraph as DocxParagraph

from ubt.analyze._identity import file_digest
from ubt.analyze.assemble import assemble, number
from ubt.analyze.normalize import collapse_whitespace as _clean
from ubt.analyze.reader_html import pipe_table
from ubt.model.ast import (
    Confidence,
    Document,
    Figure,
    Heading,
    ListItem,
    Paragraph,
    Table,
)
from ubt.model.span import Span

if TYPE_CHECKING:
    from ubt.model.ast import ElementT

#: Style-name prefixes that mark a heading (English and Chinese Word styles).
_HEADING_PREFIXES = ("heading", "标题")


def _heading_level(paragraph: DocxParagraph) -> int | None:
    """The heading level of a paragraph, from its style or outline level."""
    style = getattr(paragraph, "style", None)
    name = (getattr(style, "name", "") or "").strip().lower()
    for prefix in _HEADING_PREFIXES:
        if name.startswith(prefix):
            tail = name[len(prefix) :].strip()
            return max(1, min(6, int(tail))) if tail.isdigit() else 1
    ppr = paragraph._p.find(qn("w:pPr"))
    if ppr is not None:
        outline = ppr.find(qn("w:outlineLvl"))
        if outline is not None:
            value = outline.get(qn("w:val"))
            if value is not None and value.isdigit():
                return max(1, min(6, int(value) + 1))
    return None


def _is_list_item(paragraph: DocxParagraph) -> bool:
    style = getattr(paragraph, "style", None)
    name = (getattr(style, "name", "") or "").strip().lower()
    if name.startswith(("list", "bullet", "列表")):
        return True
    ppr = paragraph._p.find(qn("w:pPr"))
    return ppr is not None and ppr.find(qn("w:numPr")) is not None


def _image_assets(paragraph: DocxParagraph, document: _DocxDocument) -> list[str]:
    """Package part names of the images embedded in one paragraph."""
    assets: list[str] = []
    for blip in paragraph._p.findall(".//" + qn("a:blip")):
        rel_id = blip.get(qn("r:embed"))
        if not rel_id:
            continue
        try:
            part = document.part.related_parts[rel_id]
        except KeyError:
            continue
        assets.append(str(part.partname))
    return assets


def _collect_blocks(container: Any, document: _DocxDocument) -> Iterator[DocxParagraph | DocxTable]:
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P

    for child in container.iterchildren():
        if isinstance(child, CT_P) or child.tag == qn("w:p"):
            yield DocxParagraph(child, document)
        elif isinstance(child, CT_Tbl) or child.tag == qn("w:tbl"):
            yield DocxTable(child, document)
        elif child.tag == qn("w:sdt"):
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                yield from _collect_blocks(content, document)


def _iter_blocks(document: _DocxDocument) -> Iterator[DocxParagraph | DocxTable]:
    """Yield paragraphs and tables from the document body, in reading order."""
    yield from _collect_blocks(document.element.body, document)


def walk_docx_elements(
    document: _DocxDocument,
) -> list[tuple[ElementT, DocxParagraph | DocxTable]]:
    """Emit (Element, docx block) pairs for one DOCX document body, in reading order."""
    out: list[tuple[ElementT, DocxParagraph | DocxTable]] = []
    for block in _iter_blocks(document):
        if isinstance(block, DocxTable):
            rows = []
            # Walk the raw ``<w:tr>/<w:tc>`` grid rather than ``row.cells``.
            # python-docx's grid collapses both merges onto the origin cell:
            # a horizontal ``gridSpan`` cell repeats once per spanned column,
            # and a vertical ``vMerge`` continuation row resolves to the restart
            # cell above (so a merged header repeated on every row). Reading the
            # XML, a gridSpan cell is emitted once and padded, and a vMerge
            # continuation is blanked -- the grid geometry survives and no cell
            # text is double-billed. ``_Cell`` is python-docx's own text reader,
            # so paragraph/line-break handling stays identical.
            for tr in block._tbl.tr_lst:
                cells: list[str] = []
                for tc in tr.tc_lst:
                    tcpr = tc.find(qn("w:tcPr"))
                    grid_span = 1
                    continuation = False
                    if tcpr is not None:
                        gs = tcpr.find(qn("w:gridSpan"))
                        if gs is not None:
                            try:
                                grid_span = max(1, int(gs.get(qn("w:val")) or "1"))
                            except (TypeError, ValueError):
                                grid_span = 1
                        vm = tcpr.find(qn("w:vMerge"))
                        # A vMerge without val="restart" is a continuation of the
                        # cell above; its own ``<w:tc>`` carries no text.
                        if vm is not None and (vm.get(qn("w:val")) or "continue") != "restart":
                            continuation = True
                    cells.append("" if continuation else _clean(_DocxCell(tc, block).text))
                    cells.extend([""] * (grid_span - 1))
                rows.append(cells)
            markup = pipe_table(rows)
            if markup:
                out.append(
                    (
                        Table(
                            id="",
                            spine_index=0,
                            span=Span(),
                            markup=markup,
                            confidence=Confidence.VERIFIED,
                        ),
                        block,
                    )
                )
            continue
        text = _clean(block.text)
        level = _heading_level(block)
        if level is not None and text:
            out.append(
                (
                    Heading(
                        id="",
                        spine_index=0,
                        span=Span(),
                        text=text,
                        level=level,
                        confidence=Confidence.VERIFIED,
                    ),
                    block,
                )
            )
        elif _is_list_item(block) and text:
            out.append((ListItem(id="", spine_index=0, span=Span(), text=text, marker="•"), block))
        elif text:
            out.append((Paragraph(id="", spine_index=0, span=Span(), text=text), block))
        for asset_id in _image_assets(block, document):
            out.append(
                (
                    Figure(
                        id="",
                        spine_index=0,
                        span=Span(),
                        asset_id=asset_id,
                        confidence=Confidence.VERIFIED,
                    ),
                    block,
                )
            )
    return out


def read_docx(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Read a DOCX into a typed :class:`Document` with real spans."""
    docx_path = Path(path)
    document = DocxDocument(str(docx_path))
    pairs = walk_docx_elements(document)
    elements = [p[0] for p in pairs]
    numbered = number(elements, "docx")
    return assemble(numbered, doc_id=doc_id or file_digest(docx_path), path=str(docx_path))


__all__ = ["read_docx", "walk_docx_elements"]

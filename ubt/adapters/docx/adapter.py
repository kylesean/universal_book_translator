"""DOCX document adapter: typed AST extraction via reader_docx + in-place bilingual injection."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from copy import deepcopy
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph

from ubt.adapters.base import BaseDocumentAdapter, parse_pipe_table_cells
from ubt.adapters.unresolved import failure_note, is_unresolved
from ubt.analyze.assemble import number
from ubt.analyze.reader_docx import walk_docx_elements
from ubt.core.cleaners.html_sanitizer import sanitize_html_fragment, strip_html_mark_tags
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BookManifest,
    ChapterIR,
    ChapterMeta,
    IRBlock,
)
from ubt.core.ir.serializer import compute_file_sha256_cached
from ubt.model.ast import Table as ASTTable

logger = logging.getLogger(__name__)

_XML_ILLEGAL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_EAST_ASIA_FONT_MAP: dict[str, str] = {
    "zh": "SimSun",
    "zh-cn": "SimSun",
    "zh-hans": "SimSun",
    "zh-tw": "PMingLiU",
    "zh-hant": "PMingLiU",
    "zh-hk": "PMingLiU",
    "ja": "MS Mincho",
    "ko": "Malgun Gothic",
}


def _resolve_east_asia_font(target_lang: str | None) -> str | None:
    if not target_lang:
        return None
    lang_norm = target_lang.lower().replace("_", "-")
    if lang_norm in _EAST_ASIA_FONT_MAP:
        return _EAST_ASIA_FONT_MAP[lang_norm]
    primary = lang_norm.split("-")[0]
    return _EAST_ASIA_FONT_MAP.get(primary)


class DOCXAdapter(BaseDocumentAdapter):
    """Adapter for Word documents (.docx) with in-place bilingual injection."""

    output_suffixes = frozenset({".docx"})

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Extract a single-chapter manifest titled from core properties."""
        if not input_path.exists():
            raise DocumentParseError(f"DOCX file not found: {input_path}")

        doc_id = await asyncio.to_thread(compute_file_sha256_cached, input_path)
        title = input_path.stem
        try:
            doc = Document(str(input_path))
            core_title = doc.core_properties.title
            if core_title and core_title.strip():
                title = core_title.strip()
        except Exception as err:
            raise DocumentParseError(f"Failed to open DOCX document {input_path}: {err}") from err

        chapter = ChapterMeta(
            chapter_id="docx_main",
            title=title,
            spine_index=1,
            source_file=input_path.name,
        )
        return BookManifest(
            doc_id=doc_id,
            title=title,
            source_path=str(input_path),
            chapters=[chapter],
        )

    async def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]:
        """Stream the document as one ChapterIR (body walk off the event loop)."""
        manifest = await self.extract_manifest(input_path)
        chapter = manifest.chapters[0]

        loop = asyncio.get_running_loop()
        blocks = await loop.run_in_executor(None, self._parse_blocks_sync, input_path, chapter)

        yield ChapterIR(
            doc_id=manifest.doc_id,
            chapter_id=chapter.chapter_id,
            title=chapter.title,
            spine_index=chapter.spine_index,
            blocks=blocks,
        )

    def _parse_blocks_sync(self, path: Path, chapter: ChapterMeta) -> list[IRBlock]:
        """Build the typed blocks from the DOCX document structure."""
        doc = Document(str(path))
        pairs = walk_docx_elements(doc)
        elements = [p[0] for p in pairs]
        numbered = number(elements, chapter.chapter_id)
        return [IRBlock(element=elem) for elem in numbered]

    @classmethod
    def _populate_paragraph_runs(
        cls,
        para: Paragraph,
        translated_text: str,
        src_rpr: Any,
        target_lang: str | None = None,
    ) -> None:
        """Parse inline HTML formatting into native Word runs with style."""
        clean_text = _XML_ILLEGAL_CHAR_RE.sub(
            " ", sanitize_html_fragment(strip_html_mark_tags(translated_text))
        )

        east_asia_font = _resolve_east_asia_font(target_lang)

        def _apply_east_asia(r: Any) -> None:
            if not east_asia_font:
                return
            rpr = r._r.find(qn("w:rPr"))
            if rpr is None:
                rpr = r._r.makeelement(qn("w:rPr"))
                r._r.insert(0, rpr)
            rfonts = rpr.find(qn("w:rFonts"))
            if rfonts is None:
                rfonts = rpr.makeelement(qn("w:rFonts"))
                rpr.insert(0, rfonts)
            rfonts.set(qn("w:eastAsia"), east_asia_font)

        try:
            soup = BeautifulSoup(clean_text, "html.parser")

            def _walk(node: Any, is_bold: bool, is_italic: bool, is_underline: bool) -> None:
                tag = getattr(node, "name", None)
                if tag in ("b", "strong"):
                    is_bold = True
                elif tag in ("i", "em"):
                    is_italic = True
                elif tag in ("u", "ins"):
                    is_underline = True

                children = list(getattr(node, "children", []))
                if not children:
                    text = str(node) if not hasattr(node, "get_text") else node.get_text()
                    if text:
                        run = para.add_run(text)
                        if src_rpr is not None:
                            run._r.insert(0, deepcopy(src_rpr))
                        if is_bold:
                            run.bold = True
                        if is_italic:
                            run.italic = True
                        if is_underline:
                            run.underline = True
                        _apply_east_asia(run)
                else:
                    for child in children:
                        _walk(child, is_bold, is_italic, is_underline)

            contents = list(soup.contents)
            if not contents:
                contents = [NavigableString(soup.get_text())]
            for item in contents:
                _walk(item, is_bold=False, is_italic=False, is_underline=False)
        except Exception:
            run = para.add_run(BeautifulSoup(clean_text, "html.parser").get_text())
            if src_rpr is not None:
                run._r.insert(0, deepcopy(src_rpr))
            _apply_east_asia(run)

    @classmethod
    def _replace_paragraph_in_place(
        cls, para: Paragraph, translated_text: str, target_lang: str | None = None
    ) -> None:
        """Replace paragraph runs with translated text while keeping paragraph styles."""
        src_rpr = None
        for run in para.runs:
            rpr = run._r.find(qn("w:rPr"))
            if rpr is not None:
                src_rpr = deepcopy(rpr)
                break

        def _carries_graphic(el: Any) -> bool:
            return next(el.iter(qn("w:drawing"), qn("w:pict"), qn("w:object")), None) is not None

        def _text_of(el: Any) -> str:
            return "".join(node.text or "" for node in el.iter(qn("w:t")))

        p = para._p
        graphics = {child for child in p if _carries_graphic(child)}
        hyperlinks = [h for h in p.findall(qn("w:hyperlink")) if h not in graphics]
        link_text = "".join(_text_of(h) for h in hyperlinks)
        host = (
            hyperlinks[0]
            if link_text.strip() and link_text.strip() == _text_of(p).strip()
            else None
        )
        for child in list(p):
            if child in graphics:
                for t in child.iter(qn("w:t")):
                    t.text = ""
                continue
            if child is host and host is not None:
                for linked_run in list(host):
                    if linked_run.tag == qn("w:r"):
                        host.remove(linked_run)
            elif child.tag in (qn("w:r"), qn("w:hyperlink")) or (
                next(child.iter(qn("w:t")), None) is not None
            ):
                p.remove(child)
        cls._populate_paragraph_runs(para, translated_text, src_rpr, target_lang=target_lang)
        if host is not None:
            moved = 0
            for child in list(p):
                if child.tag == qn("w:r") and child not in graphics:
                    p.remove(child)
                    host.append(child)
                    moved += 1
            if not moved:
                p.remove(host)

    @classmethod
    def _insert_bilingual_after(
        cls, para: Paragraph, translated_text: str, target_lang: str | None = None
    ) -> None:
        """Insert a translated clone of ``para`` right after it (style-preserving)."""
        src_rpr = None
        for run in para.runs:
            rpr = run._r.find(qn("w:rPr"))
            if rpr is not None:
                src_rpr = deepcopy(rpr)
                break

        new_p = para._p.makeelement(qn("w:p"), {})
        src_ppr = para._p.find(qn("w:pPr"))
        if src_ppr is not None:
            copied_ppr = deepcopy(src_ppr)
            for break_el in copied_ppr.findall(qn("w:sectPr")):
                copied_ppr.remove(break_el)
            for num_el in copied_ppr.findall(qn("w:numPr")):
                copied_ppr.remove(num_el)
            if len(copied_ppr) or copied_ppr.attrib:
                new_p.append(copied_ppr)
        para._p.addnext(new_p)

        new_para = Paragraph(new_p, para._parent)
        cls._populate_paragraph_runs(new_para, translated_text, src_rpr, target_lang=target_lang)

    @classmethod
    def _render_table(
        cls,
        table: DocxTable,
        translated: str,
        is_monolingual: bool,
        target_lang: str | None = None,
    ) -> None:
        """Inject translated pipe table markup into a docx table."""
        rows = parse_pipe_table_cells(translated)
        if not rows:
            return
        seen_cells: set[int] = set()
        for row_idx, row in enumerate(table.rows):
            if row_idx >= len(rows):
                break
            target_row = rows[row_idx]
            for col_idx, cell in enumerate(row.cells):
                if col_idx >= len(target_row):
                    break
                tc_id = id(cell._tc)
                if tc_id in seen_cells:
                    continue
                seen_cells.add(tc_id)
                cell_target = target_row[col_idx].strip()
                if not cell_target:
                    continue
                if is_monolingual:
                    if cell.paragraphs:
                        cls._replace_paragraph_in_place(
                            cell.paragraphs[0], cell_target, target_lang=target_lang
                        )
                        for extra_p in list(cell.paragraphs[1:]):
                            p_el = extra_p._p
                            parent = p_el.getparent()
                            if parent is not None:
                                parent.remove(p_el)
                    else:
                        p = cell.add_paragraph()
                        cls._populate_paragraph_runs(p, cell_target, None, target_lang=target_lang)
                else:
                    p = cell.add_paragraph()
                    cls._populate_paragraph_runs(p, cell_target, None, target_lang=target_lang)

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
        """Re-open the source document and inject translated paragraphs and tables."""
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
        """Perform the actual document edit + atomic save."""
        source_path = Path(manifest.source_path)
        doc = Document(str(source_path))
        pairs = walk_docx_elements(doc)
        elements = [p[0] for p in pairs]
        chapter_id = manifest.chapters[0].chapter_id if manifest.chapters else "docx_main"
        numbered = number(elements, chapter_id)
        by_id = {b.id: b for b in blocks}

        is_monolingual = bilingual_mode in ("target", "monolingual")
        injected = 0

        for elem, (_orig_elem, block) in zip(numbered, pairs, strict=True):
            b = by_id.get(elem.id)
            if not b or b.skip_translate or not b.target_text:
                continue
            translated = strip_html_mark_tags(str(b.target_text))
            if is_unresolved(b.status):
                note = failure_note(b.status)
                translated = f"{translated}\n\n{note}".strip()

            if isinstance(elem, ASTTable) and isinstance(block, DocxTable):
                self._render_table(block, translated, is_monolingual, target_lang=target_lang)
                injected += 1
            elif isinstance(block, Paragraph):
                if is_monolingual:
                    self._replace_paragraph_in_place(block, translated, target_lang=target_lang)
                else:
                    self._insert_bilingual_after(block, translated, target_lang=target_lang)
                injected += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            doc.save(str(tmp_path))
            tmp_path.replace(output_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        if injected == 0 and by_id:
            logger.warning(
                "DOCX render injected 0 translations (empty blocks?)",
            )
        return output_path

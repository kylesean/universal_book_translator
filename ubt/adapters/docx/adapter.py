"""DOCX document adapter: paragraph + table-cell translation via python-docx (MIT).

Design notes:
- Blocks are extracted in strict document order by walking the body's ``w:p``
  and ``w:tbl`` children (python-docx's ``document.paragraphs`` and
  ``document.tables`` are separate lists and would interleave incorrectly).
- Table cells become TABLE_GRID flow blocks keyed by
  ``t{table}r{row}c{col}p{para}``; merged cells (repeated ``_tc`` elements)
  are de-duplicated by their XML identity so a merged cell is mined once.
- Rendering re-opens the original document (read-only), locates the same
  positions and inserts a bilingual paragraph right after each source
  paragraph — the source paragraph keeps its runs (with hyperlinks, images,
  formatting), the inserted copy inherits paragraph properties (pPr) and the
  first run's character properties (rPr). Everything python-docx does not
  model (headers/footers, footnotes, images, sections) is carried through
  untouched because the document is edited in memory, never rebuilt.
"""

import asyncio
import logging
import re
import tempfile
from collections.abc import AsyncIterator, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from docx import Document
from docx.document import Document as DocumentObject
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml import etree

from ubt.adapters.base import BaseDocumentAdapter
from ubt.adapters.unresolved import failure_note, is_unresolved
from ubt.core.cleaners.html_sanitizer import sanitize_html_fragment, strip_html_mark_tags
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BlockType,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    FlowID,
    IRBlock,
)
from ubt.core.ir.serializer import compute_file_sha256_cached

logger = logging.getLogger(__name__)

_HEADING_STYLE_PREFIXES = ("heading", "title")

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


def _iter_body_items(doc: DocumentObject) -> list[Paragraph | Table]:
    """Body items in true document order (paragraphs and top-level tables).

    Recurses into ``w:sdt`` content controls: their paragraphs/tables are nested
    under ``w:sdt/w:sdtContent`` rather than direct body children, so a document
    built from content controls used to be extracted (and shipped) untranslated.
    """
    items: list[Paragraph | Table] = []

    def _walk(container: Any) -> None:
        for child in container.iterchildren():
            if child.tag == qn("w:p"):
                items.append(Paragraph(child, doc))
            elif child.tag == qn("w:tbl"):
                items.append(Table(child, doc))
            elif child.tag == qn("w:sdt"):
                content = child.find(qn("w:sdtContent"))
                if content is not None:
                    _walk(content)

    _walk(doc.element.body)
    return items


#: OOXML subtrees whose ``w:t`` is a picture/text-box payload, not flow text.
_GRAPHIC_SUBTREES = (qn("w:drawing"), qn("w:pict"), qn("w:object"))


def _paragraph_text(el: Any) -> str:
    """All visible ``w:t`` text of a paragraph, excluding drawing subtrees.

    ``python-docx``'s ``Paragraph.text`` concatenates only *direct* ``w:r``
    children, so it drops runs wrapped in ``w:ins`` (tracked insertions),
    ``w:customXml``, ``w:smartTag`` or ``w:sdt`` — text that is on the page and
    that the monolingual rewrite replaces. ``w:delText`` (tracked deletions) is
    a different element and is correctly omitted. Kept in sync with the render
    leg (``_replace_paragraph_in_place``) so extraction and rewrite agree on
    what counts as the paragraph's text.
    """
    parts: list[str] = []
    for t in el.iter(qn("w:t")):
        ancestor = t.getparent()
        in_graphic = False
        while ancestor is not None and ancestor is not el:
            if ancestor.tag in _GRAPHIC_SUBTREES:
                in_graphic = True
                break
            ancestor = ancestor.getparent()
        if not in_graphic:
            parts.append(t.text or "")
    return "".join(parts)


def _walk_table_paragraphs(
    table: Table, id_prefix: str, tree: Any, seen_paths: set[str]
) -> Iterator[tuple[str, Paragraph]]:
    """Yield ``(block_id, paragraph)`` for every cell paragraph of ``table``,
    recursing into tables nested inside cells.

    Both extraction (:meth:`DOCXAdapter._extract_table_blocks`) and rendering
    (:meth:`DOCXAdapter._render_sync`) consume this *single* generator, so the
    ``block_id`` each paragraph gets is computed by identical code on both
    legs (the two can never drift apart) and nested tables are covered too.
    ``id_prefix`` encodes the path from the top-level table
    (``docx_main#t{idx:03d}``), extended by ``r{row}c{col}`` per cell and
    ``n{idx}`` per nested table. ``seen_paths`` de-duplicates merged/spanned
    cells by XML path (``row.cells`` repeats them), so each real cell is walked
    once.
    """
    for row_idx, row in enumerate(table.rows):
        for col_idx, cell in enumerate(row.cells):
            path = tree.getpath(cell._tc)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            cell_prefix = f"{id_prefix}r{row_idx:03d}c{col_idx:03d}"
            for para_idx, para in enumerate(cell.paragraphs):
                yield f"{cell_prefix}p{para_idx:03d}", para
            for nested_idx, nested in enumerate(cell.tables):
                yield from _walk_table_paragraphs(
                    nested, f"{cell_prefix}n{nested_idx:03d}", tree, seen_paths
                )


def _iter_section_parts(section: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(kind_tag, part)`` for a section's six header/footer parts.

    ``kind_tag`` is a short stable key (hdr / ftr / hdrF / ftrF / hdrE / ftrE
    for primary / first-page / even-page). Callers skip parts whose
    ``is_linked_to_previous`` is True — those inherit a previous section's part,
    so processing them would double-translate the same physical header.
    """
    for attr, tag in (
        ("header", "hdr"),
        ("footer", "ftr"),
        ("first_page_header", "hdrF"),
        ("first_page_footer", "ftrF"),
        ("even_page_header", "hdrE"),
        ("even_page_footer", "ftrE"),
    ):
        part = getattr(section, attr, None)
        if part is not None:
            yield tag, part


def _walk_header_footer_blocks(doc: DocumentObject) -> Iterator[tuple[str, Paragraph]]:
    """Yield ``(block_id, paragraph)`` for every non-inherited section
    header/footer paragraph and table-cell paragraph.

    Like :func:`_walk_table_paragraphs`, both extraction and rendering consume
    this single generator, so ids cannot drift between the two legs and nested
    tables inside a header are covered too. ``docx`` has no first-class
    footnotes/endnotes API (they live in a separate ``word/footnotes.xml``
    part), so those are intentionally out of scope here.
    """
    for sec_idx, section in enumerate(doc.sections):
        for tag, part in _iter_section_parts(section):
            if getattr(part, "is_linked_to_previous", False):
                continue
            pfx = f"docx_main#h{sec_idx:03d}{tag}"
            para_idx = 0
            for para in part.paragraphs:
                if not _paragraph_text(para._p).strip():
                    continue
                yield f"{pfx}p{para_idx:05d}", para
                para_idx += 1
            for tbl_idx, table in enumerate(part.tables):
                tree = table._element.getroottree()
                yield from _walk_table_paragraphs(table, f"{pfx}t{tbl_idx:03d}", tree, set())


# Separator/continuation footnotes/endnotes carry ids -1 and 0; they hold the
# continuation dash, not content, so they must never be translated.
_NOTE_STRUCTURAL_IDS = {"-1", "0"}


def _iter_note_parts(doc: DocumentObject) -> Iterator[tuple[str, Any]]:
    """Yield ``(kind_tag, part)`` for the footnotes / endnotes parts if present.

    python-docx has no first-class footnote API — these are generic ``Part``
    objects exposing only ``.blob`` — so callers parse the blob with lxml and
    write it back before save. Absent parts are skipped (a doc with no
    footnotes yields nothing).
    """
    for reltype, tag in (
        (RT.FOOTNOTES, "fn"),
        (RT.ENDNOTES, "en"),
    ):
        try:
            part = doc.part.part_related_by(reltype)
        except KeyError:
            continue
        yield tag, part


def _walk_note_paragraphs(root: Any, tag: str) -> Iterator[tuple[str, int, int, Any]]:
    """Yield ``(block_id, note_id, para_idx, paragraph_element)`` for the real
    footnote/endnote paragraphs in a parsed ``root``.

    Used by BOTH extraction and rendering, so ids match across the legs. The
    tag selects ``w:footnote`` vs ``w:endnote``; structural ids (-1/0) are
    skipped. ``para_idx`` is per-note so a multi-paragraph footnote yields one
    block per paragraph.
    """
    note_tag = qn("w:footnote") if tag == "fn" else qn("w:endnote")
    for note in root.findall(note_tag):
        nid = note.get(qn("w:id"))
        if nid is None or nid in _NOTE_STRUCTURAL_IDS:
            continue
        try:
            note_id = int(nid)
        except ValueError:
            continue
        for para_idx, para in enumerate(note.findall(qn("w:p"))):
            yield f"docx_main#{tag}{note_id:05d}p{para_idx:05d}", note_id, para_idx, para


# ``xml:space`` (preserve leading/trailing whitespace in <w:t>), by its real
# namespace since python-docx's qn has no "xml" prefix binding.
_XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def _apply_note_translation(
    para_el: Any,
    translated: str,
    is_monolingual: bool,
    target_lang: str | None = None,
) -> None:
    """Put ``translated`` into a footnote/endnote paragraph without breaking its
    structure.

    Only the *text* of existing ``<w:t>`` nodes is blanked (monolingual) —
    never a run removed — so the ``<w:footnoteRef/>``/``<w:endnoteRef/>``
    marker and any formatting stay intact (removing the wrong node would
    corrupt footnote numbering). The translation is then appended as one new
    run. Bilingual keeps the original text and appends the target.
    """
    clean_text = BeautifulSoup(
        sanitize_html_fragment(strip_html_mark_tags(translated)), "html.parser"
    ).get_text()

    if is_monolingual:
        for t in para_el.iter(qn("w:t")):
            t.text = ""
        prefix = ""
    else:
        existing_text = "".join(node.text or "" for node in para_el.iter(qn("w:t"))).strip()
        prefix = " " if existing_text else ""

    run = etree.SubElement(para_el, qn("w:r"))
    east_asia_font = _resolve_east_asia_font(target_lang)
    if east_asia_font:
        rpr = etree.SubElement(run, qn("w:rPr"))
        rfonts = etree.SubElement(rpr, qn("w:rFonts"))
        rfonts.set(qn("w:eastAsia"), east_asia_font)

    full_text = prefix + clean_text
    lines = full_text.split("\n")
    for i, line in enumerate(lines):
        if i > 0:
            etree.SubElement(run, qn("w:br"))
        t = etree.SubElement(run, qn("w:t"))
        t.set(_XML_SPACE, "preserve")
        t.text = line


def _classify_paragraph(para: Paragraph) -> BlockType:
    """Heading/Title styles map to HEADING; everything else is NARRATIVE."""
    try:
        style = para.style
        style_name = (style.name if style is not None else "") or ""
        style_name = style_name.lower()
    except Exception:  # malformed styles must not break ingestion
        return BlockType.NARRATIVE
    if style_name.startswith(_HEADING_STYLE_PREFIXES):
        return BlockType.HEADING
    return BlockType.NARRATIVE


class DOCXAdapter(BaseDocumentAdapter):
    """Adapter for Word documents (.docx) with in-place bilingual injection."""

    # The base's empty frozenset means "no validation", so without this a docx
    # written to a .pdf/.txt path would report success while producing a
    # mislabelled file — the class of bug markdown/epub already guard.
    output_suffixes = frozenset({".docx"})

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Extract a single-chapter manifest titled from core properties."""
        if not input_path.exists():
            raise DocumentParseError(f"DOCX file not found: {input_path}")

        # Off-loop: compute doc_id sha in thread pool to avoid blocking the event loop on large files.
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

    def _extract_blocks_sync(self, path: Path) -> list[IRBlock]:
        """Walk the document body once and collect translatable blocks."""
        doc = Document(str(path))
        blocks: list[IRBlock] = []
        global_spine = 1
        para_idx = 0
        table_idx = 0

        for item in _iter_body_items(doc):
            if isinstance(item, Paragraph):
                text = _paragraph_text(item._p).strip()
                if not text:
                    continue
                blocks.append(
                    IRBlock(
                        id=f"docx_main#p{para_idx:05d}",
                        flow_id=FlowID.MAIN_STORY,
                        spine_index=global_spine,
                        block_type=_classify_paragraph(item),
                        source_text=text,
                    )
                )
                para_idx += 1
                global_spine += 1
            else:
                table_idx, global_spine = self._extract_table_blocks(
                    item, table_idx, blocks, global_spine
                )

        # Section headers / footers (and tables inside them) are translatable
        # content and must be mined. Shared generator with the render leg keeps
        # ids aligned; inherited (linked-to-previous) parts are skipped so a
        # repeated physical header is not translated twice.
        for block_id, para in _walk_header_footer_blocks(doc):
            text = _paragraph_text(para._p).strip()
            if not text:
                continue
            blocks.append(
                IRBlock(
                    id=block_id,
                    flow_id=FlowID.MAIN_STORY,
                    spine_index=global_spine,
                    block_type=BlockType.NARRATIVE,
                    source_text=text,
                )
            )
            global_spine += 1

        # Footnotes / endnotes: text lives in a separate OPC part python-docx
        # exposes only as a raw blob. Same walker as the render leg keeps ids
        # aligned; separator notes (id -1/0) are skipped inside the walker.
        for tag, part in _iter_note_parts(doc):
            root = etree.fromstring(part.blob)
            for block_id, _nid, _pidx, para_el in _walk_note_paragraphs(root, tag):
                text = "".join(t.text or "" for t in para_el.iter(qn("w:t"))).strip()
                if not text:
                    continue
                blocks.append(
                    IRBlock(
                        id=block_id,
                        flow_id=FlowID.MAIN_STORY,
                        spine_index=global_spine,
                        block_type=BlockType.NARRATIVE,
                        source_text=text,
                    )
                )
                global_spine += 1

        return blocks

    @staticmethod
    def _extract_table_blocks(
        table: Table, table_idx: int, blocks: list[IRBlock], next_spine: int
    ) -> tuple[int, int]:
        """Append one table's cell blocks; returns (next_table_idx, next_spine).

        Traversal and id assignment go through :func:`_walk_table_paragraphs`
        — the same generator the render leg uses — so nested tables are covered
        and the two legs cannot drift apart. Merged cells are de-duplicated by
        XML tree path (``id()`` must NOT be used: transient lxml proxies can
        alias and silently drop real content).
        """
        tree = table._element.getroottree()
        id_prefix = f"docx_main#t{table_idx:03d}"
        for block_id, cell_para in _walk_table_paragraphs(table, id_prefix, tree, set()):
            text = _paragraph_text(cell_para._p).strip()
            if not text:
                continue
            blocks.append(
                IRBlock(
                    id=block_id,
                    flow_id=FlowID.TABLE_GRID,
                    spine_index=next_spine,
                    block_type=BlockType.NARRATIVE,
                    source_text=text,
                )
            )
            next_spine += 1
        return table_idx + 1, next_spine

    async def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]:
        """Stream the document as one ChapterIR (body walk off the event loop)."""
        manifest = await self.extract_manifest(input_path)
        chapter = manifest.chapters[0]

        loop = asyncio.get_running_loop()
        blocks = await loop.run_in_executor(None, self._extract_blocks_sync, input_path)

        yield ChapterIR(
            doc_id=manifest.doc_id,
            chapter_id=chapter.chapter_id,
            title=chapter.title,
            spine_index=chapter.spine_index,
            blocks=blocks,
        )

    @classmethod
    def _populate_paragraph_runs(
        cls,
        para: Paragraph,
        translated_text: str,
        src_rpr: Any,
        target_lang: str | None = None,
    ) -> None:
        """Parse inline HTML formatting into native Word runs with style."""
        # Sanitize to drop dangerous markup, then always parse the fragment:
        # sanitize_html_fragment neutralises stray ``<`` into ``&lt;`` entities,
        # which BeautifulSoup decodes back to literal text — so plain prose
        # like ``List<T>`` or ``a<b`` never loses characters, while real inline
        # ``<b>/<i>`` still becomes styled runs.
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
            # Inline pictures reach the page through w:drawing (modern) or
            # w:pict/w:object (legacy VML, embedded OLE), any depth inside the
            # run — AlternateContent wraps them in a choice/fallback pair.
            return next(el.iter(qn("w:drawing"), qn("w:pict"), qn("w:object")), None) is not None

        def _text_of(el: Any) -> str:
            return "".join(node.text or "" for node in el.iter(qn("w:t")))

        p = para._p
        # A run or hyperlink can carry drawings / pictures.
        # Graphic-carrying children are kept so pictures and clickable images survive.
        graphics = {child for child in p if _carries_graphic(child)}
        hyperlinks = [h for h in p.findall(qn("w:hyperlink")) if h not in graphics]
        # A hyperlink is a direct child of ``w:p``, so its source text would
        # stay visible in a "monolingual" export unless handled here. When the
        # paragraph's text is *entirely* link text (a linked heading or a bare
        # URL) the translation goes into the link instead of dropping it.
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
                # Remove direct runs/links AND inline wrappers that carry flow
                # text (``w:ins`` tracked insertions, ``w:customXml``,
                # ``w:smartTag``, ``w:sdt``): ``Paragraph.text`` omits their
                # runs, so leaving them behind shipped the untranslated source
                # in a monolingual export.
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
            # The loop above already removed every ``w:hyperlink`` except
            # ``host`` (they are direct children of ``w:p``), so re-removing
            # ``hyperlinks[1:]`` here raised "Element is not a child of this
            # node" on any paragraph whose text is two or more links.

    @classmethod
    def _insert_bilingual_after(
        cls, para: Paragraph, translated_text: str, target_lang: str | None = None
    ) -> None:
        """Insert a translated clone of ``para`` right after it (style-preserving)."""
        # Capture character formatting from the first source run.
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
            # A mid-document section break is stored as ``w:sectPr`` inside the
            # pPr of the section's *last* paragraph. Cloning that paragraph
            # without stripping it gave the document a second section break —
            # a two-section book rendered bilingual delivered three, with
            # duplicated page size, margins and header/footer references
            # part-way through a chapter.
            for break_el in copied_ppr.findall(qn("w:sectPr")):
                copied_ppr.remove(break_el)
            # Strip list numbering from the bilingual sibling so Word does not
            # double-increment numbered list counters or render duplicate bullets.
            for num_el in copied_ppr.findall(qn("w:numPr")):
                copied_ppr.remove(num_el)
            if len(copied_ppr) or copied_ppr.attrib:
                new_p.append(copied_ppr)
        para._p.addnext(new_p)

        new_para = Paragraph(new_p, para._parent)
        cls._populate_paragraph_runs(new_para, translated_text, src_rpr, target_lang=target_lang)

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
        """Re-open the source document and inject translated paragraphs."""
        source_path = Path(manifest.source_path)
        if bilingual_mode is None and manifest and manifest.run:
            bilingual_mode = manifest.run.bilingual_mode

        targets: dict[str, str] = {}
        for b in blocks:
            if b.skip_translate or not b.target_text:
                continue
            text = strip_html_mark_tags(str(b.target_text))
            if is_unresolved(b.status):
                # DOCX cannot store the export stage's ``<mark>`` wrapper, so a
                # failed draft would otherwise be indistinguishable from a
                # finished translation. Append the shared unresolved note.
                note = failure_note(b.status)
                text = f"{text}\n\n{note}".strip()
            targets[b.id] = text

        loop = asyncio.get_running_loop()
        injected = await loop.run_in_executor(
            None,
            self._render_sync,
            source_path,
            targets,
            output_path,
            bilingual_mode,
            target_lang,
        )

        if injected == 0 and targets:
            logger.warning(
                "DOCX render injected 0 of %d translations "
                "(block ids may not match the extracted document)",
                len(targets),
            )
        return output_path

    def _render_sync(
        self,
        source_path: Path,
        targets: dict[str, str],
        output_path: Path,
        bilingual_mode: str | None = None,
        target_lang: str | None = None,
    ) -> int:
        """Perform the actual document edit + atomic save. Returns injected count."""
        doc = Document(str(source_path))
        injected = 0
        is_monolingual = bilingual_mode in ("target", "monolingual")

        para_idx = 0
        table_idx = 0
        for item in _iter_body_items(doc):
            if isinstance(item, Paragraph):
                # Must mirror _extract_blocks_sync exactly: empty paragraphs are
                # skipped there *without* consuming a para_idx, so they must not
                # consume one here either. Divergence silently shifts every
                # subsequent block_id and injects translations into the wrong
                # Paragraph.
                if not _paragraph_text(item._p).strip():
                    continue
                block_id = f"docx_main#p{para_idx:05d}"
                para_idx += 1
                translated = targets.get(block_id)
                if translated:
                    if is_monolingual:
                        self._replace_paragraph_in_place(item, translated, target_lang=target_lang)
                    else:
                        self._insert_bilingual_after(item, translated, target_lang=target_lang)
                    injected += 1
            else:
                tree = item._element.getroottree()
                id_prefix = f"docx_main#t{table_idx:03d}"
                # Same generator as extraction → identical block ids, nested
                # tables included, merged cells de-duplicated identically.
                for block_id, cell_para in _walk_table_paragraphs(item, id_prefix, tree, set()):
                    translated = targets.get(block_id)
                    if not translated or not _paragraph_text(cell_para._p).strip():
                        continue
                    if is_monolingual:
                        self._replace_paragraph_in_place(
                            cell_para, translated, target_lang=target_lang
                        )
                    else:
                        self._insert_bilingual_after(cell_para, translated, target_lang=target_lang)
                    injected += 1
                table_idx += 1

        # Section headers / footers: same shared generator as extraction, so the
        # ids line up and inherited (linked-to-previous) parts are skipped here
        # exactly as they were when building the blocks.
        for block_id, para in _walk_header_footer_blocks(doc):
            translated = targets.get(block_id)
            if not translated or not _paragraph_text(para._p).strip():
                continue
            if is_monolingual:
                self._replace_paragraph_in_place(para, translated, target_lang=target_lang)
            else:
                self._insert_bilingual_after(para, translated, target_lang=target_lang)
            injected += 1

        # Footnotes / endnotes: rewrite the raw note-part blob (python-docx has
        # no footnote API). Same walker as extraction → identical ids; only the
        # translated text is written back, refs/formatting left intact.
        for tag, part in _iter_note_parts(doc):
            root = etree.fromstring(part.blob)
            mutated = False
            for block_id, _nid, _pidx, para_el in _walk_note_paragraphs(root, tag):
                translated = targets.get(block_id)
                if not translated:
                    continue
                _apply_note_translation(
                    para_el, translated, is_monolingual, target_lang=target_lang
                )
                injected += 1
                mutated = True
            if mutated:
                part._blob = etree.tostring(
                    root, xml_declaration=True, encoding="UTF-8", standalone=True
                )

        # Atomic POSIX-style save: write to a temp file in the target dir,
        # then swap into place. The SOURCE document is never modified.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=output_path.parent, suffix=".docx", delete=False
        ) as tf:
            temp_name = tf.name
        try:
            doc.save(temp_name)
            Path(temp_name).replace(output_path)
        except Exception:
            # A failed save must not leave an orphan .docx temp next to the output.
            Path(temp_name).unlink(missing_ok=True)
            raise
        return injected

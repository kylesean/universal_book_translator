"""HTML document adapter: leaf-block extraction with bilingual DOM injection.

Reuses the EPUB leaf-block algorithm (BLOCK_TAGS / is_leaf_block /
determine_flow_id) so standalone .html/.htm documents get exactly the same
block semantics as EPUB spine items.

Rendering is style-preserving by construction: the original document is
re-parsed and translations injected as siblings, so source markup,
stylesheets and scripts are untouched.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from bs4.element import AttributeValueList

from ubt.adapters.base import (
    BILINGUAL_TARGET_CLASS,
    BaseDocumentAdapter,
    decode_markup,
    parse_pipe_table_cells,
)
from ubt.adapters.epub.adapter import take_preserved_inline_children
from ubt.adapters.unresolved import failure_note, is_unresolved
from ubt.analyze.assemble import number
from ubt.analyze.reader_html import walk_markup_elements
from ubt.core.cleaners.html_sanitizer import sanitize_html_fragment, scrub_source_document
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BookManifest,
    ChapterIR,
    ChapterMeta,
    IRBlock,
)
from ubt.core.ir.serializer import compute_file_sha256_cached
from ubt.model.ast import ListItem, Table

logger = logging.getLogger(__name__)

# Marker class added to injected bilingual target nodes.
_TARGET_CSS_CLASS = BILINGUAL_TARGET_CLASS


class HTMLAdapter(BaseDocumentAdapter):
    """Adapter for standalone HTML documents (.html/.htm) with DOM injection."""

    output_suffixes = frozenset({".html", ".htm"})

    def _load_soup(self, input_path: Path) -> BeautifulSoup:
        if not input_path.exists():
            raise DocumentParseError(f"HTML file not found: {input_path}")
        try:
            raw = decode_markup(input_path.read_bytes())
            return BeautifulSoup(raw, "html.parser")
        except Exception as err:
            raise DocumentParseError(f"Failed to parse HTML document {input_path}: {err}") from err

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Extract a single-chapter manifest titled from <title>."""
        soup = await asyncio.to_thread(self._load_soup, input_path)
        doc_id = await asyncio.to_thread(compute_file_sha256_cached, input_path)

        title_tag = soup.title
        title = title_tag.get_text().strip() if title_tag else input_path.stem
        if not title:
            title = input_path.stem

        chapter = ChapterMeta(
            chapter_id="html_main",
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
        """Stream the document as one ChapterIR of typed blocks."""
        manifest = await self.extract_manifest(input_path)
        soup = await asyncio.to_thread(self._load_soup, input_path)
        chapter = manifest.chapters[0]

        blocks = await asyncio.to_thread(self._parse_blocks_sync, soup, chapter)

        yield ChapterIR(
            doc_id=manifest.doc_id,
            chapter_id=chapter.chapter_id,
            title=chapter.title,
            spine_index=chapter.spine_index,
            blocks=blocks,
        )

    def _parse_blocks_sync(self, soup: BeautifulSoup, chapter: ChapterMeta) -> list[IRBlock]:
        """Build the typed blocks from the DOM structure."""
        pairs = walk_markup_elements(soup.body or soup)
        elements = [p[0] for p in pairs]
        numbered = number(elements, chapter.chapter_id)
        return [IRBlock(element=elem) for elem in numbered]

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
        """Re-parse the source document and inject bilingual target siblings.

        DOM parsing and the file write are synchronous CPU + IO; run them off
        the event loop so a concurrent job/task is not stalled.
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
        """Re-parse the source document and inject bilingual target siblings."""
        soup = self._load_soup(Path(manifest.source_path))
        pairs = walk_markup_elements(soup.body or soup)
        by_id = {b.id: b for b in blocks}

        is_monolingual = bilingual_mode in ("target", "monolingual")
        chapter_id = manifest.chapters[0].chapter_id

        injected = 0
        for idx, (elem, dom_node) in enumerate(pairs):
            block_id = f"{chapter_id}#{idx:05d}"
            block = by_id.get(block_id)
            if block is None or block.skip_translate or not block.target_text:
                continue

            target_text = sanitize_html_fragment(str(block.target_text))
            if not target_text:
                continue
            if is_unresolved(block.status):
                target_text = sanitize_html_fragment(
                    f"{failure_note(block.status)}\n\n{block.target_text}"
                )
                if not target_text:
                    continue

            if isinstance(elem, Table) and isinstance(dom_node, Tag):
                rows = parse_pipe_table_cells(target_text)
                if rows:
                    if is_monolingual:
                        dom_trs = dom_node.find_all("tr")
                        if len(dom_trs) == len(rows):
                            for r_idx, tr in enumerate(dom_trs):
                                cells = tr.find_all(["th", "td"])
                                for c_idx, cell in enumerate(cells):
                                    if c_idx < len(rows[r_idx]):
                                        cell.string = rows[r_idx][c_idx]
                        else:
                            new_table = soup.new_tag("table")
                            if dom_node.get("class"):
                                new_table["class"] = dom_node["class"]
                            for row in rows:
                                tr = soup.new_tag("tr")
                                for cell_text in row:
                                    td = soup.new_tag("td")
                                    td.string = cell_text
                                    tr.append(td)
                                new_table.append(tr)
                            dom_node.replace_with(new_table)
                    else:
                        new_table = soup.new_tag("table")
                        source_classes = list(dom_node.get("class") or [])
                        new_table["class"] = AttributeValueList(
                            source_classes + [_TARGET_CSS_CLASS]
                        )
                        for row in rows:
                            tr = soup.new_tag("tr")
                            for cell_text in row:
                                td = soup.new_tag("td")
                                td.string = cell_text
                                tr.append(td)
                            new_table.append(tr)
                        dom_node.insert_after(new_table)
                injected += 1
                continue

            if isinstance(dom_node, Tag):
                if isinstance(elem, ListItem):
                    if is_monolingual:
                        preserved_inline = take_preserved_inline_children(dom_node)
                        dom_node.clear()
                        dom_node.append(BeautifulSoup(target_text, "html.parser"))
                        for node in preserved_inline:
                            dom_node.append(node)
                    else:
                        target_tag = soup.new_tag("div")
                        target_tag["class"] = AttributeValueList([_TARGET_CSS_CLASS])
                        target_tag.append(BeautifulSoup(target_text, "html.parser"))
                        dom_node.append(target_tag)
                else:
                    if is_monolingual:
                        paras = [p.strip() for p in target_text.split("\n\n") if p.strip()] or [
                            target_text
                        ]
                        preserved_inline = take_preserved_inline_children(dom_node)
                        dom_node.clear()
                        last_node: Tag = dom_node
                        for i, p_text in enumerate(paras):
                            parsed_fragment = BeautifulSoup(p_text, "html.parser")
                            container = (
                                dom_node
                                if i == 0
                                else soup.new_tag(dom_node.name if dom_node.name else "p")
                            )
                            for child in list(parsed_fragment.contents):
                                container.append(child)
                            if i > 0:
                                last_node.insert_after(container)
                            last_node = container
                        for node in preserved_inline:
                            dom_node.append(node)
                    else:
                        source_classes = list(dom_node.get("class") or [])
                        target_classes = source_classes + [_TARGET_CSS_CLASS]
                        paras = [p.strip() for p in target_text.split("\n\n") if p.strip()] or [
                            target_text
                        ]
                        last_node = dom_node
                        for p_text in paras:
                            target_tag = soup.new_tag(dom_node.name if dom_node.name else "p")
                            target_tag["class"] = AttributeValueList(target_classes)
                            parsed_fragment = BeautifulSoup(p_text, "html.parser")
                            for child in list(parsed_fragment.contents):
                                target_tag.append(child)
                            last_node.insert_after(target_tag)
                            last_node = target_tag
            elif isinstance(dom_node, list) and dom_node:
                if is_monolingual:
                    first_node = dom_node[0]
                    first_node.replace_with(BeautifulSoup(target_text, "html.parser"))
                    for rem in dom_node[1:]:
                        rem.extract()
                else:
                    last_node_str = dom_node[-1]
                    target_tag = soup.new_tag("p")
                    target_tag["class"] = AttributeValueList([_TARGET_CSS_CLASS])
                    target_tag.append(BeautifulSoup(target_text, "html.parser"))
                    last_node_str.insert_after(target_tag)
            injected += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        # The source document is copied into the deliverable largely verbatim,
        # so it goes through the same source scrub the EPUB members get: target
        # text is already sanitized, but a dirty *source* would otherwise ship
        # <script>, onload= handlers and javascript: links to the reader.
        try:
            tmp_path.write_text(scrub_source_document(str(soup)), encoding="utf-8")
            tmp_path.replace(output_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        if injected == 0:
            logger.warning(
                "HTML render injected 0 translations (empty blocks?)",
            )
        return output_path

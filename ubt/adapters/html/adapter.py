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

from ubt.adapters.base import BILINGUAL_TARGET_CLASS, BaseDocumentAdapter, decode_markup
from ubt.adapters.epub.adapter import (
    BLOCK_TAGS,
    determine_flow_id,
    is_leaf_block,
    take_preserved_inline_children,
    wrap_nested_direct_blocks,
)
from ubt.adapters.unresolved import failure_note, is_unresolved
from ubt.core.cleaners.html_sanitizer import sanitize_html_fragment, scrub_source_document
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BlockType,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    IRBlock,
)
from ubt.core.ir.serializer import compute_file_sha256_cached

logger = logging.getLogger(__name__)

# Marker class added to injected bilingual target nodes.
_TARGET_CSS_CLASS = BILINGUAL_TARGET_CLASS

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


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

    def _leaf_blocks(self, soup: BeautifulSoup) -> list[Tag]:
        """Ordered leaf blocks of the document body (EPUB-compatible algorithm)."""
        wrap_nested_direct_blocks(soup)
        body = soup.body or soup
        block_names = set(BLOCK_TAGS)
        return [t for t in body.find_all(BLOCK_TAGS) if is_leaf_block(t, block_names)]

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Extract a single-chapter manifest titled from <title>."""
        # Off-loop: soup parse and sha computation run in thread pool to prevent blocking the event loop on large HTML documents.
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
        """Stream the document as one ChapterIR of leaf blocks."""
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
        """Build the leaf blocks (synchronous: runs in a worker thread)."""
        blocks: list[IRBlock] = []
        global_spine = 1

        for leaf_idx, leaf in enumerate(self._leaf_blocks(soup)):
            text = leaf.get_text(" ", strip=True)
            if not text:
                continue

            tag_name = leaf.name or ""
            if tag_name in _HEADING_TAGS:
                b_type = BlockType.HEADING
            elif tag_name in ("pre", "code", "tt"):
                # Only a wholly-code leaf block is CODE: inline <code> inside a
                # narrative paragraph must not make the whole block CODE.
                b_type = BlockType.CODE
            else:
                b_type = BlockType.NARRATIVE

            blocks.append(
                IRBlock(
                    id=f"{chapter.chapter_id}#p{leaf_idx:05d}",
                    flow_id=determine_flow_id(leaf),
                    spine_index=global_spine,
                    block_type=b_type,
                    source_text=text,
                    skip_translate=bool(b_type == BlockType.CODE),
                )
            )
            global_spine += 1

        return blocks

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
        leaves = self._leaf_blocks(soup)

        by_id = {b.id: b for b in blocks}

        if bilingual_mode is None and manifest and manifest.run:
            bilingual_mode = manifest.run.bilingual_mode

        is_monolingual = bilingual_mode in ("target", "monolingual")
        is_source_only = bilingual_mode == "source"

        injected = 0
        if not is_source_only:
            for leaf_idx, leaf in enumerate(leaves):
                block = by_id.get(f"{manifest.chapters[0].chapter_id}#p{leaf_idx:05d}")
                if block is None or block.skip_translate or not block.target_text:
                    continue

                target_text = sanitize_html_fragment(str(block.target_text))
                if not target_text:
                    continue
                # A block whose draft never passed the quality gates must stay
                # *visible and labelled* in every deliverable (ubt.adapters.
                # unresolved). EPUB/HTML used to inject the bare machine draft as
                # if it were finished, so a reader could not tell it apart from an
                # approved translation. The note becomes the first paragraph.
                if is_unresolved(block.status):
                    target_text = sanitize_html_fragment(
                        f"{failure_note(block.status)}\n\n{block.target_text}"
                    )
                    if not target_text:
                        continue

                if is_monolingual:
                    # Split on blank lines like the bilingual branch: emitting the
                    # whole fragment let HTML collapse the paragraph breaks.
                    paras = [p.strip() for p in target_text.split("\n\n") if p.strip()] or [
                        target_text
                    ]
                    is_internal_child = leaf.name in ("td", "th", "li")
                    # Inline media/anchors must survive the text replacement (an
                    # approved DOCX parallel keeps graphic runs + hyperlinks).
                    preserved_inline = take_preserved_inline_children(leaf)
                    leaf.clear()
                    last_node: Tag = leaf
                    for i, p_text in enumerate(paras):
                        parsed_fragment = BeautifulSoup(p_text, "html.parser")
                        if i == 0:
                            container: Tag = leaf
                        elif is_internal_child:
                            # Inside a cell / list item extra paragraphs nest as a
                            # <div>; a sibling <li> would add list items.
                            container = soup.new_tag("div")
                        else:
                            container = soup.new_tag(leaf.name if leaf.name else "p")
                        for child in list(parsed_fragment.contents):
                            container.append(child)
                        if i > 0:
                            if is_internal_child:
                                leaf.append(container)
                            else:
                                last_node.insert_after(container)
                        last_node = container
                    for node in preserved_inline:
                        leaf.append(node)
                else:
                    # Inside a table cell or list item, append a <div> *inside* the element
                    # instead of a sibling <td>/<th>/<li> — a sibling cell would double the column count
                    # and a sibling <li> would add list items.
                    is_internal_child = leaf.name in ("td", "th", "li")
                    source_classes = list(leaf.get("class") or [])
                    target_classes = source_classes + [_TARGET_CSS_CLASS]

                    if "\n\n" in target_text:
                        paras = [p.strip() for p in target_text.split("\n\n") if p.strip()]
                        if is_internal_child:
                            for p_text in paras:
                                target_tag = soup.new_tag("div")
                                target_tag["class"] = AttributeValueList(target_classes)
                                parsed_fragment = BeautifulSoup(p_text, "html.parser")
                                for child in list(parsed_fragment.contents):
                                    target_tag.append(child)
                                leaf.append(target_tag)
                        else:
                            last_node = leaf
                            for p_text in paras:
                                tag_name = leaf.name if leaf.name else "p"
                                target_tag = soup.new_tag(tag_name)
                                target_tag["class"] = AttributeValueList(target_classes)
                                parsed_fragment = BeautifulSoup(p_text, "html.parser")
                                for child in list(parsed_fragment.contents):
                                    target_tag.append(child)
                                last_node.insert_after(target_tag)
                                last_node = target_tag
                    else:
                        if is_internal_child:
                            target_tag = soup.new_tag("div")
                        else:
                            tag_name = leaf.name if leaf.name else "p"
                            target_tag = soup.new_tag(tag_name)
                        target_tag["class"] = AttributeValueList(target_classes)
                        parsed_fragment = BeautifulSoup(target_text, "html.parser")
                        for child in list(parsed_fragment.contents):
                            target_tag.append(child)
                        if is_internal_child:
                            leaf.append(target_tag)
                        else:
                            leaf.insert_after(target_tag)
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

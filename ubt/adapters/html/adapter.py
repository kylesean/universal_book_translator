"""HTML document adapter: leaf-block extraction with bilingual DOM injection.

Parses standalone .html/.htm with the same shared markup walker EPUB uses
(:func:`ubt.analyze.reader_html.walk_markup_elements`), so an HTML file gets
exactly the same block semantics as EPUB spine items.

Rendering is style-preserving by construction: the original document is
re-parsed and translations injected as siblings, so source markup,
stylesheets and scripts are untouched.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from ubt.adapters.base import (
    BILINGUAL_TARGET_CLASS,
    BaseDocumentAdapter,
    decode_markup,
)
from ubt.adapters.dom_inject import inject_translated_block
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

    def _render_blocks_sync(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
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

            # The shared injection engine (EPUB's twin loop lived here until the
            # two drifted apart in sanitization and counting).
            injected += inject_translated_block(
                soup,
                elem,
                dom_node,
                target_text,
                is_monolingual=is_monolingual,
                target_css_class=_TARGET_CSS_CLASS,
            )

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

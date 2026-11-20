"""HTML inspection adapter: leaf-block extraction for read-only audits.

Parses standalone .html/.htm with the same shared markup walker EPUB uses
(:func:`ubt.analyze.reader_html.walk_markup_elements`), so an HTML file gets
exactly the same block semantics as EPUB spine items. Translation and delivery
are owned by :class:`ubt.codecs.dom.DOMCodec`; this adapter is only reached by
the explicit inspection path.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from bs4 import BeautifulSoup

from ubt.adapters.base import BaseDocumentAdapter, decode_markup
from ubt.analyze.assemble import number
from ubt.analyze.reader_html import walk_markup_elements
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BookManifest,
    ChapterIR,
    ChapterMeta,
    IRBlock,
)
from ubt.core.ir.serializer import compute_file_sha256_cached

logger = logging.getLogger(__name__)

#: Largest HTML file this adapter will read into memory. DOCX and EPUB already
#: have package-size guards; a bare .html file had none, so a multi-hundred-MB
#: page was decoded whole and expanded into a DOM (typically ~10x the bytes)
#: with nothing to stop it.
_MAX_HTML_BYTES = 64 * 1024 * 1024


class HTMLAdapter(BaseDocumentAdapter):
    """Adapter for standalone HTML documents (.html/.htm) with DOM injection."""

    output_suffixes = frozenset({".html", ".htm"})

    def _load_soup(self, input_path: Path) -> BeautifulSoup:
        if not input_path.exists():
            raise DocumentParseError(f"HTML file not found: {input_path}")
        try:
            size = input_path.stat().st_size
            if size > _MAX_HTML_BYTES:
                raise DocumentParseError(
                    f"HTML document {input_path} is {size // (1024 * 1024)} MiB, over the "
                    f"{_MAX_HTML_BYTES // (1024 * 1024)} MiB limit for a single HTML file; "
                    "split it or convert it to EPUB."
                )
            raw = decode_markup(input_path.read_bytes())
            return BeautifulSoup(raw, "html.parser")
        except DocumentParseError:
            raise
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

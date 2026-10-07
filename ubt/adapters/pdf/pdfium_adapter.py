"""Born-digital PDF fast path: the native reader + pypdfium2 geometry.

pypdfium2 is Apache-2.0 (the bundled PDFium binary is BSD-3-Clause), which
keeps the repository's Zero-AGPL delivery guarantee intact. The adapter
reuses the Docling mainline delivery pipeline (Typst reconstruction,
facing-page interleaving, rigid typesetting) so only the extraction leg
differs.

Extraction is the native reader (:mod:`ubt.analyze.reader_pdf`): it owns
reading order, paragraph grouping, heading detection and page furniture
(chrome, listings, math debris) from the page's real typography and geometry,
and emits a typed ``Document``, which the bridge projects back into the
pipeline's ``IRBlock`` contract. No flow repair runs on this path -- the
reader's output is already typed. (The plain-text pdf_oxide fallback, which has
no such model, still repairs its flow.)

Routing: ``UBT_PDF_ENGINE=auto`` picks this engine for clean single-column
born-digital PDFs (see :mod:`ubt.adapters.pdf.engine_selector`); scans and
multi-column layouts stay on Docling. ``UBT_PDF_ENGINE=pdfium`` forces the
fast path explicitly.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.adapters.base import BasePDFEngineAdapter
from ubt.adapters.pdf.docling_adapter import _PDFRenderStackMixin
from ubt.adapters.pdf.pdfium_gate import pdfium_serialized
from ubt.core.ir.models import BookManifest, ChapterIR, IRBlock

if TYPE_CHECKING:
    from ubt.cache.store import CacheStore

logger = logging.getLogger(__name__)


def _block_page_in_range(b: IRBlock, first: int, last: int) -> bool:
    """True when a block's page (from bbox or provenance) falls within [first, last]."""
    if b.bbox is not None:
        return first <= b.bbox.page <= last
    page = b.provenance.get("page")
    if page is None:
        page = b.provenance.get("source_page")
    if isinstance(page, int):
        return first <= page <= last
    if isinstance(page, str) and page.isdigit():
        return first <= int(page) <= last
    return False


def extract_blocks_with_pdfium(
    path: Path,
    page_range: tuple[int, int] | None = None,
    store: CacheStore | None = None,
) -> list[IRBlock]:
    """Read a born-digital PDF into typed blocks via the native reader.

    The reader (:mod:`ubt.analyze.reader_pdf`) owns reading order, paragraph
    grouping and heading detection -- one documented rule set over the page's
    real typography -- and emits a typed ``Document``; this projects that back
    into the pipeline's ``IRBlock`` contract. The blocks are re-numbered into
    the pipeline's own id space (``pdf_main#bNNNN``) so the ledger, the contract
    and the companion views keep joining on the same ids. ``pdfium_serialized``
    is unnecessary here: every pdfium call inside ``read_pdf`` is already
    serialized by the gate, and the lock is re-entrant.

    ``store`` is the content-addressed analyze cache (content-addressed cache layer): the
    read is a pure function of the file, page range and reader source, so a
    re-run reuses it. ``None`` computes straight through.
    """
    from ubt.adapters.pdf.analysis_cache import cached_blocks
    from ubt.analyze.reader_pdf import read_pdf_blocks

    def _read() -> list[IRBlock]:
        pages: range | None = None
        if page_range is not None:
            pages = range(max(1, page_range[0]), page_range[1] + 1)
        from ubt.core.ir.continuation import fuse_continuation_blocks

        blocks = read_pdf_blocks(path, pages=pages)
        blocks = fuse_continuation_blocks(blocks)
        for index, block in enumerate(blocks, start=1):
            block.set_id(f"pdf_main#b{index:04d}")
            block.set_spine_index(index)
            # The reader keeps a TOC row's page number on the element (the
            # bridge preserves it); move it into provenance so it survives the
            # ledger round-trip and reaches the TOC-aware renderer.
            toc_page = getattr(block.element, "toc_page", "")
            if toc_page:
                block.provenance["toc_entry"] = True
                block.provenance["toc_page"] = toc_page
        return blocks

    return cached_blocks(store, path=path, page_range=page_range, compute=_read)


class PDFiumAdapter(_PDFRenderStackMixin, BasePDFEngineAdapter):
    """Born-digital PDF fast path: CPU-cheap pypdfium2 extraction, zero model downloads.

    Shares the Docling-mainline render stack (publication Typst, rigid
    typesetting, alternating bilingual interleaving) via the render mixin —
    only extraction is replaced with geometric text harvesting.
    """

    #: Content-addressed analyze cache (set in apply_config).
    analysis_cache: CacheStore | None = None

    def __init__(self, font_family: str | None = None) -> None:
        self._init_render_stack(font_family=font_family)

    @property
    def engine_name(self) -> str:
        return "pdfium"

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Manifest identical to the Docling mainline except the engine tag."""
        from ubt.adapters.pdf.docling_parser import extract_manifest as parser_extract_manifest

        manifest = await asyncio.to_thread(
            parser_extract_manifest, input_path, is_docling_installed=False
        )
        return manifest.model_copy(
            update={"metadata": {**manifest.metadata, "pdf_parser_engine": "pdfium"}}
        )

    async def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]:
        """Stream PDF contents as a single ChapterIR partition."""
        path = Path(input_path)
        manifest = await self.extract_manifest(path)
        chapter_meta = manifest.chapters[0]

        loop = asyncio.get_running_loop()
        blocks = await loop.run_in_executor(None, self._extract_blocks_sync, path, None)

        if pages:
            blocks = [b for b in blocks if _block_page_in_range(b, min(pages), max(pages))]

        yield ChapterIR(
            doc_id=manifest.doc_id,
            chapter_id=chapter_meta.chapter_id,
            title=chapter_meta.title,
            spine_index=0,
            blocks=blocks,
        )

    def _extract_blocks_sync(
        self, path: Path, page_range: tuple[int, int] | None = None
    ) -> list[IRBlock]:
        """pypdfium2 geometric extraction; degrade to plain pdf_oxide on failure.

        The native reader owns classification -- reading order, grouping,
        headings, chrome, listings, math debris -- so its output needs no flow
        repair. The plain-text pdf_oxide fallback types directly from text
        content (the shared analyzer rules); it too has no flow-repair pass.
        """
        try:
            blocks = self._extract_with_pdfium(path, page_range)
        except Exception as exc:
            logger.warning(
                "pypdfium2 extraction failed on '%s' (%s); falling back to the "
                "plain pdf_oxide text extractor",
                path.name,
                exc,
            )
            blocks = self._extract_with_oxide(path)
        if page_range is not None:
            first, last = page_range
            blocks = [b for b in blocks if _block_page_in_range(b, first, last)]
        return blocks

    @pdfium_serialized
    def _extract_with_pdfium(
        self, path: Path, page_range: tuple[int, int] | None = None
    ) -> list[IRBlock]:
        """Geometric line harvesting with column-aware reading order."""
        return extract_blocks_with_pdfium(path, page_range, store=self.analysis_cache)

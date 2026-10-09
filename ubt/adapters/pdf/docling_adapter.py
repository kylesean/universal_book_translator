"""IBM Docling + Typst modern PDF adapter.

Permissively licensed PDF translation and reconstruction pipeline.
The adapter orchestrates three extracted modules:

- :mod:`ubt.adapters.pdf.docling_parser` — PDF → IRBlocks (Docling, pdf_oxide
  fallback, textless-page VLM tiering, page-kind profiling);
- :mod:`ubt.adapters.pdf.docling_blocks` — pure block shaping;
- :mod:`ubt.adapters.pdf.docling_render` — IRBlocks → delivered artifact
  (LayerCompositor source-canvas composition, bilingual interleaving, diagrams).

It keeps the module-level ``_docling_symbols`` / ``_has_accelerator`` import
seams and the tested private-method surface as thin delegates so existing
patch points and call sites keep working.
"""

from __future__ import annotations

import asyncio
import gc
import importlib
import importlib.util
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ubt.adapters.base import BasePDFEngineAdapter
from ubt.adapters.pdf.alternator import BilingualAlternator
from ubt.adapters.pdf.docling_parser import (
    annotate_page_kinds,
    extract_with_docling,
    extract_with_oxide,
    resolve_formula_enrichment,
    vlm_fallback_missing_pages,
)
from ubt.adapters.pdf.docling_parser import (
    extract_manifest as parser_extract_manifest,
)
from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
from ubt.adapters.pdf.font_metrics import sanitize_font_family
from ubt.adapters.pdf.page_chunking import compute_pdf_page_chunks, get_pdf_page_chunk_size
from ubt.core.env import has_accelerator as _has_accelerator
from ubt.core.ir.models import BookManifest, ChapterIR, ChapterMeta, IRBlock
from ubt.core.ir.render_plan import RenderOutcome, RenderPlan

if TYPE_CHECKING:
    from ubt.cache.store import CacheStore
    from ubt.core.ports import AdapterRuntimeConfig

logger = logging.getLogger(__name__)


def _contiguous_page_range(pages: set[int] | None) -> tuple[int, int] | None:
    """Collapse a 1-based page set into Docling's inclusive ``(start, end)``.

    Docling accepts exactly one contiguous range; a non-contiguous selection
    returns ``None`` so the caller parses the full PDF and the ingest stage
    filters the blocks afterwards. Empty or absent selections also return
    ``None`` (no restriction).
    """
    if not pages:
        return None
    start, end = min(pages), max(pages)
    if len(pages) != end - start + 1:
        return None
    return (start, end)


def _docling_symbols() -> tuple[Any, Any, Any, Any]:
    """Import hook for the Docling converter symbols (seam for tests).

    Importing ``docling`` pulls heavy native dependencies (torch/transformers);
    tests stub this hook instead of swapping ``sys.modules`` entries for the
    real packages, avoiding unsafe sys.modules manipulation of native dependencies.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    return InputFormat, PdfPipelineOptions, DocumentConverter, PdfFormatOption


class _PDFRenderStackMixin:
    """Shared PDF render infrastructure (DoclingRenderStrategy delegation).

    Both Docling and PDFium adapters compose onto the source canvas through
    the same ``DoclingRenderStrategy`` — only extraction differs. This mixin
    owns the render-stack wiring so the two adapters are siblings under
    ``BasePDFEngineAdapter`` rather than PDFium inheriting Docling.
    """

    alternator: BilingualAlternator
    last_render_skips: list[tuple[str, str]]
    last_render_flags: list[tuple[str, str]]
    last_render_outcome: RenderOutcome | None
    _renderer: DoclingRenderStrategy
    _font_family: str | None

    def _init_render_stack(
        self,
        alternator: BilingualAlternator | None = None,
        font_family: str | None = None,
    ) -> None:
        self.alternator = alternator or BilingualAlternator()
        self.allow_page_upload = False
        self._font_family = None
        self.last_render_skips = []
        self.last_render_flags = []
        self.last_render_outcome = None
        self._renderer = DoclingRenderStrategy(
            alternator=self.alternator,
            font_family=None,
        )
        self.font_family = font_family

    @property
    def font_family(self) -> str | None:
        return self._font_family

    @font_family.setter
    def font_family(self, value: str | None) -> None:
        clean = sanitize_font_family(value)
        self._font_family = clean
        self._renderer.font_family = clean

    async def render_blocks(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        render_plan: RenderPlan | None = None,
        **kwargs: Any,
    ) -> Path:
        out_path = await self._renderer.render_blocks(
            manifest,
            blocks,
            target_lang,
            output_path,
            bilingual_mode,
            render_plan=render_plan,
        )
        self.last_render_skips = list(self._renderer.last_render_skips)
        self.last_render_flags = list(self._renderer.last_render_flags)
        self.last_render_outcome = self._renderer.last_outcome
        return out_path

    def _extract_with_oxide(self, path: Path) -> list[IRBlock]:
        from ubt.core.ir.continuation import fuse_continuation_blocks

        blocks = extract_with_oxide(path)
        return fuse_continuation_blocks(blocks)


class DoclingPDFAdapter(_PDFRenderStackMixin, BasePDFEngineAdapter):
    """PDF engine adapter: IBM Docling semantic ingestion with Typst/oxide delivery."""

    #: Content-addressed analyze cache (content-addressed cache layer), set in apply_config.
    #: The Docling layout+formula pass is the heaviest step of a Docling run.
    analysis_cache: CacheStore | None = None

    def __init__(
        self,
        alternator: BilingualAlternator | None = None,
        ocr_mode: str = "auto",
        ocr_endpoint: str | None = None,
        ocr_api_key: str | None = None,
        ocr_model: str | None = None,
        formula_enrichment: str = "auto",
        formula_render: str = "witness",
        font_family: str | None = None,
        allow_page_upload: bool = False,
    ) -> None:
        self._init_render_stack(alternator, font_family)
        self.ocr_mode = ocr_mode
        self.ocr_endpoint = ocr_endpoint
        self.ocr_api_key = ocr_api_key
        self.ocr_model = ocr_model
        self.formula_enrichment = formula_enrichment
        self.formula_render = formula_render
        self.allow_page_upload = allow_page_upload

    def apply_config(self, runtime_config: AdapterRuntimeConfig) -> None:
        """Take the OCR / formula / render knobs the pipeline resolved for this run.

        These fields are owned here, so the assignment is unconditional (no
        ``hasattr`` guard). ``ocr_endpoint`` / ``ocr_api_key`` / ``ocr_model``
        stay the adapter's constructor values unless the config actually
        provides them.
        """
        self.ocr_mode = runtime_config.ocr_mode
        if runtime_config.ocr_endpoint:
            self.ocr_endpoint = runtime_config.ocr_endpoint
        if runtime_config.ocr_api_key:
            self.ocr_api_key = runtime_config.ocr_api_key
        if runtime_config.ocr_model:
            self.ocr_model = runtime_config.ocr_model
        self.formula_enrichment = runtime_config.formula_enrichment
        self.formula_render = runtime_config.formula_render
        # The page-image egress gate comes from the resolved config, not a
        # second ``os.environ`` read in the parser.
        self.allow_page_upload = runtime_config.allow_page_upload
        # Setter mirrors the sanitized name onto the render strategy.
        self.font_family = runtime_config.font_family
        # The analyze cache (content-addressed cache layer): the Docling layout+formula pass
        # is a pure function of the file, page range, enrichment policy and
        # parser code, so a resumed or re-run job reuses the extraction.
        from ubt.cache.store import DiskCacheStore

        self.analysis_cache = (
            DiskCacheStore(runtime_config.cache_dir) if runtime_config.cache_dir else None
        )

    def close(self) -> None:
        """Release adapter-owned subprocesses / resources."""
        pass

    @property
    def engine_name(self) -> str:
        return "docling"

    def is_docling_installed(self) -> bool:
        """Check if docling package is available in the current Python environment."""
        return importlib.util.find_spec("docling") is not None

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Extract lightweight PDF book manifest (delegates to docling_parser)."""
        # Off-loop: compute doc_id sha in thread pool to avoid blocking the event loop on multi-megabyte PDFs.
        return await asyncio.to_thread(
            parser_extract_manifest, input_path, is_docling_installed=self.is_docling_installed()
        )

    async def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]:
        """Stream PDF contents as structured ChapterIR partitions.

        Partitions long PDFs into out-of-core page chunks (e.g. 50 pages)
        to bound resident memory (RSS) to O(1) during full document ingest.
        AST objects and line caches are evicted between chunks.
        """
        path = Path(input_path)
        manifest = await self.extract_manifest(path)
        chunk_size = get_pdf_page_chunk_size()

        total_pages = int(manifest.metadata.get("page_count", 0))
        if total_pages <= 0:
            try:
                from ubt.adapters.pdf.short_doc import probe_pdf_pages

                total_pages, _ = probe_pdf_pages(path)
            except Exception:
                total_pages = 0

        page_chunks = compute_pdf_page_chunks(total_pages, pages=pages, chunk_size=chunk_size)
        loop = asyncio.get_running_loop()
        global_block_idx = 1

        for chunk_idx, chunk_pages in enumerate(page_chunks):
            chunk_range: tuple[int, int] | None = (
                (min(chunk_pages), max(chunk_pages)) if chunk_pages else None
            )

            chapter_meta = (
                manifest.chapters[chunk_idx]
                if chunk_idx < len(manifest.chapters)
                else ChapterMeta(
                    chapter_id=f"c{chunk_idx + 1:04d}",
                    title=f"Pages {chunk_range[0]}-{chunk_range[1]}" if chunk_range else "Main",
                    spine_index=chunk_idx + 1,
                    source_file=path.name,
                )
            )

            blocks = await loop.run_in_executor(None, self._extract_blocks_sync, path, chunk_range)

            # Filter blocks to requested pages if non-contiguous inside chunk
            if pages and chunk_pages:
                chunk_set = set(chunk_pages)
                blocks = [b for b in blocks if b.bbox is None or b.bbox.page in chunk_set]

            # Textless-page VLM fallback (default OFF via UBT_VLM_SCAN_FALLBACK).
            blocks = await asyncio.to_thread(
                self._vlm_fallback_missing_pages,
                path,
                blocks,
                self.ocr_mode,
                self.ocr_endpoint,
                self.ocr_api_key,
                self.ocr_model,
                chunk_range,
                self.allow_page_upload,
            )

            # Re-key block IDs and spine indices so they are globally unique and monotonic across chunks
            if len(page_chunks) > 1:
                for b in blocks:
                    b.set_id(f"pdf_main#b{global_block_idx:04d}")
                    b.set_spine_index(global_block_idx)
                    global_block_idx += 1
            else:
                global_block_idx += len(blocks)

            # Per-page profile kinds ride into block provenance (render routing)
            page_kinds = await loop.run_in_executor(None, self._annotate_page_kinds, path, blocks)

            metadata: dict[str, Any] = {}
            if page_kinds:
                metadata["page_kinds"] = page_kinds
            if chunk_range:
                metadata["page_range"] = list(chunk_range)

            chapter_ir = ChapterIR(
                doc_id=manifest.doc_id,
                chapter_id=chapter_meta.chapter_id,
                title=chapter_meta.title,
                spine_index=chapter_meta.spine_index,
                blocks=blocks,
                metadata=metadata,
            )
            yield chapter_ir

            # Out-of-core memory eviction: explicitly reclaim AST objects and line caches
            del blocks
            del chapter_ir
            gc.collect()

    @staticmethod
    def _annotate_page_kinds(path: Path, blocks: list[IRBlock]) -> dict[int, str]:
        """Stamp per-page provenance onto blocks (delegates to docling_parser)."""
        return annotate_page_kinds(path, blocks)

    def _extract_blocks_sync(
        self, path: Path, page_range: tuple[int, int] | None = None
    ) -> list[IRBlock]:
        """Extract blocks synchronously via Docling or geometric pypdfium2/pdf_oxide fallback."""
        if self.is_docling_installed():
            from ubt.adapters.pdf.analysis_cache import cached_blocks, docling_identity

            # The Docling pass is the run's heaviest step; cache it on the exact
            # inputs (file, page range, enrichment policy, parser code) via the content-addressed
            # cache layer. The enrichment flag is output-bearing, so it is in the key.
            enrich = self._resolve_formula_enrichment(path)
            return cached_blocks(
                self.analysis_cache,
                path=path,
                page_range=page_range,
                identity=docling_identity(),
                extra=f"enrich={enrich}",
                compute=lambda: self._extract_with_docling(path, page_range),
            )

        # Fast geometric extraction fallback via pypdfium2 (MIT/Apache-2.0, Zero-PyTorch)
        try:
            from ubt.adapters.pdf.pdfium_adapter import (
                _block_page_in_range,
                extract_blocks_with_pdfium,
            )

            logger.info(
                "Docling not installed; using lightweight geometric extraction (pypdfium2) for '%s'.",
                path.name,
            )
            blocks = extract_blocks_with_pdfium(path, page_range)
            if page_range is not None:
                first, last = page_range
                blocks = [b for b in blocks if _block_page_in_range(b, first, last)]
            return blocks
        except Exception as exc:
            logger.warning(
                "Geometric pypdfium2 extraction failed (%s); falling back to plain-text pdf_oxide",
                exc,
            )
            return self._extract_with_oxide(path)

    @classmethod
    def _vlm_fallback_missing_pages(
        cls,
        path: Path,
        blocks: list[IRBlock],
        ocr_mode: str | None = None,
        ocr_endpoint: str | None = None,
        ocr_api_key: str | None = None,
        ocr_model: str | None = None,
        page_range: tuple[int, int] | None = None,
        allow_page_upload: bool | None = None,
    ) -> list[IRBlock]:
        """Transcribe missing/weak pages via vlm/ core (delegates to docling_parser)."""
        return vlm_fallback_missing_pages(
            path,
            blocks,
            ocr_mode,
            ocr_endpoint,
            ocr_api_key,
            ocr_model,
            page_range,
            allow_page_upload,
        )

    def _resolve_formula_enrichment(self, path: Path | None = None) -> bool:
        """Resolve effective formula enrichment policy (delegates to docling_parser)."""
        return resolve_formula_enrichment(
            self.formula_enrichment,
            self.formula_render,
            _has_accelerator,
            path=path,
        )

    def _extract_with_docling(
        self, path: Path, page_range: tuple[int, int] | None = None
    ) -> list[IRBlock]:
        """Extract structured blocks using IBM Docling (delegates to docling_parser)."""
        from ubt.adapters.pdf.docling_crosscheck import (
            cross_check_blocks_with_pdfium,
            repair_math_symbols_with_lines,
            repair_missing_spaces_with_lines,
        )
        from ubt.core.ir.continuation import fuse_continuation_blocks

        blocks = extract_with_docling(
            path,
            page_range,
            symbols=_docling_symbols,
            enrich=self._resolve_formula_enrichment(path),
        )
        # Docling mis-maps TeX math symbols (e.g. '=' -> '∅', '+' -> '⊕', '−' -> 'ϒ');
        # the page's own pdfium lines restore ground-truth math glyphs first.
        blocks = repair_math_symbols_with_lines(blocks, path)
        # Docling's line join drops inter-word spaces ("A Sandbox" -> "ASandbox");
        # the page's own pdfium lines restore them before anything is translated.
        blocks = repair_missing_spaces_with_lines(blocks, path)
        blocks = cross_check_blocks_with_pdfium(blocks, path)
        return fuse_continuation_blocks(blocks)

"""IBM Docling + Typst modern PDF adapter.

Permissively licensed PDF translation and reconstruction pipeline.
The adapter orchestrates three extracted modules:

- :mod:`ubt.adapters.pdf.docling_parser` — PDF → IRBlocks (Docling, pdf_oxide
  fallback, textless-page VLM tiering, page-kind profiling);
- :mod:`ubt.adapters.pdf.docling_blocks` — pure block shaping;
- :mod:`ubt.adapters.pdf.docling_render` — IRBlocks → delivered artifact
  (Typst typesetting, bilingual per-block pairing, diagrams).

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


def _docling_symbols() -> tuple[Any, Any, Any, Any, Any]:
    """Import hook for the Docling converter symbols (seam for tests).

    Importing ``docling`` pulls heavy native dependencies (torch/transformers);
    tests stub this hook instead of swapping ``sys.modules`` entries for the
    real packages, avoiding unsafe sys.modules manipulation of native dependencies.

    ``CodeFormulaVlmOptions`` is returned so the parser can tune the formula
    crop margin through Docling's own preset API (docling#4548) instead of
    hardcoding it: the preset is what its documentation recommends.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        CodeFormulaVlmOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    return (
        InputFormat,
        PdfPipelineOptions,
        DocumentConverter,
        PdfFormatOption,
        CodeFormulaVlmOptions,
    )


class _PDFRenderStackMixin:
    """Shared PDF render infrastructure (DoclingRenderStrategy delegation).

    Both Docling and PDFium adapters rebuild the document through the same
    ``DoclingRenderStrategy`` — only extraction differs. This mixin
    owns the render-stack wiring so the two adapters are siblings under
    ``BasePDFEngineAdapter`` rather than PDFium inheriting Docling.
    """

    last_render_skips: list[tuple[str, str]]
    last_render_flags: list[tuple[str, str]]
    last_render_outcome: RenderOutcome | None
    _renderer: DoclingRenderStrategy
    _font_family: str | None
    #: Engine hooks ``parse_stream`` calls; each adapter class provides them.
    ocr_mode: str | None
    ocr_endpoint: str | None
    ocr_api_key: str | None
    ocr_model: str | None
    allow_page_upload: bool

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        raise NotImplementedError

    def _extract_blocks_sync(
        self, path: Path, page_range: tuple[int, int] | None = None
    ) -> list[IRBlock]:
        raise NotImplementedError

    @staticmethod
    def _annotate_page_kinds(path: Path, blocks: list[IRBlock]) -> dict[int, str]:
        raise NotImplementedError

    def _init_render_stack(
        self,
        font_family: str | None = None,
    ) -> None:
        self.allow_page_upload = False
        self._font_family = None
        self.last_render_skips = []
        self.last_render_flags: list[tuple[str, str]] = []
        self.last_render_outcome = None
        self._renderer = DoclingRenderStrategy(
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
        # The per-asset reconstruction verdicts ride the same side channel
        # (the strategy copies renderer.table_reconstructions after each
        # render). Without this line the base property's lazy [] answered
        # every export and the contract's RECONSTRUCTED (verified/corrupt)
        # branch was unreachable again — the exact regression P0-6 fixed
        # once already.
        self.last_render_reconstructions = list(self._renderer.last_render_reconstructions)
        self.last_render_outcome = self._renderer.last_outcome
        return out_path

    def _extract_with_oxide(self, path: Path) -> list[IRBlock]:
        from ubt.core.ir.continuation import fuse_continuation_blocks

        blocks = extract_with_oxide(path)
        return fuse_continuation_blocks(blocks)

    async def _post_extract_blocks(
        self,
        path: Path,
        blocks: list[IRBlock],
        page_range: tuple[int, int] | None,
    ) -> list[IRBlock]:
        """Engine-specific per-chunk enrichment; default is a no-op.

        Async so an override can off-load blocking work: Docling's textless-page
        VLM fallback (default OFF via ``UBT_VLM_SCAN_FALLBACK``) does network
        IO and runs in a thread; pdfium's native reader has no such pass.
        """
        return blocks

    def _evict_after_chunk(self, blocks: list[IRBlock], chapter_ir: ChapterIR) -> None:
        """Out-of-core eviction after a chunk is consumed (both engines).

        Explicitly reclaim AST objects and line caches; ``gc.collect()`` forces
        the collection rather than waiting for a generation threshold, which is
        what keeps RSS flat across a long book. Identical for both engines —
        the hook exists so an engine with different eviction needs can override
        it, not because the two differ today.
        """
        del blocks
        del chapter_ir
        gc.collect()

    async def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]:
        """Stream PDF contents as structured ChapterIR partitions.

        Partitions long PDFs into out-of-core page chunks (e.g. 50 pages)
        to bound resident memory (RSS) to O(1) during full document ingest.
        AST objects and line caches are evicted between chunks.

        Shared by both PDF engines: the loop below is identical for Docling and
        pdfium (2026-10-14 review, W3-8 — the two copies had drifted 83% byte
        for byte). The per-engine differences live in the two hooks above:
        extraction (``_extract_blocks_sync``), post-extract enrichment
        (``_post_extract_blocks``), page-kind annotation
        (``_annotate_page_kinds``) and eviction (``_evict_after_chunk``).
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

            blocks = await self._post_extract_blocks(path, blocks, chunk_range)

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
            self._evict_after_chunk(blocks, chapter_ir)


class DoclingPDFAdapter(_PDFRenderStackMixin, BasePDFEngineAdapter):
    """PDF engine adapter: IBM Docling semantic ingestion with Typst/oxide delivery."""

    #: Content-addressed analyze cache (content-addressed cache layer), set in apply_config.
    #: The Docling layout+formula pass is the heaviest step of a Docling run.
    analysis_cache: CacheStore | None = None

    def __init__(
        self,
        ocr_mode: str = "auto",
        ocr_endpoint: str | None = None,
        ocr_api_key: str | None = None,
        ocr_model: str | None = None,
        formula_enrichment: str = "auto",
        docling_device: str = "auto",
        formula_crop_margin: float = 0.08,
        font_family: str | None = None,
        allow_page_upload: bool = False,
    ) -> None:
        self._init_render_stack(font_family)
        self.ocr_mode = ocr_mode
        self.ocr_endpoint = ocr_endpoint
        self.ocr_api_key = ocr_api_key
        self.ocr_model = ocr_model
        self.formula_enrichment = formula_enrichment
        self.docling_device = docling_device
        self.formula_crop_margin = formula_crop_margin
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
        self.formula_crop_margin = runtime_config.formula_crop_margin
        # Whether Docling touches the GPU is a separate decision from whether it
        # runs at all: a local LLM on the same card needs Docling on CPU.
        self.docling_device = runtime_config.docling_device
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

    @staticmethod
    def _annotate_page_kinds(path: Path, blocks: list[IRBlock]) -> dict[int, str]:
        """Stamp per-page provenance onto blocks (delegates to docling_parser)."""
        return annotate_page_kinds(path, blocks)

    async def _post_extract_blocks(
        self,
        path: Path,
        blocks: list[IRBlock],
        page_range: tuple[int, int] | None,
    ) -> list[IRBlock]:
        """Textless-page VLM fallback (default OFF via UBT_VLM_SCAN_FALLBACK)."""
        return await asyncio.to_thread(
            self._vlm_fallback_missing_pages,
            path,
            blocks,
            self.ocr_mode,
            self.ocr_endpoint,
            self.ocr_api_key,
            self.ocr_model,
            page_range,
            self.allow_page_upload,
        )

    def _extract_blocks_sync(
        self, path: Path, page_range: tuple[int, int] | None = None
    ) -> list[IRBlock]:
        """Extract blocks synchronously via Docling or geometric pypdfium2/pdf_oxide fallback."""
        if self.is_docling_installed():
            from ubt.adapters.pdf.analysis_cache import cached_blocks, docling_identity

            # The Docling pass is the run's heaviest step; cache it on the exact
            # inputs (file, page range, enrichment policy, parser code) via the
            # content-addressed cache layer.
            enrich = self._resolve_formula_enrichment(path)
            if enrich:
                # A *requested* enrichment is deliberately not cached in this
                # outer block layer. Docling's ladder degrades inside
                # ``extract_with_docling`` (VLM OOM -> CPU, or enrichment failure
                # -> plain extraction); the degraded blocks carry the same
                # ``enrich=True`` key as a healthy run, so caching them here made
                # the degradation permanent — a later ``--formula-enrichment on``
                # run read the degraded blocks and never retried, and ``--fresh``
                # (which clears the Docling conversion cache, not this one) did
                # not help either. The inner conversion cache is keyed on the
                # *effective* options (``docling_cache_key`` folds the mutated
                # ``do_formula_enrichment``/device in), so it still dedupes the
                # expensive conversion; only the cheap block-shaping re-runs.
                return self._extract_with_docling(path, page_range)
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
            _has_accelerator,
            path=path,
        )

    def _extract_with_docling(
        self, path: Path, page_range: tuple[int, int] | None = None
    ) -> list[IRBlock]:
        """Extract structured blocks using IBM Docling (delegates to docling_parser)."""
        from ubt.adapters.pdf.docling_crosscheck import (
            cross_check_blocks_with_pdfium,
            repair_heading_numbers_with_lines,
            repair_inline_math_with_lines,
            repair_math_symbols_with_lines,
            repair_missing_spaces_with_lines,
        )
        from ubt.core.ir.continuation import fuse_continuation_blocks

        blocks = extract_with_docling(
            path,
            page_range,
            symbols=_docling_symbols,
            enrich=self._resolve_formula_enrichment(path),
            device=self.docling_device,
            formula_crop_margin=self.formula_crop_margin,
        )
        # Docling glues a heading's section number to the wrong end
        # ("3.1 Guidance Space" -> "Guidance Space 3.1"); the page's own pdfium
        # lines restore the correct order before the heading is translated.
        blocks = repair_heading_numbers_with_lines(blocks, path)
        # Docling strips inline math out of a sentence and piles it at the
        # paragraph end ("let be the initial state ... : h 0 h k k f R h R"); the
        # page's own pdfium lines put each token back in place before translation.
        blocks = repair_inline_math_with_lines(blocks, path)
        # Docling mis-maps TeX math symbols (e.g. '=' -> '∅', '+' -> '⊕', '−' -> 'ϒ');
        # the page's own pdfium lines restore ground-truth math glyphs first.
        blocks = repair_math_symbols_with_lines(blocks, path)
        # Docling's line join drops inter-word spaces ("A Sandbox" -> "ASandbox");
        # the page's own pdfium lines restore them before anything is translated.
        blocks = repair_missing_spaces_with_lines(blocks, path)
        blocks = cross_check_blocks_with_pdfium(blocks, path)
        return fuse_continuation_blocks(blocks)

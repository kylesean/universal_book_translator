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
import importlib
import importlib.util
import logging
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ubt.adapters.base import BasePDFEngineAdapter
from ubt.adapters.pdf.alternator import BilingualAlternator
from ubt.adapters.pdf.diagram_localizer import DiagramLocalizer
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
from ubt.core.env import has_accelerator as _has_accelerator
from ubt.core.ir.models import BookManifest, ChapterIR, IRBlock
from ubt.core.ir.render_plan import RenderOutcome, RenderPlan
from ubt.model.fidelity import Fidelity

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


class DoclingPDFAdapter(BasePDFEngineAdapter):
    """PDF engine adapter: IBM Docling semantic ingestion with Typst/oxide delivery."""

    #: Content-addressed analyze cache (content-addressed cache layer), set in apply_config.
    #: The Docling layout+formula pass is the heaviest step of a Docling run.
    analysis_cache: CacheStore | None = None

    def __init__(
        self,
        alternator: BilingualAlternator | None = None,
        diagram_localizer: DiagramLocalizer | None = None,
        ocr_mode: str = "auto",
        ocr_endpoint: str | None = None,
        ocr_api_key: str | None = None,
        ocr_model: str | None = None,
        formula_enrichment: str = "auto",
        render_engine: str = "auto",
        formula_render: str = "witness",
        font_family: str | None = None,
        allow_page_upload: bool = False,
    ) -> None:
        self.alternator = alternator or BilingualAlternator()
        self.diagram_localizer = diagram_localizer or DiagramLocalizer()
        self.ocr_mode = ocr_mode
        self.ocr_endpoint = ocr_endpoint
        self.ocr_api_key = ocr_api_key
        self.ocr_model = ocr_model
        self.formula_enrichment = formula_enrichment
        self.render_engine = render_engine
        self.formula_render = formula_render
        # Page-image egress gate. Overwritten by ``apply_config`` from the
        # resolved run config; the constructor default keeps a directly built
        # adapter closed.
        self.allow_page_upload = allow_page_upload
        self._font_family: str | None = None
        # Render skip side channel: plain ``(block_id, reason)`` pairs from
        # the most recent ``render_blocks`` call. The core reads them
        # duck-typed through ``ubt.core.ports.get_last_render_skips`` (DIP:
        # no core -> adapter import edge). Reset on every render so stale
        # skips can never leak across jobs.
        self.last_render_skips: list[tuple[str, str]] = []
        # Render outcome side channel (compiler render plan protocol): what the
        # renderer actually used (post rigid downgrade), read by export and the
        # visual gate. ``None`` until the first render. Reset on every render.
        self.last_render_outcome: RenderOutcome | None = None
        self._renderer = DoclingRenderStrategy(
            alternator=self.alternator,
            diagram_localizer=self.diagram_localizer,
            font_family=None,
        )
        # The setter is the only writer: it sanitizes once and mirrors onto
        # both engines, so any assignment (including the one below) routes
        # through it.
        self.font_family = font_family

    @property
    def font_family(self) -> str | None:
        """Publication font family, mirrored onto the render strategy.

        The pipeline pushes the configured value in through duck-typing
        (``hasattr(adapter, "font_family")``), so this must be a property: a
        plain attribute would stop at the adapter while the render strategy
        kept its own construction-time copy. Markup
        characters are dropped because the name is interpolated into
        ``#set text(font: "...")``.
        """
        return self._font_family

    @font_family.setter
    def font_family(self, value: str | None) -> None:
        clean = sanitize_font_family(value)
        self._font_family = clean
        self._renderer.font_family = clean

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
        self.render_engine = runtime_config.render_engine
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

        A contiguous ``pages`` selection is pushed down into Docling so the
        expensive layout plus formula-VLM pass only sees the requested range
        (the ingest stage still filters blocks, which also covers the
        non-contiguous fallback and engines that ignore the hint).
        """
        path = Path(input_path)
        manifest = await self.extract_manifest(path)
        chapter_meta = manifest.chapters[0]

        page_range = _contiguous_page_range(pages)
        if pages and page_range is None:
            logger.debug(
                "Page selection %s is non-contiguous; parsing the full PDF and "
                "filtering blocks after ingest",
                sorted(pages),
            )

        loop = asyncio.get_running_loop()
        blocks = await loop.run_in_executor(None, self._extract_blocks_sync, path, page_range)

        # Textless-page VLM fallback (default OFF via UBT_VLM_SCAN_FALLBACK).
        # Docling runs with do_ocr=False, so pages whose only content is an
        # image yield zero blocks; with the switch on, those pages are
        # transcribed through the vlm/ core (proofread→recognition) and
        # spliced back in page order. Late import: vlm stays optional.
        # ``to_thread``, not ``run_in_executor``: the OCR driver books its
        # tokens on the run's usage sink, a ContextVar that
        # ``loop.run_in_executor`` does NOT propagate (it does not copy the
        # caller's context), so an executor thread would record nothing and the
        # paid OCR channel would stay invisible to the bill.
        blocks = await asyncio.to_thread(
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

        # Per-page profile kinds ride into block provenance (render routing)
        # and chapter metadata. Best-effort: profiling must never break
        # parsing.
        page_kinds = await loop.run_in_executor(None, self._annotate_page_kinds, path, blocks)

        metadata: dict[str, Any] = {}
        if page_kinds:
            metadata["page_kinds"] = page_kinds

        chapter_ir = ChapterIR(
            doc_id=manifest.doc_id,
            chapter_id=chapter_meta.chapter_id,
            title=chapter_meta.title,
            spine_index=chapter_meta.spine_index,
            blocks=blocks,
            metadata=metadata,
        )
        yield chapter_ir

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
            self.render_engine,
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
            repair_missing_spaces_with_lines,
        )
        from ubt.core.ir.continuation import fuse_continuation_blocks

        blocks = extract_with_docling(
            path,
            page_range,
            symbols=_docling_symbols,
            enrich=self._resolve_formula_enrichment(path),
        )
        # Docling's line join drops inter-word spaces ("A Sandbox" -> "ASandbox");
        # the page's own pdfium lines restore them before anything is translated.
        blocks = repair_missing_spaces_with_lines(blocks, path)
        blocks = cross_check_blocks_with_pdfium(blocks, path)
        return fuse_continuation_blocks(blocks)

    def _extract_with_oxide(self, path: Path) -> list[IRBlock]:
        """Fallback lightweight text extractor using pdf_oxide."""
        from ubt.core.ir.continuation import fuse_continuation_blocks

        blocks = extract_with_oxide(path)
        return fuse_continuation_blocks(blocks)

    async def render_blocks(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        render_engine: str | None = None,
        render_plan: RenderPlan | None = None,
        realization_plan: Mapping[str, Fidelity] | None = None,
        **kwargs: Any,
    ) -> Path:
        """Render publication-grade output (delegates to DoclingRenderStrategy)."""
        out_path = await self._renderer.render_blocks(
            manifest,
            blocks,
            target_lang,
            output_path,
            bilingual_mode,
            render_engine,
            render_plan=render_plan,
            realization_plan=realization_plan,
        )
        self.last_render_skips = list(self._renderer.last_render_skips)
        self.last_render_outcome = self._renderer.last_outcome
        return out_path

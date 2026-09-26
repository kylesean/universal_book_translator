"""Unified Service Provider Interface (SPI) for document adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ubt.core.exceptions import RenderBlocksNotImplementedError
from ubt.core.ir.models import BookManifest, ChapterIR, IRBlock

if TYPE_CHECKING:
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ports import AdapterRuntimeConfig

#: CSS class every adapter puts on an injected bilingual target node. One name
#: for HTML and EPUB so a stylesheet (and any consumer) can target both.
BILINGUAL_TARGET_CLASS = "ubt-bilingual-target"


class BaseDocumentAdapter(ABC):
    """Unified SPI abstraction for all document format adapters.

    Primary render contract is :meth:`render_blocks` (pure: manifest +
    already-fetched blocks, no storage dependency); :meth:`render_output` is a
    compatibility shim that fetches from the ledger and delegates.
    """

    #: Suffixes this adapter's ``render_blocks`` actually writes; empty means
    #: unconstrained. The pipeline refuses an ``--output`` that contradicts them
    #: before spending a single token.
    output_suffixes: frozenset[str] = frozenset()

    #: Whether this adapter honours the 1-based ``pages`` selection in
    #: :meth:`parse_stream`. PDF engines do (they restrict the parse and/or the
    #: blocks carry a resolvable ``bbox.page``); EPUB/DOCX/HTML/Markdown do not —
    #: their blocks have no page geometry, so a ``--pages`` request would be
    #: silently ignored and the whole book billed. The ingest stage refuses
    #: page-ranged jobs on such adapters before spending instead of translating
    #: every page.
    supports_page_selection: bool = False

    def apply_config(self, runtime_config: AdapterRuntimeConfig) -> None:
        """Accept engine-level runtime knobs from the pipeline.

        Default no-op: adapters that consume none of them are unaffected. The
        Docling PDF family overrides this to take OCR / formula / font settings.
        """
        return None

    @abstractmethod
    async def extract_manifest(self, input_path: Path) -> BookManifest:
        """Extract top-level lightweight book metadata and chapter TOC index."""
        pass

    @abstractmethod
    def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]:
        """Stream chapters one by one as ChapterIR partitions for bounded memory consumption.

        ``pages`` is a 1-based page selection used by page-ranged jobs. PDF
        engines may restrict the parse itself (Docling converts only the
        contiguous range); adapters without page semantics ignore it and the
        ingest stage filters blocks afterwards.
        """
        pass

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
        """Render bilingual output from pre-fetched blocks (no ledger access).

        ``bilingual_mode`` and ``render_engine`` are accepted uniformly so
        engine-agnostic call sites typecheck; non-PDF adapters render their
        single canonical bilingual layout and ignore engine-specific options.
        """
        raise RenderBlocksNotImplementedError(
            f"{type(self).__name__} must implement render_blocks()"
        )

    async def render_output(
        self,
        manifest: BookManifest,
        ledger: SQLiteJobLedger,
        target_lang: str,
        output_path: Path,
        job_id: str | None = None,
        bilingual_mode: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Compatibility adapter: fetch blocks from ledger, then delegate to render_blocks.

        ``**kwargs`` forwards engine-specific options (e.g. PDF's
        ``render_engine``) so the PDF family need not duplicate this method.
        """
        actual_job_id = job_id or manifest.doc_id
        blocks = ledger.get_all_blocks(actual_job_id)
        return await self.render_blocks(
            manifest=manifest,
            blocks=blocks,
            target_lang=target_lang,
            output_path=output_path,
            bilingual_mode=bilingual_mode,
            **kwargs,
        )


class BasePDFEngineAdapter(BaseDocumentAdapter):
    """Specialized SPI abstraction for PDF engines supporting dual-track pluggability."""

    output_suffixes = frozenset({".pdf"})
    supports_page_selection = True

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Identifier of the PDF engine implementation (e.g. 'babeldoc', 'typst')."""
        pass

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
        """Render PDF output from pre-fetched blocks (no ledger access)."""
        raise RenderBlocksNotImplementedError(
            f"{type(self).__name__} must implement render_blocks()"
        )

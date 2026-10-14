"""Unified Service Provider Interface (SPI) for document adapters."""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ubt.core.exceptions import RenderBlocksNotImplementedError
from ubt.core.ir.models import BookManifest, ChapterIR, IRBlock
from ubt.core.text_decode import decode_markup

if TYPE_CHECKING:
    from ubt.core.ports import AdapterRuntimeConfig

__all__ = [
    "BILINGUAL_TARGET_CLASS",
    "BaseDocumentAdapter",
    "decode_markup",
    "parse_pipe_table_cells",
]

#: CSS class every adapter puts on an injected bilingual target node. One name
#: for HTML and EPUB so a stylesheet (and any consumer) can target both.
BILINGUAL_TARGET_CLASS = "ubt-bilingual-target"


def parse_pipe_table_cells(markup: str) -> list[list[str]]:
    """Parse GitHub-flavoured markdown pipe table markup into grid rows."""
    rows: list[list[str]] = []
    for line in markup.strip().splitlines():
        line = line.strip()
        if not line or not line.startswith("|"):
            continue
        stripped = line
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|") and not stripped.endswith(r"\|"):
            stripped = stripped[:-1]
        cells = [c.strip().replace(r"\|", "|") for c in re.split(r"(?<!\\)\|", stripped)]
        # Only a separator row in the canonical ``---``/``:---:`` form is
        # dropped. The previous "all of -, :, space" test also ate real data
        # rows of single-dash placeholder cells (``| - | - |``), shifting every
        # later row on write-back.
        if cells and all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows


class BaseDocumentAdapter(ABC):
    """Unified SPI abstraction for all document format adapters.

    Primary render contract is :meth:`render_blocks` (pure: manifest +
    already-fetched blocks, no storage dependency).
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

    @property
    def engine_name(self) -> str | None:
        """PDF engine identifier; ``None`` for non-PDF adapters.

        A non-``None`` name is the PDF-engine capability marker that
        :class:`BasePDFEngineAdapter` narrows to ``str``; callers tell a PDF
        engine from a text-only adapter through this, not a ``hasattr`` probe.
        """
        return None

    @property
    def last_render_skips(self) -> list[tuple[str, str]]:
        """Per-block skip ledger recorded by the last render pass."""
        if not hasattr(self, "_last_render_skips"):
            self._last_render_skips: list[tuple[str, str]] = []
        return self._last_render_skips

    @last_render_skips.setter
    def last_render_skips(self, value: list[tuple[str, str]]) -> None:
        self._last_render_skips = value

    @property
    def last_render_flags(self) -> list[tuple[str, str]]:
        """Per-block quality defect flags recorded by the last render pass."""
        if not hasattr(self, "_last_render_flags"):
            self._last_render_flags: list[tuple[str, str]] = []
        return self._last_render_flags

    @last_render_flags.setter
    def last_render_flags(self, value: list[tuple[str, str]]) -> None:
        self._last_render_flags = value

    @property
    def last_render_reconstructions(self) -> list[Any]:
        """Per-asset reconstruction verdicts recorded by the last render pass.

        Each entry is a :class:`ubt.core.content.nodes.ReconstructionRecord`
        (``block_id``/``verified``/``corrupt``). The export stage drains these
        into the delivery contract so a rebuilt table is recorded as
        RECONSTRUCTED (verified or corrupt) rather than PRESERVED_OPAQUE.
        """
        if not hasattr(self, "_last_render_reconstructions"):
            self._last_render_reconstructions: list[Any] = []
        return self._last_render_reconstructions

    @last_render_reconstructions.setter
    def last_render_reconstructions(self, value: list[Any]) -> None:
        self._last_render_reconstructions = value

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
        **kwargs: Any,
    ) -> Path:
        """Render bilingual output from pre-fetched blocks (no ledger access).

        ``bilingual_mode`` is accepted uniformly so engine-agnostic call sites
        typecheck; non-PDF adapters render their single canonical bilingual layout.

        The default implementation dispatches to :meth:`_render_blocks_sync` on
        a worker thread: DOM parsing and the file write are synchronous CPU +
        IO, and four format adapters used to carry byte-identical
        ``asyncio.to_thread`` wrappers for exactly this (a fix to one silently
        skipped the other three). An adapter whose render does not fit the
        "one sync method" shape (the PDF engines' render stack) still overrides
        this.
        """
        sync_render = getattr(self, "_render_blocks_sync", None)
        if sync_render is None:
            raise RenderBlocksNotImplementedError(
                f"{type(self).__name__} must implement render_blocks()"
            )
        return await asyncio.to_thread(
            sync_render,
            manifest,
            blocks,
            target_lang,
            output_path,
            bilingual_mode,
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
        **kwargs: Any,
    ) -> Path:
        """Render PDF output from pre-fetched blocks (no ledger access)."""
        raise RenderBlocksNotImplementedError(
            f"{type(self).__name__} must implement render_blocks()"
        )

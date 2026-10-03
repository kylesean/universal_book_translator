"""Unified Service Provider Interface (SPI) for document adapters."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import charset_normalizer

from ubt.core.exceptions import RenderBlocksNotImplementedError
from ubt.core.ir.models import BookManifest, ChapterIR, IRBlock

if TYPE_CHECKING:
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ports import AdapterRuntimeConfig

#: CSS class every adapter puts on an injected bilingual target node. One name
#: for HTML and EPUB so a stylesheet (and any consumer) can target both.
BILINGUAL_TARGET_CLASS = "ubt-bilingual-target"

# Matches both XML ``encoding="..."`` and HTML ``charset="..."`` declarations
# (``<meta charset="gbk">`` and ``content="text/html; charset=gbk"``), so the
# same sniffer serves XHTML members, standalone HTML and any other markup.
_ENCODING_DECL_RE = re.compile(
    rb"""(?:encoding|charset)\s*=\s*["']?([A-Za-z0-9._-]+)["']?""", re.IGNORECASE
)

#: Candidate encodings charset-normalizer may choose between when a source has
#: neither a BOM nor a declared encoding. Restricting the set keeps it from
#: guessing UTF-16 or Korean for a short GBK sample; on whole-file input the
#: guess is reliable, but a narrow plausible set makes short files reliable too.
_FALLBACK_ENCODINGS = ("utf_8", "gb18030", "big5", "cp932", "cp1252")


def decode_markup(raw: bytes) -> str:
    """Decode an XML/HTML/Markdown byte stream, preferring UTF-8 then its declared encoding.

    ``errors="ignore"`` silently dropped every byte a non-UTF-8 template could
    not represent. UTF-8 is tried first so a *stale* declaration on already
    UTF-8 bytes (every member the EPUB adapter re-serialises) can never
    mis-decode; only genuinely non-UTF-8 bytes fall back to the BOM / XML
    declaration / ``<meta charset>``, then to a replacement decode so nothing
    disappears without a visible marker.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    match = _ENCODING_DECL_RE.search(raw[:1024])
    if match:
        try:
            return raw.decode(match.group(1).decode("ascii", "ignore"))
        except (LookupError, UnicodeDecodeError):
            pass
    detected = _detect_and_decode(raw)
    if detected is not None:
        return detected
    return raw.decode("utf-8", errors="replace")


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
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", stripped)]
        if cells and all(set(c).issubset({"-", ":", " "}) for c in cells):
            continue
        rows.append(cells)
    return rows


def _detect_and_decode(raw: bytes) -> str | None:
    """Decode bytes with no BOM and no declared encoding via charset detection.

    A GBK/Shift-JIS ``.md``/``.txt`` carries neither, so UTF-8 + BOM +
    declaration all fail; ``charset-normalizer`` is the last resort before the
    lossy ``errors="replace"`` decode. Returns ``None`` when it cannot decide.
    """
    best = charset_normalizer.from_bytes(raw, cp_isolation=list(_FALLBACK_ENCODINGS)).best()
    return str(best) if best is not None else None


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

    @property
    def engine_name(self) -> str | None:
        """PDF engine identifier; ``None`` for non-PDF adapters.

        A non-``None`` name is the PDF-engine capability marker that
        :class:`BasePDFEngineAdapter` narrows to ``str``; callers tell a PDF
        engine from a text-only adapter through this, not a ``hasattr`` probe.
        """
        return None

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

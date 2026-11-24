"""BasePDFEngineAdapter bridge wrapping UBT v2 :class:`NativePDFCodec`.

Allows the existing ``PipelineOrchestrator``, CLI (``--pdf-engine native``),
REST API, and MCP Server to run the new 1:1 in-place PDF content-stream engine
immediately alongside the clean Layer 3 ``SurgicalTranslationService``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ubt.adapters.base import BasePDFEngineAdapter
from ubt.codecs.pdf.codec import NativePDFCodec
from ubt.codecs.protocol import DeliveryMode, OverflowItem, TranslationUnit, UnitRole
from ubt.core.ir.models import (
    BlockStatus,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    IRBlock,
    StyleMeta,
)
from ubt.core.ir.serializer import compute_file_sha256_cached
from ubt.model.ast import Heading, ListItem, Paragraph
from ubt.model.span import Span
from ubt.render.delivered_text import delivered_text, is_placeholder


def _unit_to_ir_block(unit: TranslationUnit, spine_index: int) -> IRBlock:
    bbox_list = unit.codec_meta.get("bbox")
    rect: tuple[float, float, float, float] | None = (
        (
            float(bbox_list[0]),
            float(bbox_list[1]),
            float(bbox_list[2]),
            float(bbox_list[3]),
        )
        if isinstance(bbox_list, list) and len(bbox_list) == 4
        else None
    )
    span = Span(page=unit.page_no or 1, bbox=rect)
    fs = float(unit.box_hint.get("font_size_pt", 0.0)) if unit.box_hint else 0.0
    if unit.role == UnitRole.HEADING:
        elem: Heading | ListItem | Paragraph = Heading(
            id=unit.unit_id,
            spine_index=spine_index,
            span=span,
            skip_translate=not unit.translate,
            font_size=fs,
            text=unit.source_marked,
            level=2,
        )
    elif unit.role == UnitRole.LIST_ITEM:
        elem = ListItem(
            id=unit.unit_id,
            spine_index=spine_index,
            span=span,
            skip_translate=not unit.translate,
            font_size=fs,
            text=unit.source_marked,
        )
    else:
        elem = Paragraph(
            id=unit.unit_id,
            spine_index=spine_index,
            span=span,
            skip_translate=not unit.translate,
            font_size=fs,
            text=unit.source_marked,
        )

    hint = unit.box_hint or {}
    box_w = float(hint["width_pt"]) if hint.get("width_pt") else None
    box_h = float(hint["height_pt"]) if hint.get("height_pt") else None
    line_h = float(hint["line_height_pt"]) if hint.get("line_height_pt") else None
    style = StyleMeta(
        font_size=fs if fs > 0 else None,
        alignment=str(unit.codec_meta.get("alignment") or "left"),
        box_width_pt=box_w if box_w and box_w > 0 else None,
        box_height_pt=box_h if box_h and box_h > 0 else None,
        line_height_pt=line_h if line_h and line_h > 0 else None,
    )
    return IRBlock.from_element(elem, style=style)


def _resolve_delivery_mode(bilingual_mode: str | None) -> DeliveryMode:
    if not bilingual_mode:
        return DeliveryMode.MONO
    low = bilingual_mode.strip().lower()
    if low in {"mono", "monolingual", "target_only", "auto"}:
        return DeliveryMode.MONO
    if low in {"interleaved", "dual", "bilingual"}:
        return DeliveryMode.INTERLEAVED
    if low in {"side_by_side", "side-by-side", "spread"}:
        return DeliveryMode.SIDE_BY_SIDE
    return DeliveryMode.MONO


def _native_translations(blocks: list[IRBlock]) -> dict[str, str]:
    """Collect sanitized target translations for native in-place patching.

    Quarantined (``BLOCKED_HUMAN``) and ``skip_translate`` blocks stay out of
    the patch map so their original source PDF ink remains intact on the page;
    ``FAILED`` / ``NEEDS_HUMAN`` blocks unwrap any ``<mark>`` wrapper via
    :func:`delivered_text` so HTML tags never print literally onto the page.
    """
    translations: dict[str, str] = {}
    for block in blocks:
        if block.skip_translate or block.status is BlockStatus.BLOCKED_HUMAN:
            continue
        raw_target = block.target_text or ""
        if not raw_target.strip():
            continue
        if is_placeholder(raw_target) and not (block.draft_text or "").strip():
            continue
        clean = delivered_text(block)
        if clean:
            translations[block.id] = clean
    return translations


class NativePDFAdapter(BasePDFEngineAdapter):
    """Registered PDF engine ``native`` backed by :class:`NativePDFCodec`."""

    output_suffixes = frozenset({".pdf"})
    supports_page_selection = True

    def __init__(self, codec: NativePDFCodec | None = None) -> None:
        self._codec = codec or NativePDFCodec()
        self._last_units: list[TranslationUnit] = []
        self._last_source: Path | None = None
        self._last_pages: set[int] | None = None
        self._has_parsed: bool = False
        self.last_overflows: tuple[OverflowItem, ...] = ()

    @property
    def engine_name(self) -> str:
        return "native"

    @property
    def last_units(self) -> list[TranslationUnit]:
        return self._last_units

    def close(self) -> None:
        self._last_units = []
        self._last_source = None
        self._last_pages = None
        self._has_parsed = False
        self.last_overflows = ()

    def render_page_preview_sync(
        self,
        source_path: Path,
        page_no: int,
        blocks: list[IRBlock],
        target_lang: str,
        dpi: int = 144,
    ) -> bytes:
        """Render a single live preview PNG directly from SQLite ``blocks``."""
        _, units = self._codec.extract(source_path, pages={page_no})
        by_id = _native_translations(blocks)
        return self._codec.render_page_preview(
            source_path,
            page_no,
            units,
            by_id,
            target_lang=target_lang,
            dpi=dpi,
        )

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        profile = await asyncio.to_thread(self._codec.inspect, input_path)
        file_hash = await asyncio.to_thread(compute_file_sha256_cached, input_path)
        total_pages = profile.page_count or 1
        chapters = [
            ChapterMeta(
                chapter_id=f"page_{p:04d}",
                title=f"Page {p}",
                spine_index=p - 1,
            )
            for p in range(1, total_pages + 1)
        ]
        from ubt.core.job_options import clean_source_stem

        return BookManifest(
            doc_id=f"pdf_{file_hash[:12]}",
            title=clean_source_stem(input_path),
            source_path=str(input_path),
            chapters=chapters,
            metadata={
                "source_path": str(input_path),
                "page_count": total_pages,
                "pdf_engine": "native",
                "pdf_parser_engine": "native",
                "is_scanned": profile.is_scanned,
                "pdf_recommended_mode": profile.recommended_mode.value,
            },
        )

    async def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]:
        file_hash = await asyncio.to_thread(compute_file_sha256_cached, input_path)
        doc_id = f"pdf_{file_hash[:12]}"
        _, units = await asyncio.to_thread(self._codec.extract, input_path, pages=pages)
        self._last_units = units
        self._last_source = input_path
        self._last_pages = set(pages) if pages else None
        self._has_parsed = True

        by_chap: dict[str, list[TranslationUnit]] = {}
        for u in units:
            by_chap.setdefault(u.chapter_id, []).append(u)

        spine = 0
        for chap_idx, (chap_id, chap_units) in enumerate(by_chap.items()):
            blocks: list[IRBlock] = []
            for u in chap_units:
                blocks.append(_unit_to_ir_block(u, spine))
                spine += 1
            yield ChapterIR(
                doc_id=doc_id,
                chapter_id=chap_id,
                title=chap_id,
                spine_index=chap_idx,
                blocks=blocks,
            )

    async def render_blocks(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        **kwargs: Any,
    ) -> Path:
        del kwargs
        src = self._last_source or Path(
            str(manifest.source_path or manifest.metadata.get("source_path") or "")
        )
        pages_set: set[int] | None
        if self._has_parsed and self._last_source == src:
            pages_set = self._last_pages
        else:
            raw_pages = {b.bbox.page for b in blocks if b.bbox is not None}
            total_pages = int(manifest.metadata.get("page_count") or 0)
            if total_pages > 0 and len(raw_pages) >= total_pages:
                pages_set = None
            else:
                pages_set = raw_pages or None
        if not self._last_units or self._last_source != src:
            _, self._last_units = await asyncio.to_thread(self._codec.extract, src, pages=pages_set)
            self._last_source = src

        translations = _native_translations(blocks)
        mode = _resolve_delivery_mode(bilingual_mode)
        report = await asyncio.to_thread(
            self._codec.apply,
            src,
            self._last_units,
            translations,
            target_lang=target_lang,
            mode=mode,
            output_path=output_path,
            pages=pages_set,
        )
        self.last_overflows = report.overflows
        manifest.metadata["native_overflow_count"] = len(report.overflows)
        return report.output_path

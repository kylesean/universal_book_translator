"""Bridge a v2 :class:`DocumentCodec` into the legacy IR pipeline.

The codecs own the native container patch (DOM/OOXML/AST/PDF stream). The
pipeline still speaks the IR adapter protocol, so this adapter is the one
translation layer: extract units -> IR blocks, collect delivered text -> apply.
No format-specific logic lives here.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ubt.adapters.base import BaseDocumentAdapter
from ubt.adapters.unresolved import failure_note, is_unresolved
from ubt.codecs.protocol import (
    DeliveryMode,
    DocumentCodec,
    DocumentProfile,
    OverflowItem,
    TranslationUnit,
    UnitRole,
)
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

_RIGID_ANCHOR_RE = re.compile(r"<([a-z]+\d+)/>")


def _legacy_inline_text(unit: TranslationUnit) -> str:
    """Render rigid codec anchors as their source markup for the legacy masker.

    The old segment/XLIFF path masks ``code``/``$math$``/URLs, while the codec
    contract keeps them as ``<xN/>`` tokens. Replacing a rigid anchor with its
    payload raw text lets the masker protect the same span and lets the XLIFF
    companion publish the code/math placeholder a reader expects. Pair anchors
    (links) keep their tag form; the codec restores those.
    """
    token_by_id = {token.tag_id: token for token in unit.tokens}

    def _replace(match: re.Match[str]) -> str:
        token = token_by_id.get(match.group(1))
        if token is None or token.kind != "rigid":
            return match.group(0)
        return str(token.payload.get("raw", match.group(0)))

    return _RIGID_ANCHOR_RE.sub(_replace, unit.source_marked)


def _unit_to_block(unit: TranslationUnit, spine_index: int) -> IRBlock:
    bbox_list = unit.codec_meta.get("bbox")
    rect: tuple[float, float, float, float] | None = None
    if isinstance(bbox_list, (list, tuple)) and len(bbox_list) == 4:
        rect = (
            float(bbox_list[0]),
            float(bbox_list[1]),
            float(bbox_list[2]),
            float(bbox_list[3]),
        )
    span = Span(page=unit.page_no or 1, bbox=rect)
    font_size = float(unit.box_hint.get("font_size_pt", 0.0)) if unit.box_hint else 0.0
    if unit.role == UnitRole.HEADING:
        level = int(unit.codec_meta.get("heading_level") or 2)
        element: Heading | ListItem | Paragraph = Heading(
            id=unit.unit_id,
            spine_index=spine_index,
            span=span,
            skip_translate=not unit.translate,
            font_size=font_size,
            text=_legacy_inline_text(unit),
            level=min(6, max(1, level)),
        )
    elif unit.role == UnitRole.LIST_ITEM:
        element = ListItem(
            id=unit.unit_id,
            spine_index=spine_index,
            span=span,
            skip_translate=not unit.translate,
            font_size=font_size,
            text=_legacy_inline_text(unit),
        )
    else:
        element = Paragraph(
            id=unit.unit_id,
            spine_index=spine_index,
            span=span,
            skip_translate=not unit.translate,
            font_size=font_size,
            text=_legacy_inline_text(unit),
        )
    return IRBlock.from_element(
        element,
        style=StyleMeta(
            font_size=font_size if font_size > 0 else None,
            alignment=str(unit.codec_meta.get("alignment") or "left"),
        ),
    )


def _resolve_mode(bilingual_mode: str | None) -> DeliveryMode:
    value = (bilingual_mode or "").strip().lower()
    if value in {"interleaved", "dual", "bilingual", "both"}:
        return DeliveryMode.BILINGUAL
    if value in {"side_by_side", "side-by-side", "spread"}:
        return DeliveryMode.SIDE_BY_SIDE
    return DeliveryMode.MONO


class DocumentCodecAdapter(BaseDocumentAdapter):
    """IR adapter backed by a ``DocumentCodec`` for non-PDF containers."""

    supports_page_selection = False

    def __init__(self, source_path: Path, *, codec: DocumentCodec | None = None) -> None:
        from ubt.codecs.registry import resolve_codec

        self._codec = codec or resolve_codec(source_path)
        self._source = source_path
        self._last_source: Path | None = None
        self._units: list[TranslationUnit] = []
        self._profile: DocumentProfile | None = None
        self.output_suffixes = frozenset({source_path.suffix.lower(), ".pdf"})
        self.last_overflows: tuple[OverflowItem, ...] = ()

    @property
    def engine_name(self) -> str | None:
        return None

    async def extract_manifest(self, input_path: Path) -> BookManifest:
        profile, units = await asyncio.to_thread(self._codec.extract, input_path)
        self._profile = profile
        self._units = list(units)
        file_hash = await asyncio.to_thread(compute_file_sha256_cached, input_path)
        chapter_ids = list(dict.fromkeys(unit.chapter_id for unit in units)) or ["content"]
        chapters = [
            ChapterMeta(
                chapter_id=chapter_id,
                title=chapter_id,
                spine_index=index,
            )
            for index, chapter_id in enumerate(chapter_ids)
        ]
        return BookManifest(
            doc_id=f"codec_{file_hash[:12]}",
            title=input_path.stem,
            source_path=str(input_path),
            chapters=chapters,
            metadata={
                "source_path": str(input_path),
                "page_count": profile.page_count or 0,
                "codec_format": profile.format_name,
                "is_scanned": profile.is_scanned,
                "pdf_parser_engine": "codec",
            },
        )

    async def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]:
        if not self._units or self._last_source != input_path:
            _, units = await asyncio.to_thread(self._codec.extract, input_path, pages=pages)
            self._units = list(units)
            self._last_source = input_path
        by_chapter: dict[str, list[TranslationUnit]] = {}
        for unit in self._units:
            by_chapter.setdefault(unit.chapter_id, []).append(unit)
        if not by_chapter:
            by_chapter["content"] = []
        spine = 0
        for chapter_index, (chapter_id, units) in enumerate(by_chapter.items()):
            blocks: list[IRBlock] = []
            for unit in units:
                blocks.append(_unit_to_block(unit, spine))
                spine += 1
            yield ChapterIR(
                doc_id=str(self._profile.format_name if self._profile else "codec"),
                chapter_id=chapter_id,
                title=chapter_id,
                spine_index=chapter_index,
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
        if output_path.suffix.lower() == ".pdf":
            from ubt.adapters.pdf.docling_render import DoclingRenderStrategy

            mode = _resolve_mode(bilingual_mode)
            strategy = DoclingRenderStrategy()
            self.last_overflows = ()
            return await strategy._render_publish(
                manifest,
                list(blocks),
                target_lang,
                output_path,
                bilingual=mode != DeliveryMode.MONO,
            )
        src = Path(manifest.source_path) if manifest.source_path else self._source
        translations: dict[str, str] = {}
        for block in blocks:
            if block.skip_translate:
                continue
            if block.status is BlockStatus.BLOCKED_HUMAN or is_placeholder(block.target_text or ""):
                text = (block.source_text or "").strip()
            elif block.target_text or block.draft_text:
                text = delivered_text(block)
            else:
                continue
            if not text:
                continue
            if is_unresolved(block.status):
                # A codec replaces the block's own text; the old adapters
                # rendered this note as a sibling. Keeping it in the replacement
                # is the format-agnostic equivalent, and every deliverable still
                # says a human has not approved the draft.
                text = f"{text}\n\n{failure_note(block.status)}"
            translations[block.id] = text
        units = self._units
        if not units or (self._last_source is not None and self._last_source != src):
            # Agent finish / re-export replace an existing ledger without a
            # fresh parse, so the codec has no units cached yet. Re-extract the
            # source to recover the line/byte spans `apply` patches; the block
            # ids are the same ones the original ingest wrote.
            _, units = await asyncio.to_thread(self._codec.extract, src)
            self._units = list(units)
            self._last_source = src
        report = await asyncio.to_thread(
            self._codec.apply,
            src,
            units,
            translations,
            target_lang=target_lang,
            mode=_resolve_mode(bilingual_mode),
            output_path=output_path,
        )
        self.last_overflows = report.overflows
        return report.output_path


__all__ = ["DocumentCodecAdapter"]

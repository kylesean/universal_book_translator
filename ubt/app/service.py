"""Unified Layer 3 orchestrator connecting DocumentCodecs with Translation Core.

Strictly unidirectional: imports ``ubt.codecs`` (Layer 1) and ``ubt.core``
(Layer 2), while neither Layer 1 nor Layer 2 imports ``ubt.app``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from ubt.app.spec import JobSpec
from ubt.codecs.protocol import (
    ApplyReport,
    DocumentCodec,
    DocumentProfile,
    TranslationUnit,
)
from ubt.codecs.registry import resolve_codec
from ubt.core.engine.overflow_compactor import CompactionSummary, compact_overflowing_units
from ubt.core.validators.inline_tags import validate_inline_tags

UnitTranslateFn = Callable[[TranslationUnit, str, str], Awaitable[str]]
CompactCompleteFn = Callable[[str, str], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class SurgicalTranslationResult:
    """End-to-end outcome of a UBT v2 surgical translation run."""

    profile: DocumentProfile
    report: ApplyReport
    units: tuple[TranslationUnit, ...]
    translations: dict[str, str]
    compaction: CompactionSummary | None = None
    tag_validation_failures: tuple[str, ...] = ()


class SurgicalTranslationService:
    """Format-agnostic orchestrator for extract -> translate -> validate -> apply -> compact."""

    def __init__(
        self,
        *,
        codec_resolver: Callable[[Path], DocumentCodec] = resolve_codec,
    ) -> None:
        self._resolve_codec = codec_resolver

    def inspect(self, source_path: Path) -> DocumentProfile:
        codec = self._resolve_codec(source_path)
        return codec.inspect(source_path)

    async def run(
        self,
        spec: JobSpec,
        *,
        translate_unit: UnitTranslateFn,
        compact_complete: CompactCompleteFn | None = None,
        concurrency: int = 8,
    ) -> SurgicalTranslationResult:
        codec = self._resolve_codec(spec.source_path)
        pages = spec.page_set if codec.supports_page_selection else None
        profile, units = await asyncio.to_thread(codec.extract, spec.source_path, pages=pages)

        sem = asyncio.Semaphore(max(1, concurrency))
        translations: dict[str, str] = {}
        tag_failures: list[str] = []

        async def _translate_one(unit: TranslationUnit) -> tuple[str, str | None, bool]:
            if not unit.translate:
                return unit.unit_id, None, True
            async with sem:
                raw = await translate_unit(unit, spec.source_lang, spec.target_lang)
            check = validate_inline_tags(unit.source_marked, raw)
            if not check.valid:
                return unit.unit_id, raw, False
            return unit.unit_id, check.normalized_text, True

        results = await asyncio.gather(*(_translate_one(u) for u in units))
        for uid, text, ok in results:
            if text is None:
                continue
            if ok:
                translations[uid] = text
            else:
                tag_failures.append(uid)

        out_path = spec.resolved_output_path()
        report = await asyncio.to_thread(
            codec.apply,
            spec.source_path,
            units,
            translations,
            target_lang=spec.target_lang,
            mode=spec.codec_mode,
            output_path=out_path,
            pages=pages,
        )

        compaction: CompactionSummary | None = None
        if (
            spec.allow_overflow_compaction
            and compact_complete is not None
            and report.overflows
            and spec.max_overflow_compaction_units > 0
        ):
            units_by_id = {u.unit_id: u for u in units}
            compaction = await compact_overflowing_units(
                overflows=report.overflows,
                units_by_id=units_by_id,
                translations=translations,
                complete_fn=compact_complete,
                target_lang=spec.target_lang,
                max_units=spec.max_overflow_compaction_units,
            )
            if compaction.compacted:
                translations.update(compaction.compacted)
                report = await asyncio.to_thread(
                    codec.apply,
                    spec.source_path,
                    units,
                    translations,
                    target_lang=spec.target_lang,
                    mode=spec.codec_mode,
                    output_path=out_path,
                    pages=pages,
                )

        return SurgicalTranslationResult(
            profile=profile,
            report=report,
            units=tuple(units),
            translations=translations,
            compaction=compaction,
            tag_validation_failures=tuple(tag_failures),
        )

    async def render_page_preview(
        self,
        source_path: Path,
        page_no: int,
        units: list[TranslationUnit],
        translations: dict[str, str],
        *,
        target_lang: str,
        dpi: int = 144,
    ) -> bytes:
        codec = self._resolve_codec(source_path)
        return await asyncio.to_thread(
            codec.render_page_preview,
            source_path,
            page_no,
            units,
            translations,
            target_lang=target_lang,
            dpi=dpi,
        )

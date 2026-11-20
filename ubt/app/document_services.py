"""Application implementation of the core document-engine protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ubt.core.engine.adapter_runtime import DocumentAdapter


class AdapterDocumentServices:
    """Adapter/compiler-backed document operations for core."""

    def resolve_adapter(self, input_path: Path, pdf_engine: str = "auto") -> DocumentAdapter:
        from ubt.adapters.factory import get_adapter_for_path

        return get_adapter_for_path(input_path, pdf_engine=pdf_engine)

    def detect_figure_pages(self, input_path: Path, blocks: list[Any]) -> set[int]:
        from ubt.adapters.pdf.svg_diagram import detect_figure_pages

        return set(detect_figure_pages(input_path, blocks))

    def inspect_font_damage(self, pdf_path: Path) -> list[Any]:
        from ubt.adapters.pdf.extraction_witness import inspect_pdf

        return inspect_pdf(pdf_path)

    def summarize_font_damage(self, verdicts: Any) -> dict[str, int]:
        from ubt.adapters.pdf.extraction_witness import summarize

        return summarize(verdicts)

    def flag_font_damage(self, blocks: list[Any], verdicts: Any) -> list[Any]:
        from ubt.adapters.pdf.extraction_witness import annotate_blocks

        return annotate_blocks(blocks, verdicts)

    def is_fast_lane_eligible(self, input_path: Path) -> bool:
        from ubt.adapters.pdf.short_doc import is_fast_lane_eligible

        return bool(is_fast_lane_eligible(input_path))

    def crop_block_image(self, pdf_path: Path, block: Any, dpi: int = 150) -> str | None:
        from ubt.adapters.pdf.visual_scalpel import crop_ir_block_image

        return crop_ir_block_image(pdf_path, block, dpi=dpi)

    def is_visual_scalpel_applicable(
        self,
        block: Any,
        source_pdf_path: Path | str | None = None,
        vision_threshold: float = 0.60,
    ) -> bool:
        from ubt.adapters.pdf.visual_scalpel import is_visual_scalpel_applicable

        return is_visual_scalpel_applicable(
            block,
            source_pdf_path=source_pdf_path,
            vision_threshold=vision_threshold,
        )

    def supported_suffixes(self) -> set[str]:
        from ubt.adapters.factory import supported_suffixes

        return set(supported_suffixes())

    def reject_encrypted_pdf(self, path: Path) -> None:
        from ubt.adapters.pdf.encryption import reject_encrypted_pdf

        reject_encrypted_pdf(path)

    def resolve_font_for_render(self, target_lang: str, font_override: str | None = None) -> Any:
        from ubt.adapters.pdf.font_probe import resolve_for_render

        return resolve_for_render(target_lang, font_override=font_override)

    def placeholder_engine(self) -> Any:
        from ubt.segment.placeholders import default_placeholder_engine

        return default_placeholder_engine()

    def masked_source(self, **kwargs: Any) -> Any:
        from ubt.segment.placeholders import MaskedSource

        return MaskedSource(**kwargs)

    def translation_engine(self, **kwargs: Any) -> Any:
        from ubt.translate.engine import TranslationEngine

        return TranslationEngine(**kwargs)

    def export_xliff(
        self,
        blocks: list[Any],
        *,
        src_lang: str,
        trg_lang: str,
        original: str = "",
        job_id: str = "",
    ) -> tuple[str, int] | None:
        from ubt.segment.document import segments_from_blocks
        from ubt.segment.placeholders import default_placeholder_engine
        from ubt.segment.xliff import to_xliff

        engine = default_placeholder_engine()
        segments = segments_from_blocks(blocks, engine=engine)
        if not segments:
            return None
        xml = to_xliff(
            segments,
            src_lang=src_lang,
            trg_lang=trg_lang,
            original=original,
            job_id=job_id,
        )
        return xml, len(segments)

    def optimize_pdf_deliverable(self, pdf_path: Path, profile: str = "lossless") -> Any:
        from ubt.adapters.pdf.pdf_optimizer import optimize_pdf_in_place

        return optimize_pdf_in_place(pdf_path, profile=profile)


__all__ = ["AdapterDocumentServices"]

"""Core protocol for the remaining document-engine operations.

The last adapter/compiler bridges (adapter resolution, figure detection, font
witnesses, visual scalpel, font resolution, PDF optimization, and the
segment/translate helpers) are named here as a protocol. The application
composition root installs the implementation; core never imports an adapter or
the compiler packages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ubt.core.engine import _document_services_registry as _registry
from ubt.core.engine.adapter_runtime import AdapterRuntimeConfig, DocumentAdapter


class DocumentServices(Protocol):
    """The document-engine operations core is allowed to call."""

    def resolve_adapter(
        self,
        input_path: Path,
        pdf_engine: str = "auto",
    ) -> DocumentAdapter: ...

    def resolve_inspection_adapter(
        self, input_path: Path, pdf_engine: str = "auto"
    ) -> DocumentAdapter: ...

    def detect_figure_pages(self, input_path: Path, blocks: list[Any]) -> set[int]: ...

    def inspect_font_damage(self, pdf_path: Path) -> list[Any]: ...

    def summarize_font_damage(self, verdicts: Any) -> dict[str, int]: ...

    def flag_font_damage(self, blocks: list[Any], verdicts: Any) -> list[Any]: ...

    def is_fast_lane_eligible(self, input_path: Path) -> bool: ...

    def crop_block_image(self, pdf_path: Path, block: Any, dpi: int = 150) -> str | None: ...

    def is_visual_scalpel_applicable(
        self,
        block: Any,
        source_pdf_path: Path | str | None = None,
        vision_threshold: float = 0.60,
    ) -> bool: ...

    def supported_suffixes(self) -> set[str]: ...

    def reject_encrypted_pdf(self, path: Path) -> None: ...

    def resolve_font_for_render(
        self, target_lang: str, font_override: str | None = None
    ) -> Any: ...

    def placeholder_engine(self) -> Any: ...

    def masked_source(self, **kwargs: Any) -> Any: ...

    def translation_engine(self, **kwargs: Any) -> Any: ...

    def export_xliff(
        self,
        blocks: list[Any],
        *,
        src_lang: str,
        trg_lang: str,
        original: str = "",
        job_id: str = "",
    ) -> tuple[str, int] | None: ...

    def optimize_pdf_deliverable(self, pdf_path: Path, profile: str = "lossless") -> Any: ...


def configure_document_services(provider: DocumentServices) -> None:
    """Install the app's implementation (the composition root calls this once)."""
    _registry.provider = provider


def _provider() -> DocumentServices:
    provider = _registry.provider
    if provider is None:
        raise RuntimeError(
            "Document services are not configured; import the application composition root "
            "(ubt.app.bootstrap) before running a job"
        )
    return provider


def resolve_adapter(input_path: Path, pdf_engine: str = "auto") -> DocumentAdapter:
    """Resolve the translation adapter: the codec for every non-PDF format."""
    return _provider().resolve_adapter(input_path, pdf_engine=pdf_engine)


def resolve_inspection_adapter(input_path: Path, pdf_engine: str = "auto") -> DocumentAdapter:
    """Resolve the typed-reader adapter for read-only CLI/audit surfaces."""
    return _provider().resolve_inspection_adapter(input_path, pdf_engine=pdf_engine)


def detect_figure_pages(input_path: Path, blocks: list[Any]) -> set[int]:
    return _provider().detect_figure_pages(input_path, blocks)


def inspect_font_encoding_damage(pdf_path: Path) -> list[Any]:
    return _provider().inspect_font_damage(pdf_path)


def summarize_font_encoding_damage(verdicts: Any) -> dict[str, int]:
    return _provider().summarize_font_damage(verdicts)


def flag_font_encoding_damage(blocks: list[Any], verdicts: Any) -> list[Any]:
    return _provider().flag_font_damage(blocks, verdicts)


def is_fast_lane_eligible(input_path: Path) -> bool:
    return _provider().is_fast_lane_eligible(input_path)


def crop_block_image(pdf_path: Path, block: Any, dpi: int = 150) -> str | None:
    return _provider().crop_block_image(pdf_path, block, dpi=dpi)


def is_visual_scalpel_applicable(
    block: Any,
    source_pdf_path: Path | str | None = None,
    vision_threshold: float = 0.60,
) -> bool:
    return _provider().is_visual_scalpel_applicable(
        block,
        source_pdf_path=source_pdf_path,
        vision_threshold=vision_threshold,
    )


def supported_suffixes() -> set[str]:
    return _provider().supported_suffixes()


def reject_encrypted_pdf(path: Path) -> None:
    _provider().reject_encrypted_pdf(path)


def resolve_font_for_render(target_lang: str, font_override: str | None = None) -> Any:
    return _provider().resolve_font_for_render(target_lang, font_override=font_override)


def placeholder_engine() -> Any:
    return _provider().placeholder_engine()


def masked_source(
    *,
    text: str,
    code_map: dict[str, str],
    math_map: dict[str, str],
    soup_map: dict[str, str],
    cite_map: dict[str, str],
    email_map: dict[str, str] | None = None,
) -> Any:
    return _provider().masked_source(
        text=text,
        code_map=code_map,
        math_map=math_map,
        soup_map=soup_map,
        cite_map=cite_map,
        email_map=email_map,
    )


def translation_engine(
    *,
    placeholders: Any,
    model: str = "",
    prompt_version: str = "",
    cache: Any = None,
    run_kind: str = "real",
) -> Any:
    return _provider().translation_engine(
        placeholders=placeholders,
        model=model,
        prompt_version=prompt_version,
        cache=cache,
        run_kind=run_kind,
    )


def export_xliff(
    blocks: list[Any],
    *,
    src_lang: str,
    trg_lang: str,
    original: str = "",
    job_id: str = "",
) -> tuple[str, int] | None:
    return _provider().export_xliff(
        blocks,
        src_lang=src_lang,
        trg_lang=trg_lang,
        original=original,
        job_id=job_id,
    )


def optimize_pdf_deliverable(pdf_path: Path, profile: str = "lossless") -> Any:
    return _provider().optimize_pdf_deliverable(pdf_path, profile=profile)


__all__ = [
    "AdapterRuntimeConfig",
    "DocumentAdapter",
    "DocumentServices",
    "configure_document_services",
    "crop_block_image",
    "detect_figure_pages",
    "export_xliff",
    "flag_font_encoding_damage",
    "inspect_font_encoding_damage",
    "is_fast_lane_eligible",
    "is_visual_scalpel_applicable",
    "masked_source",
    "optimize_pdf_deliverable",
    "placeholder_engine",
    "reject_encrypted_pdf",
    "resolve_adapter",
    "resolve_inspection_adapter",
    "resolve_font_for_render",
    "summarize_font_encoding_damage",
    "supported_suffixes",
    "translation_engine",
]

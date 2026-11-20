"""Core-owned adapter runtime contract and pure side-channel accessors.

These are the parts of the old ``ports`` bridge that do not name an adapter:
the runtime config dataclass, the adapter protocol core is typed against, and
the accessors for an adapter's render side channels. Keeping them here leaves
``ubt.core.engine.document_services`` only the lazy adapter forwarders it exists for.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ubt.core.ir.models import BookManifest, ChapterIR, IRBlock


@dataclass(frozen=True)
class AdapterRuntimeConfig:
    """Engine-level knobs the pipeline hands to an adapter at run start.

    Decouples the pipeline from per-field ``hasattr(adapter, ...)`` probing: the
    adapter owns which of these it consumes. ``ocr_api_key`` is the already
    unwrapped secret string, never a ``SecretStr``.
    """

    ocr_mode: str
    ocr_endpoint: str | None
    ocr_api_key: str
    #: Vision model the OCR/VLM channel bills. Owned by ``UBTConfig.ocr_model``
    #: and carried here for the same reason ``ocr_endpoint`` is: the assessor
    #: and the spend pre-flight price *this* model, so the driver must bill it
    #: too. Reading it from ``os.environ`` at the driver instead would let an
    #: environment-only setting quote one model and pay for another.
    ocr_model: str
    formula_enrichment: str
    #: Crop margin (fraction of the formula bbox) Docling expands on each side
    #: before the formula VLM reads it. Carried here so the adapter reads the
    #: resolved config value instead of hardcoding the parser's default: one knob,
    #: one source. Only consulted when ``formula_enrichment`` resolves to on.
    formula_crop_margin: float
    #: ``auto`` | ``on`` | ``off`` for placing a recovered table of contents.
    #: Carried here so the render strategy reads the resolved config value; the
    #: default (``auto``) places none for a paper and one for a book that has one.
    toc_mode: str
    #: Accelerator device Docling uses (``auto`` / ``cpu`` / ``cuda``). Carried
    #: here so the adapter reads the resolved config value instead of the
    #: parser re-deriving it: one knob, one source.
    docling_device: str
    font_family: str | None
    #: Whether page images may leave this machine (visual repair / cloud OCR /
    #: VLM judge). Carried here so the adapter reads the resolved config value
    #: instead of re-deriving it from the environment — one gate, one source.
    allow_page_upload: bool
    #: How eagerly the VLM fallback reads pages on top of the text layer
    #: (``off`` / ``missing`` / ``weak`` / ``all``). Owned by
    #: ``UBTConfig.ocr_fallback`` and carried here for the same one-knob-one-
    #: source reason as the other OCR fields: the adapter passes the resolved
    #: value down to the parser instead of the parser re-reading an env var.
    #: Declared after ``allow_page_upload`` because a defaulted field may not
    #: precede a required one; the value and its default are unchanged.
    ocr_fallback: str = "off"
    #: Content-addressed step cache root (content-addressed cache layer). Empty disables it;
    #: the adapter builds a store from it for the render path's pixel witnesses.
    cache_dir: str = ""
    #: Whether Docling conversions run in an isolated worker process.
    docling_isolate_process: bool = True
    #: Enable adaptive layout auto-healing pass during Typst publication render.
    enable_layout_auto_heal: bool = True
    layout_auto_heal_max_passes: int = 2
    #: Enable PDF size miniaturization and lossless stream compression.
    enable_pdf_optimization: bool = True
    pdf_optimize_profile: str = "lossless"
    #: Whether wide tables in single-column pages may flip to dedicated landscape pages.
    allow_landscape_tables: bool | None = None


def adapter_runtime_config(config: Any) -> AdapterRuntimeConfig:
    """Build the adapter runtime config from the engine config.

    One factory because there are two callers that must agree: the live pipeline
    at run start and the re-export that re-renders from a ledger.
    """
    return AdapterRuntimeConfig(
        ocr_mode=config.ocr_mode,
        ocr_endpoint=config.ocr_endpoint,
        ocr_api_key=config.ocr_api_key.get_secret_value(),
        ocr_model=config.ocr_model,
        ocr_fallback=getattr(config, "ocr_fallback", "off"),
        formula_enrichment=config.formula_enrichment,
        docling_device=config.docling_device,
        formula_crop_margin=config.formula_crop_margin,
        toc_mode=config.toc_mode,
        font_family=config.font_family,
        allow_page_upload=config.allow_page_upload,
        cache_dir=str(config.cache_dir) if config.cache_enabled else "",
        docling_isolate_process=config.docling_isolate_process,
        enable_layout_auto_heal=config.enable_layout_auto_heal,
        layout_auto_heal_max_passes=config.layout_auto_heal_max_passes,
        enable_pdf_optimization=config.enable_pdf_optimization,
        pdf_optimize_profile=config.pdf_optimize_profile,
        allow_landscape_tables=getattr(config, "allow_landscape_tables", None),
    )


def apply_runtime_config(adapter: Any, runtime_config: AdapterRuntimeConfig) -> None:
    """Push :class:`AdapterRuntimeConfig` into an adapter through its SPI hook.

    Replaces the pipeline's seven-field ``hasattr(adapter, field)`` chain with a
    single capability probe on one stable method name. Adapters that own these
    fields (the Docling PDF family) implement ``apply_config`` and translate the
    subset they support; adapters without the hook are left untouched.
    """
    apply = getattr(adapter, "apply_config", None)
    if callable(apply):
        apply(runtime_config)


@runtime_checkable
class DocumentAdapter(Protocol):
    """Protocol for document format adapters accessed through the ports bridge.

    Signatures mirror the real SPI (:class:`ubt.adapters.base.BaseDocumentAdapter`),
    including adapter-specific ``**kwargs``, so a call site
    typed against this protocol can name the engine knobs without a ``cast(Any)``.
    ``engine_name`` is the PDF-engine capability: a non-``None`` name marks a
    ``BasePDFEngineAdapter``.
    """

    @property
    def engine_name(self) -> str | None: ...

    async def extract_manifest(self, input_path: Path) -> BookManifest: ...

    def parse_stream(
        self, input_path: Path, pages: set[int] | None = None
    ) -> AsyncIterator[ChapterIR]: ...

    async def render_blocks(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        **kwargs: Any,
    ) -> Path: ...

    def apply_config(self, runtime_config: AdapterRuntimeConfig) -> None: ...
    @property
    def last_render_skips(self) -> list[tuple[str, str]]: ...
    @property
    def last_render_flags(self) -> list[tuple[str, str]]: ...
    @property
    def last_render_reconstructions(self) -> list[Any]: ...


def get_last_render_skips(adapter: Any) -> list[tuple[str, str]]:
    """Return the adapter's last-render per-block skip ledger.

    Adapters declare ``last_render_skips`` as part of :class:`DocumentAdapter`.
    This helper provides safe access, dropping malformed entries so an
    inspection issue cannot disrupt the render pipeline.
    """
    raw = getattr(adapter, "last_render_skips", None)
    if not isinstance(raw, list):
        return []
    skips: list[tuple[str, str]] = []
    for entry in raw:
        if (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], str)
        ):
            skips.append((entry[0], entry[1]))
    return skips


def get_last_render_flags(adapter: Any) -> list[tuple[str, str]]:
    """Return the adapter's last-render per-block *quality* flag ledger.

    Adapters declare ``last_render_flags`` as part of :class:`DocumentAdapter`.
    This helper provides safe access, dropping malformed entries so an
    inspection issue cannot disrupt the render pipeline.
    """
    raw = getattr(adapter, "last_render_flags", None)
    if not isinstance(raw, list):
        return []
    flags: list[tuple[str, str]] = []
    for entry in raw:
        if (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], str)
        ):
            flags.append((entry[0], entry[1]))
    return flags


def get_last_render_reconstructions(adapter: Any) -> list[Any]:
    """Return the adapter's last-render per-asset reconstruction verdicts.

    Adapters declare ``last_render_reconstructions`` as part of
    :class:`DocumentAdapter`. This helper provides safe access, dropping
    malformed entries (anything that is not a
    :class:`ubt.core.content.nodes.ReconstructionRecord`) so an inspection
    issue cannot disrupt the render pipeline.
    """
    from ubt.core.content.nodes import ReconstructionRecord

    raw = getattr(adapter, "last_render_reconstructions", None)
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, ReconstructionRecord)]

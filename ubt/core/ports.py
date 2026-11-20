"""Dependency-inversion ports for the core domain.

``ubt.core`` must never import a package that depends back on it at module
scope. Two families matter: the adapters (``ubt.adapters`` -- they already
depend on core's IR models and exceptions, so the reverse edge is a
package-level cycle that also drags adapter-weight dependencies such as
docling and typst tooling into the core import graph) and the compiler
packages (``ubt.segment`` / ``ubt.translate`` -- a
module-level edge from core closes the ``core.engine <-> pipeline`` cycle).

This module is the single sanctioned bridge. Implementations resolve through
FUNCTION-LEVEL lazy imports (no module-level edge); the layering invariant
forbids *module-level* edges only, so a function-body import is not one — and
unlike the string-keyed SPI registry this file used to route through, a lazy
import keeps signatures mypy-checkable, fails with a real ImportError at
first call, and does not need 23 string keys plus entry-point discovery plus
an in-tree bootstrap that named three downstream modules as strings.

Scope note: keep this file to what is actually resolved through it — it is
load-bearing architecture, and unused wrappers here read as endorsed capability.
Most bridges are thin forwarders by design: the point is *where* the adapter is
named, not how much logic sits behind the port. Do not add wrappers nobody
resolves through, pure re-exports of adapter helpers, or ``set_*`` hooks whose
globals nothing ever sets (they leave the override permanently ``None`` at every
call site). If nothing calls a piece of surface, it does not belong here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ubt.core.engine.adapter_runtime import DocumentAdapter
from ubt.core.ir.models import IRBlock

if TYPE_CHECKING:
    # Type-only edges: they keep the port's signature checked against the
    # stage contracts without entering the runtime import graph. The layering
    # invariant forbids *module-level* edges only; a TYPE_CHECKING edge is not one.
    from ubt.adapters.pdf.extraction_witness import PageVerdict
    from ubt.adapters.pdf.font_probe import FontStack
    from ubt.segment.placeholders import MaskedSource, PlaceholderEngine
    from ubt.translate.engine import TranslationEngine


def resolve_adapter(input_path: Path, pdf_engine: str = "auto") -> DocumentAdapter:
    """Return the document adapter for ``input_path`` without a core import edge.

    The registry lookup is gone: the SPI string registry (23 keys, entry
    points, an in-tree bootstrap naming three downstream modules as strings)
    cost more than the decoupling it bought and defeated mypy on every port
    (17 casts here). A function-level import keeps the module-level graph
    clean — the layering invariant only forbids module-level edges — and the
    failure mode is a real ImportError at first call instead of a KeyError
    from a half-populated registry.
    """
    from ubt.adapters.factory import get_adapter_for_path

    return cast(DocumentAdapter, get_adapter_for_path(input_path, pdf_engine=pdf_engine))


def detect_figure_pages(input_path: Path, blocks: list[Any]) -> set[int]:
    """Return 1-based figure-page numbers."""
    from ubt.adapters.pdf.svg_diagram import detect_figure_pages as _detect

    return set(_detect(input_path, blocks))


def inspect_font_encoding_damage(pdf_path: Path) -> list[PageVerdict]:
    """Run the extraction witness; returns per-page verdicts."""
    from ubt.adapters.pdf.extraction_witness import inspect_pdf

    return inspect_pdf(pdf_path)


def summarize_font_encoding_damage(verdicts: Any) -> dict[str, int]:
    """Aggregate witness verdicts into report metadata."""
    from ubt.adapters.pdf.extraction_witness import summarize

    return summarize(verdicts)


def flag_font_encoding_damage(blocks: list[Any], verdicts: Any) -> list[Any]:
    """Flag blocks on confirmed-damaged pages; returns the flagged subset."""
    from ubt.adapters.pdf.extraction_witness import annotate_blocks

    return cast("list[Any]", annotate_blocks(blocks, verdicts))


def is_fast_lane_eligible(input_path: Path) -> bool:
    """Fast-lane probe."""
    from ubt.adapters.pdf.short_doc import is_fast_lane_eligible as _probe

    return bool(_probe(input_path))


def crop_block_image(pdf_path: Path, block: Any, dpi: int = 150) -> str | None:
    """Crop block BBox from source PDF as base64 PNG."""
    from ubt.adapters.pdf.visual_scalpel import crop_ir_block_image

    return crop_ir_block_image(pdf_path, block, dpi=dpi)


def is_visual_scalpel_applicable(
    block: Any,
    source_pdf_path: Path | str | None = None,
    vision_threshold: float = 0.60,
) -> bool:
    """Check if block qualifies for visual scalpel repair (lazy adapter bridge)."""
    from ubt.adapters.pdf.visual_scalpel import is_visual_scalpel_applicable as _check

    return _check(block, source_pdf_path=source_pdf_path, vision_threshold=vision_threshold)


def supported_suffixes() -> set[str]:
    """File suffixes the adapter registry can open (from the factory)."""
    from ubt.adapters.factory import supported_suffixes as _suffixes

    return set(_suffixes())


def reject_encrypted_pdf(path: Path) -> None:
    """Refuse an encrypted PDF early with an actionable message."""
    from ubt.adapters.pdf.encryption import reject_encrypted_pdf as _reject

    _reject(path)


def resolve_font_for_render(
    target_lang: str,
    font_override: str | None = None,
) -> FontStack:
    """Resolve font availability for a given target language."""
    from ubt.adapters.pdf.font_probe import resolve_for_render as _resolve

    # No cast: ``resolve_for_render`` already declares ``-> FontStack``. The
    # adapter import is deferred to keep this module free of a hard dependency
    # on the PDF stack, not because the type was unknown.
    return _resolve(target_lang, font_override=font_override)


# Translation-unit layer bridges (``ubt.segment`` / ``ubt.translate``).
# --------------------------------------------------------------------------- #


def placeholder_engine() -> PlaceholderEngine:
    """The default placeholder engine (the one mask-order owner in ``ubt.segment``)."""
    from ubt.segment.placeholders import default_placeholder_engine

    return default_placeholder_engine()


def masked_source(
    *,
    text: str,
    code_map: dict[str, str],
    math_map: dict[str, str],
    soup_map: dict[str, str],
    cite_map: dict[str, str],
    email_map: dict[str, str] | None = None,
) -> MaskedSource:
    """Rebuild a ``MaskedSource`` from the per-family maps a draft carries."""
    from ubt.segment.placeholders import MaskedSource

    return MaskedSource(
        text=text,
        email_map=email_map or {},
        code_map=code_map,
        math_map=math_map,
        soup_map=soup_map,
        cite_map=cite_map,
    )


def translation_engine(
    *,
    placeholders: PlaceholderEngine,
    model: str = "",
    prompt_version: str = "",
    cache: Any = None,
    run_kind: str = "real",
) -> TranslationEngine:
    """Build the per-unit transform engine (mask -> restore -> judge).

    ``run_kind`` namespaces the content cache. A rehearsal must pass something
    other than ``"real"`` (see :func:`ubt.core.engine.dry_run.rehearsal_cache_dir`)
    so its echo replies can never be served to a real job.
    """
    from ubt.translate.engine import TranslationEngine

    return TranslationEngine(
        placeholders=placeholders,
        model=model,
        prompt_version=prompt_version,
        cache=cache,
        run_kind=run_kind,
    )


def export_xliff(
    blocks: list[IRBlock],
    *,
    src_lang: str,
    trg_lang: str,
    original: str = "",
    job_id: str = "",
) -> tuple[str, int] | None:
    """Generate XLIFF companion XML content and segment count without a core import edge."""
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


def optimize_pdf_deliverable(pdf_path: Path, profile: str = "lossless") -> Any:
    """Miniaturize and compress a delivered PDF through the adapter optimizer."""
    from ubt.adapters.pdf.pdf_optimizer import optimize_pdf_in_place

    return optimize_pdf_in_place(pdf_path, profile=profile)

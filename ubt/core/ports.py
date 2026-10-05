"""Dependency-inversion ports for the core domain.

``ubt.core`` must never import a package that depends back on it at module
scope. Two families matter: the adapters (``ubt.adapters`` -- they already
depend on core's IR models and exceptions, so the reverse edge is a
package-level cycle that also drags adapter-weight dependencies such as
docling and typst tooling into the core import graph) and the compiler
packages (``ubt.pipeline`` / ``ubt.segment`` / ``ubt.translate`` -- a
module-level edge from core closes the ``core.engine <-> pipeline`` cycle).

This module is the single sanctioned bridge. Implementations resolve through
function-level lazy imports (no module-level edge); the one remaining ``set_*``
hook exists because its target is exercised with a fake in tests
(``set_visual_gate_runner``, reset by ``reset_ports``). This file is the only
place in ``ubt/core/`` allowed to name an external module at runtime --
inside a function body, or under ``TYPE_CHECKING`` (type-only imports never
enter the runtime import graph).

Scope note: keep this file to what is actually resolved through it — it is
load-bearing architecture, and unused wrappers here read as endorsed capability.
Most bridges are thin forwarders by design: the point is *where* the adapter is
named, not how much logic sits behind the port. Do not add wrappers nobody
resolves through, pure re-exports of adapter helpers, or ``set_*`` hooks whose
globals nothing ever sets (they leave the override permanently ``None`` at every
call site). If nothing calls a piece of surface, it does not belong here.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ubt.core.ir.models import BookManifest, ChapterIR, IRBlock

if TYPE_CHECKING:
    # Type-only edges: they keep the port's signature checked against the
    # stage contracts without entering the runtime import graph. The layering
    # invariant forbids *module-level* edges only; a TYPE_CHECKING edge is not one.
    from ubt.adapters.pdf.engine_selector import PDFRoutePlan
    from ubt.adapters.pdf.extraction_witness import PageVerdict
    from ubt.adapters.pdf.page_profiler import PageKind
    from ubt.adapters.pdf.visual_gate import VisualGateResult
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.segment.placeholders import MaskedSource, PlaceholderEngine
    from ubt.translate.engine import TranslationEngine


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
    render_engine: str
    formula_render: str
    font_family: str | None
    math_backend: str
    #: Whether page images may leave this machine (visual repair / cloud OCR /
    #: VLM judge). Carried here so the adapter reads the resolved config value
    #: instead of re-deriving it from the environment — one gate, one source.
    allow_page_upload: bool
    #: Content-addressed step cache root (content-addressed cache layer). Empty disables it;
    #: the adapter builds a store from it for the render path's pixel witnesses.
    cache_dir: str = ""


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
    including ``render_engine`` and adapter-specific ``**kwargs``, so a call site
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
        render_engine: str | None = None,
        **kwargs: Any,
    ) -> Path: ...

    async def render_output(
        self,
        manifest: BookManifest,
        ledger: SQLiteJobLedger,
        target_lang: str,
        output_path: Path,
        job_id: str | None = None,
        bilingual_mode: str | None = None,
        **kwargs: Any,
    ) -> Path: ...

    def apply_config(self, runtime_config: AdapterRuntimeConfig) -> None: ...


VisualGateRunnerFn = Callable[..., Any]

_visual_gate_runner: VisualGateRunnerFn | None = None


def set_visual_gate_runner(fn: VisualGateRunnerFn | None) -> None:
    """Override the post-render visual gate runner (None restores default)."""
    global _visual_gate_runner
    _visual_gate_runner = fn


def reset_ports() -> None:
    """Restore all production defaults (test teardown helper)."""
    global _visual_gate_runner
    _visual_gate_runner = None


def resolve_adapter(input_path: Path, pdf_engine: str = "auto") -> DocumentAdapter:
    """Return the document adapter for ``input_path`` without a core import edge."""
    from ubt.adapters.factory import get_adapter_for_path

    res = get_adapter_for_path(input_path, pdf_engine=pdf_engine)
    return res


def detect_figure_pages(input_path: Path, blocks: list[Any]) -> set[int]:
    """Return 1-based figure-page numbers."""
    from ubt.adapters.pdf.svg_diagram import detect_figure_pages as _detect

    return set(_detect(input_path, blocks))


def get_visual_gate_runner() -> VisualGateRunnerFn:
    """Return the post-render visual gate entry point."""
    if _visual_gate_runner is not None:
        return _visual_gate_runner
    from ubt.adapters.pdf.visual_gate import run_visual_gate

    return run_visual_gate


def crashed_visual_gate_result(message: str) -> VisualGateResult:
    """Gate result reporting the gate itself as failed (visual state unknown).

    The gate dataclasses live in the adapters layer, so core stages build the
    crash record through this bridge; a crash must not degrade to "no gate",
    which skips blocking enforcement and reads as a perfect KPI pass.
    """
    from ubt.adapters.pdf.visual_gate import VisualFinding, VisualGateResult

    return VisualGateResult(
        passed=False,
        findings=(VisualFinding(severity="critical", code="visual_gate_crashed", message=message),),
        stats={"total_pages": 0},
    )


def artifact_parity_findings(
    *,
    source_pdf: Path,
    artifact_pdf: Path,
    target_lang: str,
    keeps_source_geometry: bool,
    selected_pages: Sequence[int] | None = None,
) -> list[Any]:
    """T0.5 physical-evidence diff between delivered PDF and source.

    Duck-typed findings: (severity, code, message); see visual_gate for the
    contract. Empty list when the probes cannot answer.
    """
    from ubt.adapters.pdf.artifact_parity import check_artifact_parity

    return list(
        check_artifact_parity(
            source_pdf=source_pdf,
            artifact_pdf=artifact_pdf,
            target_lang=target_lang,
            keeps_source_geometry=keeps_source_geometry,
            selected_pages=selected_pages,
        )
    )


def asset_skip_findings(blocks: list[Any], skips: list[tuple[str, str]]) -> list[Any]:
    """Classify render asset skips into visual-gate findings.

    Duck-typed findings: (severity, code, message). A content figure/table that
    could not be embedded is ``major``/``content_asset_missing``; a decorative
    or cover asset is ``info``. Empty list when there were no skips.
    """
    from ubt.adapters.pdf.artifact_parity import asset_skip_findings as _classify

    return list(_classify(blocks, skips))


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

    return annotate_blocks(blocks, verdicts)


def blocking_gate_tripped(findings: Any, total_pages: int, enabled: bool) -> list[Any]:
    """Opt-in blocking gate.

    Returns the CRITICAL findings that refuse export for short docs when
    enabled; otherwise []. Duck-typed over findings (severity/code/page).
    """
    from ubt.adapters.pdf.visual_gate import blocking_gate_tripped as _impl

    return list(_impl(findings, total_pages, enabled))


def is_fast_lane_eligible(input_path: Path) -> bool:
    """Fast-lane probe."""
    from ubt.adapters.pdf.short_doc import is_fast_lane_eligible as _probe

    return bool(_probe(input_path))


def crop_block_image(pdf_path: Path, block: Any, dpi: int = 150) -> str | None:
    """Crop block BBox from source PDF as base64 PNG."""
    from ubt.adapters.pdf.visual_scalpel import crop_ir_block_image as _crop

    return _crop(pdf_path, block, dpi=dpi)


def is_visual_scalpel_applicable(
    block: Any,
    source_pdf_path: Path | str | None = None,
    vision_threshold: float = 0.60,
) -> bool:
    """Check if block qualifies for visual scalpel repair (lazy adapter bridge)."""
    from ubt.adapters.pdf.visual_scalpel import is_visual_scalpel_applicable as _check

    return _check(block, source_pdf_path=source_pdf_path, qe_threshold_for_vision=vision_threshold)


def get_last_render_skips(adapter: Any) -> list[tuple[str, str]]:
    """Return the adapter's last-render per-block skip ledger.

    Duck-typed side channel: adapters that track fail-closed render skips
    (the PDF compositor's unstageable blocks) expose
    them as plain ``(block_id, reason)`` string tuples via
    ``last_render_skips``.
    Adapters without the attribute yield []. Never raises: malformed
    entries are dropped so a reporting bug can never fail a render.
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


def probe_pdf_pages(input_path: Path) -> tuple[int, int]:
    """Return (page_count, char_count) via short_doc probe."""
    from ubt.adapters.pdf.short_doc import probe_pdf_pages as _probe

    return _probe(input_path)


@dataclass(frozen=True)
class PdfStructureFacts:
    """Page-level structure census feeding render-engine routing.

    All ratios are over *all* pages (not a sample) and are page-level, so the
    signal does not drift when the parser re-chunks text into more IR blocks.
    ``has_scan`` / ``formula_heavy`` keep their historical majority semantics.
    """

    has_scan: bool
    formula_heavy: bool
    multicolumn_page_share: float
    structural_page_share: float


def classify_pdf_structure(input_path: Path) -> PdfStructureFacts:
    """Census the document's pages once and return the routing signals.

    Fail-open stays (a probe failure is conservative), but it is logged: a
    silent default quietly distorts the ``pdf_engine='auto'`` routing decision.
    """
    try:
        from ubt.adapters.pdf.page_profiler import (
            PageKind,
            classify_page,
            collect_page_facts,
            structural_page_shares,
        )

        facts = collect_page_facts(input_path)
        kinds = [classify_page(f) for f in facts]
        n = len(kinds)
        has_scan = n > 0 and sum(1 for k in kinds if k == PageKind.SCAN_IMAGE) * 2 >= n
        formula_heavy = n > 0 and sum(1 for k in kinds if k == PageKind.MIXED_COMPLEX) * 2 >= n
        multicolumn_share, structural_share = structural_page_shares(facts)
        return PdfStructureFacts(
            has_scan=has_scan,
            formula_heavy=formula_heavy,
            multicolumn_page_share=multicolumn_share,
            structural_page_share=structural_share,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "PDF content classification failed for %s (routing probe degraded): %s",
            input_path,
            exc,
        )
        return PdfStructureFacts(False, False, 0.0, 0.0)


def classify_pdf_content(input_path: Path) -> tuple[bool, bool]:
    """Return (has_scan, formula_heavy) via page_profiler.

    Both flags aggregate with a >=50% page share, matching the sampled-page
    majority the engine selector applies to the same signals: a lone blank
    page in a born-digital book must not route the whole title down the
    long-chain/VLM path.
    """
    facts = classify_pdf_structure(input_path)
    return facts.has_scan, facts.formula_heavy


def extract_pdf_figures(
    pdf_path: Path | str,
    output_assets_dir: Path | str,
    dpi: int = 300,
) -> dict[str, Any]:
    """Extract figures from PDF via pypdfium2."""
    from ubt.adapters.pdf.asset_extractor import extract_pdf_figures as _extract

    return _extract(pdf_path, output_assets_dir, dpi=dpi)


def is_typst_math_well_formed(expr: str) -> bool:
    """Check if Typst math delimiters and quotes are well-formed."""
    from ubt.adapters.pdf.typst_math import is_typst_math_well_formed as _check

    return _check(expr)


# ---------------------------------------------------------------------------
# PDF route-assessment bridges: thin lazy helpers that keep ``ubt/core`` from
# naming adapter modules (ubt.adapters.pdf.*) or heavy PDF dependencies
# (pypdfium2, pypdf) directly. Routing through here preserves the "core names
# adapters only inside ports.py" discipline without changing behaviour, and
# hides the adapter-owned constants (PROFILE_CACHE_DIR / PageKind)
# so core never imports them.
# ---------------------------------------------------------------------------


def inspect_pdf_route_plan(input_path: Path) -> PDFRoutePlan:
    """Return the doc-wide PDFRoutePlan (primary_engine / has_* flags)."""
    from ubt.adapters.pdf.engine_selector import PROFILE_CACHE_DIR
    from ubt.adapters.pdf.engine_selector import inspect_pdf_route_plan as _plan

    return _plan(input_path, cache_dir=PROFILE_CACHE_DIR)


def profile_pdf_pages(input_path: Path) -> list[Any]:
    """Return the per-page profile list from the page profiler."""
    from ubt.adapters.pdf.engine_selector import PROFILE_CACHE_DIR
    from ubt.adapters.pdf.page_profiler import profile_pdf as _profile

    return list(_profile(input_path, cache_dir=PROFILE_CACHE_DIR))


def page_kind_enum() -> type[PageKind]:
    """Return the adapter-side PageKind enum (kept out of the core import graph)."""
    from ubt.adapters.pdf.page_profiler import PageKind

    return PageKind


def supported_suffixes() -> set[str]:
    """File suffixes the adapter registry can open (from the factory)."""
    from ubt.adapters.factory import supported_suffixes as _supported

    return set(_supported())


def render_fidelity_stats(
    source_pdf: Path,
    artifact_pdf: Path,
    blocks: list[Any],
    *,
    dpi: int = 300,
) -> dict[str, Any]:
    """Measure rigid-render fidelity (non-text residual + painted coverage).

    Advisory bridge to the adapter's pdfium+Pillow diff so ``ubt/core`` stays
    free of the heavy raster import edge. Returns a plain stats dict.
    """
    from ubt.adapters.pdf.render_fidelity import compute_render_fidelity

    return compute_render_fidelity(source_pdf, artifact_pdf, blocks, dpi=dpi)


def render_fidelity_findings(stats: dict[str, Any]) -> list[Any]:
    """Advisory ``info`` ParityFindings from a fidelity stats dict (never blocking)."""
    from ubt.adapters.pdf.render_fidelity import fidelity_findings

    return list(fidelity_findings(stats))


def sample_pdf_pages(input_path: Path) -> tuple[int, bool, str]:
    """Sample a PDF into ``(page_count, is_scanned, text_preview)``.

    Thin bridge over the adapter's sampler so ``ubt/core/archetype.py`` never
    names an adapter module or a heavy PDF dependency (the pypdfium2 gate +
    pdf_oxide fallback live in ``ubt.adapters.pdf.plain_text_extractor``). Failures
    degrade to ``(1, False, "")``.
    """
    from ubt.adapters.pdf.plain_text_extractor import sample_pdf_pages as _sample

    return _sample(input_path)


async def interleave_bilingual_pdf(
    source_pdf: Path,
    translated_pdf: Path,
    output_pdf: Path,
    facing_spread: bool,
) -> str:
    """Interleave source pages with rendered target pages (bilingual companion).

    Thin bridge over the adapter's alternator so the export stage never names
    an adapter module; returns the companion PDF's output path.
    """
    from ubt.adapters.pdf.alternator import BilingualAlternator

    result = await BilingualAlternator().interleave_pages_async(
        source_pdf=source_pdf,
        translated_pdf=translated_pdf,
        output_pdf=output_pdf,
        facing_spread=facing_spread,
    )
    return str(result.output_path)


# --------------------------------------------------------------------------- #
# Translation-unit layer bridges (``ubt.segment`` / ``ubt.translate``).
#
# The mask order, its exact reverse, and the per-unit judgement live in the
# compiler. ``ubt/core`` names them only here, inside function bodies, so the
# module-level core import graph never reaches the compiler packages -- the
# reverse edge is what makes the ``core.engine <-> pipeline`` cycle.
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
) -> TranslationEngine:
    """Build the per-unit transform engine (mask -> restore -> judge)."""
    from ubt.translate.engine import TranslationEngine

    return TranslationEngine(
        placeholders=placeholders,
        model=model,
        prompt_version=prompt_version,
        cache=cache,
    )

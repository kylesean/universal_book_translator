"""SPI provider registrations for Document Adapters.

Registers all adapter-side capabilities with :class:`ubt.core.spi.SPIRegistry`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ubt.core.spi import SPIRegistry

logger = logging.getLogger(__name__)


def _crashed_visual_gate_result(message: str) -> Any:
    from ubt.adapters.pdf.visual_gate import VisualFinding, VisualGateResult

    return VisualGateResult(
        passed=False,
        findings=(VisualFinding(severity="critical", code="visual_gate_crashed", message=message),),
        stats={"total_pages": 0},
    )


def _classify_pdf_structure(input_path: Path) -> Any:
    from ubt.adapters.pdf.page_profiler import (
        classify_page,
        collect_page_facts,
        majority_flags,
        structural_page_shares,
    )
    from ubt.core.ports import PdfStructureFacts

    try:
        facts = collect_page_facts(input_path)
        has_scan, formula_heavy = majority_flags([classify_page(f) for f in facts])
        multicolumn_share, structural_share = structural_page_shares(facts)
        return PdfStructureFacts(
            has_scan=has_scan,
            formula_heavy=formula_heavy,
            multicolumn_page_share=multicolumn_share,
            structural_page_share=structural_share,
        )
    except Exception as exc:
        logger.warning(
            "PDF content classification failed for %s (routing probe degraded): %s",
            input_path,
            exc,
        )
        return PdfStructureFacts(False, False, 0.0, 0.0)


def _inspect_pdf_route_plan(input_path: Path) -> Any:
    from ubt.adapters.pdf.engine_selector import PROFILE_CACHE_DIR
    from ubt.adapters.pdf.engine_selector import inspect_pdf_route_plan as _plan

    return _plan(input_path, cache_dir=PROFILE_CACHE_DIR)


def _profile_pdf_pages(input_path: Path) -> list[Any]:
    from ubt.adapters.pdf.engine_selector import PROFILE_CACHE_DIR
    from ubt.adapters.pdf.page_profiler import profile_pdf as _profile

    return list(_profile(input_path, cache_dir=PROFILE_CACHE_DIR))


def _artifact_parity_findings(
    *,
    source_pdf: Path,
    artifact_pdf: Path,
    target_lang: str,
    keeps_source_geometry: bool,
    selected_pages: Sequence[int] | None = None,
) -> list[Any]:
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


def register_spi_providers(registry: SPIRegistry) -> None:
    """Register all adapter-provided SPI services into the registry."""
    from ubt.adapters.factory import get_adapter_for_path
    from ubt.adapters.factory import supported_suffixes as _supported_suffixes
    from ubt.adapters.pdf.artifact_parity import ParityFinding
    from ubt.adapters.pdf.extraction_witness import annotate_blocks, inspect_pdf, summarize
    from ubt.adapters.pdf.page_profiler import PageKind
    from ubt.adapters.pdf.plain_text_extractor import sample_pdf_pages
    from ubt.adapters.pdf.render_fidelity import compute_render_fidelity, fidelity_findings
    from ubt.adapters.pdf.short_doc import is_fast_lane_eligible, probe_pdf_pages
    from ubt.adapters.pdf.svg_diagram import detect_figure_pages
    from ubt.adapters.pdf.visual_gate import (
        blocking_gate_tripped,
        run_visual_gate,
    )
    from ubt.adapters.pdf.visual_scalpel import (
        crop_ir_block_image,
        is_visual_scalpel_applicable,
    )

    registry.register_many(
        {
            "adapter_resolver": get_adapter_for_path,
            "supported_suffixes": _supported_suffixes,
            "detect_figure_pages": detect_figure_pages,
            "visual_gate_runner": run_visual_gate,
            "crashed_visual_gate_result": _crashed_visual_gate_result,
            "artifact_parity_findings": _artifact_parity_findings,
            "probe_unavailable_finding": lambda code, message: ParityFinding("info", code, message),
            "inspect_font_encoding_damage": inspect_pdf,
            "summarize_font_encoding_damage": summarize,
            "flag_font_encoding_damage": annotate_blocks,
            "blocking_gate_tripped": lambda findings, total, enabled: list(
                blocking_gate_tripped(findings, total, enabled)
            ),
            "is_fast_lane_eligible": is_fast_lane_eligible,
            "crop_block_image": crop_ir_block_image,
            "is_visual_scalpel_applicable": is_visual_scalpel_applicable,
            "probe_pdf_pages": probe_pdf_pages,
            "classify_pdf_structure": _classify_pdf_structure,
            "inspect_pdf_route_plan": _inspect_pdf_route_plan,
            "profile_pdf_pages": _profile_pdf_pages,
            "page_kind_enum": PageKind,
            "render_fidelity_stats": compute_render_fidelity,
            "render_fidelity_findings": lambda stats: list(fidelity_findings(stats)),
            "sample_pdf_pages": sample_pdf_pages,
        }
    )

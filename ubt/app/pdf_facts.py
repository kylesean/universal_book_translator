"""Application implementation of the core PDF fact queries.

This is the only place core's :class:`PdfFactsProvider` is backed by the PDF
adapter stack. The composition root installs it once; core never imports this
module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ubt.core.engine.pdf_facts import PdfStructureFacts

logger = logging.getLogger(__name__)


class AdapterPdfFacts:
    """PDF facts from the current adapter stack."""

    def classify_structure(self, input_path: Path) -> PdfStructureFacts:
        from ubt.adapters.pdf.page_profiler import (
            majority_flags,
            profile_pdf,
            structural_page_shares,
        )
        from ubt.adapters.pdf.pdf_profiler import PROFILE_CACHE_DIR

        try:
            profiles = profile_pdf(input_path, cache_dir=PROFILE_CACHE_DIR)
            if not profiles:
                return PdfStructureFacts(False, False, 0.0, 0.0, 0)
            has_scan, formula_heavy = majority_flags([p.kind for p in profiles])
            multicolumn_share, structural_share = structural_page_shares(
                [p.facts for p in profiles]
            )

            damage_pages = 0
            try:
                from ubt.adapters.pdf.extraction_witness import inspect_pdf, summarize

                damage_pages = summarize(inspect_pdf(input_path)).get("confirmed_pages", 0)
            except Exception as exc:  # noqa: BLE001 - a witness failure is not a routing veto
                logger.debug("Font witness in structure classify: %s", exc)

            return PdfStructureFacts(
                has_scan=has_scan,
                formula_heavy=formula_heavy,
                multicolumn_page_share=multicolumn_share,
                structural_page_share=structural_share,
                confirmed_font_damage_pages=damage_pages,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PDF content classification failed for %s (routing probe degraded): %s",
                input_path,
                exc,
            )
            return PdfStructureFacts(False, False, 0.0, 0.0, 0)

    def probe_pages(self, input_path: Path) -> tuple[int, int]:
        from ubt.adapters.pdf.short_doc import probe_pdf_pages

        return probe_pdf_pages(input_path)

    def sample_pages(self, input_path: Path) -> tuple[int, bool, str]:
        from ubt.adapters.pdf.plain_text_extractor import sample_pdf_pages

        return sample_pdf_pages(input_path)

    def probe_feasibility(self, input_path: Path) -> Any:
        from ubt.adapters.pdf.native_preflight import measure_native_feasibility

        return measure_native_feasibility(input_path)

    def page_count(self, input_path: Path) -> int:
        from ubt.adapters.pdf.pdf_struct import page_count

        return page_count(input_path)

    def inspect_profile(self, input_path: Path) -> Any:
        from ubt.adapters.pdf.pdf_profiler import inspect_pdf_profile

        return inspect_pdf_profile(input_path)

    def profile_pages(self, input_path: Path) -> list[Any]:
        from ubt.adapters.pdf.page_profiler import profile_pdf
        from ubt.adapters.pdf.pdf_profiler import PROFILE_CACHE_DIR

        return list(profile_pdf(input_path, cache_dir=PROFILE_CACHE_DIR))

    def page_kind(self) -> Any:
        from ubt.adapters.pdf.page_profiler import PageKind

        return PageKind


__all__ = ["AdapterPdfFacts"]

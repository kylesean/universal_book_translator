"""``NativePDFCodec``: production in-place surgical PDF translation codec.

Implements :class:`ubt.codecs.protocol.DocumentCodec` using:
- ``RapidPPDocLayoutDetector`` (PP-DocLayoutV2 ONNX, Apache-2.0) for macro layout
- ``stream_analyzer`` (``pikepdf`` + optional ``pypdfium2`` ToUnicode recovery)
- ``layout_fusion`` (protected formulas/figures, table cell splitting, inline formula islands)
- ``formula_box`` (200-byte instruction-slice Form XObjects)
- ``font_engine`` (TTF + CFF/OTF dual-format subsetting + HarfBuzz shaping)
- ``paragraph_typesetter`` & ``stream_patcher`` (Mono / Interleaved / Side-by-Side)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pikepdf
from PIL import Image

from ubt.codecs.pdf.layout_detector import (
    HeuristicLayoutDetector,
    LayoutDetector,
    RapidPPDocLayoutDetector,
)
from ubt.codecs.pdf.layout_fusion import FusedBlock, fuse_page
from ubt.codecs.pdf.stream_analyzer import PageStreamAnalysis, analyze_page_stream
from ubt.codecs.pdf.stream_patcher import apply_pdf_patches, render_single_page_png
from ubt.codecs.protocol import (
    ApplyReport,
    DeliveryMode,
    DocumentProfile,
    TranslationUnit,
)


@dataclass(slots=True)
class ExtractedPDFDocument:
    """Cached extraction state allowing fast apply() and live single-page previews."""

    profile: DocumentProfile
    analyses: dict[int, PageStreamAnalysis]
    fused_by_page: dict[int, list[FusedBlock]]
    units: list[TranslationUnit]


def _render_page_for_layout(pdf_path: Path, page_no: int, dpi: int = 144) -> Image.Image:
    import pypdfium2 as pdfium

    from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK

    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            page = doc[page_no - 1]
            try:
                pil: Image.Image = page.render(scale=dpi / 72.0).to_pil()
                return pil
            finally:
                page.close()
        finally:
            doc.close()


class NativePDFCodec:
    """1:1 in-place PDF content-stream translation codec (default ``DeliveryMode.MONO``)."""

    supported_suffixes: frozenset[str] = frozenset({".pdf"})
    supports_page_selection: bool = True

    def __init__(
        self,
        *,
        layout_detector: LayoutDetector | None = None,
        layout_dpi: int = 144,
        fonts_dir: Path | None = None,
    ) -> None:
        self._detector: LayoutDetector = layout_detector or RapidPPDocLayoutDetector()
        self._layout_dpi = layout_dpi
        self._fonts_dir = fonts_dir
        self._profile_cache: dict[str, DocumentProfile] = {}
        self._page_cache: dict[tuple[str, int], tuple[PageStreamAnalysis, list[FusedBlock]]] = {}

    def inspect(self, source_path: Path) -> DocumentProfile:
        """Fast preflight probe: page count, encryption, scanned detection, math presence."""
        resolved_key = str(source_path.resolve())
        cached_prof = self._profile_cache.get(resolved_key)
        if cached_prof is not None:
            return cached_prof
        try:
            with pikepdf.open(source_path) as pdf:
                total = len(pdf.pages)
                sample_indices = sorted({0, total // 2, max(0, total - 1)}) if total > 0 else []
                sample_chars = 0
                math_ops = 0
                for idx in sample_indices:
                    ana = analyze_page_stream(pdf.pages[idx], page_no=idx + 1)
                    sample_chars += sum(len(o.text.strip()) for o in ana.text_ops)
                    math_ops += sum(1 for o in ana.text_ops if o.is_math_font)
        except pikepdf.PasswordError:
            return DocumentProfile(
                format_name="pdf",
                is_encrypted=True,
                recommended_mode=DeliveryMode.MONO,
            )

        avg_chars = sample_chars / max(1, len(sample_indices))
        is_scanned = total > 0 and avg_chars < 25.0
        estimated_chars = int(avg_chars * total)
        prof = DocumentProfile(
            format_name="pdf",
            page_count=total,
            estimated_chars=estimated_chars,
            is_scanned=is_scanned,
            is_encrypted=False,
            has_math=math_ops > 0,
            recommended_mode=DeliveryMode.REFLOW if is_scanned else DeliveryMode.MONO,
            metadata={"avg_chars_per_sampled_page": round(avg_chars, 1)},
        )
        self._profile_cache[resolved_key] = prof
        return prof

    def extract_detailed(
        self,
        source_path: Path,
        *,
        pages: set[int] | None = None,
    ) -> ExtractedPDFDocument:
        """Extract stream analyses, layout fusion blocks, and :class:`TranslationUnit` list."""
        resolved_key = str(source_path.resolve())
        profile = self.inspect(source_path)
        analyses: dict[int, PageStreamAnalysis] = {}
        fused_by_page: dict[int, list[FusedBlock]] = {}
        units: list[TranslationUnit] = []
        needs_image = not isinstance(self._detector, HeuristicLayoutDetector)

        total = profile.page_count or 0
        target_pages = (
            sorted(p for p in pages if 1 <= p <= total) if pages else list(range(1, total + 1))
        )
        missing_pages = [pno for pno in target_pages if (resolved_key, pno) not in self._page_cache]

        if missing_pages:
            with pikepdf.open(source_path) as pdf:
                for pno in missing_pages:
                    analysis = analyze_page_stream(
                        pdf.pages[pno - 1],
                        page_no=pno,
                        pdfium_fallback_path=source_path,
                    )
                    img = (
                        _render_page_for_layout(source_path, pno, dpi=self._layout_dpi)
                        if needs_image
                        else None
                    )
                    regions = self._detector.detect_page(img, analysis)
                    if not regions and analysis.text_ops:
                        regions = HeuristicLayoutDetector().detect_page(None, analysis)
                    fused = fuse_page(analysis, regions)
                    self._page_cache[(resolved_key, pno)] = (analysis, fused)

        for pno in target_pages:
            analysis, fused = self._page_cache[(resolved_key, pno)]
            analyses[pno] = analysis
            fused_by_page[pno] = fused
            units.extend(fb.unit for fb in fused)

        return ExtractedPDFDocument(
            profile=profile,
            analyses=analyses,
            fused_by_page=fused_by_page,
            units=units,
        )

    def extract(
        self,
        source_path: Path,
        *,
        pages: set[int] | None = None,
    ) -> tuple[DocumentProfile, list[TranslationUnit]]:
        doc = self.extract_detailed(source_path, pages=pages)
        return doc.profile, doc.units

    def apply(
        self,
        source_path: Path,
        units: list[TranslationUnit],
        translations: dict[str, str],
        *,
        target_lang: str,
        mode: DeliveryMode = DeliveryMode.MONO,
        output_path: Path,
        pages: set[int] | None = None,
    ) -> ApplyReport:
        del units  # Fused blocks already carry exact op_indices & geometry
        doc = self.extract_detailed(source_path, pages=pages)
        return apply_pdf_patches(
            source_path=source_path,
            analyses=doc.analyses,
            fused_by_page=doc.fused_by_page,
            translations=translations,
            target_lang=target_lang,
            mode=mode,
            output_path=output_path,
            pages=pages,
            fonts_dir=self._fonts_dir,
        )

    def render_page_preview(
        self,
        source_path: Path,
        page_no: int,
        units: list[TranslationUnit],
        translations: dict[str, str],
        *,
        target_lang: str,
        dpi: int = 144,
    ) -> bytes:
        del units
        doc = self.extract_detailed(source_path, pages={page_no})
        analysis = doc.analyses[page_no]
        blocks = doc.fused_by_page.get(page_no, [])
        return render_single_page_png(
            source_path=source_path,
            page_no=page_no,
            analysis=analysis,
            blocks=blocks,
            translations=translations,
            target_lang=target_lang,
            dpi=dpi,
            fonts_dir=self._fonts_dir,
        )

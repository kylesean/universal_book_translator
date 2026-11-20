"""Native in-place PDF content-stream translation codec (UBT v2)."""

from ubt.codecs.pdf.codec import ExtractedPDFDocument, NativePDFCodec
from ubt.codecs.pdf.font_engine import EmbeddedFont, FontEngine, FontSlot, ShapedGlyph
from ubt.codecs.pdf.formula_box import InlineFormulaBox, build_formula_xobject
from ubt.codecs.pdf.layout_detector import (
    HeuristicLayoutDetector,
    LayoutDetector,
    LayoutRegion,
    RapidPPDocLayoutDetector,
    RegionKind,
)
from ubt.codecs.pdf.layout_fusion import FusedBlock, ParagraphGeometry, fuse_page
from ubt.codecs.pdf.paragraph_typesetter import PositionedRun, TypesetParagraph, typeset_paragraph
from ubt.codecs.pdf.stream_analyzer import PageStreamAnalysis, StreamTextOp, analyze_page_stream
from ubt.codecs.pdf.stream_patcher import apply_pdf_patches, render_single_page_png

__all__ = [
    "EmbeddedFont",
    "ExtractedPDFDocument",
    "FontEngine",
    "FontSlot",
    "FusedBlock",
    "HeuristicLayoutDetector",
    "InlineFormulaBox",
    "LayoutDetector",
    "LayoutRegion",
    "NativePDFCodec",
    "PageStreamAnalysis",
    "ParagraphGeometry",
    "PositionedRun",
    "RapidPPDocLayoutDetector",
    "RegionKind",
    "ShapedGlyph",
    "StreamTextOp",
    "TypesetParagraph",
    "analyze_page_stream",
    "apply_pdf_patches",
    "build_formula_xobject",
    "fuse_page",
    "render_single_page_png",
    "typeset_paragraph",
]

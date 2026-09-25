"""VLM transcription core: registry + driver contract + anchoring.

Scans enter here because the previous scan path is dead in practice
(Docling is installed but neither of its OCR backends — easyocr nor
tesserocr — is, so ``do_ocr=True`` goes nowhere).

Architecture (roadmap P9):
- Drivers RECOGNIZE text (reading order). They never own geometry.
- Anchoring owns geometry: the pdfium text layer votes first (proofread
  mode); only pages WITHOUT a usable text layer use measured driver boxes
  (recognition mode), and only from drivers whose boxes are MEASURED
  (detector output like rapidocr), never hallucinated (LLM-VLM).
- Swapping DeepSeek-OCR / PaddleOCR-VL / anything else later = registering
  a new driver. The pipeline above never changes.
"""

from ubt.adapters.pdf.vlm.anchor import AnchoredLine, anchor_transcript
from ubt.adapters.pdf.vlm.registry import default_driver_name, get_driver, list_drivers
from ubt.adapters.pdf.vlm.transcribe import (
    FALLBACK_ENV_VAR,
    fallback_enabled,
    transcribe_page_to_blocks,
)
from ubt.adapters.pdf.vlm.types import PageTranscript, VlmDriver, VlmLine

__all__ = [
    "AnchoredLine",
    "FALLBACK_ENV_VAR",
    "PageTranscript",
    "VlmDriver",
    "VlmLine",
    "anchor_transcript",
    "default_driver_name",
    "fallback_enabled",
    "get_driver",
    "list_drivers",
    "transcribe_page_to_blocks",
]

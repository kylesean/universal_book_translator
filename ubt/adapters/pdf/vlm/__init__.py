"""VLM transcription core: registry + driver contract + anchoring.

Scans enter here because this project deliberately keeps Docling's own OCR leg
switched off: ``docling_parser`` pins ``do_ocr = False``, and neither of
Docling's classical backends is a declared dependency, so turning Docling's OCR
on would not be a fallback but a new install requirement. Transcribing a page
that has no text layer is this registry's job.

Architecture:
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
    "get_driver",
    "list_drivers",
    "transcribe_page_to_blocks",
]

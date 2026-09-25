"""Rapidocr driver: classical detector OCR as the first working core.

rapidocr returns MEASURED boxes (detector output, not hallucination), so
``measured_boxes=True`` and recognition mode may use them. Lazy import AND
lazy init: environments without rapidocr (or minutes away from a model
download) fail only when the driver actually recognizes.
"""

from __future__ import annotations

from typing import Any

from ubt.adapters.pdf.vlm.types import PageTranscript, VlmLine


class RapidOcrDriver:
    """ONNX detector+recognizer behind the VlmDriver contract."""

    name = "rapidocr"
    measured_boxes = True

    def __init__(self) -> None:
        self._engine: Any = None  # constructed on first recognize

    def _get_engine(self) -> Any:
        if self._engine is None:
            try:
                from rapidocr import RapidOCR as _RapidOCRMain

                self._engine = _RapidOCRMain()
            except ImportError:
                from rapidocr_onnxruntime import RapidOCR as _RapidOCROnnx

                self._engine = _RapidOCROnnx()
        return self._engine

    def recognize(
        self,
        image: object,
        page_size_pt: tuple[float, float],
        scale: float,
    ) -> PageTranscript:
        import numpy as np

        img = np.asarray(image)
        # rapidocr's LoadImage receives an ndarray unchanged (no channel
        # conversion for ndarray inputs) and its ONNX models expect BGR, the
        # cv2 convention. Callers hand us a PIL RGB image, so swap R/B;
        # ascontiguousarray keeps the negative-stride view safe for cv2.
        if img.ndim == 3 and img.shape[2] == 3:
            img = np.ascontiguousarray(img[:, :, ::-1])
        raw: Any = self._get_engine()(img)
        lines: list[VlmLine] = []
        for box, text, conf in _iter_predictions(raw):
            if not (text or "").strip():
                continue
            try:
                xs = [float(p[0]) / scale for p in box]
                ys = [float(p[1]) / scale for p in box]
                confidence = float(conf)
            except (TypeError, ValueError, IndexError):
                continue
            if not xs or not ys:
                continue
            # Pixel top-left origin -> PDF bottom-left points.
            _, height_pt = page_size_pt
            x0, x1 = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            lines.append(
                VlmLine(
                    text=str(text).strip(),
                    reading_index=len(lines),
                    confidence=confidence,
                    measured_box=(x0, height_pt - bottom, x1, height_pt - top),
                )
            )
        return PageTranscript(lines=tuple(lines), engine=self.name, measured_boxes=True)


def _iter_predictions(raw: Any) -> Any:
    """Yield (box, text, conf) across rapidocr output shapes (duck-typed).

    rapidocr 2.x returns ([[box, text, conf], ...], elapse); 3.x may return
    typed output objects — accept either, ignore anything else.
    """
    candidate = raw[0] if isinstance(raw, tuple) else raw
    boxes = getattr(candidate, "boxes", None)
    texts = getattr(candidate, "txts", getattr(candidate, "texts", None))
    confs = getattr(candidate, "scores", getattr(candidate, "confs", None))
    if boxes is not None and texts is not None:
        if confs is None:
            confs = [1.0] * len(texts)
        return list(zip(boxes, texts, confs, strict=False))
    return candidate or []

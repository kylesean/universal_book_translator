"""Rapidocr driver: classical detector OCR as the first working core.

rapidocr returns MEASURED boxes (detector output, not hallucination), so
``measured_boxes=True`` and recognition mode may use them. Lazy import AND
lazy init: environments without rapidocr (or minutes away from a model
download) fail only when the driver actually recognizes.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

from ubt.adapters.pdf.vlm.types import PageTranscript, VlmLine


class RapidOcrDriver:
    """ONNX detector+recognizer behind the VlmDriver contract."""

    name = "rapidocr"
    measured_boxes = True

    def __init__(self) -> None:
        self._engine: Any = None  # constructed on first recognize

    @classmethod
    def is_available(cls) -> bool:
        """Fast check to verify rapidocr and its inference backend (e.g. onnxruntime) can initialize."""
        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                driver = cls()
                engine = driver._get_engine()
                return engine is not None
        except Exception:
            return False

    def _get_engine(self) -> Any:
        if self._engine is None:
            err: Exception | None = None
            try:
                from rapidocr import RapidOCR as _RapidOCRMain

                self._engine = _RapidOCRMain()
                return self._engine
            except Exception as exc:
                err = exc

            try:
                from rapidocr_onnxruntime import RapidOCR as _RapidOCROnnx

                self._engine = _RapidOCROnnx()
                return self._engine
            except Exception as exc:
                err = exc

            raise ImportError(
                f"rapidocr engine could not be initialized ({err}). "
                "Ensure onnxruntime and rapidocr are installed: pip install onnxruntime rapidocr"
            ) from err
        return self._engine

    def recognize(
        self,
        image: object,
        page_size_pt: tuple[float, float],
        scale: float,
        rotation: int = 0,
    ) -> PageTranscript:
        import numpy as np

        from ubt.adapters.pdf.coordinate_resolver import (
            rotate_rect_clockwise,
            undo_page_rotation,
        )

        img = np.asarray(image)
        # rapidocr's LoadImage receives an ndarray unchanged (no channel
        # conversion for ndarray inputs) and its ONNX models expect BGR, the
        # cv2 convention. Callers hand us a PIL RGB image, so swap R/B;
        # ascontiguousarray keeps the negative-stride view safe for cv2.
        if img.ndim == 3 and img.shape[2] == 3:
            img = np.ascontiguousarray(img[:, :, ::-1])
        raw: Any = self._get_engine()(img)
        lines: list[VlmLine] = []
        # The pixel->point step below lands in the DISPLAY frame (the bitmap is
        # the displayed page), and only the last line of this method is allowed
        # to leave it: every box goes through the same display->user rotation,
        # so a /Rotate page is not handed to anchoring sideways.
        to_user_space = undo_page_rotation(rotation)
        page_width_pt, height_pt = page_size_pt
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
            x0, x1 = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            measured = rotate_rect_clockwise(
                (x0, height_pt - bottom, x1, height_pt - top),
                to_user_space,
                page_width_pt,
                height_pt,
            )
            lines.append(
                VlmLine(
                    text=str(text).strip(),
                    reading_index=len(lines),
                    confidence=confidence,
                    measured_box=measured,
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

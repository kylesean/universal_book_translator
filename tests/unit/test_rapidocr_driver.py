"""Regression tests for the rapidocr VLM driver image conventions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from ubt.adapters.pdf.vlm.drivers.rapidocr_driver import RapidOcrDriver

if TYPE_CHECKING:  # Numpy ships in the optional `ocr` extra
    import numpy as np


def test_recognize_converts_rgb_to_bgr() -> None:
    """rapidocr's ONNX models expect BGR (cv2 convention); callers pass PIL RGB."""
    # Kept as an in-test skip rather than a module-level importorskip: a
    # module-level skip drops every case in this file from collection, so a
    # missing optional extra would look like "no such tests" instead of the
    # visible skips this reports.
    pytest.importorskip("numpy", reason="numpy ships in the optional `ocr` extra")

    driver = RapidOcrDriver()
    captured: list[np.ndarray[Any, Any]] = []

    def fake_engine(img: np.ndarray[Any, Any]) -> tuple[list[Any], None]:
        captured.append(img)
        return [], None

    driver._engine = fake_engine

    rgb = Image.new("RGB", (2, 1))
    rgb.putpixel((0, 0), (10, 20, 30))
    rgb.putpixel((1, 0), (40, 50, 60))

    transcript = driver.recognize(rgb, (2.0, 1.0), 1.0)

    assert transcript.lines == ()
    assert len(captured) == 1
    arr = captured[0]
    assert list(arr[0, 0]) == [30, 20, 10]
    assert list(arr[0, 1]) == [60, 50, 40]

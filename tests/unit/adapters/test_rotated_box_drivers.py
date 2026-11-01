from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ubt.adapters.pdf.vlm.drivers.sidecar_driver import SidecarOcrDriver
from ubt.adapters.pdf.vlm.types import VlmDriver

pytestmark = pytest.mark.fast

_DISPLAY_PT = (500.0, 300.0)
_INK_PX = [445.0, 54.0, 456.0, 102.0]
_USER_PT = (54.0, 445.0, 102.0, 456.0)


def _http_client(payload: dict[str, object]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    client = MagicMock()
    client.post.return_value = resp
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    return client


def test_the_sidecar_driver_undoes_the_page_rotation() -> None:
    driver = SidecarOcrDriver(endpoint="http://localhost:8765")
    payload: dict[str, object] = {
        "lines": [{"text": "hello", "box": _INK_PX, "confidence": 0.9}],
        "coord_system": "image_pixel",
    }

    with patch(
        "ubt.adapters.pdf.vlm.drivers.sidecar_driver.httpx.Client",
        return_value=_http_client(payload),
    ):
        transcript = driver.recognize(Image.new("RGB", (500, 300)), _DISPLAY_PT, 1.0, rotation=90)

    (line,) = transcript.lines
    assert line.measured_box == pytest.approx(_USER_PT, abs=0.5)


def test_the_rapidocr_driver_undoes_the_page_rotation() -> None:
    pytest.importorskip("numpy")
    from ubt.adapters.pdf.vlm.drivers.rapidocr_driver import RapidOcrDriver

    corners = [[445.0, 54.0], [456.0, 54.0], [456.0, 102.0], [445.0, 102.0]]
    driver = RapidOcrDriver()
    driver._engine = lambda img: ([[corners, "hello", 0.9]], 0.0)  # noqa: SLF001

    transcript = driver.recognize(Image.new("RGB", (500, 300)), _DISPLAY_PT, 1.0, rotation=90)

    (line,) = transcript.lines
    assert line.measured_box == pytest.approx(_USER_PT, abs=0.5)


def test_the_sidecar_driver_keeps_its_boxes_on_an_unrotated_page() -> None:
    payload: dict[str, object] = {
        "lines": [{"text": "hello", "box": _INK_PX, "confidence": 0.9}],
        "coord_system": "image_pixel",
    }
    driver: VlmDriver = SidecarOcrDriver(endpoint="http://localhost:8765")

    with patch(
        "ubt.adapters.pdf.vlm.drivers.sidecar_driver.httpx.Client",
        return_value=_http_client(payload),
    ):
        transcript = driver.recognize(Image.new("RGB", (500, 300)), _DISPLAY_PT, 1.0)

    (line,) = transcript.lines
    assert line.measured_box == pytest.approx((445.0, 198.0, 456.0, 246.0), abs=0.5)


def test_the_sidecar_driver_applies_cropbox_origin() -> None:
    driver = SidecarOcrDriver(endpoint="http://localhost:8765")
    payload: dict[str, object] = {
        "lines": [{"text": "hello", "box": _INK_PX, "confidence": 0.9}],
        "coord_system": "image_pixel",
    }

    with patch(
        "ubt.adapters.pdf.vlm.drivers.sidecar_driver.httpx.Client",
        return_value=_http_client(payload),
    ):
        transcript = driver.recognize(
            Image.new("RGB", (500, 300)),
            _DISPLAY_PT,
            1.0,
            rotation=90,
            origin_pt=(50.0, 60.0),
        )

    (line,) = transcript.lines
    expected = (_USER_PT[0] + 50.0, _USER_PT[1] + 60.0, _USER_PT[2] + 50.0, _USER_PT[3] + 60.0)
    assert line.measured_box == pytest.approx(expected, abs=0.5)

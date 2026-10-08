"""Measuring drivers must convert a rotated page's boxes into user space.

The bitmap a driver sends (or POSTs) is the *displayed* page, so the boxes the
OCR comes back with are in the display frame. Every consumer downstream --
anchoring, block geometry, the render compositor's masks -- reads unrotated
user space, so the driver owes it the conversion. Both HTTP drivers do it by
construction now: the ``PageBBoxResolver`` they already ran the boxes through
is told the page's rotation.

The numbers are a real /Rotate 90 page: a 300x500 media box displayed as
500x300, with a line of text whose ink sits at image pixels (445, 54)-(456, 102)
and at PDF points (54, 446)-(102, 455) in user space.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ubt.adapters.pdf.vlm.drivers.cloud_driver import CloudOcrDriver
from ubt.adapters.pdf.vlm.drivers.sidecar_driver import SidecarOcrDriver

pytestmark = pytest.mark.fast

#: Displayed page size (the media box is 300x500, turned for display).
_DISPLAY_PT = (500.0, 300.0)
#: The text's ink in the rendered bitmap, 1 px to the point at scale 1.
_INK_PX = [445.0, 54.0, 456.0, 102.0]
#: The same text, as pdfium reports it in unrotated user space.
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


def test_the_cloud_rest_driver_undoes_the_page_rotation() -> None:
    driver = CloudOcrDriver(endpoint="https://ocr.example.com/api", api_key="k", provider="paddle")
    payload = {
        "lines": [{"text": "hello", "box": _INK_PX, "confidence": 0.9}],
        "coord_system": "image_pixel",
    }

    with patch(
        "ubt.adapters.pdf.vlm.drivers.cloud_driver.httpx.Client",
        return_value=_http_client(payload),
    ):
        transcript = driver.recognize(Image.new("RGB", (500, 300)), _DISPLAY_PT, 1.0, rotation=90)

    (line,) = transcript.lines
    assert line.measured_box == pytest.approx(_USER_PT, abs=0.5)


def test_the_sidecar_driver_undoes_the_page_rotation() -> None:
    driver = SidecarOcrDriver(endpoint="http://localhost:8765")
    payload = {
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
    # This driver converts its own boxes (pixel -> point -> rotate) rather than
    # routing them through PageBBoxResolver, so it is a second place the
    # conversion can be missed. The engine is stubbed: the conversion is the
    # subject, not the detector.
    pytest.importorskip("numpy")
    from ubt.adapters.pdf.vlm.drivers.rapidocr_driver import RapidOcrDriver

    corners = [[445.0, 54.0], [456.0, 54.0], [456.0, 102.0], [445.0, 102.0]]
    driver = RapidOcrDriver()
    driver._engine = lambda img: ([[corners, "hello", 0.9]], 0.0)  # noqa: SLF001

    transcript = driver.recognize(Image.new("RGB", (500, 300)), _DISPLAY_PT, 1.0, rotation=90)

    (line,) = transcript.lines
    assert line.measured_box == pytest.approx(_USER_PT, abs=0.5)


@pytest.mark.parametrize("driver_kind", ["cloud", "sidecar"])
def test_an_unrotated_page_keeps_the_drivers_old_boxes(driver_kind: str) -> None:
    # The regression guard for every document that was never turned: the same
    # payload with rotation=0 must come back as the display-frame box, exactly
    # as it did before the rotation was threaded through.
    payload = {
        "lines": [{"text": "hello", "box": _INK_PX, "confidence": 0.9}],
        "coord_system": "image_pixel",
    }
    if driver_kind == "cloud":
        driver = CloudOcrDriver(
            endpoint="https://ocr.example.com/api", api_key="k", provider="paddle"
        )
        target = "ubt.adapters.pdf.vlm.drivers.cloud_driver.httpx.Client"
    else:
        driver = SidecarOcrDriver(endpoint="http://localhost:8765")
        target = "ubt.adapters.pdf.vlm.drivers.sidecar_driver.httpx.Client"

    with patch(target, return_value=_http_client(payload)):
        transcript = driver.recognize(Image.new("RGB", (500, 300)), _DISPLAY_PT, 1.0)

    (line,) = transcript.lines
    # z0/y0 flipped out of image space, no rotation applied.
    assert line.measured_box == pytest.approx((445.0, 198.0, 456.0, 246.0), abs=0.5)

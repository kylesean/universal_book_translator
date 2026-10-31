"""HTTP sidecar OCR driver, registered as ``sidecar``.

Talks to a local or remote Docker/Podman OCR microservice over REST (PaddleOCR,
MinerU, Surya, or an enterprise gateway), so the page-level OCR models and their
CUDA dependencies stay outside this process. The wire contract is what
:meth:`SidecarOcrDriver.recognize` sends and reads: a JPEG multipart POST to
``/v1/ocr`` returning ``{"lines": [{"text", "box" or "bbox", "confidence"}],
"coord_system"}``, and ``/health`` for the availability probe used by driver
selection. Boxes may arrive in any of the coordinate spaces
:class:`PageBBoxResolver` understands, because the sidecars do not agree on one.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any

import httpx

from ubt.adapters.pdf.coordinate_resolver import PageBBoxResolver, undo_page_rotation
from ubt.adapters.pdf.vlm.types import PageTranscript, VlmLine

logger = logging.getLogger(__name__)

DEFAULT_SIDECAR_ENDPOINT = "http://localhost:8765"


class SidecarOcrDriver:
    """HTTP client driver for Docker/Podman OCR sidecars."""

    name = "sidecar"
    measured_boxes = True

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.endpoint = (
            endpoint or os.environ.get("UBT_OCR_ENDPOINT", DEFAULT_SIDECAR_ENDPOINT)
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("UBT_OCR_API_KEY")
        self.timeout = timeout

    @classmethod
    def is_healthy(
        cls,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float = 0.5,
    ) -> bool:
        """Blocking ``/health`` GET: True only on HTTP 200, False on any error.

        The call itself is synchronous, so callers on an event loop must not put
        it on a hot path; it is kept off them by the short default timeout (the
        driver-selection probe passes 0.3s) and by the fact that it runs in the
        CLI doctor and the pre-parse registry check rather than per page.
        """
        target = (endpoint or os.environ.get("UBT_OCR_ENDPOINT", DEFAULT_SIDECAR_ENDPOINT)).rstrip(
            "/"
        )
        health_url = f"{target}/health"
        # A sidecar behind UBT_OCR_SIDECAR_TOKEN requires the bearer on /health
        # too (any caller); send the same client key the OCR request uses.
        key = api_key or os.environ.get("UBT_OCR_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else None
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(health_url, headers=headers)
                return resp.status_code == 200
        except Exception:
            return False

    def recognize(
        self,
        image: Any,
        page_size_pt: tuple[float, float],
        scale: float,
        rotation: int = 0,
        origin_pt: tuple[float, float] = (0.0, 0.0),
    ) -> PageTranscript:
        """Send the page image to the OCR sidecar and return measured VlmLine items."""
        buf = io.BytesIO()
        if hasattr(image, "convert"):
            rgb_img = image.convert("RGB")
            rgb_img.save(buf, format="JPEG", quality=90)
        elif hasattr(image, "save"):
            image.save(buf, format="JPEG", quality=90)
        else:
            raise TypeError(f"Expected PIL Image, got {type(image)}")
        img_bytes = buf.getvalue()

        img_w = float(getattr(image, "width", page_size_pt[0] * scale))
        img_h = float(getattr(image, "height", page_size_pt[1] * scale))

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        ocr_url = f"{self.endpoint}/v1/ocr"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                files = {"file": ("page.jpg", img_bytes, "image/jpeg")}
                data = {
                    "width_pt": str(page_size_pt[0]),
                    "height_pt": str(page_size_pt[1]),
                    "scale": str(scale),
                }
                resp = client.post(ocr_url, files=files, data=data, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"OCR Sidecar connection failed at {ocr_url}. "
                f"Ensure your OCR container is running (e.g. `docker run -d -p 8765:8765 ubt-ocr-sidecar:latest`)."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"OCR Sidecar request failed at {ocr_url}: {exc}") from exc

        raw_lines = payload.get("lines") or []
        resolver = PageBBoxResolver(
            page_width=page_size_pt[0],
            page_height=page_size_pt[1],
            image_width=img_w,
            image_height=img_h,
            # The sidecar read the rendered (displayed) page, so its boxes are
            # in the display frame; undo the page's /Rotate to land in the
            # unrotated user-space frame block geometry and masks live in.
            rotation=undo_page_rotation(rotation),
            origin_x=origin_pt[0],
            origin_y=origin_pt[1],
        )
        coord_system = payload.get("coord_system")
        if not coord_system:
            # Decide the space once for the whole page: a per-box ``auto``
            # misreads a normalized-1000 box that happens to fit inside the page
            # as native points, scaling only some boxes and misplacing them.
            coord_system = PageBBoxResolver.infer_coord_system(
                [item.get("box") or item.get("bbox") for item in raw_lines],
                page_size_pt[0],
                page_size_pt[1],
                img_w,
                img_h,
            )

        vlm_lines: list[VlmLine] = []
        for item in raw_lines:
            text = str(item.get("text", "")).strip()
            if not text:
                continue

            raw_box = item.get("box") or item.get("bbox")
            measured_box: tuple[float, float, float, float] | None = None
            if raw_box and len(raw_box) == 4:
                measured_box = resolver.resolve_bbox(raw_box, coord_system=coord_system)

            confidence = float(item.get("confidence", 1.0))
            vlm_lines.append(
                VlmLine(
                    text=text,
                    confidence=confidence,
                    measured_box=measured_box,
                )
            )

        logger.info(
            "Sidecar OCR at %s recognized %d lines for page (%.1f x %.1f pt)",
            self.endpoint,
            len(vlm_lines),
            page_size_pt[0],
            page_size_pt[1],
        )
        has_measured = any(line.measured_box is not None for line in vlm_lines)
        self.measured_boxes = has_measured
        return PageTranscript(
            lines=tuple(vlm_lines),
            engine=f"sidecar:{self.endpoint}",
            measured_boxes=has_measured,
        )

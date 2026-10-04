"""Zero-AGPL pluggable HTTP Sidecar OCR Driver.

Connects UBT to a local or remote Docker sidecar microservice (e.g., PaddleOCR,
MinerU, Surya, or enterprise OCR gateway) over HTTP/REST.

Features:
- Completely decoupled: UBT stays lightweight without heavy C++/CUDA dependencies.
- Standard JSON wire format: accepts image, returns lines and bounding boxes.
- Zero-AGPL coordinate normalization: maps [0, 1], [0, 1000] (PaddleOCR grid),
  or pixel coordinates to canonical PDF points via PageBBoxResolver.
- Auto-probe / health check: ping /health to detect running sidecars.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any

import httpx

from ubt.adapters.pdf.coordinate_resolver import PageBBoxResolver
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
        """Fast non-blocking probe to check if the sidecar service is online."""
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
    ) -> PageTranscript:
        """Send page image to OCR sidecar and return measured VlmLine items."""
        # Convert PIL image to JPEG bytes
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
        for idx, item in enumerate(raw_lines):
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
                    reading_index=idx,
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

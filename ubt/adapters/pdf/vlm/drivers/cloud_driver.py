"""Zero-AGPL pluggable Cloud & Vision LLM OCR Driver.

Connects UBT to public cloud OCR APIs (e.g. Baidu, Tencent, Aliyun, Azure)
and OpenAI-compatible Vision LLMs (e.g. Qwen-VL, GPT-4o-mini, DeepSeek-VL, Claude).

Features:
- OpenAI Vision compatibility: sends base64 data URL to chat completions.
- Public Cloud REST API compatibility: normalizes Baidu, Azure, Tencent, or standard REST payloads.
- BBox normalization via PageBBoxResolver.
- Seamless fallback for environments without local GPU or Docker.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

import httpx

from ubt.adapters.pdf.coordinate_resolver import PageBBoxResolver
from ubt.adapters.pdf.vlm.types import PageTranscript, VlmLine

logger = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1"


def _record_ocr_usage(model: str, usage: Any) -> None:
    """Attribute one OCR call to the current run's usage sink.

    This driver reaches the provider over its own httpx client, so its tokens
    never pass ``ModelProvider._record_usage``; without this the whole OCR
    channel is invisible to ``estimate_cost_usd``, to UBT_BUDGET_USD and to the
    quality report, even though the pre-flight quote already prices it.

    A response with no ``usage`` block is booked as *unmeasured* rather than as
    zero tokens, so the cost reports as unknown instead of a fake $0.00.

    Scope note: only the vision-LLM path is metered here. The cloud-REST path
    (Baidu/Azure/Tencent) returns no token counts at all and has no price-table
    entry, so it cannot be sized; it stays unpriced rather than being guessed.
    """
    from ubt.core.router.provider import _extract_cached_tokens, record_external_usage

    if isinstance(usage, dict) and usage:
        record_external_usage(
            model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            # The shared helper, not a flat key: OpenAI reports the cache
            # discount under ``prompt_tokens_details``, which a bare
            # ``prompt_cache_hit_tokens`` read silently drops (over-billing the
            # quote against what the channel actually spent).
            cached_tokens=_extract_cached_tokens(usage),
        )
    else:
        record_external_usage(model, measured=False)


class CloudOcrDriver:
    """Universal cloud OCR driver supporting both Vision LLMs and Cloud REST APIs."""

    name = "cloud"

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        provider: str = "auto",
        model: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.endpoint = (
            endpoint
            or os.environ.get("UBT_OCR_ENDPOINT")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_OPENAI_ENDPOINT
        ).rstrip("/")
        self.api_key = (
            api_key or os.environ.get("UBT_OCR_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        self.provider = provider or os.environ.get("UBT_OCR_PROVIDER", "auto")
        self.model = model or os.environ.get("UBT_OCR_MODEL", DEFAULT_VISION_MODEL)
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        # Determine whether this driver yields measured boxes or vision text.
        # An explicit provider is authoritative; only "auto" falls back to
        # endpoint heuristics. Otherwise `--ocr cloud` against a default /
        # OpenAI-looking endpoint was silently treated as a vision LLM, which
        # returns no measured boxes and so cannot transcribe a scanned page.
        provider_kind = (self.provider or "auto").strip().lower()
        explicit_vision = provider_kind in ("vlm", "openai_vision")
        self._vision_llm = explicit_vision or (
            provider_kind == "auto"
            and (
                "/chat/completions" in self.endpoint
                or "api.openai.com" in self.endpoint
                or "api.deepseek.com" in self.endpoint
            )
        )
        self.measured_boxes = not self._vision_llm

    def recognize(
        self,
        image: Any,
        page_size_pt: tuple[float, float],
        scale: float,
    ) -> PageTranscript:
        """Route to Vision LLM or Cloud REST API based on provider."""
        # Convert PIL image to JPEG bytes
        buf = io.BytesIO()
        if hasattr(image, "convert"):
            rgb_img = image.convert("RGB")
            rgb_img.save(buf, format="JPEG", quality=85)
        elif hasattr(image, "save"):
            image.save(buf, format="JPEG", quality=85)
        else:
            raise TypeError(f"Expected PIL Image, got {type(image)}")
        img_bytes = buf.getvalue()

        if self._vision_llm:
            return self._recognize_via_vision_llm(img_bytes)
        return self._recognize_via_cloud_rest(img_bytes, image, page_size_pt, scale)

    def _recognize_via_vision_llm(self, img_bytes: bytes) -> PageTranscript:
        """Transcribe text using OpenAI-compatible multimodal vision endpoint."""
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64_str}"

        headers = {"Content-Type": "application/json"}
        if self.extra_headers:
            headers.update(self.extra_headers)
        if self.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"

        chat_url = (
            self.endpoint
            if self.endpoint.endswith("/chat/completions")
            else f"{self.endpoint}/chat/completions"
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert high-fidelity document transcription OCR engine. "
                        "Transcribe all readable text from this document image in strict natural reading order. "
                        "Output pure text line by line. Do not add markdown commentary or explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text lines from this document page verbatim.",
                        },
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            "temperature": 0.0,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(chat_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                _record_ocr_usage(self.model, data.get("usage"))
                content = data["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            raise RuntimeError(f"Vision LLM OCR request failed at {chat_url}: {exc}") from exc

        raw_lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
        vlm_lines = [
            VlmLine(text=text, reading_index=i, confidence=1.0, measured_box=None)
            for i, text in enumerate(raw_lines)
        ]

        logger.info(
            "Vision LLM OCR (%s) transcribed %d lines",
            self.model,
            len(vlm_lines),
        )
        return PageTranscript(
            lines=tuple(vlm_lines),
            engine=f"vlm:{self.model}",
            measured_boxes=False,
        )

    def _recognize_via_cloud_rest(
        self,
        img_bytes: bytes,
        image: Any,
        page_size_pt: tuple[float, float],
        scale: float,
    ) -> PageTranscript:
        """Call standard cloud OCR endpoint and parse returned lines + bounding boxes."""
        headers: dict[str, str] = {}
        if self.extra_headers:
            headers.update(self.extra_headers)
        if self.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"

        img_w = float(getattr(image, "width", page_size_pt[0] * scale))
        img_h = float(getattr(image, "height", page_size_pt[1] * scale))

        try:
            with httpx.Client(timeout=self.timeout) as client:
                files = {"file": ("page.jpg", img_bytes, "image/jpeg")}
                resp = client.post(self.endpoint, files=files, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"Cloud OCR request failed at {self.endpoint}: {exc}") from exc

        # Flexible parsing of varied cloud OCR shapes (standard / Baidu / Tencent / Azure)
        resolver = PageBBoxResolver(
            page_width=page_size_pt[0],
            page_height=page_size_pt[1],
            image_width=img_w,
            image_height=img_h,
        )

        vlm_lines: list[VlmLine] = []

        # 1. Standard format: {"lines": [{"text": ..., "box": ...}]}
        if "lines" in data and isinstance(data["lines"], list):
            coord_system = data.get("coord_system", "auto")
            for idx, item in enumerate(data["lines"]):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                raw_box = item.get("box") or item.get("bbox")
                box = resolver.resolve_bbox(raw_box, coord_system=coord_system) if raw_box else None
                vlm_lines.append(
                    VlmLine(
                        text=text,
                        reading_index=idx,
                        confidence=float(item.get("confidence", 1.0)),
                        measured_box=box,
                    )
                )

        # 2. Baidu format: {"words_result": [{"words": ..., "location": {"left": ..., "top": ..., "width": ..., "height": ...}}]}
        elif "words_result" in data and isinstance(data["words_result"], list):
            for idx, item in enumerate(data["words_result"]):
                text = str(item.get("words", "")).strip()
                if not text:
                    continue
                loc = item.get("location") or {}
                left, t, w, h = (
                    loc.get("left", 0),
                    loc.get("top", 0),
                    loc.get("width", 0),
                    loc.get("height", 0),
                )
                raw_box = (left, t, left + w, t + h)
                box = resolver.resolve_bbox(raw_box, coord_system="image_pixel", origin="top-left")
                vlm_lines.append(
                    VlmLine(text=text, reading_index=idx, confidence=1.0, measured_box=box)
                )

        # 3. Azure format: {"readResults": [{"lines": [{"text": ..., "boundingBox": [...]}]}]}
        elif "readResults" in data:
            results = data.get("readResults", [])
            line_idx = 0
            for page_res in results:
                for ln in page_res.get("lines", []):
                    text = str(ln.get("text", "")).strip()
                    if not text:
                        continue
                    pts = ln.get("boundingBox", [])
                    box = None
                    if len(pts) >= 8:
                        xs = [pts[i] for i in range(0, 8, 2)]
                        ys = [pts[i] for i in range(1, 8, 2)]
                        raw_box = (min(xs), min(ys), max(xs), max(ys))
                        box = resolver.resolve_bbox(
                            raw_box, coord_system="image_pixel", origin="top-left"
                        )
                    vlm_lines.append(
                        VlmLine(text=text, reading_index=line_idx, confidence=1.0, measured_box=box)
                    )
                    line_idx += 1

        else:
            # Flat text fallback
            raw_text = str(data.get("text") or data.get("result") or "")
            for idx, ln in enumerate(raw_text.splitlines()):
                if ln.strip():
                    vlm_lines.append(
                        VlmLine(
                            text=ln.strip(), reading_index=idx, confidence=1.0, measured_box=None
                        )
                    )

        return PageTranscript(
            lines=tuple(vlm_lines),
            engine=f"cloud:{self.endpoint}",
            measured_boxes=any(line.measured_box is not None for line in vlm_lines),
        )

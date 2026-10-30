"""OpenAI-compatible vision-endpoint OCR driver.

One wire protocol, one class: POST the page image as a base64 data URL to
``{endpoint}/chat/completions`` and read the lines back. Every modern
inference server speaks this contract — llama-server/llama-swap, SGLang,
vLLM, Ollama on localhost, or a hosted VLM — so locality, not the driver,
decides privacy: the same class serves ``ocr_mode='local'`` (a localhost
endpoint, no egress) and ``ocr_mode='cloud'`` (a remote endpoint, gated by
``UBT_ALLOW_PAGE_UPLOAD``).

Like every LLM-VLM driver, ``measured_boxes=False``: the model emits text in
reading order and no trustworthy coordinates. The anchoring layer refuses it
in recognition mode (a true scan cannot be transcribed by a text-only
witness) and uses it in proofread mode (pages that already have a text
layer, where the driver's job is cross-checking the stronger pdfium witness).
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

import httpx

from ubt.adapters.pdf.vlm.types import PageTranscript, VlmLine

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1"


def _record_ocr_usage(model: str, usage: Any) -> None:
    """Attribute one OCR call to the current run's usage sink.

    This driver reaches the provider over its own httpx client, so its tokens
    never pass ``ModelProvider._record_usage``; without this the whole OCR
    channel is invisible to ``estimate_cost_usd``, to UBT_BUDGET_USD and to the
    quality report, even though the pre-flight quote already prices it.

    A response with no ``usage`` block is booked as *unmeasured* rather than as
    zero tokens, so the cost reports as unknown instead of a fake $0.00.
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


class VisionEndpointDriver:
    """OCR over an OpenAI-compatible vision chat endpoint (local or remote)."""

    name = "vision"

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.endpoint = (
            endpoint or os.environ.get("UBT_OCR_ENDPOINT") or DEFAULT_OPENAI_ENDPOINT
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("UBT_OCR_API_KEY")
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.measured_boxes = False
        # No vendor default: the vision route must be told which model to call
        # (and, on a multi-model gateway like llama-swap, *which model to
        # route to*). Fail closed here rather than ship ``"model": ""``.
        self.model = (model or os.environ.get("UBT_OCR_MODEL") or "").strip()
        if not self.model:
            raise ValueError(
                "Vision OCR needs an explicit model: this channel does not fall "
                "back to a vendor default. Set UBT_OCR_MODEL (or ocr_model in the "
                "run config) to the vision model the endpoint should serve; "
                "choose a measured-box engine (ocr_mode='local' with rapidocr or "
                "a sidecar) or set ocr_mode='off' to skip OCR instead."
            )

    def recognize(
        self,
        image: Any,
        page_size_pt: tuple[float, float],
        scale: float,
        rotation: int = 0,
        origin_pt: tuple[float, float] = (0.0, 0.0),
    ) -> PageTranscript:
        """Transcribe one rendered page image via the vision chat endpoint."""
        # Convert PIL image to JPEG bytes
        buf = io.BytesIO()
        if hasattr(image, "convert"):
            rgb_img = image.convert("RGB")
            rgb_img.save(buf, format="JPEG", quality=85)
        elif hasattr(image, "save"):
            image.save(buf, format="JPEG", quality=85)
        else:
            raise TypeError(f"Expected PIL Image, got {type(image)}")
        return self._transcribe(buf.getvalue())

    def _transcribe(self, img_bytes: bytes) -> PageTranscript:
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

        finish_reason: str | None = None
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(chat_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                _record_ocr_usage(self.model, data.get("usage"))
                choice = data["choices"][0]
                content = choice["message"]["content"] or ""
                finish_reason = choice.get("finish_reason")
        except Exception as exc:
            raise RuntimeError(f"Vision LLM OCR request failed at {chat_url}: {exc}") from exc

        # A length-capped answer is a truncated page, not a full transcription:
        # without this the driver returned half a page as if it were complete.
        # OCR has no continuation seam, so surface it (and flag the transcript)
        # instead of silently shipping missing lines.
        truncated = str(finish_reason or "") == "length"
        if truncated:
            logger.warning(
                "Vision LLM OCR (%s) hit the output token limit "
                "(finish_reason=length); the transcription is truncated and its "
                "tail is missing.",
                self.model,
            )

        raw_lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
        vlm_lines = [
            VlmLine(text=text, confidence=1.0, measured_box=None)
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
            truncated=truncated,
        )

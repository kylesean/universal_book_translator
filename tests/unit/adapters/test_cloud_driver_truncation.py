"""Cloud vision-LLM OCR must surface an output-limit truncation, not ship it."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ubt.adapters.pdf.vlm.drivers.cloud_driver import CloudOcrDriver

pytestmark = pytest.mark.fast


def _driver() -> CloudOcrDriver:
    return CloudOcrDriver(
        endpoint="https://api.openai.com/v1",
        api_key="k",
        provider="vlm",
        model="gpt-4o-mini",
    )


def _client(payload: dict[str, object]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    client = MagicMock()
    client.post.return_value = resp
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    return client


def _payload(content: str, finish_reason: str) -> dict[str, object]:
    return {
        "choices": [
            {"message": {"content": content}, "finish_reason": finish_reason},
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_a_length_capped_ocr_response_is_flagged_and_warned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _client(_payload("line1\nline2", "length"))
    with (
        patch(
            "ubt.adapters.pdf.vlm.drivers.cloud_driver.httpx.Client",
            return_value=client,
        ),
        caplog.at_level(logging.WARNING, logger="ubt.adapters.pdf.vlm.drivers.cloud_driver"),
    ):
        transcript = _driver().recognize(Image.new("RGB", (4, 4)), (600.0, 800.0), 2.0)

    assert transcript.truncated is True
    assert len(transcript.lines) == 2
    assert any("finish_reason=length" in record.getMessage() for record in caplog.records)


def test_a_complete_ocr_response_is_not_flagged(caplog: pytest.LogCaptureFixture) -> None:
    client = _client(_payload("line1\nline2", "stop"))
    with (
        patch(
            "ubt.adapters.pdf.vlm.drivers.cloud_driver.httpx.Client",
            return_value=client,
        ),
        caplog.at_level(logging.WARNING, logger="ubt.adapters.pdf.vlm.drivers.cloud_driver"),
    ):
        transcript = _driver().recognize(Image.new("RGB", (4, 4)), (600.0, 800.0), 2.0)

    assert transcript.truncated is False
    assert len(transcript.lines) == 2
    assert caplog.records == []

from __future__ import annotations

import httpx
import pytest

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.transports.gemini import GeminiTransport

pytestmark = pytest.mark.fast


@pytest.mark.asyncio
async def test_gemini_blocked_200_still_bills_its_prompt_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "promptFeedback": {"blockReason": "SAFETY"},
                "usageMetadata": {"promptTokenCount": 42, "candidatesTokenCount": 0},
            },
        )

    transport = GeminiTransport(
        api_key="k",
        default_model="gemini-x",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ModelProviderError, match="blocked the prompt"):
        await transport.generate_with_finish_reason("hello")
    await transport.aclose()

    assert transport.usage_log
    assert transport.usage_log[0]["prompt_tokens"] == 42

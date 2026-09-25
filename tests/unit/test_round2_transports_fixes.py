"""Regression tests for Batch 1 fixes: Transports, Extractor, Pricing & Rate Limiter."""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.extractor import TranslationOutputExtractor
from ubt.core.router.pricing import resolve_model_prices
from ubt.core.router.provider import OpenAICompatibleProvider
from ubt.core.router.rate_limiter import SqliteTokenBucket
from ubt.core.router.transports.openai_chat import OpenAIChatTransport
from ubt.core.router.transports.openai_responses import OpenAIResponsesTransport


@pytest.mark.fast
def test_responses_api_reasoning_nested_format() -> None:
    transport = OpenAIResponsesTransport(api_key="mock", base_url="https://api.openai.com/v1")
    captured_payloads: list[dict[str, Any]] = []

    mock_client = AsyncMock()

    async def fake_post(url: str, **kwargs: Any) -> MagicMock:
        json_payload = kwargs.get("json")
        if isinstance(json_payload, dict):
            captured_payloads.append(json_payload)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello"}],
                }
            ],
        }
        return resp

    mock_client.post = fake_post
    mock_client.is_closed = False
    transport._client = mock_client
    transport._owned_client = False

    asyncio.run(
        transport.generate(
            prompt="test",
            model="o3-mini",
            reasoning_effort="high",
        )
    )

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert "reasoning_effort" not in payload, (
        "Responses API must not have top-level reasoning_effort"
    )
    assert payload.get("reasoning") == {"effort": "high"}


@pytest.mark.fast
def test_responses_api_incomplete_reasoning_returns_length_finish() -> None:
    transport = OpenAIResponsesTransport(api_key="mock", base_url="https://api.openai.com/v1")
    mock_client = AsyncMock()

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "still thinking..."}],
            }
        ],
    }
    mock_client.post.return_value = resp
    mock_client.is_closed = False
    transport._client = mock_client
    transport._owned_client = False

    result, finish_reason = asyncio.run(
        transport.generate_with_finish_reason(
            prompt="test",
            model="o3-mini",
        )
    )
    assert finish_reason == "length"
    assert result == ""


@pytest.mark.fast
def test_openai_chat_batch_input_file_cleanup_on_error() -> None:
    transport = OpenAIChatTransport(api_key="mock", base_url="https://api.openai.com/v1")
    mock_client = AsyncMock()

    # Upload succeeds
    upload_resp = MagicMock()
    upload_resp.status_code = 200
    upload_resp.json.return_value = {"id": "file-12345"}

    # Batch creation fails with 400
    create_resp = MagicMock()
    create_resp.status_code = 400
    create_resp.text = "Invalid batch"

    mock_client.post.side_effect = [upload_resp, create_resp]
    delete_called_with = []

    async def fake_delete(url: str, **kwargs: object) -> MagicMock:
        delete_called_with.append(url)
        del_resp = MagicMock()
        del_resp.status_code = 200
        return del_resp

    mock_client.delete = fake_delete
    mock_client.is_closed = False
    transport._client = mock_client
    transport._owned_client = False

    with pytest.raises(ModelProviderError):
        asyncio.run(
            transport.create_batch_job(
                [
                    {
                        "custom_id": "c1",
                        "body": {
                            "model": "gpt-4o",
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    }
                ],
            )
        )

    assert any("file-12345" in url for url in delete_called_with), (
        "file_id must be cleaned up on batch creation failure"
    )


@pytest.mark.fast
def test_openai_chat_batch_cleanup_handles_error_file_id() -> None:
    transport = OpenAIChatTransport(api_key="mock", base_url="https://api.openai.com/v1")
    mock_client = AsyncMock()

    job_status_resp = MagicMock()
    job_status_resp.status_code = 200
    job_status_resp.json.return_value = {
        "id": "batch_abc",
        "input_file_id": "file-in",
        "output_file_id": "file-out",
        "error_file_id": "file-err",
        "error": None,
    }
    mock_client.get.return_value = job_status_resp

    deleted_urls = []

    async def fake_delete(url: str, **kwargs: object) -> MagicMock:
        deleted_urls.append(url)
        del_resp = MagicMock()
        del_resp.status_code = 200
        return del_resp

    mock_client.delete = fake_delete
    mock_client.is_closed = False
    transport._client = mock_client
    transport._owned_client = False

    asyncio.run(transport.cleanup_batch_files("batch_abc"))
    assert any("file-err" in url for url in deleted_urls), (
        "error_file_id must be deleted during batch cleanup"
    )


@pytest.mark.fast
def test_openai_chat_batch_result_preserves_http_error() -> None:
    transport = OpenAIChatTransport(api_key="mock", base_url="https://api.openai.com/v1")
    mock_client = AsyncMock()

    batch_status_resp = MagicMock()
    batch_status_resp.status_code = 200
    batch_status_resp.json.return_value = {
        "id": "batch_abc",
        "status": "completed",
        "output_file_id": "file-out",
    }

    # One line failed with HTTP 400
    line_json = {
        "custom_id": "req-1",
        "response": {
            "status_code": 400,
            "body": {
                "error": {
                    "message": "Context length exceeded",
                    "type": "invalid_request_error",
                }
            },
        },
        "error": None,
    }
    file_content_resp = MagicMock()
    file_content_resp.status_code = 200
    file_content_resp.text = json.dumps(line_json) + "\n"

    async def fake_get(url: str, **kwargs: object) -> MagicMock:
        if "/batches/" in url:
            return batch_status_resp
        return file_content_resp

    mock_client.get = fake_get
    mock_client.is_closed = False
    transport._client = mock_client
    transport._owned_client = False

    results = asyncio.run(transport.fetch_batch_results("batch_abc"))
    assert "req-1" in results
    err = results["req-1"].get("error")
    assert err is not None
    assert "Context length exceeded" in err, f"Expected actual error message, got {err}"


@pytest.mark.fast
def test_extractor_prioritizes_final_translation() -> None:
    text = (
        "Here is my thinking:\n"
        "<translation>粗糙草稿：这是测试</translation>\n"
        "After careful reflection, I should improve this:\n"
        "<final_translation>精修定稿：这是高保真测试</final_translation>\n"
    )
    extracted = TranslationOutputExtractor.extract(text, strategy="xml_tags")
    assert extracted == "精修定稿：这是高保真测试"


@pytest.mark.fast
def test_extractor_cleans_reasoning_with_attributes_and_whitespace() -> None:
    text = (
        '<think class="r1">\nStep 1: analyze.\nStep 2: conclude.\n</think >\n这是真正的翻译输出。'
    )
    extracted = TranslationOutputExtractor.extract(text, strategy="raw")
    assert extracted == "这是真正的翻译输出。"
    assert "Step 1" not in extracted


@pytest.mark.fast
def test_pricing_multi_segment_namespace() -> None:
    standard = resolve_model_prices("claude-3-5-sonnet")
    namespaced = resolve_model_prices("openrouter/anthropic/claude-3-5-sonnet")
    assert standard != (0.0, 0.0)
    assert namespaced == standard, f"Namespaced model prices {namespaced} should match {standard}"


@pytest.mark.fast
def test_sqlite_token_bucket_preserves_negative_debt_on_429(tmp_path: Path) -> None:
    db_path = tmp_path / "rate_limiter.sqlite"
    bucket = SqliteTokenBucket(
        path=db_path,
        initial_rpm=60,
        initial_tpm=100000,
        min_rpm=10,
        min_tpm=1000,
    )
    # Put bucket into debt
    with bucket._txn() as conn:
        state, last_update = bucket._load(conn)
        state.rpm_tokens = -15.0
        state.tpm_tokens = -5000.0
        bucket._store(conn, state, last_update)

    bucket.report_429()

    with bucket._txn() as conn:
        state, _ = bucket._load(conn)
        assert state.rpm_tokens <= -10.0 or state.rpm_tokens < 0.0, (
            f"Expected rpm_tokens to retain negative debt, got {state.rpm_tokens}"
        )
        assert state.tpm_tokens <= -2000.0 or state.tpm_tokens < 0.0, (
            f"Expected tpm_tokens to retain negative debt, got {state.tpm_tokens}"
        )


@pytest.mark.fast
def test_provider_aclose_closes_all_transports() -> None:
    provider = OpenAICompatibleProvider(api_key="mock", base_url="https://api.openai.com/v1")
    t1 = provider._chat_transport
    t2 = provider._anthropic_transport
    t3 = provider._responses_transport

    m1 = AsyncMock()
    m2 = AsyncMock()
    m3 = AsyncMock()
    t1.__dict__["aclose"] = m1
    t2.__dict__["aclose"] = m2
    t3.__dict__["aclose"] = m3

    asyncio.run(provider.aclose())

    m1.assert_awaited_once()
    m2.assert_awaited_once()
    m3.assert_awaited_once()

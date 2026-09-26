"""Unit tests for Model Capabilities, Registry, and Strategy-based Router dispatch."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from ubt.api.app import create_app
from ubt.core.exceptions import ModelProviderError
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
from ubt.core.router import (
    ExtractionStrategy,
    MockModelProvider,
    ModelCapabilityRegistry,
    ModelProfile,
    ModelRouter,
    PromptStrategy,
    TranslationOutputExtractor,
    get_default_registry,
)


def test_registry_builtin_profiles_resolution() -> None:
    """Verify built-in profiles resolve expected capability strategies."""
    registry = ModelCapabilityRegistry()

    # 1. General LLM
    deepseek_prof = registry.resolve("deepseek-v4-flash")
    assert deepseek_prof.prompt_strategy == PromptStrategy.RICH
    assert deepseek_prof.extraction_strategy == ExtractionStrategy.AUTO
    assert deepseek_prof.supports_reasoning_effort is True
    assert deepseek_prof.supports_system_prompt is True

    # 2. General open instruction LLMs (Qwen 2.5, Llama 3) use RICH prompt strategy
    qwen_prof = registry.resolve("Qwen/Qwen2.5-7B-Instruct")
    assert qwen_prof.prompt_strategy == PromptStrategy.RICH
    assert qwen_prof.extraction_strategy == ExtractionStrategy.AUTO
    assert qwen_prof.supports_system_prompt is True

    llama_prof = registry.resolve("meta-llama/Llama-3.1-8B-Instruct")
    assert llama_prof.prompt_strategy == PromptStrategy.RICH
    assert llama_prof.extraction_strategy == ExtractionStrategy.AUTO
    assert llama_prof.supports_system_prompt is True

    # 4. Unknown model safe fallback
    unknown_prof = registry.resolve("unseen-proprietary-model-v1")
    assert unknown_prof.prompt_strategy == PromptStrategy.RICH
    assert unknown_prof.extraction_strategy == ExtractionStrategy.AUTO
    assert unknown_prof.supports_reasoning_effort is False


def test_registry_enterprise_custom_override() -> None:
    """Verify enterprise custom models take precedence and can be resolved dynamically."""
    registry = ModelCapabilityRegistry()

    custom = ModelProfile(
        model_pattern="corp-fintech-mt-v3",
        prompt_strategy=PromptStrategy.MINIMAL,
        extraction_strategy=ExtractionStrategy.RAW,
        supports_system_prompt=False,
        supports_reasoning_effort=False,
        display_name="Enterprise In-House MT",
    )
    registry.register(custom)

    resolved = registry.resolve("corp-fintech-mt-v3-checkpoint-1000")
    assert resolved.display_name == "Enterprise In-House MT"
    assert resolved.prompt_strategy == PromptStrategy.MINIMAL
    assert resolved.extraction_strategy == ExtractionStrategy.RAW
    assert resolved.supports_system_prompt is False


def test_extractor_strategies() -> None:
    """Verify TranslationOutputExtractor behaviors across all strategies."""
    # 1. RAW
    raw_input = "  Here is the translation: 你好世界  "
    assert (
        TranslationOutputExtractor.extract(raw_input, strategy=ExtractionStrategy.RAW)
        == "Here is the translation: 你好世界"
    )

    # 2. XML_TAG
    xml_input = "<think>reasoning</think><translation>这是一个标准翻译</translation>"
    assert (
        TranslationOutputExtractor.extract(xml_input, strategy=ExtractionStrategy.XML_TAG)
        == "这是一个标准翻译"
    )

    # 3. CONVERSATIONAL_PREFIX
    prefix_input = "Here is the final translation: 这是前缀去除测试"
    assert (
        TranslationOutputExtractor.extract(
            prefix_input, strategy=ExtractionStrategy.CONVERSATIONAL_PREFIX
        )
        == "这是前缀去除测试"
    )

    # 4. AUTO with XML tags present
    auto_xml = "<translation>优先提取标签</translation>"
    assert (
        TranslationOutputExtractor.extract(auto_xml, strategy=ExtractionStrategy.AUTO)
        == "优先提取标签"
    )

    # 5. AUTO fallback without XML tags
    auto_fallback = "### Translation:\n这是回退清理测试"
    assert (
        TranslationOutputExtractor.extract(auto_fallback, strategy=ExtractionStrategy.AUTO)
        == "这是回退清理测试"
    )


def test_router_prompt_construction_strategies() -> None:
    """Verify ModelRouter dispatches to MINIMAL, HYBRID, and RICH prompt builders correctly."""
    registry = ModelCapabilityRegistry()
    registry.register(
        ModelProfile(
            model_pattern="custom-minimal",
            prompt_strategy=PromptStrategy.MINIMAL,
            extraction_strategy=ExtractionStrategy.RAW,
            supports_system_prompt=False,
        )
    )
    registry.register(
        ModelProfile(
            model_pattern="custom-hybrid",
            prompt_strategy=PromptStrategy.HYBRID,
            extraction_strategy=ExtractionStrategy.AUTO,
            supports_system_prompt=True,
        )
    )
    mock_provider = MockModelProvider()
    router = ModelRouter(provider=mock_provider, registry=registry)

    # MINIMAL (via custom profile)
    sys_min, user_min = router.build_draft_prompt(
        source_text="Hello world",
        model="custom-minimal",
    )
    assert sys_min == ""
    assert "<translation>" not in user_min
    assert "Translate the following" in user_min
    assert "Hello world" in user_min

    # HYBRID (via custom profile)
    sys_hyb, user_hyb = router.build_draft_prompt(
        source_text="Hello world",
        model="custom-hybrid",
    )
    assert "You are a professional book translator" in sys_hyb
    assert "Provide the direct" in user_hyb
    assert "Hello world" in user_hyb

    # RICH (Qwen and DeepSeek now default to RICH)
    sys_qwen, user_qwen = router.build_draft_prompt(
        source_text="Hello world",
        model="qwen-2.5-7b",
    )
    assert "<translation>...</translation>" in sys_qwen
    assert "### Source Paragraph to Translate" in user_qwen

    sys_rich, user_rich = router.build_draft_prompt(
        source_text="Hello world",
        model="deepseek-v4-flash",
    )
    assert "<translation>...</translation>" in sys_rich
    assert "### Source Paragraph to Translate" in user_rich


@pytest.mark.asyncio
async def test_router_parameter_pre_cleaning_for_unsupported_features() -> None:
    """Verify that parameters like system_prompt and reasoning_effort are pre-cleaned based on profile."""
    registry = ModelCapabilityRegistry()
    mock_provider = MockModelProvider(default_response="你好")

    # Enterprise model without system prompt and without reasoning effort
    profile = ModelProfile(
        model_pattern="custom-ollama-mt",
        prompt_strategy=PromptStrategy.MINIMAL,
        extraction_strategy=ExtractionStrategy.RAW,
        supports_system_prompt=False,
        supports_reasoning_effort=False,
        supports_temperature=False,
    )
    registry.register(profile)

    router = ModelRouter(
        provider=mock_provider,
        draft_model="custom-ollama-mt",
        registry=registry,
        draft_reasoning_effort="high",
    )

    block = IRBlock(
        id="b01",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Test source text",
        status=BlockStatus.PENDING,
    )

    result = await router.draft(block=block, temperature=0.7, reasoning_effort="high")
    assert result == "你好"

    call = mock_provider.call_history[0]
    # System prompt should be None because supports_system_prompt is False
    assert call["system_prompt"] is None
    # Reasoning effort should be stripped (None) because supports_reasoning_effort is False
    assert call["reasoning_effort"] is None


@pytest.mark.asyncio
async def test_router_preserves_supported_features_for_frontier_models() -> None:
    """Verify that reasoning_effort and system_prompt are preserved when model supports them."""
    registry = ModelCapabilityRegistry()
    mock_provider = MockModelProvider(default_response="<translation>你好世界</translation>")

    router = ModelRouter(
        provider=mock_provider,
        draft_model="deepseek-v4-flash",
        registry=registry,
        draft_reasoning_effort="low",
    )

    block = IRBlock(
        id="b02",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="Test source text",
    )

    result = await router.draft(block=block)
    assert result == "你好世界"

    call = mock_provider.call_history[0]
    assert call["system_prompt"] is not None
    assert call["reasoning_effort"] == "low"


def test_api_model_profiles_rest_endpoints() -> None:
    """Verify FastAPI endpoints for reading and registering model profiles."""
    app = create_app()
    client = TestClient(app)

    # 1. GET list of profiles
    res_get = client.get("/api/v1/model-profiles")
    assert res_get.status_code == 200
    profiles = res_get.json()
    assert isinstance(profiles, list)
    assert any(p["model_pattern"] == "deepseek" for p in profiles)

    # 2. POST new custom profile
    new_profile = {
        "model_pattern": "test-fintech-mt-2026",
        "prompt_strategy": "minimal",
        "extraction_strategy": "raw",
        "supports_system_prompt": False,
        "supports_reasoning_effort": False,
        "display_name": "Fintech MT 2026",
    }
    res_post = client.post("/api/v1/model-profiles", json=new_profile)
    assert res_post.status_code == 200
    assert res_post.json()["model_pattern"] == "test-fintech-mt-2026"

    # 3. Verify registry reflects the posted profile
    reg = get_default_registry()
    resolved = reg.resolve("test-fintech-mt-2026-v1")
    assert resolved.display_name == "Fintech MT 2026"
    assert resolved.prompt_strategy == PromptStrategy.MINIMAL


@pytest.mark.asyncio
async def test_router_repair_with_image_b64() -> None:
    """Verify that router.repair uses complete_with_images when image_b64 is passed."""
    provider = MockModelProvider(
        default_response="<final_translation>视觉修复成功</final_translation>"
    )
    router = ModelRouter(
        provider=provider, repair_model="mock-vision-repair", allow_page_upload=True
    )

    block = IRBlock(
        id="b_math",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="where Vtm = kBT / q is the thermal voltage",
    )

    repaired = await router.repair(
        block=block,
        draft_text="其中 Vtm 是电压",
        error_flags=["formula_defect"],
        image_b64="fake_base64_crop",
    )

    assert repaired == "视觉修复成功"
    # Verify vision was invoked
    vision_calls = [c for c in provider.call_history if c.get("image_count", 0) > 0]
    assert len(vision_calls) == 1
    assert vision_calls[0]["image_count"] == 1
    assert "visual crop" in vision_calls[0]["prompt"]


@pytest.mark.asyncio
async def test_router_repair_non_vision_model_repairs_from_text() -> None:
    """A non-vision repair model ignores the crop and repairs from text."""

    class UnsupportedVisionProvider(MockModelProvider):
        async def generate_with_images(self, *args: Any, **kwargs: Any) -> str:
            raise NotImplementedError("Vision not supported")

    provider = UnsupportedVisionProvider(
        default_response="<final_translation>纯文本修复成功</final_translation>"
    )
    router = ModelRouter(provider=provider, repair_model="mock-repair")

    block = IRBlock(
        id="b_math",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="where Vtm = kBT / q is the thermal voltage",
    )

    repaired = await router.repair(
        block=block,
        draft_text="其中 Vtm 是电压",
        error_flags=["formula_defect"],
        image_b64="fake_base64_crop",
    )

    assert repaired == "纯文本修复成功"


@pytest.mark.asyncio
async def test_router_repair_never_egresses_pages_when_upload_disabled() -> None:
    """allow_page_upload=false must skip the visual crop even for a
    vision-capable repair model and degrade to text repair (privacy
    kill-switch, 2026-09 review)."""
    from ubt.core.exceptions import ModelProviderError

    class ExplodingVisionProvider(MockModelProvider):
        async def generate_with_images(self, *args: Any, **kwargs: Any) -> str:
            raise AssertionError("vision egress attempted while allow_page_upload=false")

    provider = ExplodingVisionProvider(
        default_response="<final_translation>纯文本降级修复</final_translation>"
    )
    router = ModelRouter(
        provider=provider,
        repair_model="mock-vision-repair",
        allow_page_upload=False,
    )
    block = IRBlock(
        id="b_math",
        spine_index=1,
        source_text="where Vtm = kBT / q is the thermal voltage",
    )
    repaired = await router.repair(
        block=block,
        draft_text="其中 Vtm 是电压",
        error_flags=["formula_defect"],
        image_b64="crop-b64",
    )
    assert repaired == "纯文本降级修复"
    assert not [c for c in provider.call_history if c.get("image_count", 0) > 0]

    with pytest.raises(ModelProviderError, match="page-image egress disabled"):
        await router.complete_with_images(prompt="p", images_b64_png=["x"])


def test_provider_sanitize_respects_xml_tag_and_raw_profiles() -> None:
    """Regression (2026-09 review): provider-side AUTO unwrap used to run
    before the router's strategy-aware extraction, so an XML_TAG profile saw
    no tags left to find and got "" — an empty translation for the whole
    book. RAW profiles were likewise rewritten (prefix stripped) despite
    declaring verbatim passthrough."""
    from ubt.core.router.provider import sanitize_thought_output

    reg = get_default_registry()
    reg.register(
        ModelProfile(
            model_pattern="unittest-xmltag",
            extraction_strategy=ExtractionStrategy.XML_TAG,
        ),
        override=True,
    )
    reg.register(
        ModelProfile(
            model_pattern="unittest-raw",
            extraction_strategy=ExtractionStrategy.RAW,
        ),
        override=True,
    )
    try:
        tagged = "分析完毕。\n<translation>天空是蓝的</translation>"
        cleaned = sanitize_thought_output(tagged, "unittest-xmltag")
        assert "<translation>天空是蓝的</translation>" in cleaned
        # The router-layer strategy pass now finds the content instead of "".
        assert (
            TranslationOutputExtractor.extract(cleaned, strategy=ExtractionStrategy.XML_TAG)
            == "天空是蓝的"
        )
        # reasoning traces are still a transport-layer concern for XML_TAG:
        assert (
            sanitize_thought_output(
                "<think>推理</think><translation>好</translation>", "unittest-xmltag"
            )
            == "<translation>好</translation>"
        )

        # RAW: verbatim, no conversational-prefix stripping.
        assert (
            sanitize_thought_output("中文翻译：逐字原样 <b>标签</b>", "unittest-raw")
            == "中文翻译：逐字原样 <b>标签</b>"
        )

        # AUTO (unknown model): unwrap behavior unchanged.
        assert (
            sanitize_thought_output("<translation>甲</translation>", "unittest-unmatched") == "甲"
        )
        # No model at all: legacy AUTO default.
        assert sanitize_thought_output("<translation>乙</translation>") == "乙"
    finally:
        reg._profiles = [p for p in reg._profiles if not p.model_pattern.startswith("unittest-")]


class _RecordingLimiter:
    """Stands in for the AIMD bucket so the test can see every signal."""

    def __init__(self) -> None:
        self.acquired: list[int] = []
        self.signals: list[str] = []

    async def acquire(self, estimated_tokens: int = 500) -> None:
        self.acquired.append(estimated_tokens)

    def report_success(self) -> None:
        self.signals.append("success")

    def report_429(self) -> None:
        self.signals.append("429")


class _VisionProvider(MockModelProvider):
    def __init__(self, *, fail_with_429: bool = False) -> None:
        super().__init__()
        self._fail_with_429 = fail_with_429

    async def generate_with_images(
        self,
        prompt: str,
        images_b64_png: list[str],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        if self._fail_with_429:
            raise ModelProviderError("rate limit exceeded", details={"status_code": 429})
        return "图已读"


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_vision_channel_uses_the_rate_limiter(fail: bool) -> None:
    """Page crops are the heaviest requests and must not skip the bucket.

    ``complete_with_images`` used to call the provider directly: no token
    reserved, no 429 fed back, and repair() swallowed the failure into a log line
    — so the most expensive channel neither respected nor influenced the AIMD
    throttling the text path relied on (2026-09 review).
    """
    from ubt.core.router.router import _VISION_TOKENS_PER_PAGE_IMAGE

    limiter = _RecordingLimiter()
    router = ModelRouter(
        provider=_VisionProvider(fail_with_429=fail),
        draft_model="mock-draft",
        repair_model="mock-repair",
        rate_limiter=limiter,  # type: ignore[arg-type]
        allow_page_upload=True,
    )

    if fail:
        with pytest.raises(ModelProviderError):
            await router.complete_with_images("读这页", ["iVBORw0KGgo", "iVBORw0KGgo"])
        assert limiter.signals == ["429"]
    else:
        assert await router.complete_with_images("读这页", ["iVBORw0KGgo"]) == "图已读"
        assert limiter.signals == ["success"]

    assert len(limiter.acquired) == 1
    # Base64 characters are not text tokens: the reservation must be dominated
    # by the per-image tile allowance, not by len(images) // 4.
    assert limiter.acquired[0] >= _VISION_TOKENS_PER_PAGE_IMAGE

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ubt.core.qe.heuristic import HeuristicQERunner
from ubt.core.qe.llm_judge import (
    JUDGE_SYSTEM_PROMPT,
    LLMJudgeQERunner,
    TieredQERunner,
    build_judge_prompts,
)

pytestmark = pytest.mark.fast

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "measured_gate_false_positives.json"


def _semantic_defect_cases() -> list[dict[str, Any]]:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return [case for case in payload["cases"] if case.get("_defect") == "semantic-loss"]


def test_the_fixture_still_holds_the_defects_this_layer_exists_for() -> None:
    cases = _semantic_defect_cases()
    assert {case["block_id"] for case in cases} == {"pdf_main#b0010", "pdf_main#b0015"}
    assert "公开" not in cases[0]["target_text"]
    assert "内容" not in cases[1]["target_text"]


@pytest.mark.asyncio
async def test_the_heuristic_cannot_see_either_defect() -> None:
    runner = HeuristicQERunner()
    cases = _semantic_defect_cases()
    scores = await runner.score_pairs(
        [{"src": case["source_text"], "mt": case["target_text"]} for case in cases]
    )
    for case, score in zip(cases, scores, strict=True):
        assert score == pytest.approx(0.92), f"{case['block_id']} scored {score}"


def test_a_clean_pass_is_never_judged_without_a_sample_rate() -> None:
    tier = TieredQERunner(heuristic=HeuristicQERunner(), judge=None, pass_sample=0.0)
    for case in _semantic_defect_cases():
        assert tier._needs_judge(0.92, case["source_text"]) is False


def test_a_clean_pass_is_judged_when_sampling_is_on() -> None:
    tier = TieredQERunner(heuristic=HeuristicQERunner(), judge=None, pass_sample=1.0)
    for case in _semantic_defect_cases():
        assert tier._needs_judge(0.92, case["source_text"]) is True


def test_a_hard_defect_is_not_sent_to_the_judge() -> None:
    tier = TieredQERunner(heuristic=HeuristicQERunner(), judge=None, pass_sample=1.0)
    assert tier._needs_judge(0.55, "any source text") is False


def test_the_rubric_names_compression_that_reads_well() -> None:
    lowered = JUDGE_SYSTEM_PROMPT.lower()
    assert "clause by clause" in lowered
    assert "compression" in lowered
    assert "drops source content" in lowered
    assert "publicly available" in JUDGE_SYSTEM_PROMPT
    assert "content interference" in JUDGE_SYSTEM_PROMPT


def test_the_judge_prompt_carries_both_sides_verbatim() -> None:
    system, user = build_judge_prompts("publicly available", "获取")
    assert "publicly available" in user
    assert "获取" in user
    assert "score:" in user
    assert system == JUDGE_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_a_judge_that_sees_the_loss_lowers_the_score() -> None:
    case = _semantic_defect_cases()[0]

    async def judge_fn(**kwargs: Any) -> str:
        assert kwargs["user_prompt"].strip()
        return "score: 55"

    runner = LLMJudgeQERunner(judge_fn=judge_fn).with_languages("en", "zh")
    scores = await runner.score_pairs([{"src": case["source_text"], "mt": case["target_text"]}])
    assert scores[0] == pytest.approx(0.55)


def test_sentences_split_on_cjk_punctuation_without_a_space() -> None:
    from ubt.core.qe.llm_judge import split_sentences

    assert split_sentences("这是第一句。这是第二句！这是第三句？") == [
        "这是第一句。",
        "这是第二句！",
        "这是第三句？",
    ]


def test_sentences_do_not_split_at_decimals_or_abbreviations() -> None:
    from ubt.core.qe.llm_judge import split_sentences

    assert split_sentences("The rate is 0.85. See Fig. 3. Done.") == [
        "The rate is 0.85.",
        "See Fig. 3. Done.",
    ]
    assert split_sentences("The result is 0.85 higher.") == ["The result is 0.85 higher."]


def test_a_block_score_is_the_worst_sentence() -> None:
    import asyncio

    async def judge_fn(**kwargs: Any) -> str:
        return "score: 95" if "publicly" not in kwargs["user_prompt"] else "score: 40"

    runner = LLMJudgeQERunner(judge_fn=judge_fn, compare_sentences=True)
    src = (
        ". ".join(f"Sentence {i} is fine" for i in range(9))
        + ". The weights are publicly available."
    )
    mt = ". ".join(f"第{i}句没问题" for i in range(9)) + "。权重可获取。"
    scores = asyncio.run(runner.score_pairs([{"src": src, "mt": mt}]))
    assert scores[0] == pytest.approx(0.40)


def test_a_misaligned_pair_falls_back_instead_of_inventing_a_defect() -> None:
    import asyncio

    calls: list[str] = []

    async def judge_fn(**kwargs: Any) -> str:
        calls.append(kwargs["user_prompt"])
        return "score: 30"

    runner = LLMJudgeQERunner(judge_fn=judge_fn, compare_sentences=True)
    asyncio.run(runner.score_pairs([{"src": "A one. B two. C three.", "mt": "一句话。"}]))
    assert len(calls) == 1


def test_sentence_comparison_can_be_switched_off() -> None:
    import asyncio

    calls: list[str] = []

    async def judge_fn(**kwargs: Any) -> str:
        calls.append(kwargs["user_prompt"])
        return "score: 90"

    runner = LLMJudgeQERunner(judge_fn=judge_fn, compare_sentences=False)
    asyncio.run(runner.score_pairs([{"src": "A one. B two.", "mt": "一。二。"}]))
    assert len(calls) == 1


def test_the_judge_endpoint_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    from ubt.core.config import UBTConfig

    monkeypatch.setenv("UBT_QE_JUDGE_BASE_URL", "https://judge.example/v1")
    monkeypatch.setenv("UBT_QE_JUDGE_MODEL", "some-strong-model")
    monkeypatch.setenv("UBT_QE_JUDGE_API_MODE", "openai-responses")
    monkeypatch.setenv("UBT_QE_JUDGE_SENTENCES", "true")
    cfg = UBTConfig()
    assert cfg.qe_judge_base_url == "https://judge.example/v1"
    assert cfg.qe_judge_model == "some-strong-model"
    assert cfg.qe_judge_api_mode == "openai-responses"
    assert cfg.qe_judge_sentences is True


def test_the_judge_falls_back_to_the_translation_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from ubt.core.config import UBTConfig

    monkeypatch.setenv("UBT_BASE_URL", "http://127.0.0.1:9090/v1")
    for name in (
        "UBT_QE_JUDGE_BASE_URL",
        "UBT_QE_JUDGE_API_MODE",
        "UBT_QE_JUDGE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = UBTConfig()
    assert cfg.qe_judge_base_url is None
    assert cfg.qe_judge_api_mode is None
    assert cfg.qe_judge_sentences is True


def test_a_judge_verdict_survives_the_calibration_check() -> None:
    import asyncio

    async def judge_fn(**kwargs: Any) -> str:
        return "score: 78"

    tier = TieredQERunner(
        heuristic=HeuristicQERunner(),
        judge=LLMJudgeQERunner(judge_fn=judge_fn),
        pass_sample=1.0,
    )
    assert tier.judged_this_run is False
    asyncio.run(tier.score_pairs([{"src": "A. B.", "mt": "甲。乙。"}]))
    assert tier.judged_this_run is True
    assert tier.is_calibrated() is False


def test_a_judge_verdict_below_the_line_demotes_a_passing_block() -> None:
    import asyncio

    from ubt.core.engine.stages.quality_gate import _audit_pass_sample
    from ubt.core.ir.models import BlockType, FlowID, IRBlock, make_element

    async def judge_fn(**kwargs: Any) -> str:
        return "score: 78"

    tier = TieredQERunner(
        heuristic=HeuristicQERunner(),
        judge=LLMJudgeQERunner(judge_fn=judge_fn),
        pass_sample=1.0,
    )
    block = IRBlock(
        element=make_element(
            id="t#b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="The weights are publicly available.",
        )
    )
    block.target_text = "权重可获取。"
    demoted, scored = asyncio.run(_audit_pass_sample(tier, [block], 0.80))
    assert [b.id for b in demoted] == ["t#b1"]
    assert [b.id for b in scored] == ["t#b1"]


def test_a_good_block_is_not_demoted() -> None:
    import asyncio

    from ubt.core.engine.stages.quality_gate import _audit_pass_sample
    from ubt.core.ir.models import BlockType, FlowID, IRBlock, make_element

    async def judge_fn(**kwargs: Any) -> str:
        return "score: 95"

    tier = TieredQERunner(
        heuristic=HeuristicQERunner(),
        judge=LLMJudgeQERunner(judge_fn=judge_fn),
        pass_sample=1.0,
    )
    block = IRBlock(
        element=make_element(
            id="t#b2",
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="The weights are publicly available.",
        )
    )
    block.target_text = "权重公开可用。"
    demoted, scored = asyncio.run(_audit_pass_sample(tier, [block], 0.80))
    assert demoted == []
    assert [b.mtqe_score for b in scored] == [pytest.approx(0.92)]


def test_an_unguided_runner_never_promises_calibration() -> None:
    assert HeuristicQERunner().may_be_calibrated() is False


def test_a_judge_provider_block_is_a_first_class_setting() -> None:
    from ubt.core.providers import PROVIDER_ALLOWED_KEYS

    assert "qe_judge_provider" in PROVIDER_ALLOWED_KEYS
    assert "repair_provider" in PROVIDER_ALLOWED_KEYS


def test_a_judge_provider_block_resolves_like_any_other(tmp_path: Any) -> None:
    from ubt.core.providers import load_provider_block

    config = tmp_path / "config.toml"
    config.write_text(
        "[providers.qe-judge]\n"
        'base_url = "https://judge.example/v1"\n'
        'api_mode = "openai-responses"\n'
        'api_key = "${JUDGE_KEY}"\n'
        'repair_model = "some-strong-model"\n',
        encoding="utf-8",
    )
    fields = load_provider_block("qe-judge", custom_path=config)
    assert fields["base_url"] == "https://judge.example/v1"
    assert fields["api_mode"] == "openai-responses"
    assert fields["repair_model"] == "some-strong-model"


def test_no_judge_endpoint_means_the_translation_provider_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ubt.core.config import UBTConfig

    for name in ("UBT_QE_JUDGE_PROVIDER", "UBT_QE_JUDGE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    cfg = UBTConfig()
    assert cfg.qe_judge_provider is None
    assert cfg.qe_judge_base_url is None


def test_an_unknown_judge_provider_fails_loudly() -> None:
    from ubt.core.providers import ProviderNotFoundError, load_layer

    with pytest.raises(ProviderNotFoundError):
        load_layer("no-such-judge-provider")


def test_defaults_may_enable_the_semantic_judge(tmp_path: Any) -> None:
    from ubt.core.providers import load_defaults_block

    config = tmp_path / "config.toml"
    config.write_text(
        "[defaults]\n"
        'qe_engine = "tiered"\n'
        "qe_judge_enabled = true\n"
        "qe_judge_pass_sample = 0.15\n"
        "qe_judge_sentences = true\n"
        'qe_judge_provider = "my-judge"\n'
        "qe_threshold = 0.8\n",
        encoding="utf-8",
    )
    fields = load_defaults_block(custom_path=config)
    assert fields["qe_engine"] == "tiered"
    assert fields["qe_judge_enabled"] is True
    assert fields["qe_judge_pass_sample"] == 0.15
    assert fields["qe_judge_provider"] == "my-judge"
    assert fields["qe_threshold"] == 0.8


def test_a_secret_is_still_rejected_in_defaults(tmp_path: Any) -> None:
    from ubt.core.providers import ProviderConfigError, load_defaults_block

    config = tmp_path / "config.toml"
    config.write_text('[defaults]\nservice_api_key = "nope"\n', encoding="utf-8")
    with pytest.raises(ProviderConfigError):
        load_defaults_block(custom_path=config)


def test_a_judge_provider_naming_nothing_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ubt.core.config import UBTConfig
    from ubt.core.providers import ProviderNotFoundError

    monkeypatch.setenv("UBT_QE_JUDGE_PROVIDER", "no-such-provider")
    cfg = UBTConfig()
    assert cfg.qe_judge_provider == "no-such-provider"
    with pytest.raises(ProviderNotFoundError):
        from ubt.core.providers import load_layer

        load_layer("no-such-provider")


def test_a_judge_provider_block_supplies_its_own_model_name(tmp_path: Any) -> None:
    from ubt.core.providers import load_provider_block

    config = tmp_path / "config.toml"
    config.write_text(
        "[providers.my-judge]\n"
        'base_url = "https://judge.example/v1"\n'
        'repair_model = "judge-model-name"\n',
        encoding="utf-8",
    )
    fields = load_provider_block("my-judge", custom_path=config)
    assert fields["repair_model"] == "judge-model-name"
    assert fields["repair_model"] != "gemma4-e4b"


def _block(block_id: str, block_type: Any, source: str, target: str, *, skip: bool) -> Any:
    from ubt.core.ir.models import FlowID, IRBlock, make_element

    block = IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=block_type,
            flow_id=FlowID.MAIN_STORY,
            source_text=source,
        )
    )
    block.target_text = target
    block.skip_translate = skip
    return block


def test_an_asset_path_is_not_judged() -> None:
    from ubt.core.engine.stages.quality_gate import _is_unscorable
    from ubt.core.ir.models import BlockType

    path = "/home/kyle/.cache/ubt/docling_cache/assets/6331259d7f49aa708.png"
    assert _is_unscorable(_block("img_1", BlockType.IMAGE, path, path, skip=True)) is True


def test_a_verbatim_kept_block_is_not_judged() -> None:
    from ubt.core.engine.stages.quality_gate import _is_unscorable
    from ubt.core.ir.models import BlockType

    assert (
        _is_unscorable(
            _block(
                "b1",
                BlockType.NARRATIVE,
                "Cheng Cui, Tingquan Gao, Xueqing Wang",
                "Cheng Cui, Tingquan Gao, Xueqing Wang",
                skip=True,
            )
        )
        is True
    )


def test_real_translations_are_still_judged() -> None:
    from ubt.core.engine.stages.quality_gate import _is_unscorable
    from ubt.core.ir.models import BlockType

    assert (
        _is_unscorable(
            _block(
                "b15",
                BlockType.NARRATIVE,
                "current methodologies follow two suboptimal paths",
                "当前方法通常沿两条次优路径展开",
                skip=False,
            )
        )
        is False
    )
    assert (
        _is_unscorable(
            _block(
                "b69",
                BlockType.HEADING,
                "3.4. Physical Spatial Aware Data Augmentation",
                "3.4. 物理空间感知数据增强",
                skip=False,
            )
        )
        is False
    )


def test_the_audit_skips_unscorable_blocks_before_calling_the_judge() -> None:
    import asyncio

    from ubt.core.engine.stages.quality_gate import _audit_pass_sample
    from ubt.core.ir.models import BlockType
    from ubt.core.qe.heuristic import HeuristicQERunner
    from ubt.core.qe.llm_judge import LLMJudgeQERunner, TieredQERunner

    calls: list[str] = []

    async def judge_fn(**kwargs: Any) -> str:
        calls.append(kwargs["user_prompt"])
        return "score: 30"

    tier = TieredQERunner(
        heuristic=HeuristicQERunner(),
        judge=LLMJudgeQERunner(judge_fn=judge_fn, compare_sentences=False),
        pass_sample=1.0,
    )
    path = "/tmp/assets/x.png"
    blocks = [
        _block("img_1", BlockType.IMAGE, path, path, skip=True),
        _block(
            "b1",
            BlockType.NARRATIVE,
            "the weights are publicly available",
            "权重可获取",
            skip=False,
        ),
    ]
    demoted, scored = asyncio.run(_audit_pass_sample(tier, blocks, 0.75))
    assert [b.id for b in demoted] == ["b1"]
    assert [b.id for b in scored] == ["b1"]
    assert len(calls) == 1


def test_glossary_is_pushed_into_the_omission_gate() -> None:
    from ubt.core.qe.heuristic import HeuristicQERunner

    glossary = [{"source": "state-of-the-art", "translation": "当前最先进的"}]
    unbound = HeuristicQERunner()
    assert unbound.fast_pass.omission_gate._glossary_surfaces == []

    bound = unbound.with_glossary(glossary)
    assert bound.fast_pass.omission_gate._glossary_surfaces == [
        ("state-of-the-art", "当前最先进的")
    ]


def test_an_alias_lets_the_omission_gate_accept_a_rendered_abbreviation() -> None:
    from ubt.core.qe.heuristic import HeuristicQERunner

    source = (
        "Extensive experiments demonstrate that RT-DocLayout achieves SOTA accuracy "
        "while maintaining real-time inference efficiency."
    )
    target = "大量实验表明，RT-DocLayout 在保持实时推理效率的同时实现了当前最先进的准确性。"
    glossary = [{"source": "state-of-the-art", "translation": "当前最先进的", "aliases": ["SOTA"]}]
    gate = HeuristicQERunner().with_glossary(glossary).fast_pass.omission_gate
    assert gate.evaluate(source, target).passed is True


def test_the_message_points_at_the_fix_when_a_glossary_is_bound() -> None:
    from ubt.core.qe.heuristic import HeuristicQERunner

    source = "The method achieves SOTA accuracy and RT-DocLayout efficiency."
    target = "该方法实现了当前最先进的准确性以及 RT-DocLayout 的效率。"
    glossary = [{"source": "state-of-the-art", "translation": "当前最先进的"}]
    decision = (
        HeuristicQERunner().with_glossary(glossary).fast_pass.omission_gate.evaluate(source, target)
    )
    assert decision.passed is False
    assert "alias" in decision.reason

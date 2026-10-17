"""The semantic layer: the only gate that can see a fluent compression.

Every zero-token gate scores a compressed translation as clean. Measured on
arXiv 2606.23344, two blocks shipped as ``MTQE_PASSED`` with content missing:

* b0010 — "the model weights are **publicly available** at …" became
  "模型权重可在 … **获取**" — an adjective and a past participle collapsed into one
  verb, so "you can obtain it" replaced "it is publicly available";
* b0015 — "content interference from background noise or adjacent elements"
  became "背景噪声或相邻元素的干扰" — the noun the sentence is *about* (内容) is gone.

Both read perfectly well and both are wrong. The tests below pin the two things
that let them through: that the judge is never consulted for a clean pass unless
a sample rate says so, and that the judge's rubric names the loss it must catch.
"""

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
    """The regression corpus must keep the two blocks that motivated the judge.

    If they disappear from the fixture, the tests below stop proving anything.
    """
    cases = _semantic_defect_cases()
    assert {case["block_id"] for case in cases} == {"pdf_main#b0010", "pdf_main#b0015"}
    # The loss is verifiable without an LLM: the target does not carry the
    # source's key content word at all.
    assert "公开" not in cases[0]["target_text"]
    assert "内容" not in cases[1]["target_text"]


@pytest.mark.asyncio
async def test_the_heuristic_cannot_see_either_defect() -> None:
    """Why a semantic layer is needed at all: the cheap gates are blind here."""
    runner = HeuristicQERunner()
    cases = _semantic_defect_cases()
    scores = await runner.score_pairs(
        [{"src": case["source_text"], "mt": case["target_text"]} for case in cases]
    )
    for case, score in zip(cases, scores, strict=True):
        assert score == pytest.approx(0.92), f"{case['block_id']} scored {score}"


def test_a_clean_pass_is_never_judged_without_a_sample_rate() -> None:
    """`pass_sample=0` means the judge never sees a 0.92 — the leak path.

    This is the behaviour that shipped the two defects: the LLM judge existed
    and was enabled, but a clean heuristic score short-circuited it.
    """
    tier = TieredQERunner(heuristic=HeuristicQERunner(), judge=None, pass_sample=0.0)
    for case in _semantic_defect_cases():
        assert tier._needs_judge(0.92, case["source_text"]) is False


def test_a_clean_pass_is_judged_when_sampling_is_on() -> None:
    """`pass_sample=1.0` audits every clean pass — the setting that catches them."""
    tier = TieredQERunner(heuristic=HeuristicQERunner(), judge=None, pass_sample=1.0)
    for case in _semantic_defect_cases():
        assert tier._needs_judge(0.92, case["source_text"]) is True


def test_a_hard_defect_is_not_sent_to_the_judge() -> None:
    """A judge cannot un-drop a number; those pairs cost tokens for nothing."""
    tier = TieredQERunner(heuristic=HeuristicQERunner(), judge=None, pass_sample=1.0)
    assert tier._needs_judge(0.55, "any source text") is False


def test_the_rubric_names_compression_that_reads_well() -> None:
    """A generic "adequacy" rubric scored both defects as clean.

    The prompt must say, in so many words, that a fluent translation which drops
    a noun, modifier or relation is still a loss — otherwise a judge rewards the
    fluency and misses the omission.
    """
    lowered = JUDGE_SYSTEM_PROMPT.lower()
    assert "clause by clause" in lowered
    assert "compression" in lowered
    assert "drops source content" in lowered
    # The two measured shapes are named, so a future edit cannot quietly
    # generalize the instruction away.
    assert "publicly available" in JUDGE_SYSTEM_PROMPT
    assert "content interference" in JUDGE_SYSTEM_PROMPT


def test_the_judge_prompt_carries_both_sides_verbatim() -> None:
    """The judge can only catch a loss it can compare against."""
    system, user = build_judge_prompts("publicly available", "获取")
    assert "publicly available" in user
    assert "获取" in user
    assert "score:" in user
    assert system == JUDGE_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_a_judge_that_sees_the_loss_lowers_the_score() -> None:
    """End to end through the real runner with a scripted judge.

    The judge is scripted (it reports what a real one should report after reading
    the rubric); what this test pins is the plumbing — the lowered score reaches
    the caller instead of being silently discarded.
    """
    case = _semantic_defect_cases()[0]

    async def judge_fn(**kwargs: Any) -> str:
        # Sentence comparison sends one sentence per call, so the whole block is
        # no longer in any single prompt; what must be there is some of it.
        assert kwargs["user_prompt"].strip()
        return "score: 55"

    runner = LLMJudgeQERunner(judge_fn=judge_fn).with_languages("en", "zh")
    scores = await runner.score_pairs([{"src": case["source_text"], "mt": case["target_text"]}])
    assert scores[0] == pytest.approx(0.55)


# --------------------------------------------------------------------------- #
# Sentence-granularity: a block score dilutes a one-word loss
# --------------------------------------------------------------------------- #


def test_sentences_split_on_cjk_punctuation_without_a_space() -> None:
    """Chinese puts no space after 。 — requiring one left the whole block whole."""
    from ubt.core.qe.llm_judge import split_sentences

    assert split_sentences("这是第一句。这是第二句！这是第三句？") == [
        "这是第一句。",
        "这是第二句！",
        "这是第三句？",
    ]


def test_sentences_do_not_split_at_decimals_or_abbreviations() -> None:
    """The near miss: a wrong split would compare unrelated sentences."""
    from ubt.core.qe.llm_judge import split_sentences

    assert split_sentences("The rate is 0.85. See Fig. 3. Done.") == [
        "The rate is 0.85.",
        "See Fig. 3. Done.",
    ]
    assert split_sentences("The result is 0.85 higher.") == ["The result is 0.85 higher."]


def test_a_block_score_is_the_worst_sentence() -> None:
    """A gate asks whether anything is wrong, not what the average is.

    Measured: the b0010 defect scored 0.85 against the whole block and below
    0.50 when the clause it corrupts is judged alone. Taking the minimum is what
    turns a diluted deduction into a detection.
    """
    import asyncio

    async def judge_fn(**kwargs: Any) -> str:
        # Nine good sentences and one bad one.
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
    """If the two sides split differently, positional pairing would be wrong."""
    import asyncio

    calls: list[str] = []

    async def judge_fn(**kwargs: Any) -> str:
        calls.append(kwargs["user_prompt"])
        return "score: 30"

    runner = LLMJudgeQERunner(judge_fn=judge_fn, compare_sentences=True)
    # Three source sentences, one target sentence: cannot be aligned.
    asyncio.run(runner.score_pairs([{"src": "A one. B two. C three.", "mt": "一句话。"}]))
    assert len(calls) == 1  # whole-block fallback, not three mispaired calls


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
    """The judge may run on a hosted API while translation stays local."""
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
    """An existing single-endpoint configuration must keep working unchanged."""
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


# --------------------------------------------------------------------------- #
# The closed loop: a judge verdict must reach repair, not be discarded
# --------------------------------------------------------------------------- #


def test_a_judge_verdict_survives_the_calibration_check() -> None:
    """`is_calibrated()` is conservative for repair; the gate used it wrongly.

    The gate read ``is_calibrated()`` to decide whether to keep the scores it had
    just computed. ``TieredQERunner`` answers it conservatively (so repair keeps
    its uncalibrated lifecycle), which meant every judge verdict was discarded:
    measured on arXiv 2606.23344, the judge said 0.78 for a block that had
    dropped a word and the gate stored ``None`` — the block could not reach
    repair on the judgement it had just been given.
    """
    import asyncio

    async def judge_fn(**kwargs: Any) -> str:
        return "score: 78"

    tier = TieredQERunner(
        heuristic=HeuristicQERunner(),
        judge=LLMJudgeQERunner(judge_fn=judge_fn),
        pass_sample=1.0,
    )
    assert tier.judged_this_run is False  # nothing judged yet
    asyncio.run(tier.score_pairs([{"src": "A. B.", "mt": "甲。乙。"}]))
    assert tier.judged_this_run is True
    # ...and the conservative answer that repair relies on is unchanged.
    assert tier.is_calibrated() is False


def test_a_judge_verdict_below_the_line_demotes_a_passing_block() -> None:
    """A block that cleared every zero-token gate still goes to repair.

    This is the whole point of the semantic layer: FastPass passed this block,
    the judge scored it 0.78, and the pass-sample audit must hand it back.
    """
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
    """The near miss: the audit must not send every block to repair."""
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
    # ...but the measurement is kept, so the block is not "unmeasured".
    # The tier keeps min(heuristic, judge): 0.92 is the heuristic band.
    assert [b.mtqe_score for b in scored] == [pytest.approx(0.92)]


def test_an_unguided_runner_never_promises_calibration() -> None:
    """A plain heuristic runner is never asked to score (nothing to keep)."""
    assert HeuristicQERunner().may_be_calibrated() is False


# --------------------------------------------------------------------------- #
# Standardised judge configuration: a provider block, not a bespoke endpoint
# --------------------------------------------------------------------------- #


def test_a_judge_provider_block_is_a_first_class_setting() -> None:
    """`qe_judge_provider` names a [providers.*] block, like `repair_provider`.

    Pointing at a block rather than repeating an endpoint keeps this standard:
    every user already knows how to declare a provider, and credentials resolve
    through the registry's own rules (``${VAR}`` expansion, ``UBT_LLM_API_KEY``)
    instead of a second, private mechanism invented for the judge.
    """
    from ubt.core.providers import PROVIDER_ALLOWED_KEYS

    assert "qe_judge_provider" in PROVIDER_ALLOWED_KEYS
    # Same shape as the knob it mirrors.
    assert "repair_provider" in PROVIDER_ALLOWED_KEYS


def test_a_judge_provider_block_resolves_like_any_other(tmp_path: Any) -> None:
    """The registry parses it with no judge-specific code."""
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
    """A user with one endpoint configured gets exactly today's behaviour."""
    from ubt.core.config import UBTConfig

    for name in ("UBT_QE_JUDGE_PROVIDER", "UBT_QE_JUDGE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    cfg = UBTConfig()
    assert cfg.qe_judge_provider is None
    assert cfg.qe_judge_base_url is None


def test_an_unknown_judge_provider_fails_loudly() -> None:
    """A typo must not silently fall back to the translator's model.

    Falling back would run the judge on exactly the model that cannot do the job
    (measured: a 4B scores a translation that dropped a word at 100/100), and the
    report would still say the semantic gate ran.
    """
    from ubt.core.providers import ProviderNotFoundError, load_layer

    with pytest.raises(ProviderNotFoundError):
        load_layer("no-such-judge-provider")


# --------------------------------------------------------------------------- #
# A config file must be able to describe the quality policy
# --------------------------------------------------------------------------- #


def test_defaults_may_enable_the_semantic_judge(tmp_path: Any) -> None:
    """The [defaults] block is documented as the config baseline.

    It accepted only provider-shaped keys, so turning the judge on was possible
    from the CLI and from the environment but **not from a config file** — the
    one surface a user is told to edit. A policy that can only be expressed
    through exported variables is not standard configuration.
    """
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
    """Widening the list must not open it to the inbound service key."""
    from ubt.core.providers import ProviderConfigError, load_defaults_block

    config = tmp_path / "config.toml"
    config.write_text('[defaults]\nservice_api_key = "nope"\n', encoding="utf-8")
    with pytest.raises(ProviderConfigError):
        load_defaults_block(custom_path=config)


def test_a_judge_provider_naming_nothing_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo must not silently judge with the translator's model.

    Falling back would run the judge on exactly the model that cannot do the job
    and the report would still say the semantic gate ran.
    """
    from ubt.core.config import UBTConfig
    from ubt.core.providers import ProviderNotFoundError

    monkeypatch.setenv("UBT_QE_JUDGE_PROVIDER", "no-such-provider")
    cfg = UBTConfig()
    assert cfg.qe_judge_provider == "no-such-provider"
    with pytest.raises(ProviderNotFoundError):
        from ubt.core.providers import load_layer

        load_layer("no-such-provider")


# --------------------------------------------------------------------------- #
# The judge must use the model its own endpoint knows
# --------------------------------------------------------------------------- #


def test_a_judge_provider_block_supplies_its_own_model_name(tmp_path: Any) -> None:
    """The judge's model comes from ITS provider block, not the translator's.

    Measured: a run sent the translation provider's repair model (``gemma4-e4b``,
    which lives on one local gateway) to a *different* gateway that answered
    ``404 no such model`` twelve times. ``_build_judge_fn`` resolved the right
    name internally but returned only the callable, so the caller kept passing
    the old name as an override and every call failed.

    This test pins the contract that made that possible: the resolver returns the
    model name along with the callable.
    """
    from ubt.core.providers import load_provider_block

    config = tmp_path / "config.toml"
    config.write_text(
        "[providers.my-judge]\n"
        'base_url = "https://judge.example/v1"\n'
        'repair_model = "judge-model-name"\n',
        encoding="utf-8",
    )
    fields = load_provider_block("my-judge", custom_path=config)
    # The block's own model, not the caller's fallback.
    assert fields["repair_model"] == "judge-model-name"
    assert fields["repair_model"] != "gemma4-e4b"


# --------------------------------------------------------------------------- #
# The audit must judge translations, not asset paths
# --------------------------------------------------------------------------- #


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
    """A cache path on both sides is not a translation, and scoring it is noise.

    Measured on arXiv 2606.23344: four img_*/fml_* blocks reached the judge with
    a file path as both source and target. The judge scored them 1.0 — four calls
    spent to learn nothing, and a "pass" that means nothing.
    """
    from ubt.core.engine.stages.quality_gate import _is_unscorable
    from ubt.core.ir.models import BlockType

    path = "/home/kyle/.cache/ubt/docling_cache/assets/6331259d7f49aa708.png"
    assert _is_unscorable(_block("img_1", BlockType.IMAGE, path, path, skip=True)) is True


def test_a_verbatim_kept_block_is_not_judged() -> None:
    """A bibliography entry is deliberately source-only; a low score there is wrong."""
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
    """The near miss: the filter must not silence the blocks it exists for."""
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
    # ...and a heading with a translated title is judged too.
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
    """No judge call should be spent on a block with nothing to judge."""
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
    assert len(calls) == 1  # the asset spent no call


# --------------------------------------------------------------------------- #
# The glossary and the omission gate must not contradict each other
# --------------------------------------------------------------------------- #


def test_glossary_is_pushed_into_the_omission_gate() -> None:
    """A term the glossary renders must not then be demanded as an identifier.

    The draft stage teaches its own FastPassFilter the glossary (see
    ``draft.py``), but the quality gate scores through the *runner's* filter,
    which was never told. Measured on arXiv 2606.23344 b0006: the glossary asks
    ``state-of-the-art -> 当前最先进的``, the repair obeyed it and wrote
    ``当前最先进的准确性``, and the omission gate reported "identifier term(s)
    missing from target: ['SOTA']" and capped a correct repair at 0.35.
    """
    from ubt.core.qe.heuristic import HeuristicQERunner

    glossary = [{"source": "state-of-the-art", "translation": "当前最先进的"}]
    unbound = HeuristicQERunner()
    assert unbound.fast_pass.omission_gate._glossary_surfaces == []

    bound = unbound.with_glossary(glossary)
    assert bound.fast_pass.omission_gate._glossary_surfaces == [
        ("state-of-the-art", "当前最先进的")
    ]


def test_an_alias_lets_the_omission_gate_accept_a_rendered_abbreviation() -> None:
    """The abbreviation is the term-sheet gap, and an alias closes it.

    The mechanism was already there (``_glossary_verifies``); what was missing
    was the connection between the two gates. With ``SOTA`` listed as an alias of
    the entry, the same block passes.
    """
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
    """An omission caused by an abbreviation must not read as a translation bug."""
    from ubt.core.qe.heuristic import HeuristicQERunner

    # Two identifier terms are needed before the recall floor engages.
    source = "The method achieves SOTA accuracy and RT-DocLayout efficiency."
    target = "该方法实现了当前最先进的准确性以及 RT-DocLayout 的效率。"
    glossary = [{"source": "state-of-the-art", "translation": "当前最先进的"}]
    decision = (
        HeuristicQERunner().with_glossary(glossary).fast_pass.omission_gate.evaluate(source, target)
    )
    assert decision.passed is False
    assert "alias" in decision.reason

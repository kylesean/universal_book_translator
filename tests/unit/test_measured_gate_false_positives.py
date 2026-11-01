from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ubt.core.qe.fast_pass import FastPassFilter, is_verbatim_echo
from ubt.core.qe.term_drift import detect_term_drift, normalize_glossary
from ubt.core.validators.consistency import (
    GlossaryConsistencyValidator,
    NumericConsistencyValidator,
)

pytestmark = pytest.mark.fast

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "measured_gate_false_positives.json"


def _cases() -> dict[str, dict[str, Any]]:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return {case["block_id"]: case for case in payload["cases"]}


CASES = _cases()

SEMANTIC_DEFECTS = frozenset(
    block_id for block_id, case in CASES.items() if case.get("_defect") == "semantic-loss"
)


@pytest.mark.parametrize("block_id", sorted(set(CASES) - SEMANTIC_DEFECTS))
def test_a_correct_translation_is_not_flagged_by_any_gate(block_id: str) -> None:
    case = CASES[block_id]
    src, tgt = case["source_text"], case["target_text"]
    block_type = case["block_type"]

    assert not is_verbatim_echo(src, tgt), f"{block_id}: reported as an untranslated echo"

    numeric = NumericConsistencyValidator().validate(src, tgt)
    assert numeric.is_valid, f"{block_id}: {numeric.message}"

    decision = FastPassFilter().evaluate(src, tgt, block_type=block_type)
    assert decision.passed, f"{block_id}: {decision.reason}"


_TERMINOLOGY_FALSE_POSITIVES = {
    "pdf_main#b0029": {"source": "Detection", "translation": "检测"},
    "pdf_main#b0049": {"source": "Mask", "translation": "掩码"},
    "pdf_main#b0099": {"source": "Mask", "translation": "掩码"},
}


@pytest.mark.parametrize("block_id", sorted(_TERMINOLOGY_FALSE_POSITIVES))
def test_the_terminology_gate_passes_the_blocks_it_wrongly_rejected(block_id: str) -> None:
    case = CASES[block_id]
    result = GlossaryConsistencyValidator([_TERMINOLOGY_FALSE_POSITIVES[block_id]]).validate(
        case["source_text"], case["target_text"]
    )
    assert result.is_valid, f"{block_id}: {result.message}"


def test_the_terminology_flags_that_were_right_still_fail() -> None:
    for block_id, entry in (
        ("pdf_main#b0010", {"source": "Publicly Available", "translation": "公开可用"}),
        ("pdf_main#b0015", {"source": "Content Interference", "translation": "内容干扰"}),
    ):
        case = CASES[block_id]
        result = GlossaryConsistencyValidator([entry]).validate(
            case["source_text"], case["target_text"]
        )
        assert not result.is_valid, f"{block_id}: the defect was expected to be caught"
        assert entry["translation"] not in case["target_text"]


def test_a_term_only_kept_as_a_label_does_not_excuse_prose() -> None:
    glossary = normalize_glossary([{"source": "Mask", "translation": "掩码"}])
    (finding,) = detect_term_drift(
        "The mask head predicts masks for every query.",
        "头部为每个查询预测结果。",
        glossary,
    )
    assert finding.drifted is True


def test_a_count_suffix_still_fails_when_the_count_is_dropped() -> None:
    assert (
        not NumericConsistencyValidator()
        .validate("we scaled our corpus to over 38k samples.", "我们扩大了训练语料库。")
        .is_valid
    )


def test_a_scientific_exponent_does_not_excuse_a_dropped_minus() -> None:
    assert (
        not NumericConsistencyValidator()
        .validate("the temperature is -4 degrees", "温度是 4 度")
        .is_valid
    )


def test_an_identifier_heading_still_fails_when_it_has_prose() -> None:
    assert is_verbatim_echo("3. Proposed Approach", "3. Proposed Approach")


def test_an_untranslated_table_still_fails_the_density_gate() -> None:
    src = "| Model | GLUE | SQuAD |\n|---|---|---|\n| T5-Base | 84.3 | 85.5 |"
    decision = FastPassFilter().evaluate(src, src, block_type="table")
    assert not decision.passed
    assert "script density" in decision.reason


def test_a_url_terminates_at_a_cjk_glyph() -> None:
    from ubt.core.cleaners.cjk_spacing import normalize_publishing_cjk
    from ubt.core.qe.fast_pass import _URL_RE
    from ubt.core.validators.glossary_enforcer import extract_protected_spans

    text = "模型权重可在 https://huggingface.co/PaddlePaddle/PP-DocLayoutV3获取。"
    url = "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3"

    assert _URL_RE.findall(text) == [url]
    assert [text[s:e] for s, e in extract_protected_spans(text)] == [url]
    assert normalize_publishing_cjk(text) == f"模型权重可在 {url} 获取。"


def test_a_url_path_is_still_read_whole() -> None:
    from ubt.core.qe.fast_pass import _URL_RE

    for url in (
        "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3",
        "https://arxiv.org/abs/2606.23344v1",
        "https://example.org/a_b-c~d/e?x=1&y=2#frag",
        "https://example.org/path_with_underscores/file.tar.gz",
    ):
        assert _URL_RE.findall(f"see {url} for details") == [url]


def test_the_url_swallowed_word_is_now_caught_as_a_lost_term() -> None:
    from ubt.core.validators.consistency import GlossaryConsistencyValidator

    case = CASES["pdf_main#b0010"]
    assert "公开" not in case["target_text"]
    result = GlossaryConsistencyValidator(
        [{"source": "Publicly Available", "translation": "公开可用"}]
    ).validate(case["source_text"], case["target_text"])
    assert not result.is_valid

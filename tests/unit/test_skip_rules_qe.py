"""Unit tests for F2 skip classification and F3 QE short-block calibration."""

import pytest

from ubt.core.cleaners.skip_rules import classify_skip
from ubt.core.qe.fast_pass import FastPassFilter, is_near_verbatim_echo

pytestmark = pytest.mark.fast


def _fp() -> FastPassFilter:
    return FastPassFilter(source_lang="en", target_lang="zh")


# ---------------------------------------------------------------------------
# F2: skip_translate pre-classification
# ---------------------------------------------------------------------------


HANDLE = "social-handle watermark / running-head chrome"
DEBRIS = "pure symbol/numeric debris (no translatable words)"


# The reason string is the product: `classify_skip` returning *a* reason proves
# nothing about which rule fired, so an over-eager rule (flagging real prose, or
# flagging it as the wrong family) passed every assertion below until they
# named the rule. Bibliography and byline are owned by test_skip_rules_bib.py.
@pytest.mark.parametrize(
    "text",
    ["@techNmak", "@techNmak Understanding KV Cache 12", "  @author-name  "],
)
def test_skip_handle_is_classified_by_its_own_rule(text: str) -> None:
    assert classify_skip(text) == HANDLE


@pytest.mark.parametrize(
    "text",
    ["K , V , x , t t t", "741 824 073 . . .", ". . .", "= [ ] k K K , V1 ; ; 1 : 1"],
)
def test_skip_symbol_debris_is_classified_by_its_own_rule(text: str) -> None:
    assert classify_skip(text) == DEBRIS


def test_skip_keeps_real_prose() -> None:
    assert classify_skip("Prefill") is None
    assert classify_skip("Why decoding is sequential") is None
    assert classify_skip("Research anchors: [1], [2], [5], [6]") is None
    assert classify_skip("考虑一个自注意力层。") is None
    assert classify_skip("| Architecture | 8K | 32K |") is None
    assert classify_skip("") is None


@pytest.mark.fast
def test_skip_rules_does_not_skip_narrative_prose_with_etal_or_url() -> None:
    # Narrative sentence mentioning authors, year and et al.
    academic_narrative = (
        "According to Vaswani et al. (2017), the transformer architecture relies on "
        "self-attention mechanisms without recurrent layers. We extend their approach here."
    )
    assert classify_skip(academic_narrative) is None

    # Narrative sentence mentioning URL and year
    url_narrative = (
        "In 2023, the benchmark dataset was released at https://huggingface.co/datasets/example. "
        "We evaluate our system on this benchmark."
    )
    assert classify_skip(url_narrative) is None


# ---------------------------------------------------------------------------
# F3: script-density measured over the translatable residue
# ---------------------------------------------------------------------------


def test_density_passes_citation_lines() -> None:
    d = _fp().evaluate("Research anchors: [1], [2], [5], [6]", "研究依据：[1], [2], [5], [6]")
    assert d.passed, d.reason


def test_density_passes_unit_heavy_tables() -> None:
    src = "| Architecture | 8K | 32K | 128K | | MHA ( H KV = 32) | 4 GiB | 16 GiB |"
    tgt = "| 架构 | 8K | 32K | 128K | | MHA (H KV = 32) | 4 GiB | 16 GiB |"
    d = _fp().evaluate(src, tgt)
    assert d.passed, d.reason


def test_verbatim_echo_rejects_english_passthrough() -> None:
    d = _fp().evaluate(
        "The quick brown fox jumps over the lazy dog today.",
        "The quick brown fox jumps over the lazy dog today.",
    )
    assert not d.passed
    # A whole-block echo is now classified by the verbatim-echo gate, which
    # also covers same-script pairs where density is blind. Partial residue is
    # still a density case (test_density_still_rejects_partially_translated_prose).
    assert "identical to source" in d.reason


def test_density_still_rejects_partially_translated_prose() -> None:
    # Half-English prose keeps multi-letter English words in the residue.
    d = _fp().evaluate(
        "The cat sat on the mat yesterday evening.",
        "The cat 坐在垫子上 yesterday evening",
    )
    assert not d.passed


def test_density_passes_identifier_heavy_short_block() -> None:
    d = _fp().evaluate(
        "'PagedAttention is FlashAttention.'",
        "PagedAttention 相当于 FlashAttention。",
    )
    assert d.passed, d.reason


def test_density_still_rejects_ordinary_word_passthrough() -> None:
    # Titlecase/lowercase English words are NOT identifier-shaped.
    d = _fp().evaluate("Many happy returns.", "Many happy returns.")
    assert not d.passed


def test_density_passes_clean_prose_unchanged() -> None:
    d = _fp().evaluate("Hello world, this is a test.", "你好，世界，这是一个测试。")
    assert d.passed
    assert d.target_ratio > 0.8


def _fp_latin_pair() -> FastPassFilter:
    return FastPassFilter(source_lang="en", target_lang="de")


def test_near_verbatim_echo_rejects_untranslated_latin_passage() -> None:
    """A one-character-retouched English echo into German must not ride the
    Latin->Latin blind spot to a flawless pass (2026-09 review)."""
    src = "The transistor characteristics were measured at room temperature across the entire sample set."
    near = "The transistor characteristics were measured at room temperature across the entire sample seri!"
    d = _fp_latin_pair().evaluate(src, near)
    assert not d.passed
    assert "keeps nearly all source words" in d.reason


def test_real_translation_of_latin_pair_still_passes() -> None:
    src = "The quick brown fox jumps over the lazy dog near the riverbank."
    tgt = "Der schnelle braune Fuchs springt über den faulen Hund am Ufer."
    d = _fp_latin_pair().evaluate(src, tgt)
    assert d.passed, d.reason


def test_near_echo_ignores_contractually_verbatim_math_and_code() -> None:
    """Math spans and inline code are kept verbatim by contract in every
    translation; they must not count toward near-echo retention."""
    src = r"Let $\mathcal{D} = \{b_1, b_2, \dots, b_N\}$ denote an ordered sequence with attention weights."
    tgt = r"设 $\mathcal{D} = \{b_1, b_2, \dots, b_N\}$ 表示一个带有注意力权重的有序序列，其中每个元素都在检索阶段参与打分与排序。"
    d = _fp_latin_pair().evaluate(src, tgt)
    assert d.passed, d.reason


def test_near_echo_never_fires_on_a_cjk_target_full_of_latin_proper_nouns() -> None:
    """A Chinese translation of a technical caption carries its Latin proper
    nouns verbatim (SoL-Pi, EdgeBench, GPT-5, API); the Latin-only retention
    test used to read ~0.96 and quarantine the block as untranslated (arXiv
    2609.20519 overlay run). A CJK-script target is translated by definition."""
    src = (
        "Figure 1 SoL-Pi discovers a more token-efficient harness through automated "
        "research on EdgeBench, reducing API cost for Codex with GPT-5 relative to Claude Code."
    )
    tgt = (
        "图 1 SoL-Pi 通过自动化研究发现了 token 效率更高的 harness,在 EdgeBench 上 "
        "相对 Codex 与 GPT-5 降低了 API 成本,优于 Claude Code。"
    )
    assert sum("\u4e00" <= c <= "\u9fff" for c in tgt) >= 8
    assert not is_near_verbatim_echo(src, tgt)
    assert _fp().evaluate(src, tgt).passed


def test_near_echo_still_fires_on_a_latin_target_with_no_cjk() -> None:
    """The guard is CJK-presence only: a Latin echo that never reached the
    target language still has no CJK and must stay quarantined."""
    src = "The harness reduces token cost across every benchmark in the evaluation suite."
    tgt = "The harness reduces token cost across every benchmark in the evaluation seri."
    assert is_near_verbatim_echo(src, tgt)

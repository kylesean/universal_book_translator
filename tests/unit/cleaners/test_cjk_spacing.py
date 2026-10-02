"""CJK spacing and publishing punctuation.

MT models emit ASCII spaces where Chinese typesetting uses none, and ASCII
dashes/ellipses where the publishing standard wants ``——``/``……``. These
cleaners fix only the unambiguous cases, are gated on the target language, and
must never touch protected spans (code, masks, ``$…$`` math) or weld a Korean
target's word boundaries together.

One documented behaviour is currently missing and is recorded with a strict
``xfail`` (see the last test) rather than asserted as correct.
"""

from __future__ import annotations

import pytest

from ubt.core.cleaners.cjk_spacing import (
    apply_pangu_spacing,
    normalize_cjk_punctuation,
    normalize_cjk_spacing,
    normalize_publishing_cjk,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# Space removal around CJK.
# --------------------------------------------------------------------------- #


def test_removes_space_between_two_cjk_characters() -> None:
    assert normalize_cjk_spacing("改进， 而不是") == "改进，而不是"


def test_removes_space_before_cjk_punctuation() -> None:
    assert normalize_cjk_spacing("测试 ，继续") == "测试，继续"


def test_removes_space_after_cjk_punctuation() -> None:
    assert normalize_cjk_spacing("测试， 继续") == "测试，继续"
    assert normalize_cjk_spacing("（ 测试") == "（测试"


def test_removes_space_before_ascii_punctuation_after_cjk() -> None:
    assert normalize_cjk_spacing("测试 , 继续") == "测试, 继续"


def test_keeps_the_space_between_cjk_and_latin() -> None:
    assert normalize_cjk_spacing("FinFET 和 GAA") == "FinFET 和 GAA"


def test_preserves_newlines_and_only_strips_trailing_horizontal_space() -> None:
    assert normalize_cjk_spacing("甲  \n乙") == "甲\n乙"
    assert normalize_cjk_spacing("甲\n乙") == "甲\n乙"


def test_is_gated_to_spaceless_scripts() -> None:
    # Korean separates words with spaces; running the removal rules would weld
    # them ("이것은 테스트 입니다" -> "이것은테스트입니다").
    assert normalize_cjk_spacing("이것은 테스트 입니다", target_lang="ko") == "이것은 테스트 입니다"
    assert normalize_cjk_spacing("テスト 。", target_lang="ja") == "テスト 。"


def test_is_idempotent() -> None:
    once = normalize_cjk_spacing("改进， 而不是")
    assert normalize_cjk_spacing(once) == once


# --------------------------------------------------------------------------- #
# Pangu spacing (CJK <-> Latin/number/math).
# --------------------------------------------------------------------------- #


def test_pangu_inserts_space_between_cjk_and_latin() -> None:
    assert apply_pangu_spacing("FinFET是 一种") == "FinFET 是 一种"
    assert apply_pangu_spacing("GPU加速") == "GPU 加速"


def test_pangu_handles_percent_and_bracketed_references() -> None:
    assert apply_pangu_spacing("15%的性能") == "15% 的性能"
    assert apply_pangu_spacing("式(3.1)出发") == "式 (3.1) 出发"


def test_pangu_does_not_break_function_calls() -> None:
    # A letter before a bracket is an identifier (sin(x)), not a bracketed ref.
    assert apply_pangu_spacing("计算 sin(x) 的值") == "计算 sin(x) 的值"


def test_pangu_leaves_protected_code_spans_untouched() -> None:
    assert apply_pangu_spacing("代码 `a  b` 结束") == "代码 `a  b` 结束"


def test_pangu_is_a_noop_for_a_non_chinese_target() -> None:
    assert apply_pangu_spacing("中文GPU", target_lang="en") == "中文GPU"


def test_unspaced_cjk_currency_is_not_treated_as_math() -> None:
    # `$5到$10` is money, not a formula, so it may be spaced like prose.
    assert apply_pangu_spacing("价格$5到$10之间") == "价格$5 到$10 之间"


# --------------------------------------------------------------------------- #
# Publishing punctuation.
# --------------------------------------------------------------------------- #


def test_converts_ascii_dash_and_ellipsis_next_to_cjk() -> None:
    assert normalize_cjk_punctuation("中文--继续") == "中文——继续"
    assert normalize_cjk_punctuation("中文...继续") == "中文……继续"
    assert normalize_cjk_punctuation("中文………继续") == "中文……继续"
    assert normalize_cjk_punctuation("中文 —— 继续") == "中文——继续"


def test_leaves_markdown_table_separator_rows_byte_identical() -> None:
    assert normalize_cjk_punctuation("|---|---|") == "|---|---|"


def test_never_rewrites_inside_math_or_mask_spans() -> None:
    assert normalize_cjk_punctuation("中文 $a--b$ 继续") == "中文 $a--b$ 继续"
    masked = normalize_cjk_punctuation("⟦CITE_MASK_0001-abc⟧--中文")
    assert "0001-abc" in masked  # span contents intact
    assert "——" in masked  # the dash outside the span is normalized


def test_punctuation_is_a_noop_for_a_non_chinese_target() -> None:
    assert normalize_cjk_punctuation("中文--继续", target_lang="en") == "中文--继续"


# --------------------------------------------------------------------------- #
# Composition.
# --------------------------------------------------------------------------- #


def test_composed_pipeline_applies_all_three() -> None:
    assert normalize_publishing_cjk("FinFET是 一种--继续") == "FinFET 是一种——继续"


def test_composed_pipeline_is_idempotent() -> None:
    once = normalize_publishing_cjk("FinFET是 一种--继续")
    assert normalize_publishing_cjk(once) == once


# --------------------------------------------------------------------------- #
# Confirmed defect: the documented ideographic full stop is not covered.
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Confirmed: ubt/core/cleaners/cjk_spacing.py:25 _CJK_PUNCT omits U+3002 '。' "
        "even though the module docstring (lines 10-11) lists it among the punctuation "
        "whose adjacent spaces are stripped, so '测试 。继续' keeps its stray space. "
        "Add '。' to _CJK_PUNCT and remove this marker."
    ),
)
def test_documented_ideographic_period_spacing_is_normalized() -> None:
    assert normalize_cjk_spacing("测试 。继续") == "测试。继续"
    assert normalize_cjk_spacing("测试。 继续") == "测试。继续"

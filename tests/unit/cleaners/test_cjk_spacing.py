"""CJK spacing and publishing punctuation.

MT models emit ASCII spaces where Chinese typesetting uses none, and ASCII
dashes/ellipses where the publishing standard wants ``——``/``……``. These
cleaners fix only the unambiguous cases, are gated on the target language, and
must never touch protected spans (code, masks, ``$…$`` math) or weld a Korean
target's word boundaries together.

The ideographic full stop (``。``) is part of the documented punctuation set and
is asserted directly (see the last test).
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
    assert normalize_cjk_spacing("テスト 、", target_lang="ja") == "テスト、"


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
# The ideographic full stop is part of the documented punctuation set.
# --------------------------------------------------------------------------- #


def test_documented_ideographic_period_spacing_is_normalized() -> None:
    assert normalize_cjk_spacing("测试 。继续") == "测试。继续"
    assert normalize_cjk_spacing("测试。 继续") == "测试。继续"


def test_normalize_cjk_spacing_never_rewrites_protected_spans() -> None:
    # Module contract: code/math span bytes are untouched. The four de-spacing
    # rules used to run on the whole string, swallowing the word gap inside a
    # CJK-bearing code span.
    text = "代码 `中文 中文` 结尾，你好 世界"
    out = normalize_cjk_spacing(text, "zh")
    assert "`中文 中文`" in out
    assert "你好世界" in out


def test_cjk_spacing_formula_and_html_protection() -> None:
    """Verify LaTeX math environments and HTML tags are protected from pangu spacing insertion."""
    text = "这是公式\\(x = 1\\)测试。"
    cleaned = apply_pangu_spacing(text)
    assert "\\(x = 1\\)" in cleaned

    text2 = "爱因斯坦提出\\[ E = mc^2 \\]公式。"
    cleaned2 = apply_pangu_spacing(text2)
    assert "\\[ E = mc^2 \\]" in cleaned2

    text3 = "这是<code>int a = 1;</code>代码。"
    cleaned3 = apply_pangu_spacing(text3)
    assert "<code>int a = 1;</code>" in cleaned3


def test_quoted_prose_is_cleaned_like_any_other_prose() -> None:
    """A quotation is prose, not a code-like shield.

    The cleaner reused the glossary enforcer's protected spans, which include
    paired quotes. Inside a quoted span every spacing rule was skipped (so
    ``你好， 世界`` kept its stray space), and outside it the cleaner added a
    space on both sides of the "protected" span that the text never had
    (``这是“X”模型`` became ``这是 “X” 模型``). Dialogue-heavy books are mostly
    quoted prose, so this was the common case, not an edge.
    """
    from ubt.core.cleaners.cjk_spacing import normalize_publishing_cjk as normalize

    # No space invented outside the delimiters.
    assert normalize("文中称之为“Transformer”模型。") == "文中称之为“Transformer”模型。"
    assert normalize("他说：“好。”然后离开。") == "他说：“好。”然后离开。"
    # Stray space inside the delimiters is removed, like anywhere else.
    assert normalize("他说：“这是一个模型， 很好。”然后离开了。") == (
        "他说：“这是一个模型，很好。”然后离开了。"
    )
    assert normalize("他问“你 好 吗”。") == "他问“你好吗”。"

    # Code and math ARE still shielded: their bytes must not be rewritten.
    assert normalize("这是 `code` 与 $x$ 的例子。") == "这是 `code` 与 $x$ 的例子。"
    assert normalize("用 `a  b` 表示。") == "用 `a  b` 表示。"


def test_the_glossary_enforcer_still_shields_quoted_terms() -> None:
    """The two span sets serve different callers and must stay apart.

    Dropping quotes from the *cleaner* must not drop them from the enforcer: a
    term inside a deliberate quotation is a citation of the original wording,
    and substituting it there is the Chinglish the enforcer exists to prevent.
    """
    from ubt.core.validators.glossary_enforcer import (
        extract_code_like_spans,
        extract_protected_spans,
    )

    text = "该组件名为“Data Loader”，见 `code` 与 $x$。"
    quoted = [text[a:b] for a, b in extract_protected_spans(text)]
    code_like = [text[a:b] for a, b in extract_code_like_spans(text)]

    assert any(span.startswith("“") for span in quoted), "enforcer shields the quote"
    assert not any(span.startswith("“") for span in code_like), "cleaner must not"
    assert any(span.startswith("`") for span in code_like), "code stays shielded in both"

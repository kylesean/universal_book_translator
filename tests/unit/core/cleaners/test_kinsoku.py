"""Unit tests for CJK Kinsoku Shori and punctuation compression engine."""

from __future__ import annotations

import pytest

from ubt.core.cleaners.kinsoku import (
    WORD_JOINER,
    KinsokuRuleType,
    apply_kinsoku_word_joiners,
    audit_kinsoku_typography,
    compress_cjk_punctuations,
    fix_kinsoku_prose,
    prevent_orphan_tails,
)

pytestmark = pytest.mark.fast


def test_compress_cjk_punctuations_deduplicates_stutter() -> None:
    text = "这是一个测试，，发现很多问题。。真的吗？？？是的！！！"
    compressed = compress_cjk_punctuations(text)
    assert compressed == "这是一个测试，发现很多问题。真的吗？是的！"


def test_compress_cjk_punctuations_preserves_dashes_and_ellipses() -> None:
    text = "这是一种思考——跨越时空的对话……"
    compressed = compress_cjk_punctuations(text)
    assert compressed == "这是一种思考——跨越时空的对话……"


def test_compress_cjk_punctuations_protects_code_and_math() -> None:
    text = "代码 `$x,,y$` 以及 `a,,b` 中文，，段落"
    compressed = compress_cjk_punctuations(text)
    assert compressed == "代码 `$x,,y$` 以及 `a,,b` 中文，段落"


def test_apply_kinsoku_word_joiners_line_start() -> None:
    text = "技术架构，核心模块。"
    joined = apply_kinsoku_word_joiners(text)
    assert f"构{WORD_JOINER}，" in joined
    assert f"块{WORD_JOINER}。" in joined


def test_apply_kinsoku_word_joiners_line_end() -> None:
    text = "请参考《深度学习》和（系统设计）"
    joined = apply_kinsoku_word_joiners(text)
    assert "《\u2060深" in joined
    assert "（\u2060系" in joined
    assert "计\u2060）" in joined
    assert "习\u2060》" in joined


def test_apply_kinsoku_word_joiners_inseparable_units() -> None:
    text = "关键特性——高性能与低延迟……"
    joined = apply_kinsoku_word_joiners(text)
    assert f"—{WORD_JOINER}—" in joined
    assert f"…{WORD_JOINER}…" in joined


def test_apply_kinsoku_protects_inline_math_and_code() -> None:
    text = "公式 $x \\in (0, 1)$ 以及代码 `func()` 包含（中文）"
    joined = apply_kinsoku_word_joiners(text)
    assert "$x \\in (0, 1)$" in joined
    assert "`func()`" in joined
    assert f"（{WORD_JOINER}中" in joined


def test_apply_kinsoku_noop_on_english() -> None:
    text = "This is a simple English sentence without CJK."
    joined = apply_kinsoku_word_joiners(text, target_lang="en")
    assert joined == text


def test_prevent_orphan_tails() -> None:
    text = "系统拥有完善的自愈机制与自动化架构设计。\n第二段落说明。"
    fixed = prevent_orphan_tails(text)
    assert (
        f"架{WORD_JOINER}构{WORD_JOINER}设{WORD_JOINER}计\u2060。" in fixed
        or f"构{WORD_JOINER}设{WORD_JOINER}计\u2060。" in fixed
        or f"设{WORD_JOINER}计。"[0] in fixed
    )


def test_fix_kinsoku_prose_full_pipeline() -> None:
    raw = "这是一个测试，，包含《深度学习》与技术特性——高性能。\n完成。"
    fixed = fix_kinsoku_prose(raw)
    assert "，，" not in fixed
    assert "，" in fixed
    assert f"《{WORD_JOINER}深" in fixed
    assert f"—{WORD_JOINER}—" in fixed


def test_audit_kinsoku_typography_detects_violations() -> None:
    bad_text = "，这是违规行首标点\n这是违规行尾括号（\n未处理连续标点，，严重空洞\n孤\n"
    report = audit_kinsoku_typography(bad_text)
    assert not report.is_clean
    assert report.total_lines == 4
    assert len(report.violations) >= 3
    rules = [v.rule for v in report.violations]
    assert KinsokuRuleType.LINE_START_PROHIBITED in rules
    assert KinsokuRuleType.LINE_END_PROHIBITED in rules
    assert KinsokuRuleType.CONSECUTIVE_PUNCTUATION in rules
    assert KinsokuRuleType.ORPHAN_TAIL in rules
    assert report.compliance_score < 90.0


def test_audit_kinsoku_typography_clean_text() -> None:
    clean_text = (
        "第一章 绪论与背景概述\n"
        "本书系统阐述了现代编译系统与自然语言处理的核心理论。\n"
        "通过结合数学建模与工程实战，帮助读者建立端到端的认知。"
    )
    report = audit_kinsoku_typography(clean_text)
    assert report.is_clean
    assert len(report.violations) == 0
    assert report.compliance_score == 100.0

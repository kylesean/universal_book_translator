"""Unit tests for TranslationOutputExtractor (2026 Semantic XML standard)."""

import pytest

from ubt.core.router.extractor import TranslationOutputExtractor


def test_extractor_handles_standard_xml_translation_tags() -> None:
    raw = (
        "<think>Comparing potential renderings for philosophical accuracy...</think>\n"
        "<translation>\n"
        "存在与时间是存在主义哲学的奠基之作。\n"
        "</translation>\n"
        "Hope this meets your requirements!"
    )
    result = TranslationOutputExtractor.extract(raw)
    assert result == "存在与时间是存在主义哲学的奠基之作。"


def test_extractor_handles_final_translation_tags() -> None:
    raw = (
        "<reasoning>Fixing glossary term error in draft...</reasoning>\n"
        "<final_translation>修正后的高质量专业译文。</final_translation>"
    )
    result = TranslationOutputExtractor.extract(raw)
    assert result == "修正后的高质量专业译文。"


def test_extractor_handles_unclosed_truncated_tags() -> None:
    # 1. Truncated translation tag (token limit hit during translation)
    raw_trans = "<think>Fast deliberation</think><translation>这是在生成中途被截断的译文"
    assert TranslationOutputExtractor.extract(raw_trans) == "这是在生成中途被截断的译文"

    # 2. Truncated thinking tag (token limit hit during thinking)
    raw_think = "<think>Deliberating about chapter structure but run out of tokens"
    assert TranslationOutputExtractor.extract(raw_think) == ""


def test_extractor_preserves_legitimate_book_headings() -> None:
    """Critical verification: Book section headers such as '## 内容', '### 参考', '## 上下文'

    must NEVER be stripped away by prompt-cleaning regexes.
    """
    headings_text = (
        "## 内容\n\n"
        "第一章：引言与背景\n\n"
        "### 参考\n\n"
        "1. Turing, A. M. (1950).\n\n"
        "## 上下文\n\n"
        "在复杂的分布式系统中..."
    )

    # 1. When enclosed in <translation>
    wrapped = f"<translation>\n{headings_text}\n</translation>"
    assert TranslationOutputExtractor.extract(wrapped) == headings_text

    # 2. Even when tags are omitted (fallback path)
    assert TranslationOutputExtractor.extract(headings_text) == headings_text


def test_extractor_preserves_literary_word_repetitions() -> None:
    """Critical verification: Valid English grammar, poetic epizeuxis, and Chinese reduplication

    must NEVER be collapsed by word-loop replacement.
    """
    english_grammar = "He knew that he had had enough time."
    assert TranslationOutputExtractor.extract(english_grammar) == english_grammar

    poetic_repetition = "Far, far away across the vast, vast ocean, again and again."
    assert TranslationOutputExtractor.extract(poetic_repetition) == poetic_repetition

    chinese_reduplication = "这个问题我们需要商量商量，深入研究研究。"
    assert TranslationOutputExtractor.extract(chinese_reduplication) == chinese_reduplication


def test_extractor_unwraps_markdown_code_fences() -> None:
    raw_inside = "<translation>\n```markdown\n# 第一章\n宇宙始于奇点。\n```\n</translation>"
    assert TranslationOutputExtractor.extract(raw_inside) == "# 第一章\n宇宙始于奇点。"

    raw_outside = "```text\n纯文本翻译段落。\n```"
    assert TranslationOutputExtractor.extract(raw_outside) == "纯文本翻译段落。"


def test_extractor_fallback_strips_pure_conversational_prefixes() -> None:
    raw_en = "Here is the final translation: 宇宙的尽头是代码。"
    assert TranslationOutputExtractor.extract(raw_en) == "宇宙的尽头是代码。"

    raw_zh = "这是最终精修中文翻译：知识就是力量。"
    assert TranslationOutputExtractor.extract(raw_zh) == "知识就是力量。"

    raw_header = "### Translation:\n量子计算是一门跨学科前沿领域。"
    assert TranslationOutputExtractor.extract(raw_header) == "量子计算是一门跨学科前沿领域。"


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


def test_extractor_keeps_a_translation_that_mentions_the_wrapper_tag() -> None:
    """A literal ``<translation>`` mention must not truncate the answer.

    Regression: the tag pattern was non-greedy to the FIRST close tag, so an
    answer that mentions the tag ("使用 <translation> 标签") was cut at the
    mention, losing everything after it.
    """
    raw = (
        "<translation>使用 <translation> 标签来包裹输出，"
        "例如 <translation>译文</translation>。</translation>"
    )
    result = TranslationOutputExtractor.extract(raw)
    assert result.startswith("使用")
    assert result.endswith("。")


def test_extractor_does_not_treat_mid_prose_tag_mention_as_a_wrapper() -> None:
    """An unclosed mid-prose mention is content, not a truncated wrapper."""
    raw = "使用 <translation> 标签来包裹输出，没有闭合标签的时候不应截断。"
    result = TranslationOutputExtractor.extract(raw)
    assert "使用" in result
    assert "不应截断" in result

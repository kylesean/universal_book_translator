from __future__ import annotations

import pytest

from ubt.core.router.extractor import TranslationOutputExtractor as X

pytestmark = pytest.mark.fast


def test_auto_prefers_the_xml_tag_and_falls_back_to_cleaning() -> None:
    tagged = "<translation>你好</translation>"
    assert X.extract(tagged, "auto") == "你好"
    assert X.extract(tagged, "xml_tag") == "你好"

    untagged = "Here is the translation:\n你好"
    assert X.extract(untagged, "auto") == "你好"
    assert X.extract(untagged, "xml_tag") == ""


def test_a_lead_in_followed_by_a_fence_does_not_leak_backticks() -> None:
    raw = "Here is the translation:\n```\n你好，世界。\n```"
    for strategy in ("auto", "cp"):
        cleaned = X.extract(raw, strategy)
        assert "```" not in cleaned, f"{strategy} left a fence: {cleaned!r}"
        assert cleaned == "你好，世界。"


def test_a_language_tag_is_not_part_of_the_translation() -> None:
    assert X.extract("```python\nprint('hi')\n```", "auto") == "print('hi')"
    assert X.extract("```markdown\n# 标题\n```", "auto") == "# 标题"


def test_a_fence_used_as_content_is_not_mistaken_for_a_language_tag() -> None:
    assert X.extract("```code```", "auto") == "code"
    body = "示例：\n```python\nx = 1\n```\n以上。"
    assert X.extract(body, "auto") == body


def test_raw_keeps_the_lead_in_which_is_why_it_is_not_the_default() -> None:
    raw = "Here is the translation:\n```\n你好\n```"
    kept = X.extract(raw, "raw")
    assert kept.startswith("Here is the translation:")


def test_reasoning_traces_are_stripped_even_under_raw() -> None:
    raw = "<thought>let me consider</thought>你好"
    assert X.extract(raw, "raw") == "你好"
    assert X.extract(raw, "auto") == "你好"


def test_sanitize_thought_output_strips_reasoning_preserving_xml_envelope() -> None:
    from ubt.core.router.transports.base import sanitize_thought_output

    raw = "<thought>thinking process</thought><translation>你好</translation>"
    sanitized = sanitize_thought_output(raw)
    assert sanitized == "<translation>你好</translation>"
    assert X.extract(sanitized, "xml_tag") == "你好"

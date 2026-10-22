"""What each output-extraction strategy actually does to a model reply.

These were effectively untested: only the macro-block helper had coverage, so the
three issues below survived in the shipped extractor.

The console used to offer operators ``auto / xml_tag / raw / cp`` — the enum
values, unexplained — and nobody could tell that "auto" *already is* "try the
XML tag, then fall back to stripping the conversational wrapper". Worse, the
fallback's fence handling was anchored to the whole string, so the most common
real reply shape (a lead-in sentence, then a fenced block) kept its backticks
and delivered literal ``` into the book.
"""

from __future__ import annotations

import pytest

from ubt.core.router.extractor import TranslationOutputExtractor as X

pytestmark = pytest.mark.fast


def test_auto_prefers_the_xml_tag_and_falls_back_to_cleaning() -> None:
    """``auto`` is a superset of ``xml_tag`` and ``cp``, not a fourth behaviour.

    Nothing the model is asked for guarantees a tag — only the ``rich`` prompt
    strategy demands one — so a strict tag strategy returns empty for the other
    two strategies' replies. Auto tries the tag, then cleans.
    """
    tagged = "<translation>你好</translation>"
    assert X.extract(tagged, "auto") == "你好"
    assert X.extract(tagged, "xml_tag") == "你好"

    untagged = "Here is the translation:\n你好"
    assert X.extract(untagged, "auto") == "你好"
    # A strict tag strategy has nothing to extract from the same reply.
    assert X.extract(untagged, "xml_tag") == ""


def test_a_lead_in_followed_by_a_fence_does_not_leak_backticks() -> None:
    """The anchored fence pattern only fired when the fence was the whole reply.

    A lead-in sentence hid it, and the returned text carried ``` into the
    artifact. The fence must be unwrapped *after* the lead-in is removed.
    """
    raw = "Here is the translation:\n```\n你好，世界。\n```"
    for strategy in ("auto", "cp"):
        cleaned = X.extract(raw, strategy)
        assert "```" not in cleaned, f"{strategy} left a fence: {cleaned!r}"
        assert cleaned == "你好，世界。"


def test_a_language_tag_is_not_part_of_the_translation() -> None:
    """```python replies kept the word "python" as the first line.

    The tag group was limited to ``markdown|text|xml``, so any other language
    identifier became content. A book that translates code snippets delivered a
    stray "python" line above every example.
    """
    assert X.extract("```python\nprint('hi')\n```", "auto") == "print('hi')"
    assert X.extract("```markdown\n# 标题\n```", "auto") == "# 标题"


def test_a_fence_used_as_content_is_not_mistaken_for_a_language_tag() -> None:
    """``"```code```"`` is a translation *about* a fence, not a tagged block.

    The language tag is only a tag when a newline follows it; requiring that
    keeps a one-line fenced payload intact.
    """
    assert X.extract("```code```", "auto") == "code"
    # A fence in the middle of a longer reply is content and must survive whole.
    body = "示例：\n```python\nx = 1\n```\n以上。"
    assert X.extract(body, "auto") == body


def test_raw_keeps_the_lead_in_which_is_why_it_is_not_the_default() -> None:
    """``raw`` is the only strategy that changes the *quality* of the output.

    It skips cleaning entirely, so a chatty endpoint's "Here is the translation:"
    ships as the first line of the translated paragraph. That is why ``auto`` is
    the default and ``raw`` exists only for endpoints that already return the
    bare text.
    """
    raw = "Here is the translation:\n```\n你好\n```"
    kept = X.extract(raw, "raw")
    assert kept.startswith("Here is the translation:")


def test_reasoning_traces_are_stripped_even_under_raw() -> None:
    """No strategy ships a model's private reasoning; that is transport, not extraction."""
    raw = "<thought>let me consider</thought>你好"
    assert X.extract(raw, "raw") == "你好"
    assert X.extract(raw, "auto") == "你好"

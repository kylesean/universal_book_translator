"""Tests for shared adapter helpers (``ubt/adapters/base.py``)."""

import pytest

from ubt.adapters.base import decode_markup

pytestmark = pytest.mark.fast


def test_short_gbk_text_decodes_as_chinese_not_korean() -> None:
    """A short GBK sample must not be guessed as euc_kr.

    The charset-detector candidate set contained ``euc_kr`` (and ``utf_16``)
    despite the comment claiming it kept both out; an 8-byte GBK sample decoded
    to Korean mojibake (``櫓匡꿎桿``) with no replacement marker and no warning,
    so the translation pipeline silently started from corrupt source text.
    """
    text = "中文测试"
    assert decode_markup(text.encode("gbk")) == text


def test_longer_gbk_text_still_decodes() -> None:
    text = "第一章 绪论 这是一段中文正文内容测试"
    assert decode_markup(text.encode("gbk")) == text


def test_utf16_with_bom_still_decodes() -> None:
    assert decode_markup("héllo".encode("utf-16")) == "héllo"

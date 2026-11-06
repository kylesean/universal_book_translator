"""Byte-stream decoding: the BOM ladder must never raise past the adapter contract."""

from __future__ import annotations

import pytest

from ubt.core.text_decode import decode_markup

pytestmark = pytest.mark.fast


def test_utf16_bom_is_honoured() -> None:
    assert decode_markup("Hello 世界".encode("utf-16")) == "Hello 世界"


def test_truncated_utf16_does_not_raise() -> None:
    """An odd byte count used to raise a bare UnicodeDecodeError.

    That exception is outside the adapters' ``DocumentParseError`` contract, so
    a truncated UTF-16 file failed the job as an unclassified error. The BOM
    still tells us the family, so a visible replacement beats a crash.
    """
    truncated = ("Hello world".encode("utf-16"))[:-1]
    decoded = decode_markup(truncated)
    assert "Hello" in decoded


def test_unpaired_surrogate_does_not_raise() -> None:
    raw = b"\xff\xfe" + "ab".encode("utf-16-le") + b"\x00\xd8" + "cd".encode("utf-16-le")
    assert "ab" in decode_markup(raw)


def test_truncated_utf8_bom_does_not_raise() -> None:
    raw = b"\xef\xbb\xbf" + b"ok" + b"\xe4"
    decoded = decode_markup(raw)
    assert "ok" in decoded


def test_gbk_is_decoded_through_the_declared_or_detected_ladder() -> None:
    raw = '<?xml version="1.0" encoding="gb18030"?><p>中文测试</p>'.encode("gb18030")
    assert "中文测试" in decode_markup(raw)

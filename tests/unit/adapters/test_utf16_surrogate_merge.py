"""``merge_utf16_surrogates``: pdfium code units must become real characters.

``FPDFText_GetUnicode`` returns UTF-16 **code units**, so an astral character
arrives as two lone surrogates. Storing each half as its own character produced
strings that cannot be UTF-8 encoded, and persisting a block's style then raised
``UnicodeEncodeError: surrogates not allowed`` — the whole ingest failed.

Measured on ``corpus/documents/2608.25512v1.pdf``: 840 styled runs carried lone
surrogates (mathematical italic letters such as ``𝑖`` U+1D456) before the merge,
0 after.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.textgeom import merge_utf16_surrogates

pytestmark = pytest.mark.fast


def _units(text: str) -> list[int]:
    """The UTF-16 code units pdfium would report for ``text``."""
    raw = text.encode("utf-16-le")
    return [int.from_bytes(raw[i : i + 2], "little") for i in range(0, len(raw), 2)]


def test_an_astral_character_becomes_one_code_point() -> None:
    merged = merge_utf16_surrogates(_units("\U0001d456"))  # 𝑖
    assert [code for _index, code in merged] == [0x1D456]
    assert chr(merged[0][1]) == "\U0001d456"


def test_the_reported_index_is_the_high_surrogate() -> None:
    # "a𝑖b" -> 'a', high, low, 'b'. The character's box/style come from the high
    # half, so the index must point at it (1), not at the low half (2).
    merged = merge_utf16_surrogates(_units("a\U0001d456b"))
    assert [index for index, _code in merged] == [0, 1, 3]
    assert "".join(chr(c) for _i, c in merged) == "a\U0001d456b"


def test_merged_text_is_utf8_encodable() -> None:
    # The property the whole fix exists for: the result can be persisted.
    text = "".join(chr(c) for _i, c in merge_utf16_surrogates(_units("𝑥𝑖+𝑦𝑗")))
    assert text.encode("utf-8")  # must not raise


def test_a_dangling_high_surrogate_is_dropped() -> None:
    # No low half follows, so there is no character to encode.
    assert merge_utf16_surrogates([0x41, 0xD835]) == [(0, 0x41)]


def test_an_unpaired_low_surrogate_is_dropped() -> None:
    assert merge_utf16_surrogates([0x41, 0xDC56, 0x42]) == [(0, 0x41), (2, 0x42)]


def test_a_high_surrogate_followed_by_a_non_low_unit_is_dropped() -> None:
    # A high surrogate pairs only with a low one; "a" is not.
    assert merge_utf16_surrogates([0xD835, 0x41]) == [(1, 0x41)]


def test_plain_bmp_text_passes_through_unchanged() -> None:
    merged = merge_utf16_surrogates(_units("中文 abc"))
    assert "".join(chr(c) for _i, c in merged) == "中文 abc"
    assert [index for index, _code in merged] == list(range(len("中文 abc")))


def test_consecutive_astral_characters_all_merge() -> None:
    text = "\U0001d456\U0001d457\U0001d458"  # 𝑖𝑗𝑘
    merged = merge_utf16_surrogates(_units(text))
    assert "".join(chr(c) for _i, c in merged) == text

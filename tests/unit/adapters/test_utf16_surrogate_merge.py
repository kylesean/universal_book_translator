from __future__ import annotations

import pytest

from ubt.adapters.pdf.textgeom import merge_utf16_surrogates

pytestmark = pytest.mark.fast


def _units(text: str) -> list[int]:
    raw = text.encode("utf-16-le")
    return [int.from_bytes(raw[i : i + 2], "little") for i in range(0, len(raw), 2)]


def test_an_astral_character_becomes_one_code_point() -> None:
    merged = merge_utf16_surrogates(_units("\U0001d456"))
    assert [code for _index, code in merged] == [0x1D456]
    assert chr(merged[0][1]) == "\U0001d456"


def test_the_reported_index_is_the_high_surrogate() -> None:
    merged = merge_utf16_surrogates(_units("a\U0001d456b"))
    assert [index for index, _code in merged] == [0, 1, 3]
    assert "".join(chr(c) for _i, c in merged) == "a\U0001d456b"


def test_merged_text_is_utf8_encodable() -> None:
    text = "".join(chr(c) for _i, c in merge_utf16_surrogates(_units("𝑥𝑖+𝑦𝑗")))
    assert text.encode("utf-8")


def test_a_dangling_high_surrogate_is_dropped() -> None:
    assert merge_utf16_surrogates([0x41, 0xD835]) == [(0, 0x41)]


def test_an_unpaired_low_surrogate_is_dropped() -> None:
    assert merge_utf16_surrogates([0x41, 0xDC56, 0x42]) == [(0, 0x41), (2, 0x42)]


def test_a_high_surrogate_followed_by_a_non_low_unit_is_dropped() -> None:
    assert merge_utf16_surrogates([0xD835, 0x41]) == [(1, 0x41)]


def test_plain_bmp_text_passes_through_unchanged() -> None:
    merged = merge_utf16_surrogates(_units("中文 abc"))
    assert "".join(chr(c) for _i, c in merged) == "中文 abc"
    assert [index for index, _code in merged] == list(range(len("中文 abc")))


def test_consecutive_astral_characters_all_merge() -> None:
    text = "\U0001d456\U0001d457\U0001d458"
    merged = merge_utf16_surrogates(_units(text))
    assert "".join(chr(c) for _i, c in merged) == text

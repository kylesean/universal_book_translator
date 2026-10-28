"""Phonetic and alphabetical initial calculation for English and CJK terms."""

from __future__ import annotations

import unicodedata
from typing import Final

_GB2312_PINYIN_BOUNDS: Final[list[tuple[int, int, str]]] = [
    (0xB0A1, 0xB0C4, "A"),
    (0xB0C5, 0xB2C0, "B"),
    (0xB2C1, 0xB4ED, "C"),
    (0xB4EE, 0xB6E9, "D"),
    (0xB6EA, 0xB7A1, "E"),
    (0xB7A2, 0xB8C0, "F"),
    (0xB8C1, 0xB9FD, "G"),
    (0xB9FE, 0xBBF6, "H"),
    (0xBBF7, 0xBFA5, "J"),
    (0xBFA6, 0xC0AB, "K"),
    (0xC0AC, 0xC2E7, "L"),
    (0xC2E8, 0xC4C2, "M"),
    (0xC4C3, 0xC5B5, "N"),
    (0xC5B6, 0xC5BD, "O"),
    (0xC5BE, 0xC6D9, "P"),
    (0xC6DA, 0xC8BA, "Q"),
    (0xC8BB, 0xC8F5, "R"),
    (0xC8F6, 0xCBF9, "S"),
    (0xCBFA, 0xCDD9, "T"),
    (0xCDDA, 0xCEF3, "W"),
    (0xCEF4, 0xCFF3, "X"),
    (0xCFF4, 0xD4D0, "Y"),
    (0xD4D1, 0xD7F9, "Z"),
]


def get_pinyin_initial(text: str) -> str:
    """Return the uppercase Pinyin initial letter (A-Z) for a Chinese character.

    Uses GB2312 phonetic partitions with graceful fallback to ASCII or '#'.
    """
    clean = text.strip()
    if not clean:
        return "#"

    first = clean[0]
    if first.isascii():
        return first.upper() if first.isalpha() else ("0-9" if first.isdigit() else "#")

    try:
        raw = first.encode("gb2312")
    except UnicodeEncodeError:
        return "#"

    if len(raw) != 2:
        return "#"

    val = raw[0] * 256 + raw[1]
    for low, high, initial in _GB2312_PINYIN_BOUNDS:
        if low <= val <= high:
            return initial

    return "#"


def get_latin_initial(text: str) -> str:
    """Return the normalized Latin initial letter (A-Z, 0-9, or '#')."""
    clean = text.strip()
    if not clean:
        return "#"

    first = clean[0]
    if first.isascii():
        if first.isalpha():
            return first.upper()
        if first.isdigit():
            return "0-9"
        return "#"

    normalized = unicodedata.normalize("NFKD", first)
    for ch in normalized:
        if ch.isascii() and ch.isalpha():
            return ch.upper()

    return "#"

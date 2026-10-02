"""``ubt.core.cjk_ranges`` is the single source of CJK codepoint ranges.

The invariant that makes it a *source* rather than another copy: each regex
class fragment renders exactly the tuple list it was built from (a parse of
the fragment text round-trips to the same (lo, hi) pairs), and the canonical
predicate agrees with the fragment on every block edge.
"""

from __future__ import annotations

import re

import pytest

from ubt.core.cjk_ranges import (
    CJK_BMP_CLASS,
    CJK_BMP_RANGES,
    CJK_FULL_CLASS,
    CJK_RANGES,
    CJK_SCRIPT_CLASS,
    CJK_SCRIPT_RANGES,
    CJK_WIDE_CLASS,
    CJK_WIDE_RANGES,
    HAN_KANA_HANGUL_CLASS,
    HAN_KANA_HANGUL_RANGES,
    HAN_SPAN_KANA_HANGUL_CLASS,
    HAN_SPAN_KANA_HANGUL_RANGES,
    HAN_UNIFIED_CLASS,
    HAN_UNIFIED_RANGES,
    contains_cjk,
    is_cjk_char,
)

_ESCAPE_RE = re.compile(r"(\\u)([0-9a-f]{4})|(\\U)([0-9a-f]{8})")


def _parse_class(fragment: str) -> list[tuple[int, int]]:
    """Decode a class fragment back into its (lo, hi) pairs."""
    pairs: list[tuple[int, int]] = []
    lows: list[int] = []
    for match in _ESCAPE_RE.finditer(fragment):
        hex_digits = match.group(2) or match.group(4)
        lows.append(int(hex_digits, 16))
    for lo, hi in zip(lows[0::2], lows[1::2], strict=True):
        pairs.append((lo, hi))
    return pairs


_TIER_CLASSES = {
    "HAN_UNIFIED": (HAN_UNIFIED_CLASS, HAN_UNIFIED_RANGES),
    "CJK_SCRIPT": (CJK_SCRIPT_CLASS, CJK_SCRIPT_RANGES),
    "HAN_KANA_HANGUL": (HAN_KANA_HANGUL_CLASS, HAN_KANA_HANGUL_RANGES),
    "HAN_SPAN_KANA_HANGUL": (HAN_SPAN_KANA_HANGUL_CLASS, HAN_SPAN_KANA_HANGUL_RANGES),
    "CJK_BMP": (CJK_BMP_CLASS, CJK_BMP_RANGES),
    "CJK_WIDE": (CJK_WIDE_CLASS, CJK_WIDE_RANGES),
    "CJK_FULL": (CJK_FULL_CLASS, CJK_RANGES),
}

# Every block edge of every tier, plus representative non-CJK codepoints.
_EDGES = (
    [lo for pair in CJK_WIDE_RANGES for lo in pair]
    + [hi for pair in CJK_RANGES for hi in pair]
    + [0x4DC0, 0x4DFF, 0x2B740, 0x41, 0x31, 0x3B1, 0x2026, 0xFFEE]
)


@pytest.mark.fast
def test_class_fragments_render_exactly_their_tuples() -> None:
    for name, (fragment, ranges) in _TIER_CLASSES.items():
        assert _parse_class(fragment) == list(ranges), name


@pytest.mark.fast
def test_predicate_and_canonical_class_agree_on_edges() -> None:
    pattern = re.compile(f"[{CJK_FULL_CLASS}]")
    for cp in _EDGES:
        ch = chr(cp)
        assert is_cjk_char(ch) == (pattern.match(ch) is not None), f"U+{cp:04X}"


@pytest.mark.fast
def test_predicate_edge_cases() -> None:
    assert is_cjk_char("") is False
    assert is_cjk_char("一") is True
    assert is_cjk_char("\U0002a6df") is True  # Ext-C edge
    assert is_cjk_char("\U0002b740") is False  # first codepoint past Ext C
    assert is_cjk_char("\u4dc0") is False  # Yijing gap: between Ext A and Unified
    assert contains_cjk("plain text") is False
    assert contains_cjk("北京 Beijing") is True


@pytest.mark.fast
def test_glossary_enforcer_reexport_matches_source() -> None:
    from ubt.core.validators.glossary_enforcer import CJK_RANGES as reexported
    from ubt.core.validators.glossary_enforcer import is_cjk_char as reexported_pred

    assert reexported is CJK_RANGES
    assert reexported_pred is is_cjk_char

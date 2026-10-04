"""Single source for the CJK codepoint ranges every layer matches against.

These literals used to be copied per module and the copies had drifted: a
"canonical" tuple nobody else imported, regexes silently missing blocks,
per-language tables with their own block ends. Every range now lives here and
consumers compose the *tiered* sets below instead of restating numbers, so
auditing what a site matches is reading a name, not decoding hex.

Tiers are semantic, not exhaustive: a site that deliberately scopes narrower
(a math guard asking "is this Han text", a cost model billing whole blocks)
takes the tier that names its intent. Nothing here widens a site's set on its
own. This module imports nothing from the rest of ``ubt`` — every layer
(cleaners, validators, memory, qe, adapters, pipeline, engine) may depend on
it without creating a cycle.
"""

from __future__ import annotations

# --- Blocks (each (lo, hi) pair is inclusive on both ends) -------------------

CJK_EXT_A = (0x3400, 0x4DBF)
CJK_UNIFIED = (0x4E00, 0x9FFF)
CJK_COMPAT = (0xF900, 0xFAFF)
CJK_EXT_B = (0x20000, 0x2A6DF)
CJK_EXT_C = (0x2A700, 0x2B73F)
HIRAGANA = (0x3040, 0x309F)
KATAKANA = (0x30A0, 0x30FF)
#: One-span kana: HIRAGANA and KATAKANA are contiguous, and most regex sites
#: match them as a single class.
KANA = (0x3040, 0x30FF)
HANGUL_SYLLABLES = (0xAC00, 0xD7AF)
CJK_PUNCTUATION = (0x3000, 0x303F)
FULLWIDTH_FORMS = (0xFF00, 0xFFEF)
GENERAL_PUNCTUATION = (0x2000, 0x206F)
#: Astral Han billed as one block (Ext B through the Compatibility Supplement,
#: unassigned codepoints included): the cost model charges by block, not by
#: character, so the bucket is deliberately wider than the assigned ranges.
HAN_ASTRAL_WIDE = (0x20000, 0x2FA1F)
#: CJK punctuation through the Unified block as one span: the cost model's
#: wide "this is CJK-width text" bucket.
COST_CJK_WIDE = (0x3000, 0x9FFF)
#: Extension A through the Unified block as ONE span: the skeleton matcher's
#: shortcut, which deliberately sweeps the Yijing Hexagram Symbols gap
#: (U+4DC0-U+4DFF) between the two blocks. Keep spans like this explicit —
#: composing Ext A + Unified would silently exclude the gap.
HAN_SPAN_WIDE = (0x3400, 0x9FFF)

# --- Tiers -------------------------------------------------------------------

#: The canonical CJK set — ideographs (Unified, Ext A, Compat, Ext B/C),
#: kana, hangul. This is what :func:`is_cjk_char` answers.
CJK_RANGES: tuple[tuple[int, int], ...] = (
    CJK_EXT_A,
    CJK_UNIFIED,
    CJK_COMPAT,
    CJK_EXT_B,
    CJK_EXT_C,
    HIRAGANA,
    KATAKANA,
    HANGUL_SYLLABLES,
)

#: Han ideographs including Extension A, without Compat or the astral blocks:
#: the script-ratio / width-model tier.
HAN_RANGES: tuple[tuple[int, int], ...] = (CJK_EXT_A, CJK_UNIFIED)

#: Just the common Unified block: guards that only ask "is this Han text".
HAN_UNIFIED_RANGES: tuple[tuple[int, int], ...] = (CJK_UNIFIED,)

#: Han (with Extension A) + kana + hangul, no Compat, no astral blocks: the
#: skeleton / inline-math detection tier.
HAN_KANA_HANGUL_RANGES: tuple[tuple[int, int], ...] = (
    CJK_EXT_A,
    CJK_UNIFIED,
    KANA,
    HANGUL_SYLLABLES,
)

#: The BMP-only CJK class: Han + Compat + kana + hangul. Shared by the
#: spacing cleaner and the artifact CJK-range regex.
CJK_BMP_RANGES: tuple[tuple[int, int], ...] = (
    CJK_EXT_A,
    CJK_UNIFIED,
    CJK_COMPAT,
    KANA,
    HANGUL_SYLLABLES,
)

#: Script detection (letter/word classes, script ratios): Unified + kana +
#: hangul, without the rare Extension A.
CJK_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (CJK_UNIFIED, KANA, HANGUL_SYLLABLES)

#: East-Asian adjacency for term-shape work: every CJK block on the BMP plus
#: CJK punctuation and fullwidth forms, but not the astral extensions.
CJK_WIDE_RANGES: tuple[tuple[int, int], ...] = (
    CJK_PUNCTUATION,
    KANA,
    CJK_EXT_A,
    CJK_UNIFIED,
    CJK_COMPAT,
    HANGUL_SYLLABLES,
    FULLWIDTH_FORMS,
)

#: The skeleton matcher's set: the wide Han span + kana + hangul.
HAN_SPAN_KANA_HANGUL_RANGES: tuple[tuple[int, int], ...] = (
    HAN_SPAN_WIDE,
    KANA,
    HANGUL_SYLLABLES,
)


def is_cjk_char(ch: str) -> bool:
    """True when the character falls in :data:`CJK_RANGES` (ideographs + kana + hangul)."""
    if not ch:
        return False
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in CJK_RANGES)


def is_cjk_wide_char(ch: str) -> bool:
    """True when the character falls in :data:`CJK_WIDE_RANGES` (ideographs + punctuation + fullwidth forms)."""
    if not ch:
        return False
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in CJK_WIDE_RANGES)


def contains_cjk(text: str) -> bool:
    """True when any character of the string falls in :data:`CJK_RANGES`."""
    return any(is_cjk_char(ch) for ch in text)


# --- Regex fragments ---------------------------------------------------------


def _char_class(ranges: tuple[tuple[int, int], ...]) -> str:
    """Render inclusive ranges as a regex character-class fragment."""

    def escape(lo: int, hi: int) -> str:
        if hi <= 0xFFFF:
            return rf"\u{lo:04x}-\u{hi:04x}"
        return rf"\U{lo:08x}-\U{hi:08x}"

    return "".join(escape(lo, hi) for lo, hi in ranges)


#: ``[...]``-interior fragments, one per tier. Embed inside a class:
#: ``re.compile(rf"[A-Za-z{CJK_SCRIPT_CLASS}]")``.
HAN_UNIFIED_CLASS = _char_class(HAN_UNIFIED_RANGES)
CJK_SCRIPT_CLASS = _char_class(CJK_SCRIPT_RANGES)
HAN_KANA_HANGUL_CLASS = _char_class(HAN_KANA_HANGUL_RANGES)
HAN_SPAN_KANA_HANGUL_CLASS = _char_class(HAN_SPAN_KANA_HANGUL_RANGES)
CJK_BMP_CLASS = _char_class(CJK_BMP_RANGES)
CJK_WIDE_CLASS = _char_class(CJK_WIDE_RANGES)
CJK_FULL_CLASS = _char_class(CJK_RANGES)

"""CJK spacing normalization for translated target text.

MT models (especially small ones) emit ASCII spaces where Chinese
typesetting uses none: ``改进， 而不是`` or ``FinFET是 一种``. Those stray
spaces survive into the Typst PDF and read as stutter. This cleaner strips
only unambiguous cases and deliberately keeps spaces between CJK and Latin
(``FinFET 和 GAA`` stays as-is):

- space(s) between two CJK characters;
- space(s) immediately before CJK punctuation (，。；：、？！）》”’ … —);
- space(s) immediately after CJK punctuation (（《“‘【 。，；：、？！）》”’ … —).

Pure function, applied at export to non-skipped target text, and gated on the
target language: the space-removal rules only run for spaceless writing systems
(``zh``/``ja``).
"""

from __future__ import annotations

import re

from ubt.core.cleaners.inline_math import iter_inline_math

_CJK = r"一-鿿㐀-䶿豈-﫿぀-ヿ가-힯"
_CJK_PUNCT = r"，。；：、？！（）》”’】…—「『～·《“‘【〈〔［｛"

# Horizontal whitespace only — never swallow newlines or other vertical
# Whitespace. A bare ``\s+`` would also match ``\n`` and
# silently weld two paragraphs into one line.
_WS = r"[ \t\u3000]+"

_BETWEEN_CJK = re.compile(f"([{_CJK}]){_WS}(?=[{_CJK}])")
_BEFORE_PUNCT = re.compile(f"{_WS}([{_CJK_PUNCT}])")
_AFTER_PUNCT = re.compile(f"([{_CJK_PUNCT}]){_WS}")
_BEFORE_ASCII_PUNCT = re.compile(rf"(?<=[{_CJK}]){_WS}([,;:.!?])")
# Stray horizontal whitespace immediately before a hard line break is
# invisible noise in the PDF; strip it without touching the newline itself
# (Only horizontal whitespace, never the newline).
_WS_BEFORE_NEWLINE = re.compile(r"[ \t\u3000]+\n")


_HAS_CJK = re.compile(rf"[{_CJK}]")
# Writing systems that do not separate words with spaces. The rules above are
# only correct for these: ``_CJK`` covers Hangul, so running them on a Korean
# target welded every word together ("이것은 테스트 입니다" -> "이것은테스트입니다")
# in the delivered file, where the space is grammatical.
_SPACELESS_SCRIPT_LANGS = ("zh", "ja")
# Publishing-punctuation rewrites only fire immediately adjacent to CJK, so
# numeric ranges ('10--20'), URLs ('a--b') and decimal runs ('1.0...2') in a
# CJK target are never rewritten. Math/mask spans are additionally protected
# by :data:`_PROTECTED_SPAN_RE` before these patterns run.
_DOUBLE_DASH = re.compile(rf"(?<=[{_CJK}])(?:--|––)|(?:--|––)(?=[{_CJK}])")
_SINGLE_EM_DASH = re.compile(
    rf"(?<!\u2014)(?:(?<=[{_CJK}])\u2014(?!\u2014)|\u2014(?=[{_CJK}])(?!\u2014))"
)
_ASCII_ELLIPSIS = re.compile(rf"(?<=[{_CJK}])\.{{3,6}}|\.{{3,6}}(?=[{_CJK}])")
_SINGLE_ELLIPSIS = re.compile(
    rf"(?<!\u2026)(?<=[{_CJK}])\u2026(?!\u2026)|(?<!\u2026)\u2026(?=[{_CJK}])(?!\u2026)"
)
_EXCESS_ELLIPSIS = re.compile(rf"(?<=[{_CJK}])(?:\u2026){{3,}}|(?:\u2026){{3,}}(?=[{_CJK}])")
_SPACED_EM_DASH = re.compile(rf"(?<=[{_CJK}])\s*——\s*(?=[{_CJK}])")
_SPACED_ELLIPSIS = re.compile(rf"(?<=[{_CJK}])\s*……\s*(?=[{_CJK}])")
_OTHER_PROTECTED_RE = re.compile(
    r"(⟦[^⟧]*⟧|```[\w]*\n[\s\S]*?\n```|```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]*?\$\$)"
)


def _split_protected(text: str) -> list[str]:
    """Split text into alternating [unprotected, protected, ...] segments.

    Guards code blocks, code spans, maskers, display math, and genuine inline math,
    without treating unspaced CJK currency (e.g. $5到$10) as math.
    """
    intervals: list[tuple[int, int]] = []
    for m in _OTHER_PROTECTED_RE.finditer(text):
        intervals.append((m.start(), m.end()))
    for m in iter_inline_math(text):
        start, end = m.start(), m.end()
        if not any(start < prev_end and end > prev_start for prev_start, prev_end in intervals):
            intervals.append((start, end))
    if not intervals:
        return [text]
    intervals.sort()
    parts: list[str] = []
    last = 0
    for start, end in intervals:
        parts.append(text[last:start])
        parts.append(text[start:end])
        last = end
    parts.append(text[last:])
    return parts


# Markdown table separator row: pipes, colons, dashes and spaces only.
#
# Possessive quantifiers are load-bearing, not style: the non-possessive form
# partitioned a dash run in exponentially many ways and took seconds on a long
# never-matching line (``"-" * 26 + "x"``), which is reachable from LLM output.
# Possessive keeps the accepted language (verified exhaustively up to length 5
# over ``-:| \t`` plus fuzzing) while making each run consume once.
_TABLE_SEP_RE = re.compile(r"[ \t]*+(?:(?:\|?[ \t]*+:?-++:?[ \t]*+))+\|?[ \t]*+")


_LATIN_OR_NUM = r"[a-zA-Z0-9]"
_OPEN_BRACKET = r"[\(\[\{]"
_CLOSE_BRACKET = r"[\)\]\}]"


def normalize_cjk_spacing(text: str, target_lang: str = "zh") -> str:
    """Remove stray ASCII/ideographic spaces around CJK characters and punctuation.

    Only horizontal whitespace is touched; newlines and other vertical
    whitespace are preserved. Trailing whitespace before a line break is noise in
    every language, so that rule runs regardless of ``target_lang``; the
    space-removal rules need a spaceless writing system (see
    :data:`_SPACELESS_SCRIPT_LANGS`) or they delete a Korean target's word
    boundaries.
    """
    if not text or not any(c in text for c in " \t\u3000"):
        return text
    out = _WS_BEFORE_NEWLINE.sub("\n", text)
    if not (target_lang or "").lower().startswith(_SPACELESS_SCRIPT_LANGS):
        return out
    out = _BETWEEN_CJK.sub(r"\1", out)
    out = _BEFORE_PUNCT.sub(r"\1", out)
    out = _BEFORE_ASCII_PUNCT.sub(r"\1", out)
    out = _AFTER_PUNCT.sub(r"\1", out)
    return out


def apply_pangu_spacing(text: str, target_lang: str = "zh") -> str:
    """Insert a half-width space between CJK characters and Latin/numbers/math expressions.

    Conforms to W3C Requirements for Chinese Text Layout and the Pangu Spacing standard:
    - Inserts a single half-width space between CJK characters and Latin letters or digits;
    - Inserts a single half-width space between CJK characters and percentage signs (e.g., 15% 的);
    - Inserts a single half-width space between CJK characters and inline LaTeX formulas or masks;
    - Inserts spaces around half-width bracketed references (式 (3.1) 出发, 图 3.1 (a));
    - Never inserts spaces adjacent to full-width CJK punctuation (，。！？；：“”‘’（）《》 etc.);
    - Preserves existing spaces between Latin words (FinFET and GAA) and internal number structures (3.14).
    """
    if not text or not target_lang or "zh" not in target_lang.lower():
        return text
    if not _HAS_CJK.search(text):
        return text

    parts = _split_protected(text)
    for i in range(0, len(parts), 2):
        seg = parts[i]
        # 1. CJK + Latin/Num
        seg = re.sub(rf"([{_CJK}])({_LATIN_OR_NUM})", r"\1 \2", seg)
        # 2. Latin/Num + CJK
        seg = re.sub(rf"({_LATIN_OR_NUM})([{_CJK}])", r"\1 \2", seg)
        # 2b. CJK + half-width bracket with Latin/Num/Math: e.g. 式(3.1) -> 式 (3.1)
        seg = re.sub(rf"([{_CJK}])({_OPEN_BRACKET}{_LATIN_OR_NUM})", r"\1 \2", seg)
        # 2c. Half-width bracket with Latin/Num/Math + CJK: e.g. (3.1)出发 -> (3.1) 出发
        seg = re.sub(rf"({_LATIN_OR_NUM}{_CLOSE_BRACKET})([{_CJK}])", r"\1 \2", seg)
        # 2d. Num + half-width bracket with Latin/Num: e.g. 图 3.1(a) -> 图 3.1 (a)
        # Only a *digit* may precede the bracket. A letter is an identifier, and
        # ``sin(x)`` / ``f(x)`` are function calls, not bracketed references — the
        # old ``_LATIN_OR_NUM`` rule inserted a space and broke them.
        seg = re.sub(rf"(\d)({_OPEN_BRACKET}[a-zA-Z0-9])", r"\1 \2", seg)
        # 3. Number/Latin + % + CJK (e.g. 15% 的性能)
        seg = re.sub(rf"({_LATIN_OR_NUM}%)([{_CJK}])", r"\1 \2", seg)
        # 6. Clean spaces around CJK punctuation
        seg = _BEFORE_PUNCT.sub(r"\1", seg)
        seg = _AFTER_PUNCT.sub(r"\1", seg)

        # Space between CJK and adjacent protected math/mask spans
        if i + 1 < len(parts) and seg and re.search(rf"[{_CJK}]$", seg):
            seg = seg + " "
        if i > 0 and seg and re.match(rf"^[{_CJK}]", seg):
            seg = " " + seg
        parts[i] = seg

    # Collapse 2+ spaces adjacent to CJK only in the *unprotected* segments:
    # running this on the rejoined string rewrote the bytes inside an inline
    # code/math span, which the module contract says must never be touched.
    for i in range(0, len(parts), 2):
        seg = parts[i]
        seg = re.sub(rf"([{_CJK}])[ \t\u3000]{{2,}}", r"\1 ", seg)
        seg = re.sub(rf"[ \t\u3000]{{2,}}([{_CJK}])", r" \1", seg)
        parts[i] = seg
    return "".join(parts)


def _apply_punct_rules(text: str) -> str:
    """Apply publishing dash/ellipsis rewrites to a single unprotected line."""
    if not text:
        return text
    out = _DOUBLE_DASH.sub("——", text)
    out = _SINGLE_EM_DASH.sub("——", out)
    out = _ASCII_ELLIPSIS.sub("……", out)
    out = _SINGLE_ELLIPSIS.sub("……", out)
    out = _EXCESS_ELLIPSIS.sub("……", out)
    out = _SPACED_EM_DASH.sub("——", out)
    out = _SPACED_ELLIPSIS.sub("……", out)
    return out


def _normalize_punct_block(block: str) -> str:
    """Normalize a stretch of text outside math/mask spans, line by line.

    Markdown table separator rows are left byte-identical: rewriting their
    dashes would turn them into em-dashes and break the table.
    """
    return "\n".join(
        line if _TABLE_SEP_RE.fullmatch(line) else _apply_punct_rules(line)
        for line in block.split("\n")
    )


def normalize_cjk_punctuation(text: str, target_lang: str = "zh") -> str:
    """Normalize CJK punctuation to Chinese publishing standards (GB/T 15834-2011).

    Converts '--', en-dashes, or single isolated em-dashes to standard Chinese double
    em-dash '——' (U+2014 x 2), and 3-dot ellipses to standard 6-dot '……' (U+2026 x 2).
    Guards against foreign target languages and non-CJK texts, only rewrites runs
    adjacent to CJK, and never touches ``$...$`` math or ``⟦...⟧`` mask spans.
    """
    if not text or not target_lang or "zh" not in target_lang.lower():
        return text
    if not _HAS_CJK.search(text):
        return text

    parts = _split_protected(text)
    for i in range(0, len(parts), 2):
        parts[i] = _normalize_punct_block(parts[i])
    return "".join(parts)


def normalize_publishing_cjk(text: str, target_lang: str = "zh") -> str:
    """Apply CJK spacing, Pangu spacing (CJK-Latin/num/math), and publishing punctuation standards."""
    if not text:
        return text
    cleaned = normalize_cjk_spacing(text, target_lang=target_lang)
    spaced = apply_pangu_spacing(cleaned, target_lang=target_lang)
    return normalize_cjk_punctuation(spaced, target_lang=target_lang)

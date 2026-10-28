"""CJK typesetting line-breaking (Kinsoku Shori) and punctuation compression engine.

Conforms to Chinese publishing standards (GB/T 15834-2011), JIS X 4051, and
W3C Requirements for Chinese Text Layout (clreq):
1. Line-start prohibited punctuation (避头标点): closing brackets, closing quotes,
   sentence terminators, pauses, and suffix units.
2. Line-end prohibited punctuation (避尾标点): opening brackets, opening quotes,
   and currency / prefix symbols.
3. Inseparable character runs (不可拆分序列): double em-dashes (——) and
   two-em ellipses (……).
4. Punctuation compression (标点挤压): deduplicating accidental stutter punctuation
   and eliminating excessive blank gaps between consecutive full-width punctuation.
5. Orphan tail prevention (防孤字/寡行): binding the final characters of a paragraph
   with Unicode Word Joiner (U+2060) so an isolated character never wraps alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from ubt.core.cjk_ranges import CJK_BMP_CLASS
from ubt.core.validators.glossary_enforcer import extract_code_like_spans

# Unicode Word Joiner: zero-width, non-breaking formatting character (U+2060).
WORD_JOINER = "\u2060"

_CJK = CJK_BMP_CLASS
_HAS_CJK = re.compile(rf"[{_CJK}]")
_SPACELESS_LANGS = ("zh", "ja")

# Characters prohibited at line start (避头标点)
LINE_START_PROHIBITED: frozenset[str] = frozenset(
    "）】｝〉》」』〕］｝”’"  # Closing brackets & quotes
    "。，、；：！？"  # Terminators & pauses
    "%‰℃°¢￠"  # Suffix symbols & units
    "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮー"  # Small kana / prolonged sound
    ")]}>,!?:;"  # Half-width punctuation in CJK context
)

# Characters prohibited at line end (避尾标点)
LINE_END_PROHIBITED: frozenset[str] = frozenset(
    "（【｛〈《「『〔［｛“‘"  # Opening brackets & quotes
    "￥¥$£€№§"  # Prefix & currency symbols
    "([{<"  # Half-width brackets in CJK context
)

# Inseparable multi-character sequences (不可拆分字符)
INSEPARABLE_UNITS: tuple[str, ...] = ("——", "……", "──")

# Consecutive stutter punctuation patterns to deduplicate
_STUTTER_PUNCT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"，{2,}"), "，"),
    (re.compile(r"。{2,}"), "。"),
    (re.compile(r"、{2,}"), "、"),
    (re.compile(r"；{2,}"), "；"),
    (re.compile(r"：{2,}"), "："),
    (re.compile(r"！{3,}"), "！"),
    (re.compile(r"？{3,}"), "？"),
    (re.compile(r"([。，、；：！？])\s+([。，、；：！？])"), r"\1\2"),
)


class KinsokuRuleType(StrEnum):
    """Classification of Kinsoku typography rules and violations."""

    LINE_START_PROHIBITED = "line_start_prohibited"
    LINE_END_PROHIBITED = "line_end_prohibited"
    INSEPARABLE_PAIR = "inseparable_pair"
    CONSECUTIVE_PUNCTUATION = "consecutive_punctuation"
    ORPHAN_TAIL = "orphan_tail"


@dataclass(frozen=True)
class KinsokuViolation:
    """Individual violation of Kinsoku line-breaking or typography rules."""

    rule: KinsokuRuleType
    line_number: int
    column: int
    character: str
    snippet: str
    message: str
    suggested_fix: str


@dataclass(frozen=True)
class KinsokuAuditReport:
    """Aggregate audit metrics for document typography compliance."""

    total_lines: int
    total_cjk_chars: int
    violations: list[KinsokuViolation]
    consecutive_punct_count: int
    orphan_tail_count: int
    compliance_score: float

    @property
    def is_clean(self) -> bool:
        """True if no typography violations were detected."""
        return len(self.violations) == 0


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _split_protected_spans(text: str) -> list[str]:
    """Split text into alternating [unprotected, protected, ...] segments."""
    intervals = _merge_intervals(extract_code_like_spans(text))
    if not intervals:
        return [text]
    parts: list[str] = []
    last = 0
    for start, end in intervals:
        parts.append(text[last:start])
        parts.append(text[start:end])
        last = end
    parts.append(text[last:])
    return parts


def compress_cjk_punctuations(text: str, target_lang: str = "zh") -> str:
    """Deduplicate stutter punctuation and normalize whitespace between adjacent CJK punctuations."""
    if not text or not target_lang or not target_lang.lower().startswith(_SPACELESS_LANGS):
        return text
    if not _HAS_CJK.search(text):
        return text

    parts = _split_protected_spans(text)
    for i in range(0, len(parts), 2):
        seg = parts[i]
        for pattern, replacement in _STUTTER_PUNCT_PATTERNS:
            seg = pattern.sub(replacement, seg)
        parts[i] = seg
    return "".join(parts)


def apply_kinsoku_word_joiners(text: str, target_lang: str = "zh") -> str:
    r"""Inject Unicode Word Joiners (\u2060) to prevent prohibited line breaks.

    Inserts \u2060:
    1. Before characters prohibited at line start (e.g. char + \u2060 + ，);
    2. After characters prohibited at line end (e.g. “ + \u2060 + char);
    3. Between inseparable pairs (e.g. — + \u2060 + —).
    """
    if not text or not target_lang or not target_lang.lower().startswith(_SPACELESS_LANGS):
        return text
    if not _HAS_CJK.search(text):
        return text

    parts = _split_protected_spans(text)
    for i in range(0, len(parts), 2):
        seg = parts[i]
        # 1. Bind inseparable units
        for unit in INSEPARABLE_UNITS:
            if unit in seg:
                bound = unit[0] + WORD_JOINER + unit[1:]
                seg = seg.replace(unit, bound)

        # 2. Bind line-start prohibited characters to preceding character
        out: list[str] = []
        for idx, ch in enumerate(seg):
            if idx > 0 and ch in LINE_START_PROHIBITED:
                prev = seg[idx - 1]
                # Only insert if previous character is not whitespace, newline, or already a joiner
                if (
                    prev not in ("\n", "\r", " ", "\t", WORD_JOINER)
                    and prev not in LINE_END_PROHIBITED
                    and (not out or out[-1] != WORD_JOINER)
                ):
                    out.append(WORD_JOINER)
            out.append(ch)
            if ch in LINE_END_PROHIBITED and idx + 1 < len(seg):
                nxt = seg[idx + 1]
                # Only insert if next character is not whitespace, newline, or already a joiner
                if (
                    nxt not in ("\n", "\r", " ", "\t", WORD_JOINER)
                    and nxt not in LINE_START_PROHIBITED
                ):
                    out.append(WORD_JOINER)

        parts[i] = "".join(out)

    return "".join(parts)


def prevent_orphan_tails(text: str, target_lang: str = "zh") -> str:
    r"""Bind the final two content characters of a paragraph with \u2060 to avoid single-char orphans."""
    if not text or not target_lang or not target_lang.lower().startswith(_SPACELESS_LANGS):
        return text
    if not _HAS_CJK.search(text):
        return text

    lines = text.split("\n")
    fixed_lines: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        trailing_spaces = line[len(stripped) :]
        if len(stripped) >= 3 and _HAS_CJK.search(stripped):
            # Check if line ends with terminal punctuation preceded by CJK
            last_char = stripped[-1]
            if last_char in "。！？…”’）》】":
                penultimate = stripped[-2]
                antepenultimate = stripped[-3]
                if (
                    antepenultimate != WORD_JOINER
                    and _HAS_CJK.match(penultimate)
                    and _HAS_CJK.match(antepenultimate)
                ):
                    stripped = stripped[:-2] + WORD_JOINER + penultimate + last_char
        fixed_lines.append(stripped + trailing_spaces)
    return "\n".join(fixed_lines)


def fix_kinsoku_prose(text: str, target_lang: str = "zh") -> str:
    """Apply full CJK typography fix pipeline: compression, Kinsoku word joiners, and orphan prevention."""
    if not text:
        return text
    compressed = compress_cjk_punctuations(text, target_lang=target_lang)
    joined = apply_kinsoku_word_joiners(compressed, target_lang=target_lang)
    return prevent_orphan_tails(joined, target_lang=target_lang)


def audit_kinsoku_typography(text: str, target_lang: str = "zh") -> KinsokuAuditReport:
    """Audit lines and paragraphs for Kinsoku violations, consecutive punctuations, and orphans."""
    if not text:
        return KinsokuAuditReport(0, 0, [], 0, 0, 100.0)

    lines = text.splitlines()
    violations: list[KinsokuViolation] = []
    consecutive_punct_count = 0
    orphan_tail_count = 0
    total_cjk_chars = len(_HAS_CJK.findall(text))

    for line_idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        # 1. Line-start prohibited
        first_char = line[0]
        if first_char in LINE_START_PROHIBITED:
            snippet = line[: min(len(line), 20)]
            violations.append(
                KinsokuViolation(
                    rule=KinsokuRuleType.LINE_START_PROHIBITED,
                    line_number=line_idx,
                    column=1,
                    character=first_char,
                    snippet=snippet,
                    message=f"Line starts with prohibited punctuation '{first_char}'",
                    suggested_fix="Pull previous character to current line or bind with preceding line using Word Joiner",
                )
            )

        # 2. Line-end prohibited
        last_char = line[-1]
        if last_char in LINE_END_PROHIBITED:
            snippet = line[max(0, len(line) - 20) :]
            violations.append(
                KinsokuViolation(
                    rule=KinsokuRuleType.LINE_END_PROHIBITED,
                    line_number=line_idx,
                    column=len(line),
                    character=last_char,
                    snippet=snippet,
                    message=f"Line ends with prohibited opening punctuation '{last_char}'",
                    suggested_fix=f"Push '{last_char}' to next line or bind with following character",
                )
            )

        # 3. Check for broken inseparable pairs (e.g. line ends with single em-dash or 3-dot ellipsis)
        if raw_line.endswith("—") and not raw_line.endswith("——"):
            violations.append(
                KinsokuViolation(
                    rule=KinsokuRuleType.INSEPARABLE_PAIR,
                    line_number=line_idx,
                    column=len(raw_line),
                    character="—",
                    snippet=raw_line[-15:],
                    message="Broken em-dash split across line boundary",
                    suggested_fix="Keep both halves of '——' on the same line",
                )
            )

        # 4. Check consecutive punctuation runs (>= 2 fullwidth punctuations)
        consec_matches = re.finditer(r"([，。；：！？（）《》“”‘’【】]{2,})", line)
        for m in consec_matches:
            matched_str = m.group(1)
            # Legitimate pairs like —— or …… are ignored
            if matched_str not in ("——", "……", "──"):
                consecutive_punct_count += 1
                if any(c * 2 in matched_str for c in "，。；：！？"):
                    violations.append(
                        KinsokuViolation(
                            rule=KinsokuRuleType.CONSECUTIVE_PUNCTUATION,
                            line_number=line_idx,
                            column=m.start() + 1,
                            character=matched_str,
                            snippet=line[max(0, m.start() - 5) : min(len(line), m.end() + 5)],
                            message=f"Uncompressed or duplicate punctuation run '{matched_str}'",
                            suggested_fix="Deduplicate punctuation or compress spacing",
                        )
                    )

        # 5. Check for orphan tail in short line (< 3 chars ending paragraph)
        if 1 <= len(line) <= 2 and _HAS_CJK.search(line):
            orphan_tail_count += 1
            violations.append(
                KinsokuViolation(
                    rule=KinsokuRuleType.ORPHAN_TAIL,
                    line_number=line_idx,
                    column=1,
                    character=line,
                    snippet=line,
                    message=f"Orphaned short line with isolated character '{line}'",
                    suggested_fix="Pull additional character from previous line using Word Joiner",
                )
            )

    penalty = len(violations) * 5.0 + consecutive_punct_count * 1.5 + orphan_tail_count * 2.0
    denom = max(10, len(lines))
    score = max(0.0, min(100.0, 100.0 - (penalty / denom) * 20.0))

    return KinsokuAuditReport(
        total_lines=len(lines),
        total_cjk_chars=total_cjk_chars,
        violations=violations,
        consecutive_punct_count=consecutive_punct_count,
        orphan_tail_count=orphan_tail_count,
        compliance_score=round(score, 2),
    )

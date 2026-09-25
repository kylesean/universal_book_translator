"""High-performance deterministic terminology consistency enforcer using Aho-Corasick.

Prevents terminology drift, canonicalizes non-preferred aliases to approved renderings,
and replaces untranslated source term leaks with the canonical target translation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

try:
    import ahocorasick  # type: ignore[import-not-found]

    _HAS_AHOCORASICK = True
except ImportError:
    _HAS_AHOCORASICK = False


@dataclass(frozen=True, slots=True)
class EnforcementRecord:
    """Audit entry recording one deterministic terminology substitution."""

    original_span: str
    corrected_span: str
    start_pos: int
    end_pos: int
    rule_source: str


# Canonical CJK Unicode codepoint ranges, shared across glossary enforcement
# and CJK matching (ideographs, kana, and hangul).
CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0x2A700, 0x2B73F),  # CJK Extension C
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0xAC00, 0xD7AF),  # Hangul Syllables
)


def is_cjk_char(ch: str) -> bool:
    """Check whether a character falls in a shared CJK range (ideographs + kana + hangul)."""
    if not ch:
        return False
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in CJK_RANGES)


def _is_latin_word_char(ch: str) -> bool:
    """Check whether a character is a Latin/alphabetic word character or connector."""
    if ch in ("_", "-"):
        return True
    if ch.isascii():
        return ch.isalnum()
    cat = unicodedata.category(ch)
    if cat.startswith(("L", "N")):
        if is_cjk_char(ch):
            return False
        name = unicodedata.name(ch, "")
        return "LATIN" in name or "CYRILLIC" in name or "GREEK" in name
    return False


_PROTECTED_SPAN_PATTERNS = (
    re.compile(r"<[^>]+>"),  # HTML/XML tags
    re.compile(r"```[\s\S]*?```"),  # Fenced code blocks ```...```
    re.compile(r"~~~[\s\S]*?~~~"),  # Tilde fenced code blocks ~~~...~~~
    re.compile(r"`[^`\n]+`"),  # Inline code `...`
    re.compile(r"https?://[^\s<>'\"\)\]]+"),  # Standalone URLs
    re.compile(r"\$\$[\s\S]*?\$\$"),  # Display math $$...$$
    re.compile(r"\$[^\$\n]+\$"),  # Inline math $...$
    re.compile(r"\\\[[\s\S]*?\\\]"),  # LaTeX display \[...\]
    re.compile(r"\\\([\s\S]*?\\\)"),  # LaTeX inline \(...\)
    re.compile(r"\\begin\{[a-zA-Z*]+\}[\s\S]*?\\end\{[a-zA-Z*]+\}"),  # LaTeX environments
    re.compile(r"\]\([^\)\n]+\)"),  # Markdown link targets ](url)
    re.compile(r"⟦[^⟧]*⟧"),  # Masker tokens (checksum is lowercase hex + hyphen)
)


def extract_protected_spans(text: str) -> list[tuple[int, int]]:
    """Find intervals in text (HTML tags, LaTeX math, Markdown URLs) to shield from glossary substitutions."""
    spans: list[tuple[int, int]] = []
    for patt in _PROTECTED_SPAN_PATTERNS:
        for m in patt.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def find_term_occurrences(
    text: str,
    term: str,
    protected: list[tuple[int, int]] | None = None,
    *,
    case_insensitive: bool = False,
) -> list[tuple[int, int]]:
    """Boundary-aware occurrences of ``term`` in ``text``, excluding protected spans.

    Single-sources the enforcer's matching rules (Latin word boundaries;
    HTML/math/URL/masker regions are never term positions) so downstream
    consumers — e.g. the drift detector and the terminology metrics — cannot
    count a term that the enforcer itself would refuse to touch.

    ``protected`` lets a caller match many terms against one text without paying
    the nine structural patterns per term; pass ``extract_protected_spans(text)``
    when doing so.

    ``case_insensitive`` folds case through ``re.IGNORECASE`` instead of
    lowering either string, so offsets stay valid in the original ``text``.
    Only *judging* consumers (does the target render the term?) fold case —
    the rewriter below stays case-sensitive and is reached through the default.
    """
    if not text or not term:
        return []
    is_latin = bool(re.search(r"[A-Za-z]", term)) and not any(is_cjk_char(c) for c in term)
    if protected is None:
        protected = extract_protected_spans(text)
    flags = re.IGNORECASE if case_insensitive else 0
    found: list[tuple[int, int]] = []
    for match in re.finditer(re.escape(term), text, flags):
        start, end = match.start(), match.end()
        if is_latin:
            if start > 0 and _is_latin_word_char(text[start - 1]):
                continue
            if end < len(text) and _is_latin_word_char(text[end]):
                continue
        if any(start < p_end and end > p_start for p_start, p_end in protected):
            continue
        found.append((start, end))
    return found


class DeterministicGlossaryEnforcer:
    """Enforces 100% deterministic terminology consistency using an Aho-Corasick automaton.

    Features:
    1. Aho-Corasick multi-pattern dictionary tree for $O(N)$ string scanning.
    2. Boundary-aware match verification (word boundary assertions for Latin/Cyrillic scripts).
    3. Longest-match-first disambiguation for overlapping term matches.
    4. Single-pass non-destructive string slicing (immune to cyclic re-substitutions).
    5. Morphological inflection awareness: allows approved target inflected variants.
    """

    def __init__(
        self,
        glossary: list[dict[str, Any]],
        target_lang: str = "zh",
        source_lang: str = "en",
        enforce_source_leak_replacement: bool = True,
    ) -> None:
        self.target_lang = target_lang.lower()
        self.source_lang = source_lang.lower()
        self.enforce_source_leak_replacement = enforce_source_leak_replacement

        # Build replacement mapping: pattern -> (canonical_translation, rule_source, is_latin_word)
        # Also collect approved target variants to prevent overwriting correct inflections.
        self._rules: dict[str, tuple[str, str, bool]] = {}
        self._approved_target_forms: set[str] = set()

        self._compile_glossary(glossary)
        self._build_automaton()

    def _compile_glossary(self, glossary: list[dict[str, Any]]) -> None:
        for entry in glossary:
            source = str(entry.get("source", "")).strip()
            target = str(entry.get("translation", "")).strip()
            if not target:
                continue

            self._approved_target_forms.add(target)
            # Add inflected variants to approved set
            for var in entry.get("inflected_variants") or []:
                v_clean = str(var).strip()
                if v_clean:
                    self._approved_target_forms.add(v_clean)

            # 1. Map aliases -> canonical target
            for alias in entry.get("aliases") or []:
                a_clean = str(alias).strip()
                if a_clean and a_clean != target and a_clean not in self._approved_target_forms:
                    is_latin = bool(re.search(r"[A-Za-z]", a_clean)) and not any(
                        is_cjk_char(c) for c in a_clean
                    )
                    self._rules[a_clean] = (target, f"alias:{source}->{target}", is_latin)

            # 2. Map untranslated source term -> canonical target (if scripts differ or leak enforcement enabled)
            if self.enforce_source_leak_replacement and source and source != target:
                # If source is Latin and target has non-Latin (or CJK), replacing untranslated English in Chinese text is critical
                is_latin_src = bool(re.search(r"[A-Za-z]", source))
                # Avoid adding very short English words (1-2 chars) as source leaks unless uppercase acronyms
                if (
                    len(source) > 2 or source.isupper()
                ) and source not in self._approved_target_forms:
                    self._rules[source] = (target, f"source_leak:{source}->{target}", is_latin_src)

    def _build_automaton(self) -> None:
        if _HAS_AHOCORASICK and self._rules:
            self._automaton = ahocorasick.Automaton()
            for pattern, val in self._rules.items():
                self._automaton.add_word(pattern, (pattern, *val))
            self._automaton.make_automaton()
        else:
            self._automaton = None

    def _verify_boundary(self, text: str, start: int, end: int, is_latin: bool) -> bool:
        """Check whether the match satisfies word/token boundary assertions."""
        if not is_latin:
            return True

        # Left boundary check
        if start > 0:
            prev_char = text[start - 1]
            if _is_latin_word_char(prev_char):
                return False

        # Right boundary check
        if end < len(text):
            next_char = text[end]
            if _is_latin_word_char(next_char):
                return False

        return True

    def enforce(self, target_text: str) -> tuple[str, list[EnforcementRecord]]:
        """Enforce glossary consistency on target_text in a single atomic pass.

        Returns:
            (corrected_text, list_of_enforcement_records)
        """
        if not target_text or not self._rules:
            return target_text, []

        raw_matches: list[
            tuple[int, int, str, str, str]
        ] = []  # (start, end, pattern, replacement, rule)

        if self._automaton is not None:
            # Aho-Corasick scanning
            for end_idx, (pattern, repl, rule, is_latin) in self._automaton.iter(target_text):
                start_idx = end_idx - len(pattern) + 1
                end_pos = end_idx + 1
                if self._verify_boundary(target_text, start_idx, end_pos, is_latin):
                    raw_matches.append((start_idx, end_pos, pattern, repl, rule))
        else:
            # Pure Python Trie / regex fallback
            for pattern, (repl, rule, is_latin) in self._rules.items():
                patt = re.compile(re.escape(pattern))
                for m in patt.finditer(target_text):
                    if self._verify_boundary(target_text, m.start(), m.end(), is_latin):
                        raw_matches.append((m.start(), m.end(), pattern, repl, rule))

        if not raw_matches:
            return target_text, []

        # Protected structure guard: prevent substitutions inside HTML tags,
        # LaTeX math environments, markdown URLs, or deterministic maskers.
        protected_spans = extract_protected_spans(target_text)
        if protected_spans:
            filtered_by_protection: list[tuple[int, int, str, str, str]] = []
            for start, end, pattern, repl, rule in raw_matches:
                if any(start < p_end and end > p_start for p_start, p_end in protected_spans):
                    continue
                filtered_by_protection.append((start, end, pattern, repl, rule))
            raw_matches = filtered_by_protection

        if not raw_matches:
            return target_text, []

        # H12: CJK compound guard — an EXPANSION rule (replacement longer than
        # the pattern, e.g. alias '网络' -> '神经网络') inserts characters into
        # the surrounding text. When the match sits inside a CJK run, that
        # insertion corrupts correct compounds ('计算机网络' becomes
        # '计算机神经网络'). The hazard only exists when the pattern is already
        # part of its own canonical replacement; a full swap that shares no
        # text with the pattern ('关注' -> '注意力') cannot merge with flanks,
        # so it stays enabled to enforce correct term replacement.
        # Equal-length swaps and contractions never insert either.
        filtered_by_compound: list[tuple[int, int, str, str, str]] = []
        for start, end, pattern, repl, rule in raw_matches:
            if len(repl) > len(pattern) and pattern in repl:
                flanked = (start > 0 and is_cjk_char(target_text[start - 1])) or (
                    end < len(target_text) and is_cjk_char(target_text[end])
                )
                if flanked:
                    continue
            filtered_by_compound.append((start, end, pattern, repl, rule))
        raw_matches = filtered_by_compound

        if not raw_matches:
            return target_text, []

        # Idempotency guard:
        # On resume/re-run the enforcer sees text that already contains its own
        # substitutions. Skip a match when it overlaps ANY existing occurrence
        # of its canonical replacement — not merely prefix-aligned ones
        # (handles patterns that are substrings of their replacement, e.g.
        # '网络' inside '神经网络').
        filtered_matches: list[tuple[int, int, str, str, str]] = []
        for start, end, pattern, repl, rule in raw_matches:
            scan_start = max(0, start - len(repl) + 1)
            scan_end = min(len(target_text), end + len(repl) - 1)
            search_from = max(0, scan_start - len(repl))
            skip_match = False
            while True:
                rel = target_text.find(repl, search_from, scan_end + len(repl))
                if rel < 0:
                    break
                repl_start, repl_end = rel, rel + len(repl)
                overlaps = repl_start < end and repl_end > start
                inside_match = repl_start >= start and repl_end <= end
                # Skip only when an applied replacement overlaps the match
                # from OUTSIDE it. An occurrence fully contained in the match
                # span (the replacement text being a substring of the source
                # pattern, e.g. '工作记忆' inside '短时工作记忆') is inherent
                # to the pattern, not evidence of a prior substitution.
                if overlaps and not inside_match:
                    skip_match = True
                    break
                search_from = rel + 1
            if not skip_match:
                filtered_matches.append((start, end, pattern, repl, rule))
        raw_matches = filtered_matches

        if not raw_matches:
            return target_text, []

        # Disambiguate overlapping matches with true longest-match-first
        # Semantics (the doc promised longest-match, the code did
        # leftmost-then-longer; longest patterns now always win regardless of
        # their position), leftmost breaking ties.
        raw_matches.sort(key=lambda item: (-(item[1] - item[0]), item[0]))

        non_overlapping: list[tuple[int, int, str, str, str]] = []
        taken: list[tuple[int, int]] = []
        for start, end, pattern, repl, rule in raw_matches:
            # Longest-first order requires interval checks against every
            # already-taken span (a single running last_end only works when
            # matches are sorted by start position).
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            taken.append((start, end))
            non_overlapping.append((start, end, pattern, repl, rule))

        if not non_overlapping:
            return target_text, []

        # Single-pass string assembly proceeds in text order.
        non_overlapping.sort(key=lambda item: item[0])

        # Single-pass string assembly
        pieces: list[str] = []
        records: list[EnforcementRecord] = []
        curr_idx = 0

        for start, end, orig, repl, rule in non_overlapping:
            if start > curr_idx:
                pieces.append(target_text[curr_idx:start])
            pieces.append(repl)
            records.append(
                EnforcementRecord(
                    original_span=orig,
                    corrected_span=repl,
                    start_pos=start,
                    end_pos=end,
                    rule_source=rule,
                )
            )
            curr_idx = end

        if curr_idx < len(target_text):
            pieces.append(target_text[curr_idx:])

        return "".join(pieces), records

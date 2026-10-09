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

from ubt.core.cjk_ranges import CJK_RANGES, contains_cjk, is_cjk_char
from ubt.core.cleaners.inline_math import inline_math_spans

__all__ = ["CJK_RANGES", "contains_cjk", "is_cjk_char"]


@dataclass(frozen=True, slots=True)
class EnforcementRecord:
    """Audit entry recording one deterministic terminology substitution."""

    original_span: str
    corrected_span: str
    start_pos: int
    end_pos: int
    rule_source: str


def _is_latin_word_char(ch: str) -> bool:
    """Check whether a character is an alphabetic/alphanumeric word character or connector."""
    if ch in ("_", "-"):
        return True
    if ch.isascii():
        return ch.isalnum()
    cat = unicodedata.category(ch)
    if cat.startswith(("L", "N")):
        return not is_cjk_char(ch)
    return False


def _make_tokenizer(rules: dict[str, tuple[str, str, bool, bool]]) -> Any | None:
    """A private jieba tokenizer seeded with this glossary's CJK patterns.

    Returns ``None`` when jieba is unavailable. A per-instance
    :class:`jieba.Tokenizer` is used deliberately: the module-level
    ``jieba.dt`` is process-global, so seeding it made one job's glossary
    change another concurrent job's token boundaries (``worker --concurrency``)
    and made enforcement non-deterministic across resumes. ``add_word`` also
    grows ``Tokenizer.total`` without bound, so a shared instance leaked memory
    per job. A private instance keeps both problems local to the glossary that
    owns them.
    """
    try:
        import jieba  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover - jieba is a hard dependency
        return None
    tokenizer = jieba.Tokenizer()
    for pat in rules:
        if len(pat) >= 2 and all(is_cjk_char(ch) for ch in pat):
            tokenizer.add_word(pat, freq=1000000)
    return tokenizer


_PROTECTED_SPAN_PATTERNS = (
    re.compile(r"<[^>]+>"),  # HTML/XML tags
    re.compile(r"```[\s\S]*?```"),  # Fenced code blocks ```...```
    re.compile(r"~~~[\s\S]*?~~~"),  # Tilde fenced code blocks ~~~...~~~
    re.compile(r"`[^`\n]+`"),  # Inline code `...`
    re.compile(r"https?://[^\s<>'\"\)\]]+"),  # Standalone URLs
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),  # Email addresses
    re.compile(r"\$\$[\s\S]*?\$\$"),  # Display math $$...$$
    re.compile(r"\\\[[\s\S]*?\\\]"),  # LaTeX display \[...\]
    re.compile(r"\\\([\s\S]*?\\\)"),  # LaTeX inline \(...\)
    re.compile(r"\\begin\{[a-zA-Z*]+\}[\s\S]*?\\end\{[a-zA-Z*]+\}"),  # LaTeX environments
    re.compile(r"\]\([^\)\n]+\)"),  # Markdown link targets ](url)
    re.compile(r"⟦[^⟧]*⟧"),  # Masker tokens (checksum is lowercase hex + hyphen)
    # Paired quotation. A quoted span is a citation of the original wording -- a
    # product name, an identifier, a title -- so rewriting it is the Chinglish
    # the enforcer exists to prevent, not the drift it exists to fix
    # (该组件名为 "Data Loader" must keep the name it names). Delimiters are
    # matched as explicit pairs, never as one character class, so a closing
    # quote can never open a span. The straight double quote opens a span only
    # when no Latin letter or digit precedes it: 10" is an inch mark, and
    # treating it as an opener swallowed the text up to the next real quote.
    # Spans are single-line and length-capped; an over-long or unbalanced quote
    # protects nothing, which fails toward enforcement. The straight single
    # quote is absent on purpose: ' is overwhelmingly an apostrophe in the
    # contractions and possessives that fill English prose, and a pattern that
    # pairs any two of them would shield spans that no one wrote as quotes.
    re.compile(r"“[^“”\n]{1,200}”"),  # “typographic double”
    re.compile(r"‘[^‘’\n]{1,200}’"),  # ‘typographic single’
    re.compile(r"「[^「」\n]{1,200}」"),  # 「CJK corner」
    re.compile(r"『[^『』\n]{1,200}』"),  # 『CJK white corner』
    re.compile(r"«[^«»\n]{1,200}»"),  # «guillemets»
    re.compile(r'(?<![A-Za-z0-9])"[^"\n]{1,200}"'),  # "straight double"
)


def extract_protected_spans(text: str) -> list[tuple[int, int]]:
    """Find intervals in text (HTML tags, code, LaTeX math, URLs, quotes) to shield from glossary substitutions.

    Inline ``$...$`` is matched through the shared, currency-aware
    :func:`~ubt.core.cleaners.inline_math.inline_math_spans`: a naive
    ``\\$[^\\$\\n]+\\$`` paired ``"$5 and $10"`` into one "math" span and
    shielded every term between the two dollars from enforcement.

    Paired quotation is protected for the same reason code is: a quoted span
    points at the original wording, so substituting a term inside it corrupts a
    deliberate citation rather than fixing drift. See
    :data:`_PROTECTED_SPAN_PATTERNS` for the delimiter rules.
    """
    spans: list[tuple[int, int]] = []
    for patt in _PROTECTED_SPAN_PATTERNS:
        for m in patt.finditer(text):
            spans.append((m.start(), m.end()))
    spans.extend(inline_math_spans(text))
    return spans


def _term_pattern(term: str) -> str:
    """A term's regex, allowing optional whitespace at CJK<->Latin boundaries.

    CJK/Latin spacing inserts a space the term itself does not carry
    (``CPU调度`` renders as ``CPU 调度``), so an exact match reports a correctly
    rendered term as drifted. Whitespace is allowed only where the term crosses
    the CJK/Latin boundary, never inside a Latin run.
    """
    parts: list[str] = []
    previous = ""
    for ch in term:
        if previous and is_cjk_char(previous) != is_cjk_char(ch):
            # Horizontal whitespace only: ``\s`` also matches newlines, which
            # let a term's halves match across a line or paragraph break
            # ("CPU\n调度器" counting as "CPU调度").
            parts.append(r"[ \t]*")
        parts.append(re.escape(ch))
        previous = ch
    return "".join(parts)


def find_term_occurrences(
    text: str,
    term: str,
    protected: list[tuple[int, int]] | None = None,
    *,
    case_insensitive: bool = False,
    allow_cjk_latin_space: bool = False,
) -> list[tuple[int, int]]:
    """Boundary-aware occurrences of ``term`` in ``text``, excluding protected spans.

    Single-sources the enforcer's matching rules (Latin word boundaries;
    HTML/math/URL/masker regions are never term positions) so downstream
    consumers — e.g. the drift detector and the terminology metrics — cannot
    count a term that the enforcer itself would refuse to touch.

    ``protected`` lets a caller match many terms against one text without paying
    the structural patterns per term; pass ``extract_protected_spans(text)``
    when doing so.

    ``case_insensitive`` folds case through ``re.IGNORECASE`` instead of
    lowering either string, so offsets stay valid in the original ``text``.
    Only *judging* consumers (does the target render the term?) fold case —
    the rewriter below stays case-sensitive and is reached through the default.
    """
    if not text or not term:
        return []
    check_left = bool(term) and _is_latin_word_char(term[0])
    check_right = bool(term) and _is_latin_word_char(term[-1])
    if protected is None:
        protected = extract_protected_spans(text)
    flags = re.IGNORECASE if case_insensitive else 0
    pattern = _term_pattern(term) if allow_cjk_latin_space else re.escape(term)
    found: list[tuple[int, int]] = []
    for match in re.finditer(pattern, text, flags):
        start, end = match.start(), match.end()
        if check_left and start > 0 and _is_latin_word_char(text[start - 1]):
            continue
        if check_right and end < len(text) and _is_latin_word_char(text[end]):
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
    6. A private CJK tokenizer so a compound-word guard never depends on (or
       mutates) another job's process-global tokenizer state.
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

        # Build replacement mapping: pattern -> (canonical_translation, rule_source, check_left, check_right)
        # Also collect approved target variants to prevent overwriting correct inflections.
        self._rules: dict[str, tuple[str, str, bool, bool]] = {}
        self._approved_target_forms: set[str] = set()

        self._compile_glossary(glossary)
        self._build_automaton()
        self._tokenizer = _make_tokenizer(self._rules)

    def _compile_glossary(self, glossary: list[dict[str, Any]]) -> None:
        # Two passes on purpose: the alias guard below compares against every
        # *approved target form*, but the targets are only known after the whole
        # glossary is read. Compiling in one pass let an alias that collides with
        # a later entry's translation be rewritten (entry A alias "Beta", entry B
        # translation "Beta" -> "Beta" got remapped to A's target).
        for entry in glossary:
            target = str(entry.get("translation", "")).strip()
            if not target:
                continue
            self._approved_target_forms.add(target)
            # Add inflected variants to approved set
            for var in entry.get("inflected_variants") or []:
                v_clean = str(var).strip()
                if v_clean:
                    self._approved_target_forms.add(v_clean)

        for entry in glossary:
            source = str(entry.get("source", "")).strip()
            target = str(entry.get("translation", "")).strip()
            if not target:
                continue

            # 1. Map aliases -> canonical target
            for alias in entry.get("aliases") or []:
                a_clean = str(alias).strip()
                if a_clean and a_clean != target and a_clean not in self._approved_target_forms:
                    check_left = _is_latin_word_char(a_clean[0])
                    check_right = _is_latin_word_char(a_clean[-1])
                    self._rules[a_clean] = (
                        target,
                        f"alias:{source}->{target}",
                        check_left,
                        check_right,
                    )

            # 2. Map untranslated source term -> canonical target (if scripts differ or leak enforcement enabled)
            if (
                self.enforce_source_leak_replacement
                and source
                and source != target
                and (len(source) > 2 or source.isupper())
                and source not in self._approved_target_forms
            ):
                check_left_src = _is_latin_word_char(source[0])
                check_right_src = _is_latin_word_char(source[-1])
                self._rules[source] = (
                    target,
                    f"source_leak:{source}->{target}",
                    check_left_src,
                    check_right_src,
                )

    def _build_automaton(self) -> None:
        if _HAS_AHOCORASICK and self._rules:
            self._automaton = ahocorasick.Automaton()
            for pattern, val in self._rules.items():
                self._automaton.add_word(pattern, (pattern, *val))
            self._automaton.make_automaton()
        else:
            self._automaton = None

    def _verify_boundary(
        self, text: str, start: int, end: int, check_left: bool, check_right: bool
    ) -> bool:
        """Check whether the match satisfies word/token boundary assertions."""
        # Left boundary check
        if check_left and start > 0:
            prev_char = text[start - 1]
            if _is_latin_word_char(prev_char):
                return False

        # Right boundary check
        if check_right and end < len(text):
            next_char = text[end]
            if _is_latin_word_char(next_char):
                return False

        return True

    def enforce_audited(
        self, target_text: str
    ) -> tuple[str, list[EnforcementRecord], list[EnforcementRecord]]:
        """Enforce glossary consistency on target_text with compound audit.

        Returns:
            (corrected_text, list_of_applied_records, list_of_quarantined_records)
        """
        if not target_text or not self._rules:
            return target_text, [], []

        raw_matches: list[
            tuple[int, int, str, str, str]
        ] = []  # (start, end, pattern, replacement, rule)

        if self._automaton is not None:
            # Aho-Corasick scanning
            for end_idx, (pattern, repl, rule, check_left, check_right) in self._automaton.iter(
                target_text
            ):
                start_idx = end_idx - len(pattern) + 1
                end_pos = end_idx + 1
                if self._verify_boundary(target_text, start_idx, end_pos, check_left, check_right):
                    raw_matches.append((start_idx, end_pos, pattern, repl, rule))
        else:
            # Pure Python Trie / regex fallback
            for pattern, (repl, rule, check_left, check_right) in self._rules.items():
                patt = re.compile(re.escape(pattern))
                for m in patt.finditer(target_text):
                    if self._verify_boundary(
                        target_text, m.start(), m.end(), check_left, check_right
                    ):
                        raw_matches.append((m.start(), m.end(), pattern, repl, rule))

        if not raw_matches:
            return target_text, [], []

        # Protected structure guard: prevent substitutions inside HTML tags,
        # LaTeX math environments, markdown URLs, deterministic maskers, or
        # paired quotes (a quoted span cites the original wording).
        protected_spans = extract_protected_spans(target_text)
        if protected_spans:
            filtered_by_protection: list[tuple[int, int, str, str, str]] = []
            for start, end, pattern, repl, rule in raw_matches:
                if any(start < p_end and end > p_start for p_start, p_end in protected_spans):
                    continue
                filtered_by_protection.append((start, end, pattern, repl, rule))
            raw_matches = filtered_by_protection

        if not raw_matches:
            return target_text, [], []

        # CJK compound & token boundary guard.
        # A CJK alias inside a larger CJK run is almost always part of a longer word
        # (e.g., '关注' in '关注度很高', '云' in '云计算', '状态' in '初始状态').
        # Mechanically replacing it destroys natural compound words and produces severe grammatical errors.
        # 1. Single-character CJK term flanked by CJK characters is inherently unsafe.
        # 2. Longer replacement flanked by CJK characters is unsafe (prevents expanding fragments).
        # 3. Tokenizer-aware boundary check (jieba): match MUST align with token boundaries.
        # Unsafe matches are quarantined into audit records rather than violently applied.
        #
        # The guard fails CLOSED: if a CJK term is present but the tokenizer is
        # unavailable, every CJK match is quarantined rather than applied, because
        # a missing tokenizer must not silently downgrade to the mechanical
        # substitution the guard exists to prevent (``初始状态`` -> ``初始态势``).
        token_boundaries: tuple[set[int], set[int]] | None = None
        has_cjk_matches = any(
            bool(p) and all(is_cjk_char(ch) for ch in p) for _, _, p, _, _ in raw_matches
        )
        if has_cjk_matches and self._tokenizer is not None:
            tokens = list(self._tokenizer.tokenize(target_text))
            token_boundaries = ({s for _, s, _ in tokens}, {e for _, _, e in tokens})

        filtered_by_compound: list[tuple[int, int, str, str, str]] = []
        quarantined_matches: list[tuple[int, int, str, str, str]] = []

        for start, end, pattern, repl, rule in raw_matches:
            cjk_pattern = bool(pattern) and all(is_cjk_char(ch) for ch in pattern)
            is_unsafe = False
            if cjk_pattern:
                flanked = (start > 0 and is_cjk_char(target_text[start - 1])) or (
                    end < len(target_text) and is_cjk_char(target_text[end])
                )
                if (len(pattern) == 1 or len(repl) > len(pattern)) and flanked:
                    is_unsafe = True
                elif flanked:
                    # A flanked CJK term needs a tokenizer to prove the match
                    # aligns with a word boundary. Without one, quarantine
                    # rather than apply: a missing tokenizer must not silently
                    # downgrade to the mechanical compound substitution the
                    # guard exists to prevent (``初始状态`` -> ``初始态势``).
                    # A standalone (unflanked) term is unambiguous and applied.
                    if token_boundaries is None:
                        is_unsafe = True
                    else:
                        t_starts, t_ends = token_boundaries
                        if start not in t_starts or end not in t_ends:
                            is_unsafe = True

            if is_unsafe:
                quarantined_matches.append((start, end, pattern, repl, rule))
            else:
                filtered_by_compound.append((start, end, pattern, repl, rule))
        raw_matches = filtered_by_compound

        quarantined_records: list[EnforcementRecord] = [
            EnforcementRecord(
                original_span=orig,
                corrected_span=repl,
                start_pos=start,
                end_pos=end,
                rule_source=f"quarantined_compound:{rule}",
            )
            for start, end, orig, repl, rule in quarantined_matches
        ]

        if not raw_matches:
            return target_text, [], quarantined_records

        # Idempotency guard:
        # On resume/re-run the enforcer sees text that already contains its own
        # substitutions. Skip a match only when an occurrence of its canonical
        # replacement *contains* the match span — that is the precise signal the
        # text at this location was produced by a prior substitution. The old
        # "overlaps from outside" heuristic also skipped a legitimate match that
        # merely shared a boundary character with an incidental replacement
        # elsewhere ('甲乙丙' with 甲乙->乙丙 was left unenforced), so
        # consistency enforcement silently stopped enforcing.
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
                if repl_start <= start and end <= repl_end:
                    skip_match = True
                    break
                search_from = rel + 1
            if not skip_match:
                filtered_matches.append((start, end, pattern, repl, rule))
        raw_matches = filtered_matches

        if not raw_matches:
            return target_text, [], quarantined_records

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
            return target_text, [], quarantined_records

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

        return "".join(pieces), records, quarantined_records

    def enforce(self, target_text: str) -> tuple[str, list[EnforcementRecord]]:
        """Enforce glossary consistency on target_text in a single atomic pass.

        Returns:
            (corrected_text, list_of_enforcement_records)
        """
        corrected, applied, _ = self.enforce_audited(target_text)
        return corrected, applied

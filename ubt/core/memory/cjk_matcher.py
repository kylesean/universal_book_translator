import functools
import re
from typing import Any

# CJK character range definitions and ``is_cjk_char`` are centralized in
# ``ubt.core.validators.glossary_enforcer`` to guarantee consistent matching
# semantics across both the matcher and drift detectors.
from ubt.core.validators.glossary_enforcer import is_cjk_char as is_cjk_char

try:
    import ahocorasick  # type: ignore[import-not-found]

    _HAS_AHOCORASICK = True
except ImportError:
    _HAS_AHOCORASICK = False

# Below this many glossary terms the automaton build costs more than a
# direct scan; small books keep the old loop (identical semantics).
_AHO_MIN_TERMS = 8

# CJK character ranges (ideographs, kana, hangul) that constitute word boundaries for non-CJK terms:
_CJK_CHARS_REGEX = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"


@functools.lru_cache(maxsize=4096)
def _compile_boundary_pattern(source: str) -> re.Pattern[str]:
    """Precompile and cache Unicode-aware word boundary pattern."""
    escaped = re.escape(source)
    return re.compile(
        rf"(?<![^\W_{_CJK_CHARS_REGEX}]){escaped}(?![^\W_{_CJK_CHARS_REGEX}])", re.IGNORECASE
    )


def contains_cjk(s: str) -> bool:
    """True if s contains any CJK character."""
    return any(is_cjk_char(c) for c in s)


def _cjk_has_free_boundary(source: str, text: str) -> bool:
    """True when some occurrence of a CJK term touches a non-CJK edge.

    selection guard: a 2-3 char CJK term that only ever occurs
    embedded inside longer CJK runs is likely a compound fragment, not an
    independent hit. Callers deprioritize (never drop) such matches so the
    compound guard in ``glossary_enforcer`` stays the enforcement point.
    """
    if not source or not text:
        return False
    start = 0
    while True:
        idx = text.find(source, start)
        if idx < 0:
            return False
        end = idx + len(source)
        left_free = idx == 0 or not is_cjk_char(text[idx - 1])
        right_free = end >= len(text) or not is_cjk_char(text[end])
        if left_free or right_free:
            return True
        start = idx + 1


def _aho_candidate_indices(terms: list[dict[str, Any]], lowered_text: str) -> set[int] | None:
    """Single-pass multi-pattern prefilter over source+aliases.

    Returns candidate term indices that *may* match, or None when the
    optional ``ahocorasick`` dependency is unavailable. Candidates are
    always re-verified with ``term_appears_in_text`` so semantics are
    identical with or without the library.
    """
    if not _HAS_AHOCORASICK:
        return None
    sig = tuple(
        (str(term.get("source", "")), tuple(str(a) for a in term.get("aliases", []) or [] if a))
        for term in terms
    )
    automaton = _get_compiled_automaton(sig)
    candidates: set[int] = set()
    for _end, idxs in automaton.iter(lowered_text):
        candidates.update(idxs)
    return candidates


@functools.lru_cache(maxsize=32)
def _get_compiled_automaton(term_signatures: tuple[tuple[str, tuple[str, ...]], ...]) -> Any:
    """Precompile and cache ahocorasick Automaton from term signatures."""
    # One key can belong to several terms (term A's source == term B's alias).
    # ``add_word`` replaces the value for an existing key, so a per-key index
    # list is stored and the reader unions them — otherwise every term but the
    # last sharing a key was dropped from the candidate set and, once the global
    # top-N pool is exhausted, from the glossary entirely.
    by_key: dict[str, list[int]] = {}
    for idx, (source, aliases) in enumerate(term_signatures):
        keys = {source.lower()}
        keys.update(a.lower() for a in aliases)
        for key in keys:
            if key:
                by_key.setdefault(key, []).append(idx)
    automaton = ahocorasick.Automaton()
    for key, idxs in by_key.items():
        automaton.add_word(key, tuple(idxs))
    automaton.make_automaton()
    return automaton


def count_term_in_text(source: str, text: str) -> int:
    """Count occurrences of source in text.

    - CJK: uses substring matching; rejects single-character terms (returns 0)
      to prevent massive false-positive over-matching.
    - All other scripts (Latin, Cyrillic, Greek, accented...): uses the
      Unicode-aware boundary regex (?<!\\w)source(?!\\w) with case-insensitivity
      so 'cat' does not match 'category' and 'Attention' matches 'attention'.
    """
    if not source or not text:
        return 0
    if contains_cjk(source):
        if len(source) <= 1:
            return 0  # Skip single CJK characters
        return text.count(source)
    if source.lower() not in text.lower():
        return 0
    pattern = _compile_boundary_pattern(source)
    return len(pattern.findall(text))


def _term_matches(source: str, text: str) -> bool:
    """Fast early-exit check whether a term exists in text."""
    if not source or not text:
        return False
    if contains_cjk(source):
        if len(source) <= 1:
            return False
        return source in text
    if source.lower() not in text.lower():
        return False
    pattern = _compile_boundary_pattern(source)
    return pattern.search(text) is not None


def term_appears_in_text(term: dict[str, Any], text: str) -> bool:
    """Check whether term's source or any of its aliases appear in text."""
    source = str(term.get("source", ""))
    if _term_matches(source, text):
        return True
    return any(_term_matches(str(alias), text) for alias in term.get("aliases", []))


def select_terms_for_chunk(
    terms: list[dict[str, Any]],
    chunk_text: str,
    top_n: int = 20,
    max_terms: int = 50,
) -> list[dict[str, Any]]:
    """Return terms relevant to chunk_text: union of local hits and global top-N."""
    if not terms:
        return []

    def sort_key(t: dict[str, Any]) -> tuple[int, str]:
        return (-int(t.get("frequency", 0)), str(t.get("source", "")))

    candidate_ids: set[int] | None = None
    if len(terms) >= _AHO_MIN_TERMS:
        candidate_ids = _aho_candidate_indices(terms, chunk_text.lower())
    local_terms = []
    for idx, term in enumerate(terms):
        if not term.get("source"):
            continue
        if candidate_ids is not None and idx not in candidate_ids:
            continue
        if term_appears_in_text(term, chunk_text):
            local_terms.append(term)
    clean_hits = []
    embedded_hits = []
    for term in local_terms:
        source = str(term.get("source", ""))
        if (
            contains_cjk(source)
            and 1 < len(source) <= 3
            and not _cjk_has_free_boundary(source, chunk_text)
        ):
            embedded_hits.append(term)
        else:
            clean_hits.append(term)
    clean_hits.sort(key=sort_key)
    embedded_hits.sort(key=sort_key)
    local_terms = clean_hits + embedded_hits

    local_ids = {t.get("id", t.get("source")) for t in local_terms}
    top_pool = sorted(
        (t for t in terms if t.get("id", t.get("source")) not in local_ids),
        key=sort_key,
    )
    top_terms = top_pool[: max(0, top_n)]

    if max_terms <= 0:
        return []
    if len(local_terms) >= max_terms:
        return local_terms[:max_terms]
    remaining = max_terms - len(local_terms)
    return local_terms + top_terms[:remaining]


def format_terms_markdown_table(terms: list[dict[str, Any]]) -> str:
    """Format terms as a 3-column markdown table: 原文 | 别名 | 译文."""
    if not terms:
        return ""

    def _clean(val: Any) -> str:
        s = " ".join(str(val or "").split())
        return s.replace("|", "\\|")

    rows = ["| 原文 | 别名 | 译文 |", "| :--- | :--- | :--- |"]
    for t in terms:
        src = _clean(t.get("source", ""))
        aliases = ", ".join(_clean(a) for a in t.get("aliases", []) if a)
        tgt = _clean(t.get("translation", t.get("target", "")))
        rows.append(f"| {src} | {aliases} | {tgt} |")

    return "\n".join(rows)

"""0-Token abbreviation pattern miner for the Translation Bible.

Academic papers introduce terminology with the canonical
"Full Expansion (ACRONYM)" convention — e.g. "Long Short-Term Memory (LSTM)"
or "scaled dot-product attention (SDPA)". These pairs are the highest-value
glossary entries: they are author-attested, appear early (abstract/intro),
and recur constantly. This miner extracts them deterministically (no LLM)
and emits BibleEntry-compatible dicts for the pipeline's Stage 2 bible.

Precision rule — strict word-initial matching: the words immediately
preceding the parenthesised acronym (hyphen compounds split into parts,
e.g. "dot-product" → dot/product) must spell the acronym by their initials.
This handles mixed initialisms (LSTM, SDPA, BERT, CNN) exactly, and rejects
greedy false prefixes like "We employ scaled dot-product attention".
"""

import re
from collections.abc import Iterable
from typing import Any

# Acronym inside parentheses: 2-10 chars, starts uppercase, letters/digits only
_ACRONYM_IN_PARENS = re.compile(r"\(\s*([A-Z][A-Za-z0-9]{1,9})\s*\)")
_WORD_TOKEN = re.compile(r"[A-Za-z]+")

_STOP_ACRONYMS = {"TV", "US", "UK", "PC", "AI"}  # common non-terminals, avoid noise
_STOP_WORDS = {
    "from",
    "of",
    "the",
    "a",
    "an",
    "and",
    "for",
    "in",
    "on",
    "with",
    "to",
    "by",
    "using",
    "based",
    "via",
}
_MAX_EXPANSION_TOKENS = 10
_MIN_EXPANSION_TOKENS = 2


def _find_expansion_for(acronym: str, left_context: str) -> str | None:
    """Return the tightest word suffix before the acronym whose initials spell it.

    Tokens include hyphen-split parts ("dot-product" → dot, product) so mixed
    initialisms like SDPA (scaled dot-product attention) match exactly.
    Filler words ("from", "of", ...) may be skipped, so BERT
    (Bidirectional Encoder Representations from Transformers) also matches.
    """
    tokens = [(m.group(0), m.start()) for m in _WORD_TOKEN.finditer(left_context)][
        -_MAX_EXPANSION_TOKENS:
    ]

    target = acronym.lower()
    for k in range(1, len(tokens) + 1):
        suffix = tokens[-k:]
        expansion_start = suffix[0][1]

        initials_all = "".join(w[0] for w, _ in suffix).lower()
        if initials_all == target and k >= _MIN_EXPANSION_TOKENS:
            return left_context[expansion_start:].strip()

        content = [(w, s) for w, s in suffix if w.lower() not in _STOP_WORDS]
        if len(content) >= _MIN_EXPANSION_TOKENS:
            initials_filtered = "".join(w[0] for w, _ in content).lower()
            if initials_filtered == target:
                return left_context[expansion_start:].strip()
    return None


def _scan_abbreviations(text: str) -> dict[str, dict[str, Any]]:
    """Scan one text region and return acronym-keyed entries in first-attestation order."""
    found: dict[str, dict[str, Any]] = {}
    for match in _ACRONYM_IN_PARENS.finditer(text):
        acronym = match.group(1)
        if acronym in _STOP_ACRONYMS or re.search(r"\d", acronym):
            continue
        start_window = max(0, match.start() - 300)
        expansion = _find_expansion_for(acronym, text[start_window : match.start()])
        if not expansion or len(expansion) < 4:
            continue
        key = acronym.lower()
        if key in found:
            continue
        found[key] = {
            "source": expansion,
            "translation": "",
            "aliases": [acronym],
            "kind": "term",
        }
    return found


def _capped_sorted_entries(
    found: dict[str, dict[str, Any]], max_entries: int
) -> list[dict[str, Any]]:
    """Cap to the first-attested ``max_entries`` then sort deterministically by acronym."""
    capped = list(found.values())[:max_entries]
    return sorted(capped, key=lambda e: str(e["aliases"][0]))


def mine_abbreviations(
    text: str,
    *,
    max_entries: int = 60,
) -> list[dict[str, Any]]:
    """Mine 'Expansion (ACRONYM)' pairs as bible-entry dicts (first attestation wins).

    Returns dicts shaped like ``BibleEntry.model_dump()``:
    ``{"source": expansion, "translation": "", "aliases": [acronym], "kind": "term"}``.
    Translation is empty by design: target-language renderings are supplied
    later by LLM draft/repair or a user-provided glossary override.
    """
    return _capped_sorted_entries(_scan_abbreviations(text), max_entries)


def mine_abbreviations_stream(
    blocks: Iterable[str],
    *,
    max_entries: int = 60,
    chunk_chars: int = 262_144,
    tail_chars: int = 1_024,
) -> list[dict[str, Any]]:
    """Streaming variant of :func:`mine_abbreviations` with bounded memory.

    Blocks are accumulated into ~``chunk_chars`` regions and scanned region by
    region; the last ``tail_chars`` bytes carry over so "Expansion (ACRONYM)"
    pairs near block boundaries still match (the 300-char expansion window is
    well inside the tail). Equivalent to mining ``"\\n".join(blocks)`` except
    for boundary-spanning pairs wider than the tail window.
    """
    found: dict[str, dict[str, Any]] = {}
    tail = ""
    buf: list[str] = []
    size = 0

    def _consume(chunk: str) -> None:
        for key, entry in _scan_abbreviations(chunk).items():
            if key not in found:
                found[key] = entry

    for text in blocks:
        buf.append(text)
        size += len(text) + 1
        if size >= chunk_chars:
            chunk = f"{tail}\n" + "\n".join(buf) if tail else "\n".join(buf)
            _consume(chunk)
            if len(found) >= max_entries:
                return _capped_sorted_entries(found, max_entries)
            tail = chunk[-tail_chars:]
            buf, size = [], 0
    if buf:
        chunk = f"{tail}\n" + "\n".join(buf) if tail else "\n".join(buf)
        _consume(chunk)
    return _capped_sorted_entries(found, max_entries)


def format_abbreviations_markdown_table(entries: list[dict[str, Any]]) -> str:
    """Render mined abbreviation pairs as a prompt-side canonical table.

    Instructs the model to keep the acronym unchanged and translate the
    expansion consistently — the empty-translation entries never appear in
    the term glossary, only here.
    """
    if not entries:
        return ""
    rows = ["| 缩写 | 全称（保持缩写不变，全称译法全篇一致） |", "| :--- | :--- |"]
    for e in entries:
        acronym = ", ".join(str(a) for a in e.get("aliases", []) if a)
        expansion = str(e.get("source", "")).replace("|", "\\|")
        rows.append(f"| {acronym} | {expansion} |")
    return "\n".join(rows)

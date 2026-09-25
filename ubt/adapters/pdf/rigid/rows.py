"""Row matching and continuation evidence for the rigid typesetter.

The rigid engine replaces per-line pairing, but zone growth still needs
the same normalized-text rungs the previous backfill engine calibrated:
token precision with an absolute content-hit count, the reclaimability
exclusions for equation/citation debris, the strong exact-substring rung
for cross-page tails, and clause splitting for zone pagination.

Pure functions — no I/O, safe to unit-test in isolation.
"""

from __future__ import annotations

import re

from ubt.adapters.pdf.textgeom import SUPERSCRIPT_DECODE_MAP, LineBox, dehyph

# Stopwords never count as content hits: a caption riding three of them
# must not clear the hit gate against a giant OCR-soup block.
_CLAIM_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "as",
        "at",
        "by",
        "from",
        "and",
        "or",
        "but",
        "nor",
        "so",
        "yet",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "which",
        "who",
        "will",
        "would",
        "can",
        "could",
        "may",
        "might",
        "shall",
        "should",
        "also",
        "not",
        "no",
        "vs",
        "per",
        "via",
        "up",
        "out",
        "off",
        "over",
        "under",
        "more",
        "most",
        "such",
        "than",
        "then",
        "thus",
        "there",
        "here",
    ]
)

_CJK_RE = re.compile("[\u4e00-\u9fff]")
_MIN_CLAIM_CHARS = 3
# P0 breaker: isolated equation numbers (e.g. "(3.1)"), citation brackets
# (e.g. "[1]"), or bare numbers must never be reclaimed as narrative
# continuation.
_RECLAIM_EXCLUDE_RE = re.compile(
    r"^(?:\(\s*\d+(?:\.\d+)*\s*\)|\[\s*\d+(?:[\s,\-–—]\d+)*\s*\]|\d+(?:\.\d+)*)$"
)


def _row_precision_hits(line_text: str, block_source: str) -> tuple[float, int]:
    """Row precision plus absolute content-hit count.

    Ratio alone lets a caption ride 3 stopword hits ("of the ... Functions")
    to 0.60 against a giant OCR-soup block. The hits count only content
    tokens (length > 1, stopwords excluded): "Chart of the Hyperbolic
    Functions." scores a single content hit ("functions") and stays
    orphaned, while real paragraph rows carry many. Callers require both
    the precision ratio and the hit count.
    """
    norm_line = (line_text or "").translate(SUPERSCRIPT_DECODE_MAP).lower()
    lt = re.findall(r"[A-Za-z0-9]+", norm_line)
    if not lt:
        return 0.0, 0
    norm_src = (block_source or "").translate(SUPERSCRIPT_DECODE_MAP).lower()
    bt = set(re.findall(r"[A-Za-z0-9]+", norm_src))
    prec = sum(1 for w in lt if w in bt) / len(lt)
    hits = sum(1 for w in lt if len(w) > 1 and w not in _CLAIM_STOPWORDS and w in bt)
    return prec, hits


def content_token_count(line_text: str) -> int:
    """Content tokens in one row (length > 1, stopwords excluded).

    Source-independent, so a caller may use it to decide whether a row is
    *judgeable* at all: a wrapped fragment like "expressed by" carries one
    content token and can neither confirm nor veto a run, while a real
    prose row carries many.
    """
    norm_line = (line_text or "").translate(SUPERSCRIPT_DECODE_MAP).lower()
    lt = re.findall(r"[A-Za-z0-9]+", norm_line)
    return sum(1 for w in lt if len(w) > 1 and w not in _CLAIM_STOPWORDS)


def _reclaimable(line: LineBox) -> bool:
    """Row-level eligibility mirroring the row-match P0 exclusions."""

    if line.table_band:
        return False
    text = (line.text or "").strip()
    norm = dehyph(text)
    if not norm:
        return False
    # 1-2 char latin/digit shards substring-match any long block
    # ("v" in "values") — same breaker as the precision gate.
    if len(norm) < _MIN_CLAIM_CHARS and not _CJK_RE.search(text):
        return False
    if _RECLAIM_EXCLUDE_RE.match(text):
        return False
    # Narrative continuation requires word characters.
    return bool(any(c.isalpha() for c in text) or _CJK_RE.search(text))


def _row_is_continuation(row_text: str, block_source: str) -> bool:
    """Strong-evidence rung: normalized exact substring with length and alpha gate."""

    norm = dehyph(row_text or "").strip()
    if not norm or not any(c.isalpha() for c in norm):
        return False
    if len(norm) >= 8:
        if norm in dehyph(block_source or ""):
            return True
        # Robust fallback for math/subscript OCR differences between engines
        # (e.g. Docling vs pdfium dropping 1-char subscripts or flipping sub/superscript order):
        # Check if the sequence of alphabetic words in the row is a subsequence of block_source.
        row_words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", row_text or "")]
        content_words = [w for w in row_words if w not in _CLAIM_STOPWORDS]
        if len(content_words) >= 2 or len(row_words) >= 4:
            src_words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", block_source or "")]
            it = iter(src_words)
            if all(w in it for w in row_words):
                return True
    # Tail rung for short wrap fragments: a paragraph whose last line wrapped
    # to the next page can be a single word ("agent.") far below the 8-char
    # substring floor. Unanchored substring matching is ambiguous at that
    # length, so require the row to be the exact END of the block's source --
    # an unambiguous "this is where the paragraph stops".
    return len(norm) >= _MIN_CLAIM_CHARS and dehyph(block_source or "").endswith(norm)


_ABBREV_TAILS = frozenset(
    [
        "fig",
        "figs",
        "eq",
        "eqs",
        "e.g",
        "i.e",
        "vs",
        "etc",
        "al",
        "dr",
        "mr",
        "mrs",
        "st",
        "no",
        "vol",
        "sec",
        "ch",
        "ref",
        "refs",
        "cf",
        "approx",
        "u.s",
    ]
)


def _looks_like_abbrev_tail(segment: str) -> bool:
    tail = segment.strip().rsplit(None, 1)[-1].lower().strip("\"'()[]{}")
    tail = tail.rstrip(".")
    return (
        tail in _ABBREV_TAILS
        or (len(tail) == 1 and tail.isalpha())  # "Section X." / "Model C."
        or (len(tail) == 3 and tail[1] == "." and tail[0].isalpha() and tail[2].isalpha())
    )


def split_sentences(text: str) -> list[str]:
    """Sentence split keeping delimiters; CJK and latin modes.

    CJK mode (any CJK char present) splits on ``。！？…；`` + newline, so
    latin abbreviations embedded in Chinese ("e.g.") never cut. Latin mode
    splits on ``.!?`` + whitespace with an abbreviation guard
    ("Fig.", "Eq.", "e.g.", single capitals).
    """

    if not (text or "").strip():
        return []
    if re.search(r"[぀-ヿ㐀-䶿一-鿿]", text):
        parts = re.split(r"(?<=[。！？…；\n])", text)
        return [p for p in parts if p.strip()]
    # Zero-width split (before the whitespace, not consuming it) so the pieces
    # re-join to the exact source. Consuming the separator deleted a space at
    # every sentence boundary, and the painted text is this join.
    raw = re.split(r"(?<=[.!?])(?=\s)", text)
    merged: list[str] = []
    for segment in raw:
        if merged and _looks_like_abbrev_tail(merged[-1]):
            # ``segment`` still carries its leading whitespace, so appending it
            # verbatim is exact; inserting another space would double it.
            merged[-1] = merged[-1] + segment
        else:
            merged.append(segment)
    return [s for s in merged if s.strip()]


def split_clauses(text: str) -> list[str]:
    """Clause split keeping delimiters; layout allocation granularity.

    Sentence snapping alone is too coarse: a single long sentence often
    straddles the zone break. Clauses preserve reading order just as well,
    and CJK wraps at any character while latin splits keep words intact
    (comma/semicolon/colon + space only — never mid-word).
    """

    if not (text or "").strip():
        return []
    if re.search(r"[぀-ヿ㐀-䶿一-鿿]", text):
        parts = re.split(r"(?<=[。！？…；，：、\n])", text)
        return [p for p in parts if p.strip()]
    clauses: list[str] = []
    for sentence in split_sentences(text):
        # Zero-width again: keep the separator's space in the following clause
        # so ``"".join(split_clauses(t)) == t`` and the painted text is intact.
        clauses.extend(re.split(r"(?<=[:,;])(?=\s)", sentence))
    return [c for c in clauses if c.strip()]


__all__ = [
    "_reclaimable",
    "_row_is_continuation",
    "_row_precision_hits",
    "split_clauses",
    "split_sentences",
]

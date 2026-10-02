"""The single classification rule set (single classification rule set, single layout vocabulary).

Extraction is a *prior*: whether a line is a listing, a page number, or math
debris is a property of the line, decided once here -- not re-guessed by a
second pass. The native reader (:mod:`ubt.analyze.reader_pdf`) and the plain-text
fallbacks (:mod:`ubt.adapters.pdf.plain_text_extractor`) call these, so a line
cannot be prose to one and debris to the other.

Only the *text-content* rules live here. Where a line sits (the margin band that
makes a bare number a page number) is geometry, and stays with the reader that
measured it.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
#: A bare page number (arabic or roman) and nothing else.
_BARE_PAGE_NUMBER = re.compile(r"^\s*(?:\d{1,4}|[ivxlcdm]{1,7})\s*$", re.IGNORECASE)


def is_bare_page_number(text: str) -> bool:
    """True for a bare page number -- arabic or roman, nothing else."""
    return bool(_BARE_PAGE_NUMBER.match(text or ""))


# --------------------------------------------------------------------------- #
# Listings
# --------------------------------------------------------------------------- #
#: Unambiguous program/algorithm syntax. Deliberately narrow: a false positive
#: silently stops a real paragraph from being translated, so only forms that do
#: not occur in running prose qualify. A listed line almost always carries an
#: assignment arrow or a definition/call keyword; weaker hints ("for ... (",
#: "obj.method(") were removed because prose ("for the number ... (No deadlock.)")
#: matched them and was silently kept untranslated.
_LISTING_FORMS = re.compile(
    r"←|⟵|↤|▷"  # assignment / dataflow / comment markers
    r"|\bawait\s+\w[\w.]*\s*\("  # await call(...)
    r"|\bdef\s+\w+\s*\(|\bclass\s+\w+\s*[:\(]|\bfunction\s+\w+\s*\("
    r"|\bfrom\s+[\w.]+\s+import\b"
    r"|\b(?:else|elif|repeat|until)\s+\d{1,3}\b"  # bare statement + line number
)
_LISTING_MAX_CHARS = 400


def looks_like_listing(text: str) -> bool:
    """True when text carries unambiguous program/algorithm syntax."""
    body = (text or "").strip()
    return bool(body) and len(body) <= _LISTING_MAX_CHARS and bool(_LISTING_FORMS.search(body))


# --------------------------------------------------------------------------- #
# Math / algorithm debris
# --------------------------------------------------------------------------- #
_MATH_SYMBOL_CHARS = frozenset("←⟵↤↦∘≔≃≅≤≥≠⊤⊥⊢⊣∈∉⊆⊂∪∩∀∃∧∨¬≡∑∏√∫∞∂∇⋯⋃⋂′″⟨⟩")
#: Algorithm/typing tokens: ``L-Iter``, ``pr 1``, ``id Γ``. These are names, not
#: prose; translating them is meaningless and painting over them corrupts the
#: listing, so they are preserved.
_ALG_TOKEN = re.compile(
    r"^(?:[A-Z][A-Za-z]*[-_][A-Z][a-z]+"
    r"|[a-z]{1,3}\s+\d{1,3}"
    r"|[a-z]{1,3}\s*[\U0001D400-\U0001D7FFΓΔΘΛΞΠΣΦΨΩ])$"
)
_DEBRIS_MAX_CHARS = 80


def _has_math(text: str) -> bool:
    return any(ch in _MATH_SYMBOL_CHARS or 0x1D400 <= ord(ch) <= 0x1D7FF for ch in text)


def looks_like_debris(text: str) -> bool:
    """True for a short math/algorithm token that is not prose.

    Axiom B preserves non-translatable content explicitly: these are names,
    operators and equation fragments extraction typed as prose. Translating them
    yields nonsense and repainting them corrupts the listing, so they are held
    byte-identical instead. A sentence fragment (terminal punctuation) is never
    debris -- it belongs to a paragraph and must be translated.
    """
    body = (text or "").strip()
    if not body or len(body) > _DEBRIS_MAX_CHARS or body.endswith((".", "!", "?", "。")):
        return False
    if _ALG_TOKEN.fullmatch(body):
        return True
    tokens = body.split()
    if len(tokens) > 10:
        return False
    wordy = sum(1 for w in tokens if w.isalpha() and len(w) >= 3)
    if wordy <= 1 and (body.endswith(("=", "→", "↦", "≔", "<", ">")) or _has_math(body)):
        return True
    return bool(wordy <= 2 and len(tokens) <= 6 and _has_math(body))


# --------------------------------------------------------------------------- #
# Markdown content rules
# --------------------------------------------------------------------------- #
# Shared by the native Markdown reader and the Markdown adapter, so a line
# cannot be a heading to one and prose to the other (single source of truth: one owner).
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_DELIMITER = re.compile(r"^\|?(\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?$")
_MD_LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$")


def markdown_heading(line: str) -> tuple[int, str] | None:
    """``(level, text)`` for an ATX heading line, or ``None``."""
    match = _MD_HEADING.match((line or "").strip())
    if match is None:
        return None
    return len(match.group(1)), match.group(2).strip()


def markdown_list_item(line: str) -> tuple[str, str] | None:
    """``(marker, text)`` for a list-item line, or ``None``."""
    match = _MD_LIST_ITEM.match(line or "")
    if match is None:
        return None
    return match.group(2), match.group(3).strip()


def is_markdown_table(text: str) -> bool:
    """True when text is a GitHub-flavoured Markdown table."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if not all("|" in line for line in lines):
        return False
    return bool(_TABLE_DELIMITER.match(lines[1]))


def is_display_math(text: str) -> bool:
    """True for a ``$$ … $$`` display-math block."""
    body = (text or "").strip()
    return body.startswith("$$") and body.endswith("$$") and len(body) >= 4


# --------------------------------------------------------------------------- #
# Plain-text structure rules
# --------------------------------------------------------------------------- #
#: Leading glyphs a plain-text extractor treats as a list item.
_LIST_PREFIXES = ("•", "-", "*", "·", "–")
#: A heading is short; longer lines are prose whatever their punctuation.
_HEADING_MAX_CHARS = 80


def is_list_prefix(text: str) -> bool:
    """True when a line opens with a list bullet."""
    return any((text or "").startswith(prefix) for prefix in _LIST_PREFIXES)


def looks_like_heading(text: str) -> bool:
    """True for a short, bullet-free line with no terminal punctuation.

    The *text-content* rule a plain-text extractor uses when it has no
    typography to measure (single source of truth: one owner). Deliberately narrow -- a title
    never ends in sentence punctuation and a long line is a paragraph -- so a
    miss merely leaves a heading as prose, which still translates.
    """
    body = (text or "").strip()
    if not body or len(body) >= _HEADING_MAX_CHARS:
        return False
    if body.endswith((".", "!", "?", "。")):
        return False
    return not is_list_prefix(body)


# --------------------------------------------------------------------------- #
# Project Gutenberg structural markers
# --------------------------------------------------------------------------- #
#: Transcribing these verbatim is the only correct rendering, and LLM
#: translation of them is wasted spend (they also trip script-density repair
#: loops). Shared by the Markdown reader and the Markdown adapter so the two
#: cannot disagree about which lines are structural markers.
_GUTENBERG_MARKER = re.compile(r"^\[(Illustration|Footnote)\b", re.IGNORECASE)


def is_gutenberg_marker(text: str) -> bool:
    """True for a Project Gutenberg ``[Illustration …]``/``[Footnote …]`` marker."""
    return bool(_GUTENBERG_MARKER.match((text or "").strip()))


__all__ = [
    "is_bare_page_number",
    "is_display_math",
    "is_gutenberg_marker",
    "is_list_prefix",
    "is_markdown_table",
    "looks_like_debris",
    "looks_like_heading",
    "looks_like_listing",
    "markdown_heading",
    "markdown_list_item",
]

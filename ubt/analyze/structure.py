"""The single classification rule set (single classification rule set, single layout vocabulary).

Extraction is a *prior*: whether a line is a listing, a page number, or math
debris is a property of the line, decided once here -- not re-guessed by a
second pass. The native reader (:mod:`ubt.adapters.pdf.reader_pdf`) and the plain-text
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
#: not occur in running prose qualify.
#:
#: Two forms here are *anchored to a call/keyword shape* rather than left loose,
#: because the loose spelling fired on prose and equations (measured on the
#: PaddleOCR layout paper, where two full paragraphs leaked untranslated):
#:
#: * ``class`` must be followed by a Capitalised identifier and its
#:   ``:``/``(`` — the loose ``\bclass\s+\w+\s*[:\(]`` matched the sentence
#:   "…focal loss … to address **class imbalance:**", a two-word prose phrase;
#: * ``else``/``elif``/``repeat``/``until`` must carry a trailing line number
#:   *and* stand at a statement boundary — a bare ``else 1`` matched the maths
#:   "if t = 1 **else 1** − σ(z)" inside a running paragraph.
_LISTING_FORMS = re.compile(
    r"←|⟵|↤|▷"  # assignment / dataflow / comment markers
    r"|\bawait\s+\w[\w.]*\s*\("  # await call(...)
    r"|\bdef\s+[A-Za-z_]\w*\s*\("  # def name(...) — a real function signature
    r"|\bclass\s+[A-Z]\w*\s*[:\(]"  # class Name: / class Name( — a real class
    r"|\bfunction\s+[A-Za-z_]\w*\s*\("
    r"|\bfrom\s+[\w.]+\s+import\b"
    # A statement keyword with a line number, at a line start or after a
    # separator — never mid-sentence prose/equation.
    r"|(?:^|[\n;{])\s*(?:else|elif|repeat|until)\s+\d{1,3}\b"
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


#: Ordered markers a PDF line can open with: ``1.``, ``1.1.``, ``1)``, ``(1)``,
#: ``a.``, ``(a)``, ``一、``, and their full-width forms (``（1）``, ``１.``). A
#: trailing space is required so ``1.5`` stays a decimal, and dotted section
#: numbers (a TOC title) match whole.
_PDF_ORDERED_MARKER_RE = re.compile(
    r"^\s*([（(]?[0-9０-９]{1,3}(?:[.．][0-9０-９]{1,3})*[.．、)）]"
    # A bracketed citation number: "[12] Author, Title." — the shape every
    # academic reference list uses. Without it the number is dropped at
    # extraction and the delivered bibliography loses its numbering (measured on
    # arXiv 2609.20519: the source carries 108 markers, the ledger kept none).
    r"|\[[0-9０-９]{1,3}\]"
    r"|[（(]?[a-zA-ZＡ-Ｚａ-ｚ][.．)）]"
    r"|[一二三四五六七八九十百]+[、.．)）])\s+"
)
#: Unambiguous bullet glyphs -- a marker whether or not a space follows.
_PDF_BULLETS = ("•", "◦", "‣", "▪", "●")
#: Ambiguous dash/star forms -- a marker only before whitespace, so ``-3`` and
#: ``*ptr`` stay prose.
_PDF_DASH_BULLETS = ("-", "*", "·", "–")


def pdf_list_marker(text: str) -> tuple[str, str] | None:
    """``(marker, text)`` for a PDF list-item line, or ``None``.

    The native PDF reader and the renderer share this one rule, so a line
    the reader types as a list item is one the marker restorer recognises (and
    vice versa). Ordered markers keep their number/parenthesis and are stripped
    from the returned text; the reader stores that marker on the element.
    """
    stripped = (text or "").lstrip()
    if not stripped:
        return None
    if stripped[0] in _PDF_BULLETS:
        return stripped[0], stripped[1:].strip()
    if stripped[0] in _PDF_DASH_BULLETS:
        if len(stripped) > 1 and stripped[1].isspace():
            return stripped[0], stripped[1:].strip()
        return None
    match = _PDF_ORDERED_MARKER_RE.match(text)
    if match is not None:
        return match.group(1), (text or "")[match.end() :].strip()
    return None


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

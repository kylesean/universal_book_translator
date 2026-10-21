"""The text a block delivers, and the wrappers that decide it — one owner.

The PDF renderer and the companion HTML/EPUB views must agree about what a
block's delivered text is. They did not: ``document_renderer._block_text``
unwrapped the ``<mark>`` wrappers the repair stage writes and fell back to
``draft_text``, while ``html_view.element_text`` took ``target_text`` verbatim.
A failed repair then shipped the marker and its ``title`` (which carries the
provider's exception text) into the companion HTML as literal escaped markup,
and a quarantined block showed the source in HTML while the PDF showed the
Chinese draft.

This module is that rule, extracted so there is exactly one implementation. Both
renderers call :func:`delivered_text`; the mark predicates are public here
because the HTML view also needs to know whether a block is a quarantine marker.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ubt.core.ir.models import IRBlock

#: Quarantine wrapper: the payload is the SOURCE under a marker, written when a
#: crash/resume could not place the translation. It must never reach a page.
_BLOCKED_HUMAN_CLASS = "ubt-blocked-human"

#: Failed-draft wrapper: a REAL translation whose repair call failed.
#: The wrapped text is what the reader should get — only the repair failed.
#: Collapsing this with the quarantine wrapper discarded the translation and fell
#: back to ``draft_text``, which is the pre-glossary-enforcement draft.
_FAILED_DRAFT_CLASS = "ubt-failed-draft"

_MARK_RE = re.compile(r"^\s*<mark\b([^>]*)>", re.IGNORECASE)
_CLASS_ATTR_RE = re.compile(r'class\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)


def mark_class(text: str) -> str | None:
    """The ``class`` of a leading ``<mark>`` wrapper, or ``None`` if unwrapped."""
    match = _MARK_RE.match(text)
    if match is None:
        return None
    class_match = _CLASS_ATTR_RE.search(match.group(1))
    return class_match.group(1) if class_match else ""


def unwrap_mark(text: str) -> str:
    """The content inside a leading ``<mark …>`` … ``</mark>`` wrapper.

    Only the pipeline's own wrappers are unwrapped; anything else (a stray
    literal ``</mark>`` in prose that happened to start with ``<mark``) is
    returned unchanged rather than truncated.
    """
    match = _MARK_RE.match(text)
    if match is None:
        return text
    body = text[match.end() :]
    closing = body.rfind("</mark>")
    if closing == -1:
        return text
    return body[:closing]


def is_any_mark(text: str) -> bool:
    """Whether the text starts with any ``<mark>`` wrapper at all.

    A last-resort guard: a wrapper the pipeline does not recognise must still not
    reach the page as literal markup.
    """
    return _MARK_RE.match(text) is not None


def is_placeholder(text: str) -> bool:
    """Whether this text is a quarantine marker rather than block content.

    Deliberately narrow: ONLY ``ubt-blocked-human`` is a placeholder. A
    ``ubt-failed-draft`` wrapper carries the draft the reader should see, and
    treating it as a placeholder is what silently reverted the delivered text to
    the un-enforced draft.
    """
    return mark_class(text) == _BLOCKED_HUMAN_CLASS


def is_failed_draft(text: str) -> bool:
    """Whether this is a REAL draft whose repair call failed (not a quarantine)."""
    return mark_class(text) == _FAILED_DRAFT_CLASS


def delivered_text(block: IRBlock) -> str:
    """The text to render for a block: its translation, else its draft, else source.

    Two wrappers can arrive in ``target_text`` and they mean opposite things:

    - ``ubt-blocked-human`` (quarantine): the SOURCE under a marker. Rendering it
      would put the quarantine markup on the page, so fall back to the draft —
      real translated prose that a false-positive gate rejected.
    - ``ubt-failed-draft``: a real draft whose *repair call* failed (provider
      5xx). The wrapped text IS the translation and must be rendered with the
      wrapper stripped.
    - Any other wrapper is stripped too: pipeline markup must never reach the
      page as literal text, whatever class an older or newer build wrote.

    A block with no target and no draft is *not* silently dropped — the reader
    gets the source sentence.
    """
    target = (block.target_text or "").strip()
    if target and not is_placeholder(target):
        if is_failed_draft(target) or is_any_mark(target):
            return unwrap_mark(target).strip()
        return target
    draft = (block.draft_text or "").strip()
    if draft and not is_placeholder(draft):
        if is_failed_draft(draft) or is_any_mark(draft):
            return unwrap_mark(draft).strip()
        return draft
    return (block.source_text or "").strip()


#: A heading that is really running chrome (an arXiv stamp, a journal line, a
#: submission date). Kept out of the body so the document opens on its title.
_CHROME_HEADING_RE = re.compile(
    r"^(?:arXiv:|Journal of Machine Learning|Submitted \d|Preprint\b|"
    r"Under review\b|Published \d|Revised \d)",
    re.IGNORECASE,
)

#: A running head of author names in caps ("FEDUS, ZOPH AND SHAZEER") — the
#: JMLR/arXiv convention of repeating the paper's authors on every page. Matched
#: case-**sensitively**, and only over a *comma-separated* list of 2-4 short
#: all-caps words (optionally closed by "AND <NAME>"), so a genuine section
#: heading can never match — not a one-word all-caps heading ("EXPERIMENTS"),
#: not a title-case one, and not a sentence. The translated spelling
#: ("Fedus、Zoph与Shazeer") is caught through the source text in
#: :func:`dropped_page_furniture`.
_CHROME_AUTHOR_HEAD_RE = re.compile(
    r"^[A-Z][A-Z'’\-]{1,19}(?:, [A-Z][A-Z'’\-]{1,19})+"
    r"(?:,? AND [A-Z][A-Z'’\-]{1,19})?$"
)

#: The same running head as the reader lifts it from page 2 onward, where JMLR
#: sets it in *title case* ("Fedus, Zoph and Shazeer" → 译文 "Fedus、Zoph 和
#: Shazeer"): capitalized surname(s), comma-separated, closed by "and". The
#: connector accepts the translated forms (和/与/、) so the delivered text
#: matches too. Paired with a small font (see ``dropped_page_furniture``) — a
#: genuine heading is never set smaller than the body — this cannot catch a
#: real section heading ("Ross and White", a section ON someone, is larger and
#: rarely stands alone as a heading).
_CHROME_AUTHOR_HEAD_TITLE_RE = re.compile(
    r"^[A-Z][a-z'’\-]{1,19}(?:[、,] ?[A-Z][a-z'’\-]{1,19})+"
    r"(?: ?(?:and|和|与|、) ?[A-Z][a-z'’\-]{1,19})?$"
)


def dropped_page_furniture(block: IRBlock) -> bool:
    """Whether the renderer deliberately omits this block from the artifact.

    The single owner of that decision: a heading the renderer recognizes as
    running chrome (an arXiv/journal stamp, a submission date) is dropped so the
    document opens on its title (see ``build_source``). The delivered-artifact
    check must ask *this* function rather than re-test the pattern, or an audit
    would demand a realization the renderer is right not to place — the shape
    that reported a false ``[FAIL] artifact carries 280/281`` on a page-furniture
    heading.

    Kept narrow on purpose: only the chrome-heading drop counts. A block merely
    *kept in the source* (a bibliography entry, an affiliation) is still placed
    verbatim and must still appear, so it is not furniture.
    """
    if str(getattr(block.block_type, "value", block.block_type) or "") != "heading":
        return False
    # A heading block whose *region* is the page-furniture band came from the
    # layout model's running head, not from the text flow. Typesetting it
    # produces a chapter-level heading in the middle of a chapter, in the
    # source language, once per page the head spans.
    #
    # Only when the block was left in the source language. ``--translate-chrome``
    # exists so an operator can ask for the strip to be *translated*, and ingest
    # clears ``skip_translate`` when they do; dropping the block then would
    # silently discard a translation the run paid for, which is a worse defect
    # than the untranslated band this rule removes.
    region = str(getattr(block, "region", None) or "")
    if region in ("header", "footer", "page_number") and block.skip_translate:
        return True
    if _CHROME_HEADING_RE.match(delivered_text(block)):
        return True
    # The chrome is matched on the SOURCE text too: a translated heading no
    # longer spells like the stamp it came from — "FEDUS, ZOPH AND SHAZEER"
    # arrived as "Fedus、Zoph与Shazeer", which the delivered-text pattern never
    # caught, so the running header survived and opened the body on the paper's
    # author line. The source text is the stable fingerprint of what the block
    # *was*; the all-caps author-head shape (case-sensitive) keeps this narrow.
    source = (block.source_text or "").strip()
    if _CHROME_HEADING_RE.match(source) or _CHROME_AUTHOR_HEAD_RE.match(source):
        return True
    # The title-case running head ("Fedus, Zoph and Shazeer", page 2 onward) is
    # distinguished from a genuine section heading by its *size*: JMLR sets it
    # below the body size, and no real heading is set below the body. The size
    # lives on the element (0/None = unknown → keep, fail closed to placing).
    delivered = delivered_text(block)
    for text in (delivered, source):
        if not _CHROME_AUTHOR_HEAD_TITLE_RE.match(text):
            continue
        size = getattr(block.element, "font_size", None)
        if not size or size < 11.0:
            return True
    return False


def is_toc_entry(block: IRBlock) -> bool:
    """Whether the extractor tagged this block as a table-of-contents row.

    The PDF reader pairs a TOC title with its right-margin page number and
    records ``provenance.toc_entry`` (+ ``toc_page``) so a table of contents is
    not scattered into translated titles plus a run of bare page-number blocks.
    The renderer must read that flag: left unread, every TOC row renders as an
    ordinary ``#heading`` — inventing section headings that have no body (an
    orphaned, near-empty page) and dropping the page numbers the reader worked
    to recover.
    """
    return bool(block.provenance.toc_entry)


__all__ = [
    "delivered_text",
    "dropped_page_furniture",
    "is_toc_entry",
    "is_any_mark",
    "is_failed_draft",
    "is_placeholder",
    "mark_class",
    "unwrap_mark",
]

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

from ubt.render.table_compiler import normalize_table_to_source

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


#: A masker placeholder that survived its restore pass: the five token
#: families :mod:`ubt.segment.placeholders` issues (email / code / math / soup /
#: citation). Shape mirrors ``artifact_guard._SENTINEL_PREFIXES`` — one regex
#: rather than a per-family list, so a masker rename fails the parity test
#: instead of leaking silently.
#:
#: Not ``ubt.render.typst.artifact_guard.has_sentinel`` itself: that is a render
#: -layer predicate and ``ubt.render.typst`` imports *this* module, so importing
#: it back would be a cycle. The emphasis families (``⟦B⟧``, ``⟦I⟧``,
#: ``⟦SEMANTIC_BREAK⟧``) are deliberately excluded here — they carry meaning the
#: renderer still needs (``parse_emphasis`` turns them into bold runs, and the
#: bifurcation token splits a block), so blanking them would destroy content
#: rather than hide machinery. Only a masker token is proven machinery: its
#: checksum was the restore's own integrity test, and a refusal to substitute it
#: means the protected span is *not available*, not that it is unimportant.
#:
#: The delimiter alternatives include the LaTeX bracket macros. A model rewrote
#: ``⟦SOUP_MASK_0001-2a3⟧`` as ``$\\llbracket SOUP\\_MASK\\_0001-2a3
#: \\rrbracket$`` in 26-page replay block ``b0058``: the token itself is
#: unchanged, but ``[⟦\[【]`` never matched ``\\llbracket``, so the stripper
#: returned the text untouched and Typst rendered the macros back into the
#: ⟦ ⟧ glyphs the artifact contract (rightly) calls a ``placeholder_leak``. The
#: optional surrounding ``$`` is swallowed with the token because the
#: replacement is prose, not math — leaving the dollars behind would make Typst
#: parse the placeholder as a math variable.
_SENTINEL_CORE = (
    r"(?:[A-Za-z]+(?:\\?_)+)?MASK(?:\\?_)+\d{1,6}"
    r"(?:-?(?:\\?_)?[0-9A-Za-z\U0001d400-\U0001d7ff]{1,6})?"
)
_BARE_SENTINEL_CORE = (
    r"(?:SOUP|MATH|CODE|CITE|EMAIL)(?:\\?_)+MASK(?:\\?_)+\d{1,6}"
    r"(?:-?(?:\\?_)?[0-9A-Za-z\U0001d400-\U0001d7ff]{1,6})?"
)
_SENTINEL_LEFT = (
    r"(?:\$?\s*(?:[⟦\[【]|\\llbracket)"
    r"|\$(?:\\(?:text|mathrm|mathbf)\s*\{)?"
    r"|\\(?:text|mathrm|mathbf)\s*\{"
    r"|\\\()"
)
_SENTINEL_RIGHT = r"(?:[⟧】\]]|\\rrbracket|(?:\}\s*\$|\$|\}|\\\)))\s*\$?"
_PIPELINE_SENTINEL_RE = re.compile(
    rf"(?:{_SENTINEL_LEFT}\s*{_SENTINEL_CORE}[^\n⟧】\]$]{{0,40}}?\s*{_SENTINEL_RIGHT}|(?<![A-Za-z0-9]){_BARE_SENTINEL_CORE}(?![A-Za-z0-9\U0001d400-\U0001d7ff]))",
    re.IGNORECASE,
)

#: What a reader gets in place of an unrestorable span: a neutral bracketed
#: marker, the same convention :func:`ubt.render.typst.math_transpiler.clean_math_fallback_text`
#: uses for an uncompilable formula (``⟨math⟩``). Deliberately not a sentence in
#: any one language — the pipeline is multilingual and this ships in the target.
SENTINEL_PLACEHOLDER = "⟨missing⟩"


def strip_pipeline_sentinels(text: str) -> str:
    """Replace any leaked masker token with :data:`SENTINEL_PLACEHOLDER`.

    The restore pass verifies a masked token's checksum before substituting the
    span it protects. When the checksum does not match it refuses — correctly,
    because restoring anyway could put the *wrong* formula in the sentence — and
    leaves the token in the draft. Nothing downstream removed it, so
    ``⟦SOUP_MASK_0001-f2d⟧`` was measured in the delivered PDF's text layer and
    in its HTML, EPUB and XLIFF companions: internal machinery printed in the
    book the reader paid for.

    The token is not content the reader is owed (its span was never
    recoverable), but the *sentence* around it is, so the token is replaced
    rather than the block dropped — and the block keeps its ``sentinel_leaked``
    and ``soup_token_corrupt`` flags, so the loss stays attributable and lands in
    the post-edit queue.
    """
    if not text or not _PIPELINE_SENTINEL_RE.search(text):
        return text
    return _PIPELINE_SENTINEL_RE.sub(SENTINEL_PLACEHOLDER, text)


#: A stray HTML formatting tag in delivered prose. A model that saw a converter's
#: link spans in the source echoed the wrapper's closing tag into its
#: translation: the 18-page ATC'25 replay shipped ``RustyHermit-MPK [59]</span>``
#: to the page nine times across two blocks. The wrapper never existed in this
#: pipeline's markup (the render layer is Typst, not HTML), so the tag is pure
#: machinery: no content is owed for it, and unlike a masker token there is
#: nothing to restore, so the tag is deleted outright rather than replaced with
#: a marker.
#:
#: The list holds the multi-letter formatting tags a converter or a model
#: plausibly echoes. Single-letter tags (``a``/``b``/``i``/``p``/``u``) are
#: deliberately absent: ``x<a>y`` in a math-adjacent sentence is plausible prose
#: and must survive, while ``<span>``/``<div>``/``<sub>`` sequences never carry
#: meaning on their own.
_STRAY_HTML_TAG_RE = re.compile(
    r"</?(?:span|div|br|em|strong|sub|sup|small|code|font|mark)\b[^<>]*>",
    re.IGNORECASE,
)
_STRAY_HTML_TAG_NO_BR_RE = re.compile(
    r"</?(?:span|div|em|strong|sub|sup|small|code|font|mark)\b[^<>]*>",
    re.IGNORECASE,
)

#: Public alias: the delivered-artifact contract detects the same family in the
#: rendered PDF, so both sides must match on one definition rather than drift.
STRAY_HTML_TAG_RE = _STRAY_HTML_TAG_RE


def strip_stray_html_tags(text: str, *, preserve_breaks: bool = False) -> str:
    """Delete stray HTML formatting tags from delivered prose.

    See :data:`_STRAY_HTML_TAG_RE` for the failure mode. Applied after the
    sentinel strip: the sentinel rule replaces its token with a marker, and a
    deletion running first inside a sentinel could corrupt that replacement.
    When ``preserve_breaks=True`` (e.g. inside table blocks), ``<br>`` tags are
    preserved because markdown and Docling tables rely on ``<br>`` for in-cell
    line breaks.
    """
    if not text or "<" not in text:
        return text
    pattern = _STRAY_HTML_TAG_NO_BR_RE if preserve_breaks else _STRAY_HTML_TAG_RE
    return pattern.sub("", text)


#: A list marker the extractor degraded. The source PDF's filled bullet arrives
#: as a middle dot — the ATC'25 replay's list items extracted as
#: ``· Redis. We use ...`` — and a model copies the marker verbatim into its
#: translation, often dropping the space after it (``·Redis。``). The reader
#: gets a shrunken dot glued to the first word where the source had a bullet.
#: The marker is furniture, not content: for list-shaped blocks it is
#: normalized to a real bullet plus a separating space. Prose in other block
#: types keeps its middle dots — a ``·`` between clauses is a legitimate
#: typographic character.
_LIST_ITEM_TYPE = "list_item"
_LIST_MARKER_RE = re.compile(r"^\s*[·•・]\s*")


def normalize_list_marker(block_type: object, text: str) -> str:
    """Retype a degraded list marker as a proper bullet (list blocks only)."""
    if str(getattr(block_type, "value", block_type) or "") != _LIST_ITEM_TYPE:
        return text
    if not text:
        return text
    return _LIST_MARKER_RE.sub("• ", text, count=1)


def delivered_text(block: IRBlock, *, sanitize_sentinels: bool = True) -> str:
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

    ``sanitize_sentinels=False`` returns the surface *as the pipeline wrote it*,
    with any leaked masker token and stray markup tag intact. Render-layer
    diagnostics must use it: the default substitution is what stops the token
    reaching the page, so a check built on the sanitized text would stop
    reporting a leak at exactly the moment the leak happened.
    """
    raw = _delivered_surface(block)
    if not sanitize_sentinels:
        return raw
    is_table = str(getattr(block.block_type, "value", block.block_type) or "") == "table"
    text = normalize_list_marker(
        block.block_type,
        strip_stray_html_tags(strip_pipeline_sentinels(raw), preserve_breaks=is_table),
    )
    if is_table:
        # A model re-chunking one source cell into several target cells ships a
        # ragged grid the renderer lays out broken; the source's own row widths
        # are the contract, so the delivered surface is re-shaped to them
        # (content merged, never dropped) before any consumer sees it.
        text = normalize_table_to_source(block.source_text, text)
    return text


def _delivered_surface(block: IRBlock) -> str:
    """The delivered text before sentinel substitution — see :func:`delivered_text`."""
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

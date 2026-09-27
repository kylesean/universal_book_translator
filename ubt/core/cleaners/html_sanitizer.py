"""Allowlist-based HTML sanitizer for LLM-produced content.

LLM translations are untrusted input: a prompt-injected or hallucinated
response can carry ``<script>``, inline event handlers or ``javascript:``
URLs that would end up stored verbatim in EPUB/Markdown artifacts (stored
XSS). The sanitizer strips dangerous constructs while preserving safe inline
formatting so legitimate bilingual markup (``<em>``, ``<strong>``, tables,
...) survives export.

Design notes:
- Text nodes are HTML-escaped (``&``, ``<``, ``>``). ``HTMLParser`` decodes
  character references before ``handle_data`` sees them, so emitting text
  verbatim let ``&lt;script&gt;`` (or double-encoded ``&amp;lt;script&amp;gt;``)
  reconstitute a real ``<script>`` element downstream — stored XSS. Escaping
  keeps the text inert while rendering identically in any HTML/CommonMark
  consumer.
- Disallowed-but-benign tags (``<font>``, ``<div>``, ...) are dropped while
  their inner text is kept; dangerous containers (``<script>``,
  ``<iframe>``, ...) are dropped together with their content. (``<svg>``
  and its shape children are allowlisted for diagram fidelity — the
  attribute allowlist already strips event handlers, so they are safe.)
- The parser fails closed: on any parsing error all tags are stripped and the
  remaining text is escaped.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from html.parser import HTMLParser

# Inline formatting / structure that is meaningful in book translations.
ALLOWED_TAGS = frozenset(
    {
        "a",
        "abbr",
        "annotation",
        "b",
        "blockquote",
        "br",
        "circle",
        "cite",
        "code",
        "del",
        "ellipse",
        "em",
        "figcaption",
        "figure",
        "g",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "i",
        "img",
        "ins",
        "kbd",
        "li",
        "line",
        "mark",
        "math",
        "mfrac",
        "mi",
        "mn",
        "mo",
        "mroot",
        "mrow",
        "msqrt",
        "msub",
        "msubsup",
        "msup",
        "mtable",
        "mtd",
        "mtext",
        "mtr",
        "ol",
        "p",
        "path",
        "polygon",
        "polyline",
        "pre",
        "q",
        "rect",
        "s",
        "semantics",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "svg",
        "table",
        "tbody",
        "td",
        "text",
        "tfoot",
        "th",
        "thead",
        "tr",
        "tspan",
        "u",
        "ul",
    }
)

VOID_TAGS = frozenset({"br", "hr", "img"})

# Containers whose entire content must vanish (they can execute code or hide
# payloads, and none of them carries translatable book text).
DROP_WITH_CONTENT = frozenset(
    {
        "script",
        "style",
        "iframe",
        "object",
        "embed",
        "applet",
        "template",
        "noscript",
    }
)

# Allowed attributes per tag; everything else renders bare.
ALLOWED_ATTRS: dict[str, frozenset[str]] = {
    "a": frozenset({"href", "title", "id", "class"}),
    "img": frozenset({"src", "alt", "title", "width", "height", "class", "id", "loading"}),
    "svg": frozenset({"viewbox", "width", "height", "xmlns", "class", "id", "fill", "stroke"}),
    "g": frozenset({"id", "class", "transform", "fill", "stroke", "stroke-width"}),
    "path": frozenset({"d", "fill", "stroke", "stroke-width", "class", "id"}),
    "circle": frozenset({"cx", "cy", "r", "fill", "stroke", "stroke-width", "class", "id"}),
    "rect": frozenset(
        {"x", "y", "width", "height", "rx", "ry", "fill", "stroke", "stroke-width", "class", "id"}
    ),
    "line": frozenset({"x1", "y1", "x2", "y2", "stroke", "stroke-width", "class", "id"}),
    "ellipse": frozenset({"cx", "cy", "rx", "ry", "fill", "stroke", "stroke-width", "class", "id"}),
    "polygon": frozenset({"points", "fill", "stroke", "stroke-width", "class", "id", "transform"}),
    "polyline": frozenset({"points", "fill", "stroke", "stroke-width", "class", "id", "transform"}),
    "text": frozenset(
        {
            "x",
            "y",
            "dx",
            "dy",
            "text-anchor",
            "font-size",
            "font-family",
            "font-weight",
            "fill",
            "stroke",
            "class",
            "id",
            "transform",
        }
    ),
    "tspan": frozenset(
        {
            "x",
            "y",
            "dx",
            "dy",
            "text-anchor",
            "font-size",
            "font-family",
            "font-weight",
            "fill",
            "stroke",
            "class",
            "id",
        }
    ),
    "table": frozenset({"border", "cellspacing", "cellpadding", "width", "class", "id", "align"}),
    "thead": frozenset({"class", "id", "align", "valign"}),
    "tbody": frozenset({"class", "id", "align", "valign"}),
    "tfoot": frozenset({"class", "id", "align", "valign"}),
    "tr": frozenset({"class", "id", "align", "valign"}),
    "th": frozenset(
        {"colspan", "rowspan", "scope", "headers", "width", "align", "valign", "class", "id"}
    ),
    "td": frozenset({"colspan", "rowspan", "headers", "width", "align", "valign", "class", "id"}),
    "math": frozenset({"xmlns", "display", "class", "id"}),
    "span": frozenset({"class", "id"}),
    "mark": frozenset({"class", "id", "title"}),
    "p": frozenset({"class", "id"}),
    "code": frozenset({"class", "id"}),
    "pre": frozenset({"class", "id"}),
    "blockquote": frozenset({"class", "id"}),
    "ol": frozenset({"start", "type", "class", "id"}),
    "li": frozenset({"value", "class", "id"}),
    "h1": frozenset({"class", "id"}),
    "h2": frozenset({"class", "id"}),
    "h3": frozenset({"class", "id"}),
    "h4": frozenset({"class", "id"}),
    "h5": frozenset({"class", "id"}),
    "h6": frozenset({"class", "id"}),
}

SAFE_URL_SCHEMES = frozenset({"http", "https", "mailto", "data"})
_SCHEME_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*):")
# Browsers delete control characters and spaces from a URL before resolving its
# scheme, so `java<TAB>script:alert(1)` reaches the parser as `javascript:`.
# Matching the raw value would let such a URL fall through the allowlist
# unrecognised and be treated as a relative link.
_URL_CONTROL_RE = re.compile(r"[\x00-\x20\x7f]")


def _escape_attr(value: str | None) -> str:
    return (
        (value or "")
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _escape_text(value: str) -> str:
    """Escape a decoded text node so it can never reconstitute markup."""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _url_scheme_allowed(name: str, value: str) -> bool:
    """Whether an ``href``/``src`` value carries a permitted scheme."""
    probe = _URL_CONTROL_RE.sub("", value)
    match = _SCHEME_RE.match(probe)
    if match:
        scheme = match.group(1).lower()
        if scheme not in SAFE_URL_SCHEMES:
            return False
        # data: is legitimate for embedded image bytes, never for navigation.
        return not (scheme == "data" and name == "href")
    # No scheme-shaped prefix. A colon that appears before the first path
    # separator still introduces a scheme the browser may recover (an unknown
    # one is handed to the external protocol handler), so deny; after a `/` it
    # is ordinary path text (`img/a.png`, `xhtml/nav#note:2`) and is allowed.
    colon = probe.find(":")
    if colon == -1:
        return True
    slash = probe.find("/")
    return slash != -1 and slash < colon


def _srcset_allowed(value: str) -> bool:
    """Every URL candidate in a ``srcset`` must pass the scheme gate.

    ``srcset`` is ``url [descriptor], url [descriptor], …``; the legacy
    single-scheme check only looked at the first token, so a later
    ``javascript:``/``data:`` candidate shipped untouched.
    """
    for candidate in value.split(","):
        stripped = candidate.strip()
        if not stripped:
            continue
        url = stripped.split()[0]
        if not _url_scheme_allowed("src", url):
            return False
    return True


def _style_value_allowed(value: str) -> bool:
    """Inline ``style`` may stay only when it loads no disallowed URL and runs no code."""
    if _CSS_EXPRESSION_RE.search(value):
        return False
    for match in _CSS_URL_RE.finditer(value):
        url = match.group(2).strip()
        if url and not _url_scheme_allowed("src", url):
            return False
    return True


def _url_is_remote(value: str) -> bool:
    """True for an absolute/protocol-relative URL (anything that leaves the package)."""
    probe = _URL_CONTROL_RE.sub("", value).strip()
    return probe.startswith("//") or bool(_SCHEME_RE.match(probe))


def scrub_css_text(css: str) -> str:
    """Drop remote/unsafe ``@import`` and neutralise ``url()``/``expression()``.

    A kept stylesheet cannot execute in an EPUB reading system, but a *remote*
    ``@import`` fetches from the network (tracking, and CSS attribute-selector
    exfiltration), so remote targets go while relative in-package ones stay.

    Public because a standalone ``.css`` package member is copied into the
    deliverable and needs exactly the same gate as an inline ``<style>`` body;
    only this file may hold the URL policy.
    """
    lowered = css.lower()
    if "@import" not in lowered and "url(" not in lowered and "expression" not in lowered:
        return css

    def _import(match: re.Match[str]) -> str:
        target = (match.group(2) or match.group(4) or "").strip()
        if not target:
            return ""
        if not _url_scheme_allowed("src", target) or _url_is_remote(target):
            return ""
        return match.group(0)

    css = _CSS_IMPORT_RE.sub(_import, css)
    css = _CSS_EXPRESSION_RE.sub("/*expression*/(", css)

    def _gated(match: re.Match[str]) -> str:
        url = match.group(2).strip()
        if not url or _url_scheme_allowed("src", url):
            return match.group(0)
        return 'url("")'

    return _CSS_URL_RE.sub(_gated, css)


def _is_refresh_meta(attrs: list[tuple[str, str | None]]) -> bool:
    for name, value in attrs:
        if name.lower() == "http-equiv" and value:
            return re.sub(r"\s+", "", value).lower() == "refresh"
    return False


def _link_loads_resource(attrs: list[tuple[str, str | None]]) -> bool:
    """True for a ``<link>`` with an unsafe href, or a *remote* resource link.

    A relative ``<link rel="stylesheet" href="book.css">`` stays (the EPUB
    renderer itself emits one); only network-fetching targets go.
    """
    rels: set[str] = set()
    href: str = ""
    for name, value in attrs:
        if not value:
            continue
        lowered = name.lower()
        if lowered == "rel":
            rels |= set(value.lower().split())
        elif lowered == "href":
            href = value
    if href and not _url_scheme_allowed("href", href):
        return True
    return bool(rels & _LINK_RESOURCE_RELS and _url_is_remote(href))


def _attr_allowed(tag: str, name: str, value: str | None) -> bool:
    allowed = ALLOWED_ATTRS.get(tag)
    if not allowed or name not in allowed:
        return False
    if name in ("href", "src") and value:
        return _url_scheme_allowed(name, value)
    return True


class _SanitizingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._drop_depth = 0
        self._open_stack: list[str] = []

    # -- helpers ---------------------------------------------------------

    def _emit_start(
        self, tag: str, attrs: list[tuple[str, str | None]], self_closing: bool
    ) -> None:
        filtered = [(k, v) for k, v in attrs if _attr_allowed(tag, k.lower(), v)]
        attr_str = "".join(
            f' {k}="{_escape_attr(v)}"' if v is not None else f" {k}" for k, v in filtered
        )
        if tag in VOID_TAGS:
            self._out.append(f"<{tag}{attr_str}>")
            return
        if self_closing:
            self._out.append(f"<{tag}{attr_str}></{tag}>")
        else:
            self._out.append(f"<{tag}{attr_str}>")
            self._open_stack.append(tag)

    def _auto_close(self, tag: str) -> None:
        while self._open_stack:
            open_tag = self._open_stack.pop()
            self._out.append(f"</{open_tag}>")
            if open_tag == tag:
                break

    # -- HTMLParser callbacks --------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._drop_depth:
            if tag in DROP_WITH_CONTENT:
                self._drop_depth += 1
            return
        if tag in DROP_WITH_CONTENT:
            self._drop_depth = 1
            return
        if tag in ALLOWED_TAGS:
            self._emit_start(tag, attrs, self_closing=False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._drop_depth:
            return
        if tag in ALLOWED_TAGS:
            self._emit_start(tag, attrs, self_closing=tag not in VOID_TAGS)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._drop_depth:
            if tag in DROP_WITH_CONTENT:
                self._drop_depth -= 1
            return
        if tag in ALLOWED_TAGS and tag not in VOID_TAGS and tag in self._open_stack:
            self._auto_close(tag)

    def handle_data(self, data: str) -> None:
        if not self._drop_depth:
            # ``convert_charrefs=True`` has already decoded ``&lt;``/``&amp;``
            # into real characters; re-escape so text can never become markup.
            self._out.append(_escape_text(data))

    def handle_comment(self, data: str) -> None:
        """HTML comments are dropped (payload smuggling vector)."""

    def handle_decl(self, decl: str) -> None:
        """Doctype/decls are dropped."""

    def handle_pi(self, data: str) -> None:
        """Processing instructions are dropped."""

    def unknown_decl(self, data: str) -> None:
        """Unknown declarations (CDATA etc.) are dropped."""

    # -- finalization ------------------------------------------------------

    def close(self) -> None:
        super().close()
        # Close anything left open so the fragment stays well-formed.
        while self._open_stack:
            self._out.append(f"</{self._open_stack.pop()}>")

    def result(self) -> str:
        return "".join(self._out)


_TAG_STRIPPER = re.compile(r"<[^>]*>")

# A tag-shaped run: ``<name``, ``</name``, with the name captured separately.
# Attribute/body text after the name is matched loosely (no ``>``) so the
# decision to keep or neutralise is made from the name alone; ``HTMLParser``
# still parses attributes afterwards.
_TAG_NAME_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)", re.IGNORECASE)

# A well-formed tag: name plus only ``attr`` / ``attr="value"`` tokens.
# Rejects prose that merely starts with ``<`` followed by an allowlisted letter
# (e.g. ``a<b 且 b>c`` looks tag-like to HTMLParser but is not a tag).
#
# Possessive quantifiers are load-bearing: without them the attribute group
# partitioned a long name run in exponentially many ways and spent seconds on
# ``<b aaa...!>`` (LLM output is untrusted). The accepted set is unchanged for
# real tags (verified against a curated corpus incl. quoted/unquoted/empty-value
# attributes and prose bodies) while a failing body now fails in linear time.
_TAG_STRUCT_RE = re.compile(
    r"<(/?)([a-zA-Z][a-zA-Z0-9]*)"
    r"\s*"
    r"((?:[a-zA-Z_:][-a-zA-Z0-9_:.]*+"
    r"(?:\s*+=\s*+(?:\"[^\"]*\"|'[^']*'|[^\s\"'>]+))?\s*+)*)"
    r"(/?)>",
    re.IGNORECASE,
)


def _scan_tag_end(text: str, start: int) -> int:
    """Index of the ``>`` that closes the tag beginning at ``start`` (quote-aware).

    Returns ``-1`` if the tag is never closed. Attribute values may legally
    contain ``>`` (``href="data:text/html,<b>x</b>"``), so a bare ``find`` would
    stop the tag in the middle of a quoted value.
    """
    i = start + 1
    n = len(text)
    quote: str | None = None
    while i < n:
        c = text[i]
        if quote is not None:
            if c == quote:
                quote = None
        elif c in ("'", '"'):
            quote = c
        elif c == ">":
            return i
        i += 1
    return -1


# Existence queries inside ``_neutralize_pseudo_tags`` used to re-scan the whole
# fragment (``text[:i]`` / ``text[start:]``) once per tag, which made the pass
# O(n^2) on adversarial input. These scan the fragment once so every query is
# O(1); the patterns mirror the old per-name forms exactly.
_OPENER_SCAN_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)[\s/>]", re.IGNORECASE)
_CLOSER_SCAN_RE = re.compile(r"</\s*([a-zA-Z][a-zA-Z0-9]*)\b", re.IGNORECASE)


def _neutralize_pseudo_tags(text: str) -> str:
    """Escape ``<`` sequences that are not real, balanced inline markup.

    Plain translated prose routinely contains ``List<T>``, ``<stdio.h>`` or
    ``a<b``. ``HTMLParser`` treats those as (illegal-named or unpaired) tags
    and swallows their inner text, so an export would silently lose content —
    fatal for a CS/math book. This pass keeps only ``<`` that opens a genuine
    tag:

    - a dangerous container (``DROP_WITH_CONTENT``) stays markup so the parser
      drops it *with* its content (the XSS guarantee is unchanged);
    - an allowlisted tag stays markup when it is void/self-closing, or when a
      matching close tag exists later (so a lone ``<b`` from ``a<b`` cannot
      open a phantom bold span).

    Everything else — unknown names like ``T``/``stdio.h``, and unpaired
    allowlisted tags — has its ``<`` escaped to literal text. This only ever
    *preserves* content relative to the old behaviour and cannot widen the XSS
    surface: neutralised runs render as inert text.
    """
    out: list[str] = []
    n = len(text)

    # One pass answers "is there an opener ``<name[\s/>]`` before i?" and "is
    # there a closer ``</\s*name\b`` at/after p?" in O(1) each.
    first_opener: dict[str, int] = {}
    last_closer: dict[str, int] = {}
    for opener_match in _OPENER_SCAN_RE.finditer(text):
        first_opener.setdefault(opener_match.group(1).lower(), opener_match.start())
    for closer_match in _CLOSER_SCAN_RE.finditer(text):
        last_closer[closer_match.group(1).lower()] = closer_match.start()

    # ``>`` positions and the next quote, so ``tag_end`` runs the quote-aware
    # scan only when a quote actually precedes the first ``>``.
    gt_positions = [idx for idx, ch in enumerate(text) if ch == ">"]
    next_quote = [n] * (n + 1)
    next_q = n
    for idx in range(n - 1, -1, -1):
        if text[idx] in ("'", '"'):
            next_q = idx
        next_quote[idx] = next_q

    def opener_before(name: str, pos: int) -> bool:
        return first_opener.get(name, n) < pos

    def closer_after(name: str, pos: int) -> bool:
        return last_closer.get(name, -1) >= pos

    def tag_end(start: int) -> int:
        pos = bisect_left(gt_positions, start + 1)
        if pos == len(gt_positions):
            return -1
        first_gt = gt_positions[pos]
        if next_quote[start + 1] >= first_gt:
            return first_gt
        return _scan_tag_end(text, start)

    i = 0
    while i < n:
        if text[i] != "<":
            nxt = text.find("<", i)
            if nxt == -1:
                out.append(text[i:])
                break
            out.append(text[i:nxt])
            i = nxt
            continue

        m = _TAG_NAME_RE.match(text, i)
        if not m:
            out.append("&lt;")
            i += 1
            continue

        is_close = m.group(1) == "/"
        name = m.group(2).lower()
        # Escape just the leading ``<`` and continue scanning after it; the
        # remainder may legitimately contain nested markup.
        literal = "&lt;" + m.group(0)[1:]

        if name in DROP_WITH_CONTENT:
            # A dangerous name drops content only when its close tag is present:
            # a *paired* container still reaches the parser and is dropped with
            # its content (the XSS guarantee). An unpaired occurrence is prose
            # about the tag (``The <script> tag loads …``) and a void name
            # (``<embed>``) has no content at all; both are escaped to inert
            # text so the rest of the fragment survives.
            if is_close:
                opts = opener_before(name, i)
                if opts:
                    end = tag_end(i)
                    out.append(text[i:] if end == -1 else text[i : end + 1])
                    i = n if end == -1 else end + 1
                else:
                    out.append(literal)
                    i = m.end()
                continue
            end = tag_end(i)
            if end == -1:
                out.append(literal)
                i = m.end()
                continue
            if closer_after(name, end + 1):
                out.append(text[i : end + 1])
                i = end + 1
            else:
                out.append(literal)
                i = m.end()
            continue

        if is_close:
            # A closer is real markup only when its opener was (or would be)
            # kept earlier in this fragment; otherwise escape it so a stray
            # ``</b>``/``</div>`` cannot orphan-open structure downstream.
            opener = opener_before(name, i)
            if opener:
                end = tag_end(i)
                out.append(text[i:] if end == -1 else text[i : end + 1])
                i = n if end == -1 else end + 1
                continue
            out.append(literal)
            i = m.end()
            continue

        if name not in ALLOWED_TAGS:
            # A benign *paired* tag (``<div>…</div>``) is real markup: keep it
            # so the parser drops the tag and keeps its inner text (module
            # docstring). An unpaired name (``List<T>``, ``<stdio.h>``) is
            # prose, not markup: escape it so its inner text is not swallowed.
            gt = tag_end(i)
            if gt != -1 and closer_after(name, gt + 1):
                out.append(text[i : gt + 1])
                i = gt + 1
                continue
            out.append(literal)
            i = m.end()
            continue

        # Allowlisted open tag: read to its ``>`` to tell void/self-closing
        # from a bare opener that needs a matching close.
        gt = tag_end(i)
        if gt == -1:
            # Unterminated ``<`` — cannot be markup; escape it as text.
            out.append(literal)
            i = m.end()
            continue
        tag_full = text[i : gt + 1]
        if not _TAG_STRUCT_RE.match(tag_full):
            # ``<b 且 b>`` — tag-shaped name but the body is prose, not
            # attributes. Escape the ``<`` and re-scan inside.
            out.append(literal)
            i = m.end()
            continue
        if name in VOID_TAGS or tag_full.endswith("/>"):
            out.append(tag_full)
            i = gt + 1
            continue
        if closer_after(name, gt + 1):
            out.append(tag_full)
            i = gt + 1
        else:
            out.append(literal)
            i = m.end()

    return "".join(out)


def sanitize_html_fragment(text: str) -> str:
    """Return ``text`` with dangerous HTML removed (allowlist approach).

    Plain-text angle brackets that are not real, balanced inline markup are
    preserved as escaped text (see :func:`_neutralize_pseudo_tags`), so
    technical prose such as ``List<T>`` or ``a<b`` never loses characters.
    Dangerous containers are dropped with their content, and on parser
    failure every tag is stripped (fail closed). Idempotent: sanitizing an
    already-sanitized fragment is a no-op.
    """
    if not text:
        return text
    if "<" in text:
        text = _neutralize_pseudo_tags(text)
    if "<" not in text and "&" not in text:
        return text
    parser = _SanitizingParser()
    try:
        parser.feed(text)
        parser.close()
        return parser.result()
    except Exception:
        return _escape_text(_TAG_STRIPPER.sub("", text))


_MARK_TAG_RE = re.compile(r"</?mark\b[^>]*>")


#: Attributes whose value is a URL a reading system may navigate or fetch.
#: ``xlink:href`` is the standard SVG/XML link attribute and is activatable;
#: ``srcset``/``imagesrcset`` carry a comma-separated candidate list that a
#: single-scheme check cannot validate, so they are handled separately below.
_URL_ATTR_NAMES = frozenset({"href", "src", "poster", "action", "formaction", "data", "xlink:href"})
_SRCSET_ATTR_NAMES = frozenset({"srcset", "imagesrcset"})

#: Void elements dropped on sight: ``<base>`` rewrites how every relative URL
#: in the member resolves, so a malicious source can reroute all links.
_SOURCE_DROP_TAGS = frozenset({"base"})

#: SMIL animation elements, dropped on sight. Their ``attributeName``/``values``
#: pair retargets an arbitrary attribute of a *parent* element to a value the
#: URL-scheme scrubber never inspects, so ``<animate attributeName="href"
#: values="javascript:…">`` animated an already-approved ``href`` into an
#: executable URL after the fact.
_SMIL_ANIMATION_TAGS = frozenset({"animate", "animatemotion", "animatetransform", "set", "discard"})

#: ``<link>`` rel values that load a remote resource (CSS, prefetch, icons).
_LINK_RESOURCE_RELS = frozenset(
    {
        "stylesheet",
        "preload",
        "prefetch",
        "preconnect",
        "dns-prefetch",
        "modulepreload",
        "icon",
        "apple-touch-icon",
        "manifest",
    }
)

_CSS_IMPORT_RE = re.compile(
    r"@import\s+(?:url\(\s*(['\"]?)(.*?)\1\s*\)|(['\"])(.*?)\3)\s*[^;]*;?",
    re.IGNORECASE,
)
_CSS_URL_RE = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE | re.DOTALL)
_CSS_EXPRESSION_RE = re.compile(r"expression\s*\(", re.IGNORECASE)

#: Containers dropped *with their content* from a source document.
#:
#: :data:`DROP_WITH_CONTENT` minus the three whose payload is not executable
#: code: ``style`` is the author's stylesheet (dropping it silently restyles the
#: whole book, and CSS cannot execute in an EPUB reading system), ``template``
#: is inert, and ``noscript`` content is only ever markup. A ``<script>`` nested
#: inside any of them is still caught by the generic tag rule, so nothing is
#: smuggled by keeping them.
_SOURCE_DROP_WITH_CONTENT = DROP_WITH_CONTENT - {"style", "template", "noscript"}

#: Elements whose bodies are RAWTEXT, not markup: a ``<!--`` inside them is CSS
#: or literal text to the browser, so scanning the body as markup both mis-reads
#: it and lets a crafted body smuggle an unscrubbed tag past the scrubber.
_RAW_TEXT_ELEMENTS = frozenset({"style", "textarea", "title"})


def _comment_end(markup: str, start: int) -> int:
    """Index just past the comment opened at ``start`` (HTML5 abrupt-close aware).

    HTML ends a comment at the first ``-->``, ``--!>``, or an abrupt close
    (``<!-->`` / ``<!--->``). Searching only for ``-->`` copied the rest of the
    document verbatim once a browser would already have resumed markup, so a tag
    after an abrupt close skipped the attribute scrubber. An unterminated
    comment runs to EOF, matching the browser, which keeps the copied bytes
    inert.
    """
    if markup.startswith("<!-->", start):
        return start + 5
    if markup.startswith("<!--->", start):
        return start + 6
    best = len(markup)
    for marker, width in (("-->", 3), ("--!>", 4)):
        pos = markup.find(marker, start + 4)
        if pos != -1:
            best = min(best, pos + width)
    return best


class _SourceTagScrubber:
    """Decide, per tag, what a source document must lose — and change nothing else.

    ``HTMLParser`` cannot be used for this: it lowercases tag names, so
    re-serializing an XML member (ONIX metadata, SVG) would rewrite
    ``</SenderName>`` into ``</sendername>`` and corrupt the document. A byte
    scanner keeps every region it does not have to touch exactly as it was, so
    scrubbing a clean book is provably a no-op (verified against the checked-in
    EPUB baselines).

    Only one thing needs a parser: reading a tag's attributes, done here on the
    isolated tag text, because quoting and entity references make a regex
    unreliable. A tag is re-rendered only when an attribute actually had to go.
    """

    _TAG_NAME_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)")

    @classmethod
    def scan(cls, markup: str) -> str:
        out: list[str] = []
        index = 0
        length = len(markup)
        while index < length:
            open_bracket = markup.find("<", index)
            if open_bracket == -1:
                out.append(markup[index:])
                break
            out.append(markup[index:open_bracket])

            if markup.startswith("<!--", open_bracket):
                comment_end = _comment_end(markup, open_bracket)
                out.append(markup[open_bracket:comment_end])
                index = comment_end
                continue

            # CDATA sections terminate at ']]>'
            if markup.startswith("<![CDATA[", open_bracket):
                cdata_end = markup.find("]]>", open_bracket + 9)
                tag_end = length if cdata_end == -1 else cdata_end + 3
                out.append(markup[open_bracket:tag_end])
                index = tag_end
                continue

            # Processing instructions terminate at '?>'
            if markup.startswith("<?", open_bracket):
                pi_end = markup.find("?>", open_bracket + 2)
                tag_end = length if pi_end == -1 else pi_end + 2
                out.append(markup[open_bracket:tag_end])
                index = tag_end
                continue

            # ``<!DOCTYPE …>`` and declarations: metadata and prologs, never executable — kept verbatim.
            if markup.startswith("<!", open_bracket):
                tag_end = _scan_tag_end(markup, open_bracket)
                tag_end = length if tag_end == -1 else tag_end + 1
                out.append(markup[open_bracket:tag_end])
                index = tag_end
                continue

            match = cls._TAG_NAME_RE.match(markup, open_bracket)
            if match is None:
                # A bare ``<`` in prose (``a<b``): copy it and move on.
                out.append(markup[open_bracket])
                index = open_bracket + 1
                continue

            name = match.group(2).lower()
            tag_end = _scan_tag_end(markup, open_bracket)
            if tag_end == -1:
                # Unterminated tag: fail closed by dropping the remainder.
                break
            raw_tag = markup[open_bracket : tag_end + 1]
            index = tag_end + 1

            if match.group(1) == "/":
                out.append(raw_tag)
                continue
            if name in _SOURCE_DROP_TAGS or name in _SMIL_ANIMATION_TAGS:
                # Void tag: skip the tag only, never ``_skip_element`` (which
                # would hunt a nonexistent end tag and drop the document tail).
                continue
            if name == "meta" and _is_refresh_meta(_collect_tag_attrs(raw_tag)):
                continue
            if name == "link" and _link_loads_resource(_collect_tag_attrs(raw_tag)):
                continue
            if name in _SOURCE_DROP_WITH_CONTENT:
                # Drop with content only when a close tag actually exists.
                # Without one, ``_skip_element`` would hunt a nonexistent end
                # tag and delete the rest of the member — the bug for void
                # ``<embed>`` and for prose about a tag name. Escape the tag to
                # inert text instead; a paired container still drops above.
                if re.search(rf"</\s*{name}\b", markup[index:], re.IGNORECASE):
                    index = cls._skip_element(markup, index, name)
                    continue
                out.append("&lt;" + raw_tag[1:-1] + "&gt;")
                continue
            out.append(cls._scrub_attributes(raw_tag))
            if name in _RAW_TEXT_ELEMENTS:
                index = cls._copy_raw_text(markup, index, name, out)

        return "".join(out)

    @classmethod
    def _copy_raw_text(cls, markup: str, start: int, name: str, out: list[str]) -> int:
        """Copy a raw-text element's body verbatim; return the index past its close.

        ``<style>``/``<textarea>``/``<title>`` bodies are RAWTEXT: a ``<!--``
        inside is CSS or literal text, not a comment. Scanning that body as
        markup is what let ``<style><!--</style><img onerror=…>`` smuggle an
        unscrubbed tag past the attribute scrubber. The first closing tag ends
        the body, matching HTML; with none, the remainder is copied inert.
        """
        closing = re.compile(rf"</\s*{name}\b[^>]*>", re.IGNORECASE)
        match = closing.search(markup, start)
        transform = scrub_css_text if name == "style" else None
        if match is None:
            body = markup[start:]
            out.append(transform(body) if transform is not None else body)
            return len(markup)
        body = markup[start : match.start()]
        out.append(transform(body) if transform is not None else body)
        out.append(match.group(0))
        return match.end()

    @classmethod
    def _skip_element(cls, markup: str, start: int, name: str) -> int:
        """Index just past the element's end tag (or end of input, failing closed).

        First-match, not depth-counted: for ``<script>``/``<style>`` HTML itself
        ends the element at the first closing tag, and for the remaining
        containers over-deleting an unpaired nesting is the safe direction.
        """
        closing = re.compile(rf"</\s*{name}\b[^>]*>", re.IGNORECASE)
        match = closing.search(markup, start)
        return len(markup) if match is None else match.end()

    @classmethod
    def _scrub_attributes(cls, raw_tag: str) -> str:
        """``raw_tag`` unchanged, or re-rendered without its dangerous attributes."""
        kept = _scrub_attr_list(_collect_tag_attrs(raw_tag))
        if kept is None:
            return raw_tag
        # EPUB members are XHTML, where ``<br>`` is not interchangeable with
        # ``<br/>``: a re-rendered tag must keep whichever form it arrived in or
        # the reading system rejects the whole member.
        self_closing = raw_tag.rstrip().endswith("/>")
        tag_name = raw_tag[1:-1].split()[0].rstrip("/") if raw_tag[1:-1].split() else ""
        rendered = "".join(
            f" {attr}" if value is None else f' {attr}="{_escape_attr(value)}"'
            for attr, value in kept
        )
        return f"<{tag_name}{rendered}{' /' if self_closing else ''}>"


def _raw_attr_names(raw_tag: str) -> list[str]:
    """Attribute names of one isolated tag, in source order, original casing.

    ``HTMLParser`` lowercases attribute names, so re-rendering a tag through its
    output rewrites ``viewBox`` to ``viewbox`` — which is a different attribute
    in SVG/XML and silently breaks the graphic. Scanning the tag text recovers
    the spelling the document used; quoted values are skipped so a ``=`` or a
    space inside a value is not mistaken for a new attribute.
    """
    names: list[str] = []
    index = 1
    length = len(raw_tag)
    # Skip the tag name (``<svg`` -> index just past ``svg``).
    while index < length and not raw_tag[index].isspace() and raw_tag[index] not in "/>":
        index += 1
    while index < length:
        char = raw_tag[index]
        if char.isspace() or char == "/":
            index += 1
            continue
        if char == ">":
            break
        start = index
        while index < length and not raw_tag[index].isspace() and raw_tag[index] not in "=/>":
            index += 1
        name = raw_tag[start:index]
        if name:
            names.append(name)
        while index < length and raw_tag[index].isspace():
            index += 1
        if index < length and raw_tag[index] == "=":
            index += 1
            while index < length and raw_tag[index].isspace():
                index += 1
            if index < length and raw_tag[index] in "\"'":
                quote = raw_tag[index]
                index += 1
                while index < length and raw_tag[index] != quote:
                    index += 1
                index += 1
            else:
                while index < length and not raw_tag[index].isspace() and raw_tag[index] != ">":
                    index += 1
    return names


def _collect_tag_attrs(raw_tag: str) -> list[tuple[str, str | None]]:
    """Attributes of one isolated tag, via the stdlib parser (quoting-aware).

    Names are re-spelled to their source casing: the parser lowercases them, and
    a re-rendered tag must not turn ``viewBox`` into ``viewbox``. When the two
    name lists disagree (a malformed tag the parser normalized) the parser's
    spelling wins — a missed attribute removal would be a security hole, a
    miscased one is cosmetic.
    """
    collector = _AttributeCollector()
    try:
        collector.feed(raw_tag)
        collector.close()
    except Exception:  # pragma: no cover - HTMLParser is total on tag text
        return []
    attrs = collector.attrs
    original_names = _raw_attr_names(raw_tag)
    if len(original_names) != len(attrs):
        return attrs
    return [(original, value) for original, (_, value) in zip(original_names, attrs, strict=True)]


class _AttributeCollector(HTMLParser):
    """Captures the attributes of the first start tag it is fed."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.attrs: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self.attrs:
            self.attrs = list(attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self.attrs:
            self.attrs = list(attrs)


def _scrub_attr_list(
    attrs: list[tuple[str, str | None]],
) -> list[tuple[str, str | None]] | None:
    """The attributes to keep, or None when nothing had to be removed.

    ``None`` is the "no change" signal that keeps :meth:`_SourceTagScrubber.scan`
    byte-exact for the overwhelming majority of tags.
    """
    kept: list[tuple[str, str | None]] = []
    removed = False
    for name, value in attrs:
        lowered = name.lower()
        if lowered.startswith("on"):
            removed = True
            continue
        if value and lowered in _SRCSET_ATTR_NAMES and not _srcset_allowed(value):
            removed = True
            continue
        if value and lowered in _URL_ATTR_NAMES and not _url_scheme_allowed(lowered, value):
            removed = True
            continue
        if value and lowered == "style" and not _style_value_allowed(value):
            removed = True
            continue
        kept.append((name, value))
    return kept if removed else None


def scrub_source_document(markup: str) -> str:
    """Strip executable constructs from a source document, structure intact.

    Source documents (EPUB members, HTML inputs) are copied into the exported
    artifact largely verbatim, so a malicious or merely dirty source can ship
    ``<script>``, ``onload=`` handlers or ``javascript:`` links to the reader.
    LLM-produced *target* text already goes through
    :func:`sanitize_html_fragment`; this is the other half, applied to markup
    the run did not write itself, providing defense-in-depth against malicious
    constructs embedded within imported source documents.

    Byte-exact wherever nothing had to be removed — including XML members whose
    tag case is significant — so scrubbing a clean book is provably a no-op.
    Idempotent, and total: a malformed document loses the tail it could not be
    parsed through rather than shipping unscrubbed.
    """
    if not markup:
        return markup
    return _SourceTagScrubber.scan(markup)


def strip_html_mark_tags(text: str) -> str:
    """Strip HTML <mark> tags (e.g. error/quarantine annotations) while preserving inner text.

    Used by non-HTML export adapters (DOCX, Markdown, Inplace PDF) so raw HTML
    formatting never leaks into compiled output or Word documents.
    """
    if not text or ("<mark" not in text and "</mark>" not in text):
        return text
    return _MARK_TAG_RE.sub("", text)

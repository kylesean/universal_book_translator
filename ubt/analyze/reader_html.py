"""Native HTML/XHTML reader: markup -> typed ``Document`` (native multi-format reader seam).

The third native reader, and the shared walker EPUB reuses. A single HTML file is
not paginated, so ``Span.page`` stays ``0``; the elements are decided from the DOM
itself -- headings keep their level, list items their marker, tables their grid
(as Markdown pipe markup, the same representation the Markdown reader uses),
figures their ``src`` asset reference, and ``pre`` blocks are code. Every element
gets a real ``Span.chars`` into the canonical text built from those elements, so
the typed document can be sliced and exported without an ``IRBlock`` detour.

The canonical text is the reading-order concatenation of the element texts, *not*
the raw HTML: it is the normalized text a reader sees, which is what a span and a
verifier should index (semantic HTML / document view index).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from ubt.analyze._identity import file_digest
from ubt.analyze.assemble import assemble, number
from ubt.model.ast import (
    Caption,
    CodeBlock,
    Confidence,
    Document,
    Figure,
    Heading,
    ListItem,
    Paragraph,
    Table,
)
from ubt.model.span import Span

if TYPE_CHECKING:
    from ubt.model.ast import ElementT

#: Heading tag -> level.
_HEADINGS: dict[str, int] = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
#: Non-content tags the walker never descends into.
_SKIP = frozenset({"script", "style", "head", "noscript", "template", "svg"})
#: Container tags the walker descends into without emitting an element itself.
_CONTAINERS = frozenset(
    {
        "body",
        "html",
        "div",
        "section",
        "article",
        "main",
        "header",
        "footer",
        "aside",
        "nav",
        "blockquote",
        "figure",
        "figcaption",
        "details",
        "summary",
        "dl",
        "dt",
        "dd",
        "fieldset",
        "form",
    }
)


def decode_bytes(raw: bytes) -> str:
    """Decode a markup byte string (BOM / declared encoding first)."""
    from ubt.adapters.base import decode_markup

    return decode_markup(raw)


def _clean(text: str) -> str:
    return " ".join(text.split())


def pipe_table(rows: list[list[str]]) -> str:
    """A grid as GitHub-flavoured Markdown pipe markup (its canonical form)."""
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)

    def _line(row: list[str]) -> str:
        padded = (row + [""] * width)[:width]
        # A literal | inside a cell would terminate the cell at parse time
        # (parse_pipe_table_cells splits on unescaped |); escape it, matching
        # the Docling table path.
        escaped = [cell.replace("|", r"\|") for cell in padded]
        return "| " + " | ".join(escaped) + " |"

    lines = [_line(rows[0]), "| " + " | ".join(["---"] * width) + " |"]
    lines.extend(_line(row) for row in rows[1:])
    return "\n".join(lines)


def _pipe_table(table: Tag) -> str:
    """A ``<table>`` as GitHub-flavoured Markdown pipe markup (its grid)."""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        row = [_clean(cell.get_text(" ", strip=True)) for cell in cells]
        if any(cell for cell in row):
            rows.append(row)
    return pipe_table(rows)


def _list_marker(item: Tag, index: int) -> str:
    parent = item.find_parent(["ul", "ol"])
    if parent is not None and parent.name == "ol":
        value = item.get("value")
        return f"{value if value else index}."
    return "•"


def walk_markup_elements(
    node: Tag,
    *,
    page: int = 0,
    resolve_asset: Callable[[str], str] | None = None,
) -> list[tuple[ElementT, Tag | list[NavigableString]]]:
    """Emit (Element, DOM node) pairs for one DOM node's block-level children, in reading order."""
    resolve = resolve_asset or (lambda src: src)
    out: list[tuple[ElementT, Tag | list[NavigableString]]] = []
    pending: list[NavigableString] = []

    def flush() -> None:
        if not pending:
            return
        text = _clean(" ".join(str(s) for s in pending))
        saved_pending = list(pending)
        pending.clear()
        if text:
            out.append(
                (Paragraph(id="", spine_index=0, span=Span(page=page), text=text), saved_pending)
            )

    for child in node.children:
        if isinstance(child, NavigableString):
            if str(child).strip():
                pending.append(child)
            continue
        if not isinstance(child, Tag):
            continue
        name = (child.name or "").lower()
        if name in _SKIP:
            continue
        if name in _HEADINGS:
            flush()
            text = _clean(child.get_text(" ", strip=True))
            if text:
                out.append(
                    (
                        Heading(
                            id="",
                            spine_index=0,
                            span=Span(page=page),
                            text=text,
                            level=_HEADINGS[name],
                            confidence=Confidence.VERIFIED,
                        ),
                        child,
                    )
                )
            continue
        if name == "p":
            flush()
            text = _clean(child.get_text(" ", strip=True))
            if text:
                out.append(
                    (Paragraph(id="", spine_index=0, span=Span(page=page), text=text), child)
                )
            continue
        if name in ("ul", "ol"):
            flush()
            for index, item in enumerate(child.find_all("li", recursive=False), start=1):
                text = _clean(item.get_text(" ", strip=True))
                if text:
                    out.append(
                        (
                            ListItem(
                                id="",
                                spine_index=0,
                                span=Span(page=page),
                                text=text,
                                marker=_list_marker(item, index),
                            ),
                            item,
                        )
                    )
            continue
        if name == "li":
            flush()
            text = _clean(child.get_text(" ", strip=True))
            if text:
                out.append(
                    (
                        ListItem(id="", spine_index=0, span=Span(page=page), text=text, marker="•"),
                        child,
                    )
                )
            continue
        if name == "pre":
            flush()
            code = child.get_text("", strip=False).rstrip("\n")
            if code.strip():
                out.append(
                    (
                        CodeBlock(
                            id="",
                            spine_index=0,
                            span=Span(page=page),
                            text=code,
                            skip_translate=True,
                            confidence=Confidence.VERIFIED,
                        ),
                        child,
                    )
                )
            continue
        if name == "table":
            flush()
            markup = _pipe_table(child)
            if markup:
                out.append(
                    (
                        Table(
                            id="",
                            spine_index=0,
                            span=Span(page=page),
                            markup=markup,
                            confidence=Confidence.VERIFIED,
                        ),
                        child,
                    )
                )
            continue
        if name == "img":
            flush()
            src = str(child.get("src") or child.get("data-src") or "").strip()
            if src:
                out.append(
                    (
                        Figure(
                            id="",
                            spine_index=0,
                            span=Span(page=page),
                            asset_id=resolve(src),
                            confidence=Confidence.VERIFIED,
                        ),
                        child,
                    )
                )
            continue
        if name == "figcaption":
            flush()
            text = _clean(child.get_text(" ", strip=True))
            if text:
                out.append((Caption(id="", spine_index=0, span=Span(page=page), text=text), child))
            continue
        if name in _CONTAINERS:
            flush()
            out.extend(walk_markup_elements(child, page=page, resolve_asset=resolve))
            continue
        # Unknown/inline tag: fold its text into the pending paragraph.
        inline = _clean(child.get_text(" ", strip=True))
        if inline:
            pending.append(NavigableString(inline))
    flush()
    return out


def _walk(
    node: Tag,
    out: list[ElementT],
    *,
    page: int,
    resolve_asset: Callable[[str], str],
) -> None:
    """Emit elements for one DOM node's block-level children, in reading order."""
    pairs = walk_markup_elements(node, page=page, resolve_asset=resolve_asset)
    out.extend(element for element, _ in pairs)


def elements_from_markup(
    soup: BeautifulSoup,
    *,
    page: int = 0,
    resolve_asset: Callable[[str], str] | None = None,
) -> list[ElementT]:
    """The block elements of a parsed HTML/XHTML document, in reading order."""
    resolve = resolve_asset or (lambda src: src)
    body = soup.body or soup
    return [elem for elem, _ in walk_markup_elements(body, page=page, resolve_asset=resolve)]


def read_html(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Read an HTML/XHTML file into a typed :class:`Document` with real spans."""
    html_path = Path(path)
    soup = BeautifulSoup(decode_bytes(html_path.read_bytes()), "html.parser")
    elements = number(elements_from_markup(soup), "html")
    return assemble(elements, doc_id=doc_id or file_digest(html_path), path=str(html_path))


__all__ = ["elements_from_markup", "pipe_table", "read_html", "walk_markup_elements"]

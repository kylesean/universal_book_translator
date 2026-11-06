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
from ubt.analyze.normalize import collapse_whitespace as _clean
from ubt.core.exceptions import DocumentParseError
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
    from ubt.core.text_decode import decode_markup

    return decode_markup(raw)


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
    """A ``<table>`` as GitHub-flavoured Markdown pipe markup (its grid).

    ``colspan``/``rowspan`` are expanded into the grid with the covered cells
    left empty. Without that, a merged cell silently shifted every later column
    of the row: a table with one ``colspan=2`` header rendered each row one cell
    short. The source and the translated grid are read by this same function, so
    the replacement verifier saw two wrong grids that agreed.

    Rows are collected from *this* table only (``find_parent("table") is table``),
    so a nested table's rows no longer appear twice — once inside the outer cell
    text and once as their own rows.
    """
    own_rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    grid: dict[tuple[int, int], str] = {}
    occupied: set[tuple[int, int]] = set()
    width = 0
    for row_index, tr in enumerate(own_rows):
        column = 0
        for cell in tr.find_all(["th", "td"]):
            if cell.find_parent("table") is not table:
                continue
            while (row_index, column) in occupied:
                column += 1
            text = _clean(cell.get_text(" ", strip=True))
            col_span = _int_attr(cell, "colspan")
            row_span = _int_attr(cell, "rowspan")
            for dr in range(row_span):
                for dc in range(col_span):
                    key = (row_index + dr, column + dc)
                    # Only the anchor cell carries the text; the cells it spans
                    # are rendered empty, which is as close as pipe markup gets
                    # to a merge.
                    grid[key] = text if (dr == 0 and dc == 0) else ""
                    occupied.add(key)
            column += col_span
            width = max(width, column)
    rows = [
        [grid.get((row_index, column), "") for column in range(width)]
        for row_index in range(len(own_rows))
    ]
    rows = [row for row in rows if any(cell for cell in row)]
    return pipe_table(rows)


#: Upper bound on one cell's span. A hostile/degenerate document can declare
#: ``colspan="100000"``; expanding it literally would allocate a list per row
#: for no reader's benefit.
_MAX_TABLE_SPAN = 64


def _int_attr(cell: Tag, name: str, *, default: int = 1, minimum: int = 1) -> int:
    """A bounded positive integer attribute, or ``default`` when unusable."""
    try:
        value = int(str(cell.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, min(value, _MAX_TABLE_SPAN))


def _list_marker(item: Tag, index: int) -> str:
    parent = item.find_parent(["ul", "ol"])
    if parent is not None and parent.name == "ol":
        value = item.get("value")
        if value:
            return f"{value}."
        # ``<ol start="5">`` continued numbering was ignored, so a list split
        # across pages restarted at 1.
        try:
            start = int(str(parent.get("start", "")).strip())
        except (TypeError, ValueError):
            start = 1
        return f"{max(1, start) + index - 1}."
    return "•"


#: Block-level tags whose text must not be glued to the surrounding inline run.
#: Without a separator ``<li>parent<ul><li>child</li></ul></li>`` flattened to
#: ``"parentchild"`` — the parent item and its sub-item fused into one word that
#: the translator then has to guess at.
_INLINE_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "div",
        "dl",
        "dd",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "ul",
    }
)

#: Nesting deeper than this is not a book — it is a crafted or broken document.
#: Recursing through it raised a bare ``RecursionError`` (which escapes the
#: adapter's ``DocumentParseError`` contract) somewhere past ~500 levels; a typed
#: refusal names the real problem instead.
_MAX_DOM_DEPTH = 200


def _inline_marked_text(node: Tag | NavigableString, *, _depth: int = 0) -> str:
    """Flatten inline emphasis into the pipeline's own marker syntax.

    The document model has no inline tree yet, but ``<em>``/``<strong>`` are
    the two inline facts a translation must preserve. Reusing the existing
    ``⟦B⟧``/``⟦I⟧`` protocol keeps the change inside the renderer's established
    emphasis path instead of inventing a second inline representation.
    """
    if isinstance(node, NavigableString):
        return str(node)
    if _depth > _MAX_DOM_DEPTH:
        raise DocumentParseError(
            f"HTML nesting exceeds {_MAX_DOM_DEPTH} levels at <{node.name}>; "
            "refusing to walk this document (it is malformed or deliberately "
            "pathological)."
        )
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = (child.name or "").lower()
        if name in ("script", "style", "noscript"):
            continue
        inner = _inline_marked_text(child, _depth=_depth + 1)
        if name in ("em", "i"):
            parts.append(f"⟦I⟧{inner}⟦/I⟧")
        elif name in ("strong", "b"):
            parts.append(f"⟦B⟧{inner}⟦/B⟧")
        elif name == "br":
            parts.append(" ")
        elif name in _INLINE_BLOCK_TAGS:
            parts.append(f" {inner} ")
        else:
            parts.append(inner)
    return "".join(parts)


def walk_markup_elements(
    node: Tag,
    *,
    page: int = 0,
    resolve_asset: Callable[[str], str] | None = None,
    _depth: int = 0,
) -> list[tuple[ElementT, Tag | list[NavigableString]]]:
    """Emit (Element, DOM node) pairs for one DOM node's block-level children, in reading order."""
    if _depth > _MAX_DOM_DEPTH:
        # A bare RecursionError escaped the adapter's DocumentParseError contract
        # and reached the job as an unclassified exception; fail with the real
        # cause while the document is still identifiable.
        raise DocumentParseError(
            f"HTML nesting exceeds {_MAX_DOM_DEPTH} levels at <{node.name}>; "
            "refusing to walk this document (it is malformed or deliberately "
            "pathological)."
        )
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
            text = _clean(_inline_marked_text(child))
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
            # An image inside a paragraph is the standard HTML/EPUB figure
            # shape (``<p><img src="fig1.png"></p>``). Taking only the
            # paragraph's *text* dropped it entirely: an ``<img>`` contributes
            # no text, so the block produced nothing and the picture was never
            # visited — the ``img`` branch below only fires for a direct child.
            images = child.find_all("img")
            text = _clean(_inline_marked_text(child))
            if text:
                out.append(
                    (Paragraph(id="", spine_index=0, span=Span(page=page), text=text), child)
                )
            for image in images:
                src = str(image.get("src") or image.get("data-src") or "").strip()
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
                            image,
                        )
                    )
            continue
        if name in ("ul", "ol"):
            flush()
            for index, item in enumerate(child.find_all("li", recursive=False), start=1):
                text = _clean(_inline_marked_text(item))
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
            text = _clean(_inline_marked_text(child))
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
            text = _clean(_inline_marked_text(child))
            if text:
                out.append((Caption(id="", spine_index=0, span=Span(page=page), text=text), child))
            continue
        if name in _CONTAINERS:
            flush()
            out.extend(
                walk_markup_elements(child, page=page, resolve_asset=resolve, _depth=_depth + 1)
            )
            continue
        # Unknown/inline tag: fold its text into the pending paragraph.
        inline = _clean(_inline_marked_text(child))
        if inline:
            pending.append(NavigableString(inline))
    flush()
    return out


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

"""Bilingual/monolingual DOM injection: the one engine behind HTML and EPUB.

The EPUB and HTML adapters both deliver by re-parsing the source document and
injecting the translation beside (bilingual) or over (monolingual) each
extracted block. The two loops were ~54% identical and had drifted: EPUB
sanitized each ``\\n\\n`` fragment separately while HTML sanitized once up
front, EPUB counted every injected paragraph and HTML counted one per block,
and the source-class merge accepted a bare-string ``class`` in one and not the
other. A fix to one loop silently skipped the other, so the two formats shipped
different escaping and different injection counts for the same book.

This module owns the per-node injection: one function per case (table, list
item, prose paragraph, bare text-run block), parameterized by the target CSS
class. The target text arrives already sanitized (the adapter sanitizes once,
before splitting), so the escaping behavior is by construction identical.

Nothing here knows about blocks, chapters or the ledger: it takes the typed
element, its DOM node, and the delivered text.
"""

from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup, Tag
from bs4.element import AttributeValueList

from ubt.model.ast import ListItem, Table


#: Classes a node carries after merging the source's own classes with the
#: injected-target marker. Handles bs4's list-shaped value, a bare string (a
#: hand-written ``class="a b"`` a parser kept as text), and anything falsy --
#: one merge for both adapters, where each used to have its own.
def merged_target_classes(dom_node: Tag, target_css_class: str) -> list[str]:
    raw = dom_node.get("class")
    if isinstance(raw, (list, AttributeValueList)):
        return list(raw) + [target_css_class]
    if raw:
        return str(raw).split() + [target_css_class]
    return [target_css_class]


def _copy_fragment(soup: BeautifulSoup, markup: str, container: Tag) -> None:
    """Append a parsed markup fragment's children onto ``container``."""
    parsed = BeautifulSoup(markup, "html.parser")
    for child in list(parsed.contents):
        container.append(child)


#: Inline tags a monolingual rewrite carries over: graphic media, embedded objects,
#: breaks and linked figures. Text-bearing links and formulas are excluded because
#: their translated prose is already placed within the paragraph body.
_MEDIA_INLINE_TAGS = frozenset(
    {"img", "svg", "picture", "source", "video", "audio", "br", "hr", "object"}
)


def _should_preserve_node(node: Tag) -> bool:
    """Whether an inline element should be detached and carried over to monolingual output."""
    if node.name in _MEDIA_INLINE_TAGS:
        return True
    if node.name == "a":
        # Keep image links (e.g. <a href="..."><img .../></a>)
        if node.find(list(_MEDIA_INLINE_TAGS)):
            return True
        # Keep empty target anchors (e.g. <a id="..."></a> or <a name="..."></a>)
        if (node.get("id") or node.get("name")) and not node.get_text(strip=True):
            return True
    return False


def take_preserved_inline_children(leaf: Tag) -> list[Tag]:
    """Detach and return inline descendants a monolingual rewrite must keep.

    Only outermost media tags and linked images/target anchors are taken; text-bearing
    <a> tags and <math> formulas are not appended to the end of translated paragraphs
    to avoid duplication against translated body text.
    """
    preserved = [
        c
        for c in leaf.find_all(lambda tag: isinstance(tag, Tag) and _should_preserve_node(tag))
        if not any(_should_preserve_node(p) for p in c.parents if isinstance(p, Tag) and p != leaf)
    ]
    for node in preserved:
        node.extract()
    return preserved


def _inject_table(
    soup: BeautifulSoup,
    dom_node: Tag,
    target_text: str,
    *,
    is_monolingual: bool,
    target_css_class: str,
) -> int:
    """A table block: rewrite the grid in place, or rebuild it as a new table.

    A translated grid whose row count still matches the DOM's is written cell by
    cell, preserving ``th``/``td`` and the table's own attributes. Anything else
    is rebuilt from the translated Markdown grid: replaced over the source in
    monolingual mode, inserted beside it (marked with the target class) in
    bilingual.
    """
    rows = None
    from ubt.adapters.base import parse_pipe_table_cells

    parsed_rows = parse_pipe_table_cells(target_text)
    if parsed_rows:
        rows = parsed_rows
    if not rows:
        return 0
    if is_monolingual:
        dom_trs = dom_node.find_all("tr")
        if len(dom_trs) == len(rows):
            for r_idx, tr in enumerate(dom_trs):
                cells = tr.find_all(["th", "td"])
                for c_idx, cell in enumerate(cells):
                    if c_idx < len(rows[r_idx]):
                        cell.string = rows[r_idx][c_idx]
        else:
            new_table = soup.new_tag("table")
            if dom_node.get("class"):
                new_table["class"] = dom_node["class"]
            for row in rows:
                tr = soup.new_tag("tr")
                for cell_text in row:
                    td = soup.new_tag("td")
                    td.string = cell_text
                    tr.append(td)
                new_table.append(tr)
            dom_node.replace_with(new_table)
    else:
        new_table = soup.new_tag("table")
        new_table["class"] = AttributeValueList(merged_target_classes(dom_node, target_css_class))
        for row in rows:
            tr = soup.new_tag("tr")
            for cell_text in row:
                td = soup.new_tag("td")
                td.string = cell_text
                tr.append(td)
            new_table.append(tr)
        dom_node.insert_after(new_table)
    return 1


def _inject_list_item(
    soup: BeautifulSoup,
    dom_node: Tag,
    target_text: str,
    *,
    is_monolingual: bool,
    target_css_class: str,
) -> int:
    """A list item: replace the text (keeping preserved inline nodes) or append a sibling.

    A ``<li>`` must keep its marker and stay inside its parent list, so
    monolingual mode clears only the text and re-hangs the preserved inline
    children (images, anchors, breaks); bilingual mode appends a ``<div>``
    sibling *inside* the ``<li>`` so the pair reads as one item.
    """
    if is_monolingual:
        preserved_inline = take_preserved_inline_children(dom_node)
        dom_node.clear()
        dom_node.append(BeautifulSoup(target_text, "html.parser"))
        for node in preserved_inline:
            dom_node.append(node)
    else:
        target_tag = soup.new_tag("div")
        target_tag["class"] = AttributeValueList([target_css_class])
        target_tag.append(BeautifulSoup(target_text, "html.parser"))
        dom_node.append(target_tag)
    return 1


def _inject_prose(
    soup: BeautifulSoup,
    dom_node: Tag,
    target_text: str,
    *,
    is_monolingual: bool,
    target_css_class: str,
) -> int:
    """A prose block: one or more ``\\n\\n``-separated paragraphs.

    Monolingual replaces the node's text with the paragraphs (the first lands in
    the node itself, the rest as same-tag siblings); bilingual inserts a
    same-tag sibling per paragraph after the source, each carrying the source's
    classes plus the target marker. Returns the number of paragraphs placed --
    the caller's injected count, so a chapter whose paragraphs split counts
    what actually shipped.
    """
    paras = [p.strip() for p in target_text.split("\n\n") if p.strip()] or [target_text]
    node_name = dom_node.name if dom_node.name else "p"
    if is_monolingual:
        preserved_inline = take_preserved_inline_children(dom_node)
        dom_node.clear()
        last_node: Tag = dom_node
        for i, p_text in enumerate(paras):
            container = dom_node if i == 0 else soup.new_tag(node_name)
            _copy_fragment(soup, p_text, container)
            if i > 0:
                last_node.insert_after(container)
            last_node = container
        for node in preserved_inline:
            dom_node.append(node)
    else:
        target_classes = merged_target_classes(dom_node, target_css_class)
        last_node = dom_node
        for p_text in paras:
            target_tag = soup.new_tag(node_name)
            target_tag["class"] = AttributeValueList(target_classes)
            _copy_fragment(soup, p_text, target_tag)
            last_node.insert_after(target_tag)
            last_node = target_tag
    return len(paras)


def _inject_text_run(
    soup: BeautifulSoup,
    dom_nodes: list[Any],
    target_text: str,
    *,
    is_monolingual: bool,
    target_css_class: str,
) -> int:
    """A bare text-run block (text directly inside ``<blockquote>`` and friends).

    Extraction counted the run as a block, so injection must too: monolingual
    replaces the first run's text and drops the rest; bilingual appends one
    ``<p>`` after the last run.
    """
    if is_monolingual:
        dom_nodes[0].replace_with(BeautifulSoup(target_text, "html.parser"))
        for rem in dom_nodes[1:]:
            rem.extract()
    else:
        target_tag = soup.new_tag("p")
        target_tag["class"] = AttributeValueList([target_css_class])
        target_tag.append(BeautifulSoup(target_text, "html.parser"))
        dom_nodes[-1].insert_after(target_tag)
    return 1


def inject_translated_block(
    soup: BeautifulSoup,
    elem: Any,
    dom_node: Any,
    target_text: str,
    *,
    is_monolingual: bool,
    target_css_class: str,
) -> int:
    """Inject one block's translation. Returns the number of injected nodes.

    ``target_text`` must already be sanitized (and carry its failure note, if
    any): this module places text, it does not decide what is safe to place --
    one sanitize at the adapter, not one per fragment, is the drift fix.
    Dispatch follows the same order extraction did: table, list item, prose,
    bare text-run list.
    """
    if isinstance(elem, Table) and isinstance(dom_node, Tag):
        return _inject_table(
            soup,
            dom_node,
            target_text,
            is_monolingual=is_monolingual,
            target_css_class=target_css_class,
        )
    if isinstance(dom_node, Tag):
        if isinstance(elem, ListItem):
            return _inject_list_item(
                soup,
                dom_node,
                target_text,
                is_monolingual=is_monolingual,
                target_css_class=target_css_class,
            )
        return _inject_prose(
            soup,
            dom_node,
            target_text,
            is_monolingual=is_monolingual,
            target_css_class=target_css_class,
        )
    if isinstance(dom_node, list) and dom_node:
        return _inject_text_run(
            soup,
            dom_node,
            target_text,
            is_monolingual=is_monolingual,
            target_css_class=target_css_class,
        )
    return 0


__all__ = ["inject_translated_block", "merged_target_classes"]

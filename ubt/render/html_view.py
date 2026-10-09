"""Semantic HTML lowering: the delivered blocks as a web view.

It lowers the run's blocks to semantic HTML, using each block's delivered text:
the translation where one was placed, the source slice where the block was kept.
A block's HTML tag follows its typed element, so a heading becomes ``<h2>``, a
list item ``<li>``, and so on. It is a *plain* view -- the AST does not model
inline structure (links, emphasis), so paragraphs become text-only ``<p>``.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path

from ubt.core.ir.models import IRBlock
from ubt.model.ast import (
    Caption,
    CodeBlock,
    Dialogue,
    Element,
    Figure,
    Formula,
    Heading,
    ListItem,
    Paragraph,
    Table,
)
from ubt.render.outputs import Composition, Placement

_TAIL = "</body>\n</html>\n"


def document_head(lang: str = "en", direction: str = "ltr") -> str:
    """The ``<html>`` head for a view, carrying the target language and direction.

    ``dir`` is only emitted for a right-to-left target, so an LTR document's
    bytes are unchanged from before direction support existed.
    """
    dir_attr = f' dir="{html.escape(direction)}"' if direction and direction != "ltr" else ""
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="{html.escape(lang)}"{dir_attr}>\n<head>\n<meta charset="utf-8">\n'
        "<title>UBT translation</title>\n</head>\n<body>\n"
    )


def element_text(block: IRBlock, translations: Mapping[str, str]) -> str:
    """The text this block delivers: its placed translation, else its source."""
    target = (translations.get(block.id) or block.target_text or "").strip()
    if target and not _kept_in_source(block):
        return target
    return block.source_text or ""


def _kept_in_source(block: IRBlock) -> bool:
    """Whether the delivery keeps this block in the source (deliberate or fail-closed)."""
    if block.skip_translate:
        return True
    flags = block.error_flags or []
    return any(
        isinstance(flag, str) and flag.startswith(("render_skip:", "inplace_skip:"))
        for flag in flags
    )


def _render_element(element: Element, body: str) -> str:
    """One element's HTML tag, with ``body`` already escaped."""
    if isinstance(element, Heading):
        level = min(max(element.level, 1), 6)
        return f"<h{level}>{body}</h{level}>"
    if isinstance(element, Paragraph):
        return f"<p>{body}</p>"
    if isinstance(element, Dialogue):
        return f'<p class="dialogue">{body}</p>'
    if isinstance(element, Caption):
        return f"<figcaption>{body}</figcaption>"
    if isinstance(element, CodeBlock):
        return f"<pre><code>{body}</code></pre>"
    if isinstance(element, Formula):
        return f'<span class="formula">{body}</span>'
    if isinstance(element, Table):
        return f'<pre class="table">{body}</pre>'
    if isinstance(element, Figure):
        return f'<figure data-asset="{html.escape(element.asset_id)}"></figure>'
    return f"<p>{body}</p>"


def render_fragment(
    blocks: Sequence[IRBlock],
    translations: Mapping[str, str],
) -> tuple[str, tuple[Placement, ...]]:
    """The element HTML and every placement, without the document wrapper.

    Shared by the HTML view and the EPUB view, so both lower blocks the same way.
    """
    parts: list[str] = []
    placements: list[Placement] = []
    list_open = False
    for block in sorted(blocks, key=lambda b: b.spine_index):
        element = block.element
        body = html.escape(element_text(block, translations))
        if isinstance(element, ListItem):
            if not list_open:
                parts.append("<ul>")
                list_open = True
            parts.append(f"<li>{body}</li>")
        else:
            if list_open:
                parts.append("</ul>")
                list_open = False
            parts.append(_render_element(element, body))
        placements.append(
            Placement(
                block.id,
                element.span.page,
                drawn=not _kept_in_source(block),
                detail=f"html:{element.kind}",
            )
        )
    if list_open:
        parts.append("</ul>")
    return "\n".join(parts), tuple(placements)


def compose_html(
    blocks: Sequence[IRBlock],
    translations: Mapping[str, str],
    output_path: str | Path,
    *,
    lang: str = "en",
    direction: str = "ltr",
) -> Composition:
    """Lower the delivered blocks to semantic HTML, recording every placement.

    ``lang``/``direction`` describe the *target* language; a right-to-left target
    gets ``dir="rtl"`` on the root element so a browser lays the document out
    right-to-left (the text stays in logical order).
    """
    fragment, placements = render_fragment(blocks, translations)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{document_head(lang, direction)}{fragment}\n{_TAIL}", encoding="utf-8")
    return Composition(output_path=output, placements=placements)


__all__ = ["compose_html", "element_text", "render_fragment"]

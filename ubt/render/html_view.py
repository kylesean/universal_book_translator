"""Semantic HTML lowering: the delivered blocks as a web view.

It lowers the run's blocks to semantic HTML, using each block's delivered text:
the translation where one was placed, the source slice where the block was kept.
A block's HTML tag follows its typed element, so a heading becomes ``<h2>``, a
list item ``<li>``, and so on. Styled runs (``StyleMeta.inline_runs`` /
``target_runs``) are re-applied as ``<strong>``/``<em>``/``<sup>``/colour
spans -- see :func:`render_fragment`; a paragraph is text-only only when its
source had no recorded styles.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path

from ubt.core.atomic import atomic_write_text
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
from ubt.render.inline_spans import SpanStyle, block_runs, styled_chunks
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
    """Whether the delivery keeps this block in the source (deliberate or fail-closed).

    Delegates to the single owner of this decision (``content.adapt``) rather
    than re-listing the skip-flag prefixes: a local copy drifted, so a
    quarantined (``BLOCKED_HUMAN``) block -- whose ``target_text`` is a ``<mark>``
    wrapper around the source, not a translation -- was rendered as its raw
    marker text in the HTML/EPUB view. One predicate, one place to keep correct.
    """
    from ubt.core.content.adapt import kept_in_source

    return kept_in_source(block)


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


def _html_style_wrap(segment: str, style: SpanStyle) -> str:
    """Wrap an escaped segment in the HTML its source style calls for.

    The HTML twin of the renderer's ``_wrap_style``: same inputs, same
    precedence (superscript innermost), a ``<span style>`` carrying the colour.
    Applied after escaping, so the injected tags are never escaped.
    """
    start, _end, bold, italic, superscript, color_hex = style
    if superscript:
        segment = f"<sup>{segment}</sup>"
    if italic:
        segment = f"<em>{segment}</em>"
    if bold:
        segment = f"<strong>{segment}</strong>"
    if color_hex:
        segment = f'<span style="color:{html.escape(color_hex)}">{segment}</span>'
    return segment


def render_fragment(
    blocks: Sequence[IRBlock],
    translations: Mapping[str, str],
) -> tuple[str, tuple[Placement, ...]]:
    """The element HTML and every placement, without the document wrapper.

    Shared by the HTML view and the EPUB view, so both lower blocks the same way.

    A block's styled runs (:data:`StyleMeta.inline_runs` for the source spans
    the probe recorded, ``target_runs`` for the model's bold markers) are
    located in the delivered text and wrapped: a heading or paragraph then keeps
    the emphasis and colour its source had, where an ``html.escape`` of the
    whole text dropped every one. A run that survives only after the target's
    CJK normalization still locates (tolerant matching); one the translation
    rewrote draws plain, the same contract the Typst renderer documents.
    """
    parts: list[str] = []
    placements: list[Placement] = []
    list_open = False
    for block in sorted(blocks, key=lambda b: b.spine_index):
        element = block.element
        text = element_text(block, translations)
        styled = styled_chunks(text, block_runs(block))
        if len(styled) == 1 and styled[0][1] is None:
            body = html.escape(text)
        else:
            body = "".join(
                html.escape(chunk) if style is None else _html_style_wrap(html.escape(chunk), style)
                for chunk, style in styled
            )
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
    atomic_write_text(
        output, f"{document_head(lang, direction)}{fragment}\n{_TAIL}", encoding="utf-8"
    )
    return Composition(output_path=output, placements=placements)


__all__ = ["compose_html", "element_text", "render_fragment"]

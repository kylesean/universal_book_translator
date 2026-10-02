"""Semantic HTML lowering: the Document as a web view (semantic document lowering layer).

The first non-PDF view, and deliberately a *view*, not a reader: it lowers the
same typed :class:`~ubt.model.ast.Document` the kernel already realizes to
semantic HTML, using each element's :class:`~ubt.model.fidelity.Attestation` to
pick its text -- the delivered translation where the element was reconstructed,
the opaque source slice where it was preserved. It is the HTML twin of
:func:`ubt.render.outputs.compose`: same coverage rule (a missing attestation is
a refusal, not a silent gap), same ``Placement`` record.

It is a *plain* view: the AST deliberately does not model inline structure
(links, emphasis) yet, so paragraphs become text-only ``<p>``. That is the honest
first step; a richer view arrives when the AST can carry what it needs.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path

from ubt.model.ast import (
    Caption,
    CodeBlock,
    Dialogue,
    Document,
    Element,
    Figure,
    Formula,
    Heading,
    ListItem,
    Paragraph,
    Table,
)
from ubt.model.fidelity import Attestation, Fidelity
from ubt.model.span import CanonicalSource
from ubt.render.outputs import Composition, LoweringUnsupported, Placement
from ubt.render.overlay_backend import source_slice

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


def _element_text(
    element: Element,
    fidelity: Fidelity,
    source: CanonicalSource,
    delivered: Mapping[str, str],
) -> str:
    """The text this element was attested to carry (translation or source slice)."""
    if fidelity > Fidelity.PRESERVED_OPAQUE:
        target = delivered.get(element.id, "")
        if target:
            return target
    return source_slice(element, source)


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
    document: Document,
    attestations: Sequence[Attestation],
    delivered: Mapping[str, str],
) -> tuple[str, tuple[Placement, ...]]:
    """The element HTML and every placement, without the document wrapper.

    Shared by the HTML view and the EPUB view, so both lower elements the same
    way and neither re-derives the attestation-to-text rule.
    """
    by_id = {attestation.element_id: attestation for attestation in attestations}
    unjudged = [element.id for element in document.elements if element.id not in by_id]
    if unjudged:
        raise LoweringUnsupported(
            f"{len(unjudged)} element(s) have no attestation: {', '.join(unjudged[:5])}"
        )

    parts: list[str] = []
    placements: list[Placement] = []
    list_open = False
    for element in document.elements:
        attestation = by_id[element.id]
        body = html.escape(_element_text(element, attestation.fidelity, document.source, delivered))
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
                element.id,
                element.span.page,
                attestation.fidelity,
                attestation.fidelity,
                f"html:{element.kind}",
            )
        )
    if list_open:
        parts.append("</ul>")
    return "\n".join(parts), tuple(placements)


def compose_html(
    document: Document,
    attestations: Sequence[Attestation],
    delivered: Mapping[str, str],
    output_path: str | Path,
    *,
    lang: str = "en",
    direction: str = "ltr",
) -> Composition:
    """Lower a realized document to semantic HTML, recording every placement.

    ``lang``/``direction`` describe the *target* language; a right-to-left target
    gets ``dir="rtl"`` on the root element so a browser lays the document out
    right-to-left (the text stays in logical order).
    """
    fragment, placements = render_fragment(document, attestations, delivered)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{document_head(lang, direction)}{fragment}\n{_TAIL}", encoding="utf-8")
    return Composition(output_path=output, placements=placements)


__all__ = ["compose_html", "render_fragment"]

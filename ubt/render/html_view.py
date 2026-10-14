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

import base64
import html
import mimetypes
from collections.abc import Callable, Mapping, Sequence
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

#: Resolves a block's ``asset_id`` to an absolute path (``None`` when the file
#: is not there). The EPUB view supplies one that copies the bytes into the
#: package and returns the in-package href; the HTML view uses the default,
#: which inlines the bytes as a data URI so the single file stays portable.
AssetResolver = Callable[[str], str | None]


def _data_uri(path: Path) -> str:
    """Inline an asset as a ``data:`` URI (self-contained HTML file)."""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _default_asset_resolver(asset: str) -> str | None:
    """Resolve an asset path relative to the CWD; inline it as a data URI."""
    if not asset:
        return None
    candidate = Path(asset)
    if not candidate.is_absolute() or not candidate.exists():
        return None
    return _data_uri(candidate)


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


def _render_element(
    element: Element, body: str, asset_href: str | None = None, raw_text: str = ""
) -> str:
    """One element's HTML tag, with ``body`` already escaped.

    ``asset_href`` is the resolved ``src`` for a figure (a data URI in the HTML
    view, an in-package path in the EPUB); ``None`` means the asset could not be
    resolved, and the figure degrades to a labelled placeholder rather than a
    silently empty box. ``raw_text`` is the unescaped delivered text, which the
    table branch parses (a grid must be read before escaping).
    """
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
        return _render_table(raw_text or body)
    if isinstance(element, Figure):
        # ``data-asset`` carries the BASENAME only: the delivered view is a
        # customer artifact, and the extractor's asset_id is an absolute host
        # path (``/home/<user>/.../pic_p1_1.png``) that has no business
        # shipping. The name is enough to correlate a figure with its block.
        asset_name = html.escape(Path(element.asset_id).name or element.asset_id)
        if asset_href:
            return (
                f'<figure data-asset="{asset_name}">'
                f'<img src="{html.escape(asset_href, quote=True)}" alt="">'
                "</figure>"
            )
        return f'<figure class="figure-missing"><p>[figure: {asset_name}]</p></figure>'
    return f"<p>{body}</p>"


def _render_table(markup: str) -> str:
    """Lower a table block's delivered grid to a real ``<table>``.

    A table block's delivered text is a Markdown pipe grid (the renderer's own
    input). The view used to wrap that grid verbatim in ``<pre class="table">``,
    so a screen reader read the pipes and the text reflowed as code. Parsing the
    grid through the renderer's own parser (one definition of "a real table")
    yields the semantic table; a grid that does not parse — a malformed
    translation the PDF renderer would also fall back on — keeps the ``<pre>``
    form rather than dropping the block.
    """
    from ubt.render.table_compiler import parse_markdown_table

    parsed = parse_markdown_table(markup, escape=False)
    if parsed is None:
        return f'<pre class="table">{html.escape(markup)}</pre>'
    headers, rows = parsed
    head = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body_rows}</tbody></table>"


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
    *,
    asset_resolver: AssetResolver | None = None,
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

    Three non-text element classes lower semantically rather than as raw markup
    (2026-10-14 review, W3-7): a table grid becomes a real ``<table>``, a
    formula's LaTeX degrades through the transpiler's own fallback to readable
    Unicode (the same guarantee the PDF renderer's degrade path gives), and a
    figure resolves through ``asset_resolver`` to an actual ``<img>`` instead of
    an empty ``<figure data-asset>`` box.
    """
    from ubt.render.typst.math_transpiler import clean_math_fallback_text, strip_math_delimiters

    resolve = asset_resolver or _default_asset_resolver
    parts: list[str] = []
    placements: list[Placement] = []
    list_open = False
    for block in sorted(blocks, key=lambda b: b.spine_index):
        element = block.element
        text = element_text(block, translations)
        if isinstance(element, Formula):
            # The delivered text of a formula block is its LaTeX (the renderer
            # never lets a model retype it). Raw LaTeX in a web view is markup
            # soup; the transpiler's fallback turns it into the readable form
            # the PDF would have drawn had the formula degraded there.
            text = clean_math_fallback_text(strip_math_delimiters(text))
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
            asset_href = resolve(element.asset_id) if isinstance(element, Figure) else None
            parts.append(_render_element(element, body, asset_href, raw_text=text))
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

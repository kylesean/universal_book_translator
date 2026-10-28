"""Publishing renderers for chapter epigraphs and introductory abstracts in Typst and HTML."""

from __future__ import annotations

import html
from collections.abc import Callable

from ubt.core.epigraph.models import ChapterAbstract, Epigraph, EpigraphConfig, EpigraphStyle


def _esc_typst(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def _format_quote(text: str, enable_decoration: bool = True) -> str:
    clean = text.strip()
    if not enable_decoration:
        return clean
    if not (clean.startswith(("“", "「", '"', "『")) and clean.endswith(("”", "」", '"', "』"))):
        # Apply standard typographic curly quotes
        is_cjk = any("\u4e00" <= c <= "\u9fff" for c in clean)
        return f"“{clean}”" if is_cjk else f'"{clean}"'
    return clean


def render_epigraph_typst(
    epigraph: Epigraph,
    config: EpigraphConfig | None = None,
    prose_renderer: Callable[[str], str] | None = None,
) -> str:
    """Render a chapter epigraph into publication-grade Typst markup."""
    cfg = config or EpigraphConfig()
    render_p = prose_renderer or _esc_typst

    quote_decorated = _format_quote(epigraph.text, cfg.enable_quotes_decoration)
    quote_rendered = render_p(quote_decorated)
    source_rendered = render_p(epigraph.source.strip())

    is_cjk = any("\u4e00" <= c <= "\u9fff" for c in epigraph.source)
    prefix = cfg.author_prefix_zh if is_cjk else cfg.author_prefix_en
    author_line = f"{prefix}{source_rendered}" if source_rendered else ""

    sz = cfg.font_size_pt
    sz_src = max(7.0, sz - 0.5)

    if cfg.style == EpigraphStyle.BORDER_LEFT:
        return (
            f"#block(\n"
            f"  stroke: (left: 2.5pt + luma(160)),\n"
            f"  inset: (left: 12pt, top: 4pt, bottom: 6pt),\n"
            f"  width: 85%,\n"
            f")[\n"
            f'  #text(size: {sz}pt, style: "italic", fill: luma(60))[{quote_rendered}]\n'
            + (
                f"  #v(3pt)\n  #align(right)[#text(size: {sz_src}pt, fill: luma(100))[{author_line}]]\n"
                if author_line
                else ""
            )
            + "]"
        )

    if cfg.style == EpigraphStyle.CALLOUT_BOX:
        return (
            f"#block(\n"
            f"  fill: luma(250),\n"
            f"  inset: 10pt,\n"
            f"  radius: 3pt,\n"
            f"  stroke: 0.5pt + luma(220),\n"
            f"  width: 90%,\n"
            f")[\n"
            f'  #text(size: {sz}pt, style: "italic", fill: luma(50))[{quote_rendered}]\n'
            + (
                f"  #v(4pt)\n  #align(right)[#text(size: {sz_src}pt, fill: luma(90))[{author_line}]]\n"
                if author_line
                else ""
            )
            + "]"
        )

    if cfg.style == EpigraphStyle.MINIMAL:
        return (
            f"#align(center)[\n"
            f"  #block(width: 80%)[\n"
            f'    #text(size: {sz}pt, style: "italic", fill: luma(60))[{quote_rendered}]\n'
            + (
                f"    #v(3pt)\n    #text(size: {sz_src}pt, fill: luma(100))[{author_line}]\n"
                if author_line
                else ""
            )
            + "  ]\n"
            "]"
        )

    # Default: EpigraphStyle.CLASSIC_RIGHT
    return (
        f"#align(right)[\n"
        f"  #block(width: 75%, inset: (top: 4pt, bottom: 8pt))[\n"
        f'    #align(left)[#text(size: {sz}pt, style: "italic", fill: luma(70))[{quote_rendered}]]\n'
        + (
            f"    #v(4pt)\n    #align(right)[#text(size: {sz_src}pt, fill: luma(100))[{author_line}]]\n"
            if author_line
            else ""
        )
        + "  ]\n"
        "]"
    )


def render_abstract_typst(
    abstract: ChapterAbstract,
    config: EpigraphConfig | None = None,
    prose_renderer: Callable[[str], str] | None = None,
) -> str:
    """Render a chapter abstract briefing container into Typst markup."""
    cfg = config or EpigraphConfig()
    render_p = prose_renderer or _esc_typst

    title_rendered = render_p(abstract.title.strip())
    body_rendered = render_p(abstract.text.strip())

    fill_line = "  fill: luma(248),\n" if cfg.abstract_box_shaded else ""
    return (
        f"#block(\n"
        f"{fill_line}"
        f"  inset: 10pt,\n"
        f"  radius: 3pt,\n"
        f"  stroke: 0.5pt + luma(210),\n"
        f"  width: 100%,\n"
        f")[\n"
        f'  #align(center)[#text(weight: "bold", size: 9.5pt)[{title_rendered}]]\n'
        f"  #v(4pt)\n"
        f"  #text(size: 9pt, fill: luma(60))[{body_rendered}]\n"
        f"]"
    )


def render_epigraph_html(
    epigraph: Epigraph,
    config: EpigraphConfig | None = None,
) -> str:
    """Render an epigraph into semantic HTML5 blockquote with cite."""
    cfg = config or EpigraphConfig()
    quote_decorated = _format_quote(epigraph.text, cfg.enable_quotes_decoration)
    q_esc = html.escape(quote_decorated)
    s_esc = html.escape(epigraph.source.strip())
    prefix = (
        cfg.author_prefix_zh
        if any("\u4e00" <= c <= "\u9fff" for c in epigraph.source)
        else cfg.author_prefix_en
    )

    cite_part = f'\n  <cite class="ubt-epigraph-source">{prefix}{s_esc}</cite>' if s_esc else ""
    return (
        f'<blockquote class="ubt-epigraph ubt-epigraph-{cfg.style.value}">\n'
        f'  <p class="ubt-epigraph-text">{q_esc}</p>{cite_part}\n'
        f"</blockquote>"
    )


def render_abstract_html(
    abstract: ChapterAbstract,
    config: EpigraphConfig | None = None,
) -> str:
    """Render a chapter abstract into semantic HTML5 aside container."""
    _ = config or EpigraphConfig()
    t_esc = html.escape(abstract.title.strip())
    b_esc = html.escape(abstract.text.strip())

    return (
        f'<aside class="ubt-chapter-abstract">\n'
        f'  <header class="ubt-abstract-title">{t_esc}</header>\n'
        f'  <div class="ubt-abstract-body">{b_esc}</div>\n'
        f"</aside>"
    )

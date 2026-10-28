"""Publication-grade bilingual concordance and index rendering for Typst, HTML, and Markdown."""

from __future__ import annotations

import html
from typing import Final

from ubt.core.index.models import ConcordanceIndex, IndexEntry

_TYPST_SPECIALS: Final[dict[str, str]] = {
    "[": r"\[",
    "]": r"\]",
    "#": r"\#",
    "$": r"\$",
    "@": r"\@",
    "&": r"\&",
    "<": r"\<",
    ">": r"\>",
}


def _escape_typst(text: str) -> str:
    """Escape Typst syntax characters in plain term names."""
    res = text
    for char, rep in _TYPST_SPECIALS.items():
        res = res.replace(char, rep)
    return res


def render_index_typst(
    index: ConcordanceIndex,
    language_first: str = "en",
    columns: int = 3,
    show_locations: bool = True,
) -> str:
    """Render a publication-grade multi-column index section in Typst."""
    if not index.entries:
        return ""

    lines: list[str] = [
        "// --- Publication Bilingual Concordance Index ---",
        "#pagebreak(weak: true)",
        "#heading(level: 1, numbering: none)[索引 (Index)]",
        "#v(0.6em)",
        f"#columns({columns}, gutter: 12pt)[",
        "  #set text(size: 8.5pt, spacing: 120%)",
        "  #set par(justify: false, leading: 0.55em, first-line-indent: 0pt)",
    ]

    grouped = (
        index.grouped_by_source_initial()
        if language_first.lower() == "en"
        else index.grouped_by_target_initial()
    )

    for letter, entries in grouped.items():
        lines.append("  #block(breakable: false)[")
        lines.append(f'    #text(weight: "bold", size: 1.25em, fill: rgb("#1a365d"))[{letter}]')
        lines.append("    #v(1pt)")
        lines.append('    #line(length: 100%, stroke: 0.5pt + rgb("#cbd5e1"))')
        lines.append("    #v(3pt)")
        lines.append("  ]")

        for entry in entries:
            term_line = _render_typst_entry_line(entry, language_first, show_locations)
            lines.append(f"  {term_line}")

        lines.append("  #v(8pt)")

    lines.append("]")
    lines.append("// --- End Bilingual Index ---\n")
    return "\n".join(lines)


def _render_typst_entry_line(entry: IndexEntry, language_first: str, show_locations: bool) -> str:
    if language_first.lower() == "en":
        primary = _escape_typst(entry.source_term)
        secondary = _escape_typst(entry.target_term)
        display = f"*{primary}*" + (f" ({secondary})" if secondary and secondary != primary else "")
    else:
        primary = _escape_typst(entry.target_term)
        secondary = _escape_typst(entry.source_term)
        display = f"*{primary}*" + (f" ({secondary})" if secondary and secondary != primary else "")

    loc_links: list[str] = []
    if show_locations and entry.locations:
        for idx, loc in enumerate(entry.locations[:5]):
            label = (
                str(loc.page)
                if loc.page
                else (f"§{idx + 1}" if not loc.section_title else loc.section_title[:10])
            )
            anchor = loc.section_anchor or loc.block_id
            if anchor:
                loc_links.append(f"#link(<{anchor}>)[{_escape_typst(label)}]")
            else:
                loc_links.append(f"[{_escape_typst(label)}]")

    if loc_links:
        return f"{display} · " + ", ".join(loc_links) + " \\"
    return f"{display} \\"


def render_index_html(index: ConcordanceIndex, language_first: str = "en", columns: int = 3) -> str:
    """Render an interactive responsive HTML index section with alphabet navigation."""
    if not index.entries:
        return ""

    grouped = (
        index.grouped_by_source_initial()
        if language_first.lower() == "en"
        else index.grouped_by_target_initial()
    )

    nav_links = "".join(
        f'<a href="#idx-{html.escape(k)}" class="index-nav-item">{html.escape(k)}</a> '
        for k in grouped
    )

    parts: list[str] = [
        '<section class="ubt-concordance-index">',
        '  <h2 class="index-heading">索引 (Index)</h2>',
        f'  <nav class="index-quick-jump">{nav_links}</nav>',
        f'  <div class="index-grid" style="column-count: {columns}; column-gap: 1.5rem;">',
    ]

    for letter, entries in grouped.items():
        parts.append(
            f'    <div id="idx-{html.escape(letter)}" class="index-letter-group" style="break-inside: avoid; margin-bottom: 1.25rem;">'
        )
        parts.append(
            f'      <h3 class="letter-badge" style="border-bottom: 2px solid #3b82f6; color: #1e40af; margin-bottom: 0.5rem;">{html.escape(letter)}</h3>'
        )
        parts.append(
            '      <ul style="list-style: none; padding-left: 0; margin: 0; font-size: 0.9rem; line-height: 1.6;">'
        )

        for entry in entries:
            if language_first.lower() == "en":
                name = f"<strong>{html.escape(entry.source_term)}</strong>"
                if entry.target_term and entry.target_term != entry.source_term:
                    name += f" <span class='sub-term' style='color: #64748b;'>({html.escape(entry.target_term)})</span>"
            else:
                name = f"<strong>{html.escape(entry.target_term)}</strong>"
                if entry.source_term and entry.source_term != entry.target_term:
                    name += f" <span class='sub-term' style='color: #64748b;'>({html.escape(entry.source_term)})</span>"

            loc_strs: list[str] = []
            for idx, loc in enumerate(entry.locations[:5]):
                lbl = (
                    str(loc.page)
                    if loc.page
                    else (loc.section_title[:10] if loc.section_title else f"§{idx + 1}")
                )
                if loc.section_anchor:
                    loc_strs.append(
                        f'<a href="#{html.escape(loc.section_anchor)}" class="index-link">{html.escape(lbl)}</a>'
                    )
                else:
                    loc_strs.append(f"<span>{html.escape(lbl)}</span>")

            links_span = f" · {' '.join(loc_strs)}" if loc_strs else ""
            parts.append(f"        <li style='margin-bottom: 0.25rem;'>{name}{links_span}</li>")

        parts.append("      </ul>")
        parts.append("    </div>")

    parts.append("  </div>")
    parts.append("</section>")
    return "\n".join(parts)


def render_index_markdown(index: ConcordanceIndex, language_first: str = "en") -> str:
    """Render a GitHub-flavored markdown concordance index."""
    if not index.entries:
        return ""

    grouped = (
        index.grouped_by_source_initial()
        if language_first.lower() == "en"
        else index.grouped_by_target_initial()
    )

    lines: list[str] = [
        "## 📑 术语索引 (Index)",
        "",
        "> 本索引按字母顺序汇总全书核心概念与专业术语中英双语对照及出处位置。",
        "",
    ]

    for letter, entries in grouped.items():
        lines.append(f"### {letter}\n")
        for entry in entries:
            if language_first.lower() == "en":
                display = f"**{entry.source_term}**" + (
                    f" ({entry.target_term})" if entry.target_term else ""
                )
            else:
                display = f"**{entry.target_term}**" + (
                    f" ({entry.source_term})" if entry.source_term else ""
                )

            loc_refs = []
            for idx, loc in enumerate(entry.locations[:4]):
                lbl = str(loc.page) if loc.page else (loc.section_title or f"§{idx + 1}")
                loc_refs.append(lbl)

            loc_str = f" — *{', '.join(loc_refs)}*" if loc_refs else ""
            lines.append(f"- {display}{loc_str}")
        lines.append("")

    return "\n".join(lines)

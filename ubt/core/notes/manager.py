"""Footnote and endnote flow management, transformation, and multi-target rendering."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence

from ubt.core.notes.models import (
    FootnoteItem,
    NoteNumberingStyle,
    NotesConfig,
)


def _esc_typst(s: str) -> str:
    return s.replace("[", r"\[").replace("]", r"\]").replace("#", r"\#")


def render_footnote_inline_typst(note: FootnoteItem) -> str:
    """Render a native Typst page-bottom footnote inline call."""
    return f"#footnote[{_esc_typst(note.content)}]"


def render_endnote_section_typst(
    notes: Sequence[FootnoteItem], config: NotesConfig | None = None
) -> str:
    """Render a publication-grade endnote section in Typst."""
    cfg = config or NotesConfig()
    if not notes:
        return ""

    items: list[str] = []
    for n in notes:
        marker = n.formatted_marker(cfg.numbering_style)
        backlink = (
            f' #link(<{n.note_id}-ref>)[#text(fill: rgb("#2563eb"))[↩]]'
            if cfg.enable_backlinks
            else ""
        )
        content = _esc_typst(n.content)
        items.append(
            f'  #block(inset: (bottom: 0.35em))[[#metadata(()) <{n.note_id}>] #text(weight: "bold")[{marker}.] {content}{backlink}]'
        )

    joined_items = "\n".join(items)
    return (
        f"#v(1.2em)\n"
        f"#line(length: 30%, stroke: 0.5pt + luma(150))\n"
        f"#v(0.4em)\n"
        f"#heading(level: 2)[{cfg.section_title}]\n"
        f"#v(0.3em)\n"
        f"#set text(size: {cfg.font_size_pt}pt, fill: luma(60))\n"
        f"{joined_items}\n"
    )


def render_endnote_section_html(
    notes: Sequence[FootnoteItem], config: NotesConfig | None = None
) -> str:
    """Render a publication-grade endnote section in semantic HTML5."""
    cfg = config or NotesConfig()
    if not notes:
        return ""

    items: list[str] = []
    for n in notes:
        content = html.escape(n.content)
        backlink = (
            f' <a href="#{n.note_id}-ref" class="ubt-backlink" title="Jump back">↩</a>'
            if cfg.enable_backlinks
            else ""
        )
        items.append(
            f'    <li id="{n.note_id}" value="{n.index_num}">'
            f'<span class="ubt-endnote-text">{content}</span>{backlink}</li>'
        )

    joined_items = "\n".join(items)
    return (
        f'<section class="ubt-endnotes" role="doc-endnotes">\n'
        f'  <hr class="ubt-notes-divider" />\n'
        f'  <h2 class="ubt-notes-heading">{html.escape(cfg.section_title)}</h2>\n'
        f'  <ol class="ubt-endnotes-list">\n'
        f"{joined_items}\n"
        f"  </ol>\n"
        f"</section>"
    )


def render_endnote_section_markdown(
    notes: Sequence[FootnoteItem], config: NotesConfig | None = None
) -> str:
    """Render standard Markdown footnote definitions."""
    if not notes:
        return ""
    lines = [f"[^{n.index_num}]: {n.content}" for n in notes]
    return "\n".join(lines)


def inject_typst_footnotes_into_text(text: str, notes_in_block: Sequence[FootnoteItem]) -> str:
    """Replace original footnote callout markers in prose with native Typst `#footnote[...]`."""
    out = text
    for note in notes_in_block:
        pattern = re.escape(note.callout_marker)
        esc_variant = pattern.replace(r"\[", r"\\\[").replace(r"\]", r"\\\]")
        combined = f"(?:{esc_variant}|{pattern})"
        replacement = render_footnote_inline_typst(note)
        out = re.sub(combined, replacement, out, count=1)
    return out


def inject_endnote_links_into_text(
    text: str, notes_in_block: Sequence[FootnoteItem], style: NoteNumberingStyle | None = None
) -> str:
    """Replace original footnote callout markers in prose with superscript anchor links."""
    st = style or NoteNumberingStyle.ARABIC
    out = text
    for note in notes_in_block:
        marker = note.formatted_marker(st)
        link_callout = (
            f"[#metadata(()) <{note.note_id}-ref>]#link(<{note.note_id}>)[#super[{marker}]]"
        )
        pattern = re.escape(note.callout_marker)
        esc_variant = pattern.replace(r"\[", r"\\\[").replace(r"\]", r"\\\]")
        combined = f"(?:{esc_variant}|{pattern})"
        out = re.sub(combined, link_callout, out, count=1)
    return out


class FootnoteFlowManager:
    """Holds the note placement configuration used by the render legs."""

    def __init__(self, config: NotesConfig | None = None) -> None:
        self.config = config or NotesConfig()

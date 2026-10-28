"""Footnote and endnote flow management, transformation, and multi-target rendering."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.core.notes.models import (
    FootnoteItem,
    NoteNumberingStyle,
    NotePlacement,
    NotesConfig,
    NotesTopology,
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
    """High-level orchestrator for footnote and endnote transformation across IR blocks."""

    def __init__(self, config: NotesConfig | None = None) -> None:
        self.config = config or NotesConfig()

    def transform_blocks(self, blocks: Sequence[IRBlock], topology: NotesTopology) -> list[IRBlock]:
        """Transform IRBlocks according to the configured note placement policy."""
        if not topology.notes:
            return list(blocks)

        # Map source_block_id -> list of FootnoteItem
        notes_by_source: dict[str, list[FootnoteItem]] = {}
        for n in topology.notes:
            if n.source_block_id:
                notes_by_source.setdefault(n.source_block_id, []).append(n)

        # Map chapter_index -> list of FootnoteItem
        notes_by_chapter: dict[int, list[FootnoteItem]] = {}
        for n in topology.notes:
            notes_by_chapter.setdefault(n.chapter_index, []).append(n)

        result: list[IRBlock] = []
        current_chapter_idx = 1

        for i, block in enumerate(blocks):
            # Check if this block is a consumed footnote definition block
            if block.id in topology.def_block_ids:
                # Omit definition block from the main body stream
                continue

            b_type = str(getattr(block.block_type, "value", block.block_type) or "")
            lvl = (
                getattr(getattr(block, "element", None), "level", None)
                or getattr(block, "level", None)
                or 1
            )
            is_h1 = b_type == "heading" and lvl == 1

            if is_h1 and self.config.placement == NotePlacement.CHAPTER_ENDNOTE and i > 0:
                prev_notes = notes_by_chapter.get(current_chapter_idx, [])
                if prev_notes:
                    endnote_markup = render_endnote_section_typst(prev_notes, self.config)
                    el = make_element(
                        id=f"endnotes-ch{current_chapter_idx}",
                        spine_index=1,
                        block_type=BlockType.NARRATIVE,
                        source_text=endnote_markup,
                    )
                    result.append(
                        IRBlock(
                            id=f"endnotes-ch{current_chapter_idx}",
                            block_type=BlockType.NARRATIVE,
                            source_text=endnote_markup,
                            target_text=endnote_markup,
                            element=el,
                        )
                    )
                current_chapter_idx += 1

            # Inlining for page-bottom footnote placement in Typst
            if self.config.placement == NotePlacement.FOOTNOTE and block.id in notes_by_source:
                text = block.target_text or block.source_text or ""
                transformed_text = inject_typst_footnotes_into_text(text, notes_by_source[block.id])
                new_block = IRBlock(
                    id=block.id,
                    block_type=block.block_type,
                    source_text=block.source_text,
                    target_text=transformed_text,
                    element=block.element,
                    bbox=block.bbox,
                    provenance=block.provenance,
                )
                result.append(new_block)
                continue

            # Inlining link anchors for endnote placement
            if (
                self.config.placement in (NotePlacement.CHAPTER_ENDNOTE, NotePlacement.BOOK_ENDNOTE)
                and block.id in notes_by_source
            ):
                text = block.target_text or block.source_text or ""
                transformed_text = text
                for note in notes_by_source[block.id]:
                    marker = note.formatted_marker(self.config.numbering_style)
                    # Insert superscript link to note definition
                    link_callout = f"[#metadata(()) <{note.note_id}-ref>]#link(<{note.note_id}>)[#super[{marker}]]"
                    transformed_text = re.sub(
                        re.escape(note.callout_marker), link_callout, transformed_text, count=1
                    )
                new_block = IRBlock(
                    id=block.id,
                    block_type=block.block_type,
                    source_text=block.source_text,
                    target_text=transformed_text,
                    element=block.element,
                    bbox=block.bbox,
                    provenance=block.provenance,
                )
                result.append(new_block)
                continue

            result.append(block)

        # Flush trailing endnotes
        if self.config.placement == NotePlacement.CHAPTER_ENDNOTE:
            last_notes = notes_by_chapter.get(current_chapter_idx, [])
            if last_notes:
                endnote_markup = render_endnote_section_typst(last_notes, self.config)
                el = make_element(
                    id=f"endnotes-ch{current_chapter_idx}",
                    spine_index=1,
                    block_type=BlockType.NARRATIVE,
                    source_text=endnote_markup,
                )
                result.append(
                    IRBlock(
                        id=f"endnotes-ch{current_chapter_idx}",
                        block_type=BlockType.NARRATIVE,
                        source_text=endnote_markup,
                        target_text=endnote_markup,
                        element=el,
                    )
                )
        elif self.config.placement == NotePlacement.BOOK_ENDNOTE:
            all_notes = topology.notes
            if all_notes:
                endnote_markup = render_endnote_section_typst(all_notes, self.config)
                el = make_element(
                    id="endnotes-book",
                    spine_index=1,
                    block_type=BlockType.NARRATIVE,
                    source_text=endnote_markup,
                )
                result.append(
                    IRBlock(
                        id="endnotes-book",
                        block_type=BlockType.NARRATIVE,
                        source_text=endnote_markup,
                        target_text=endnote_markup,
                        element=el,
                    )
                )

        return result

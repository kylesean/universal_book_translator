"""Footnote & Endnote Flow Manager for professional book publication."""

from __future__ import annotations

from ubt.core.notes.extractor import extract_notes_topology
from ubt.core.notes.manager import (
    FootnoteFlowManager,
    inject_endnote_links_into_text,
    inject_typst_footnotes_into_text,
    render_endnote_section_html,
    render_endnote_section_markdown,
    render_endnote_section_typst,
    render_footnote_inline_typst,
)
from ubt.core.notes.models import (
    FootnoteItem,
    NoteNumberingScope,
    NoteNumberingStyle,
    NotePlacement,
    NotesConfig,
    NotesTopology,
)

__all__ = [
    "FootnoteFlowManager",
    "FootnoteItem",
    "NoteNumberingScope",
    "NoteNumberingStyle",
    "NotePlacement",
    "NotesConfig",
    "NotesTopology",
    "extract_notes_topology",
    "inject_endnote_links_into_text",
    "inject_typst_footnotes_into_text",
    "render_endnote_section_html",
    "render_endnote_section_markdown",
    "render_endnote_section_typst",
    "render_footnote_inline_typst",
]

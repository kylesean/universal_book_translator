"""Data models for footnote and endnote life-cycle flow management."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NotePlacement(StrEnum):
    """Where footnote and endnote explanations are typeset and presented."""

    FOOTNOTE = "footnote"  # Inlined as page-bottom notes (Typst #footnote[...] or EPUB popup)
    CHAPTER_ENDNOTE = "chapter_endnote"  # Gathered at the end of each chapter
    BOOK_ENDNOTE = "book_endnote"  # Gathered at the end of the entire book / backmatter


class NoteNumberingScope(StrEnum):
    """Scope over which note numerical indices increment."""

    PER_CHAPTER = "per_chapter"  # Resets to 1 at each Level-1 Chapter heading
    CONTINUOUS = "continuous"  # Increments monotonically 1..N across the entire document


class NoteNumberingStyle(StrEnum):
    """Format and typographic symbol style for note reference callouts."""

    ARABIC = "arabic"  # 1, 2, 3...
    CIRCLED = "circled"  # ①, ②, ③, ④...
    ROMAN = "roman"  # i, ii, iii, iv...
    SYMBOL = "symbol"  # *, †, ‡, §...


@dataclass(slots=True)
class FootnoteItem:
    """A matched footnote/endnote pair linking callout citation to explanatory text."""

    note_id: str  # Unique anchor id (e.g. 'fn-ch1-1', 'fn-2')
    index_num: int  # Normalized 1-based index within numbering scope
    callout_marker: str  # Original marker found in prose (e.g. '[^1]', '1', '①', '*')
    content: str  # Note definition body text
    chapter_index: int = 1  # 1-based chapter counter
    chapter_title: str = ""  # Title of the enclosing chapter
    source_block_id: str | None = None  # Block ID where the callout citation was placed
    def_block_id: str | None = None  # Block ID where the footnote definition was declared
    page: int | None = None  # Page number if known from extraction provenance

    def formatted_marker(self, style: NoteNumberingStyle = NoteNumberingStyle.ARABIC) -> str:
        """Render index number in the specified typographic symbol style."""
        if style == NoteNumberingStyle.CIRCLED:
            _CIRCLED = [
                "①",
                "②",
                "③",
                "④",
                "⑤",
                "⑥",
                "⑦",
                "⑧",
                "⑨",
                "⑩",
                "⑪",
                "⑫",
                "⑬",
                "⑭",
                "⑮",
                "⑯",
                "⑰",
                "⑱",
                "⑲",
                "⑳",
            ]
            idx = self.index_num - 1
            if 0 <= idx < len(_CIRCLED):
                return _CIRCLED[idx]
            return f"({self.index_num})"

        if style == NoteNumberingStyle.ROMAN:
            _ROMAN = [
                "i",
                "ii",
                "iii",
                "iv",
                "v",
                "vi",
                "vii",
                "viii",
                "ix",
                "x",
                "xi",
                "xii",
                "xiii",
                "xiv",
                "xv",
            ]
            idx = self.index_num - 1
            if 0 <= idx < len(_ROMAN):
                return _ROMAN[idx]
            return str(self.index_num)

        if style == NoteNumberingStyle.SYMBOL:
            _SYMBOLS = ["*", "†", "‡", "§", "**", "††", "‡‡", "§§"]
            idx = self.index_num - 1
            if 0 <= idx < len(_SYMBOLS):
                return _SYMBOLS[idx]
            return str(self.index_num)

        return str(self.index_num)

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "index_num": self.index_num,
            "callout_marker": self.callout_marker,
            "content": self.content,
            "chapter_index": self.chapter_index,
            "chapter_title": self.chapter_title,
            "source_block_id": self.source_block_id,
            "def_block_id": self.def_block_id,
            "page": self.page,
        }


@dataclass(slots=True)
class NotesConfig:
    """Configuration options for footnote & endnote handling."""

    placement: NotePlacement = NotePlacement.FOOTNOTE
    numbering_scope: NoteNumberingScope = NoteNumberingScope.PER_CHAPTER
    numbering_style: NoteNumberingStyle = NoteNumberingStyle.ARABIC
    section_title: str = "注释"  # Title used when emitting chapter or book endnotes
    enable_backlinks: bool = True  # Inject jump-back links (↩) in endnotes
    font_size_pt: float = 8.5  # Typesetting font size for note bodies

    def to_dict(self) -> dict[str, Any]:
        return {
            "placement": str(self.placement),
            "numbering_scope": str(self.numbering_scope),
            "numbering_style": str(self.numbering_style),
            "section_title": self.section_title,
            "enable_backlinks": self.enable_backlinks,
            "font_size_pt": self.font_size_pt,
        }


@dataclass(slots=True)
class NotesTopology:
    """The complete cross-document notes topology and linking manifest."""

    notes: list[FootnoteItem] = field(default_factory=list)
    placement: NotePlacement = NotePlacement.FOOTNOTE
    numbering_scope: NoteNumberingScope = NoteNumberingScope.PER_CHAPTER
    numbering_style: NoteNumberingStyle = NoteNumberingStyle.ARABIC
    orphan_callouts: list[dict[str, Any]] = field(default_factory=list)
    orphan_definitions: list[dict[str, Any]] = field(default_factory=list)
    def_block_ids: set[str] = field(default_factory=set)

    @property
    def total_notes(self) -> int:
        return len(self.notes)

    @property
    def resolution_rate(self) -> float:
        total = self.total_notes + len(self.orphan_callouts) + len(self.orphan_definitions)
        if total == 0:
            return 1.0
        return self.total_notes / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_notes": self.total_notes,
            "resolution_rate": round(self.resolution_rate, 4),
            "placement": str(self.placement),
            "numbering_scope": str(self.numbering_scope),
            "numbering_style": str(self.numbering_style),
            "notes": [n.to_dict() for n in self.notes],
            "orphan_callouts": self.orphan_callouts,
            "orphan_definitions": self.orphan_definitions,
        }

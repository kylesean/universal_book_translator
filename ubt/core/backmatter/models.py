"""Data models for book appendix, bibliography, and backmatter layout partitioning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BackmatterKind(StrEnum):
    """Semantic category of backmatter book elements."""

    APPENDIX = "appendix"  # Appendices (A, B, C...)
    BIBLIOGRAPHY = "bibliography"  # References and citations list
    GLOSSARY = "glossary"  # Term definitions and notation dictionary
    INDEX = "index"  # Subject and term concordance index
    AFTERWORD = "afterword"  # Epilogue, afterword, or author notes
    COLOPHON = "colophon"  # CIP metadata and publication colophon
    OTHER_BACKMATTER = "other_backmatter"  # Miscellaneous post-text material


@dataclass(slots=True)
class BackmatterConfig:
    """Publishing configuration for backmatter formatting and numbering."""

    enabled: bool = True
    appendix_numbering: str = "A.1"  # Typst heading numbering pattern for appendices
    appendix_prefix_zh: str = "附录"
    appendix_prefix_en: str = "Appendix"
    auto_pagebreak: bool = True  # Inject weak pagebreak before backmatter sections
    reset_heading_counter: bool = True  # Reset heading counter to 0 for Appendix A
    suppress_backmatter_numbering: bool = True  # Set numbering: none for references/index/afterword

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "appendix_numbering": self.appendix_numbering,
            "appendix_prefix_zh": self.appendix_prefix_zh,
            "appendix_prefix_en": self.appendix_prefix_en,
            "auto_pagebreak": self.auto_pagebreak,
            "reset_heading_counter": self.reset_heading_counter,
            "suppress_backmatter_numbering": self.suppress_backmatter_numbering,
        }


@dataclass(slots=True)
class BackmatterSection:
    """A detected backmatter partition section."""

    section_id: str  # Unique anchor id (e.g. 'bm-app-a', 'bm-bib')
    kind: BackmatterKind
    raw_title: str
    clean_title: str
    letter_index: str = ""  # "A", "B", "C" for appendices, empty for other backmatter
    heading_block_id: str = ""
    block_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "kind": str(self.kind),
            "raw_title": self.raw_title,
            "clean_title": self.clean_title,
            "letter_index": self.letter_index,
            "heading_block_id": self.heading_block_id,
            "total_blocks": len(self.block_ids),
        }


@dataclass(slots=True)
class BackmatterManifest:
    """Document-wide manifest of all backmatter sections and transitions."""

    sections: list[BackmatterSection] = field(default_factory=list)
    first_appendix_block_id: str | None = None
    first_backmatter_block_id: str | None = None

    @property
    def total_sections(self) -> int:
        return len(self.sections)

    @property
    def appendix_count(self) -> int:
        return sum(1 for s in self.sections if s.kind == BackmatterKind.APPENDIX)

    @property
    def has_bibliography(self) -> bool:
        return any(s.kind == BackmatterKind.BIBLIOGRAPHY for s in self.sections)

    @property
    def has_glossary(self) -> bool:
        return any(s.kind == BackmatterKind.GLOSSARY for s in self.sections)

    @property
    def has_index(self) -> bool:
        return any(s.kind == BackmatterKind.INDEX for s in self.sections)

    @property
    def has_afterword(self) -> bool:
        return any(s.kind == BackmatterKind.AFTERWORD for s in self.sections)

    def section_for_heading(self, block_id: str) -> BackmatterSection | None:
        """Find the backmatter section corresponding to a heading block ID."""
        for s in self.sections:
            if s.heading_block_id == block_id:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_sections": self.total_sections,
            "appendix_count": self.appendix_count,
            "has_bibliography": self.has_bibliography,
            "has_glossary": self.has_glossary,
            "has_index": self.has_index,
            "has_afterword": self.has_afterword,
            "first_appendix_block_id": self.first_appendix_block_id,
            "first_backmatter_block_id": self.first_backmatter_block_id,
            "sections": [s.to_dict() for s in self.sections],
        }

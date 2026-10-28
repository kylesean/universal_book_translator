"""Data models for cross-chapter internal anchors and citation links."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ReferenceKind(StrEnum):
    """Types of cross-reference targets in a book or paper."""

    BIBLIOGRAPHY = "bibliography"
    HEADING = "heading"
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    FOOTNOTE = "footnote"


@dataclass(frozen=True, slots=True)
class AnchorTarget:
    """A registered jump target / bookmark in the document."""

    anchor_id: str  # Typst/HTML-safe unique label (e.g. 'bib-1', 'sec-2-1', 'fig-3')
    kind: ReferenceKind
    key: str  # Lookup key (e.g. '1', '2.1', 'figure 3', 'table 2')
    block_id: str
    page: int | None = None
    title: str = ""


@dataclass(frozen=True, slots=True)
class CitationLink:
    """An identified reference or citation in body prose."""

    source_block_id: str
    target_anchor_id: str
    matched_text: str  # Exact text in prose (e.g. '[1]', 'Figure 2', '第 3 章')
    kind: ReferenceKind
    span_start: int
    span_end: int


@dataclass(slots=True)
class CitationTopology:
    """The document's complete internal link graph."""

    anchors: dict[str, AnchorTarget] = field(default_factory=dict)  # anchor_id -> AnchorTarget
    key_to_anchor: dict[tuple[ReferenceKind, str], str] = field(default_factory=dict)
    total_citations_found: int = 0
    resolved_citations_count: int = 0
    links: list[CitationLink] = field(default_factory=list)
    unresolved_mentions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def resolution_rate(self) -> float:
        if self.total_citations_found == 0:
            return 1.0
        return self.resolved_citations_count / self.total_citations_found

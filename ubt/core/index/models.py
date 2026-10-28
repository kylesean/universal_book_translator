"""Data models for publication concordance, bilingual index, and term occurrences."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ubt.core.index.phonetics import get_latin_initial, get_pinyin_initial


@dataclass(frozen=True, slots=True)
class IndexEntryLocation:
    """A specific occurrence of an indexed term in the document."""

    block_id: str
    section_title: str = ""
    section_anchor: str = ""
    page: int | None = None


@dataclass(slots=True)
class IndexEntry:
    """An indexed term with its bilingual translation and document locations."""

    source_term: str
    target_term: str
    category: str = ""
    locations: list[IndexEntryLocation] = field(default_factory=list)
    occurrences_count: int = 1
    primary_anchor: str = ""

    @property
    def source_initial(self) -> str:
        """First letter initial for English/source sorting (A-Z, 0-9, #)."""
        return get_latin_initial(self.source_term)

    @property
    def target_initial(self) -> str:
        """Pinyin or Latin initial for Chinese/target sorting (A-Z, 0-9, #)."""
        return get_pinyin_initial(self.target_term)

    @property
    def display_en_zh(self) -> str:
        """Standard English-first representation: 'Source Term (Target Term)'."""
        if self.target_term and self.target_term.lower() != self.source_term.lower():
            return f"{self.source_term} ({self.target_term})"
        return self.source_term

    @property
    def display_zh_en(self) -> str:
        """Standard Chinese-first representation: 'Target Term (Source Term)'."""
        if self.source_term and self.source_term.lower() != self.target_term.lower():
            return f"{self.target_term} ({self.source_term})"
        return self.target_term or self.source_term


@dataclass(slots=True)
class ConcordanceIndex:
    """Full bilingual concordance index across a book or monograph."""

    entries: list[IndexEntry] = field(default_factory=list)
    doc_id: str = ""
    title: str = ""

    @property
    def total_terms(self) -> int:
        return len(self.entries)

    @property
    def total_occurrences(self) -> int:
        return sum(e.occurrences_count for e in self.entries)

    def grouped_by_source_initial(self) -> dict[str, list[IndexEntry]]:
        """Return entries grouped by English/source initial (A-Z), sorted alphabetically."""
        groups: dict[str, list[IndexEntry]] = defaultdict(list)
        for entry in sorted(self.entries, key=lambda e: (e.source_initial, e.source_term.lower())):
            groups[entry.source_initial].append(entry)
        # Order keys: A-Z, then 0-9, then #
        sorted_keys = sorted(
            groups.keys(),
            key=lambda k: (0 if k.isalpha() else (1 if k == "0-9" else 2), k),
        )
        return {k: groups[k] for k in sorted_keys}

    def grouped_by_target_initial(self) -> dict[str, list[IndexEntry]]:
        """Return entries grouped by Chinese/target initial (Pinyin A-Z), sorted alphabetically."""
        groups: dict[str, list[IndexEntry]] = defaultdict(list)
        for entry in sorted(self.entries, key=lambda e: (e.target_initial, e.target_term.lower())):
            groups[entry.target_initial].append(entry)
        sorted_keys = sorted(
            groups.keys(),
            key=lambda k: (0 if k.isalpha() else (1 if k == "0-9" else 2), k),
        )
        return {k: groups[k] for k in sorted_keys}

    def filter(self, min_occurrences: int = 1, category: str | None = None) -> ConcordanceIndex:
        """Filter entries by minimum frequency or category."""
        filtered = [
            e
            for e in self.entries
            if e.occurrences_count >= min_occurrences
            and (category is None or e.category.lower() == category.lower())
        ]
        return ConcordanceIndex(entries=filtered, doc_id=self.doc_id, title=self.title)

    def top_terms(self, n: int = 20) -> list[IndexEntry]:
        """Return top N most frequent terms."""
        return sorted(self.entries, key=lambda e: -e.occurrences_count)[:n]

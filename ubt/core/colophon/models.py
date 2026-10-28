"""Data models for publication colophons, CIP cataloging, and copyright pages."""

from __future__ import annotations

from dataclasses import dataclass, field

from ubt.core.colophon.isbn import format_isbn, validate_isbn


@dataclass
class ColophonMetadata:
    """Formal publication colophon and CIP cataloging metadata."""

    title: str = "未命名图书"
    subtitle: str = ""
    original_title: str = ""
    authors: list[str] = field(default_factory=list)
    translators: list[str] = field(default_factory=list)
    reviewers: list[str] = field(default_factory=list)
    publisher: str = "开源翻译出版工坊"
    original_publisher: str = ""
    original_pub_year: str = ""
    publication_date: str = "2026-10"
    edition: str = "第 1 版"
    impression: str = "第 1 次印刷"
    isbn: str = "978-7-123-45678-9"
    clc_code: str = "TP311"
    subject_headings: list[str] = field(
        default_factory=lambda: ["程序设计", "人工智能", "自然语言处理"]
    )
    estimated_words: int = 150000
    format_size: str = "16开 (185mm × 260mm)"
    sheets: str = "18.5"
    legal_disclaimer: str = "版权所有 · 侵权必究。印装差错，负责调换。"
    printer: str = "中华商务联合印刷有限公司"
    price: str = "89.00 元"

    @property
    def has_valid_isbn(self) -> bool:
        """True if the ISBN checksum is strictly valid."""
        return validate_isbn(self.isbn)

    @property
    def display_isbn(self) -> str:
        """Normalized hyphenated ISBN representation."""
        return format_isbn(self.isbn)

    @property
    def authors_display(self) -> str:
        """Formatted authors list."""
        if not self.authors:
            return "著"
        return "、".join(self.authors) + " 著"

    @property
    def translators_display(self) -> str:
        """Formatted translators list."""
        if not self.translators:
            return ""
        return "、".join(self.translators) + " 译"

    @property
    def reviewers_display(self) -> str:
        """Formatted reviewers list."""
        if not self.reviewers:
            return ""
        return "、".join(self.reviewers) + " 审校"

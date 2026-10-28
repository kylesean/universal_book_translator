"""Data models for multi-page table continuation styling and header management."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TableContinuationConfig:
    """Configuration for publication-grade repeated table headers and continuation notices."""

    enable_continued_header: bool = True
    enable_next_page_footer: bool = True
    continued_header_text_zh: str = "（续表）"
    continued_header_text_en: str = "(Continued from previous page)"
    next_page_footer_text_zh: str = "（续下页）"
    next_page_footer_text_en: str = "(Continued on next page)"
    min_rows_for_continuation: int = 4
    font_size_pt: float = 7.5
    text_color_luma: int = 100

    def get_continued_header_text(self, lang: str = "zh") -> str:
        """Resolve continuation header text based on target language."""
        if str(lang).lower().startswith("zh"):
            return self.continued_header_text_zh
        return self.continued_header_text_en

    def get_next_page_footer_text(self, lang: str = "zh") -> str:
        """Resolve next page footer text based on target language."""
        if str(lang).lower().startswith("zh"):
            return self.next_page_footer_text_zh
        return self.next_page_footer_text_en

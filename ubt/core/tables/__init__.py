"""Publication-grade multi-page table continuation styling and header repetition."""

from __future__ import annotations

from ubt.core.tables.models import TableContinuationConfig
from ubt.core.tables.styler import (
    generate_html_table_continuation_css,
    generate_typst_table_continuation_elements,
    sanitize_table_uid,
)

__all__ = [
    "TableContinuationConfig",
    "generate_html_table_continuation_css",
    "generate_typst_table_continuation_elements",
    "sanitize_table_uid",
]

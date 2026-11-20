"""Format-agnostic data contracts for UBT v2 surgical in-place document codecs.

Every document format (native PDF, scanned PDF via Typst, EPUB/HTML DOM,
DOCX OOXML, Markdown AST) implements :class:`DocumentCodec` and communicates
with the translation core exclusively through :class:`TranslationUnit` and
lightweight XML-like inline anchors (``<b1>...</b1>``, ``<i2>...</i2>``,
``<a3>...</a3>``, ``<f1/>``, ``<n1/>``, ``<x1/>``).

Zero format-specific libraries may be imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable


class DeliveryMode(StrEnum):
    """How a translated document is assembled back into its native container."""

    MONO = "mono"
    INTERLEAVED = "interleaved"
    SIDE_BY_SIDE = "side_by_side"
    BILINGUAL = "bilingual"
    REFLOW = "reflow"


class UnitRole(StrEnum):
    """Semantic role of a translation unit within its source container."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    TOC_ENTRY = "toc_entry"
    HEADER_FOOTER = "header_footer"


@dataclass(frozen=True, slots=True)
class InlineToken:
    """Metadata binding a lightweight markup anchor back to native container state.

    - ``kind == "pair"`` corresponds to ``<{tag_id}>...</{tag_id}>`` (e.g. bold,
      italic, color, hyperlink, Word ``<w:rPr>``).
    - ``kind == "rigid"`` corresponds to self-closing ``<{tag_id}/>`` (e.g. an
      inline PDF formula Form XObject, Word ``<m:oMath>``, MathML, footnote ref).
    """

    tag_id: str
    kind: Literal["pair", "rigid"]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationUnit:
    """Minimal format-agnostic unit entering the translation core."""

    unit_id: str
    chapter_id: str
    role: UnitRole
    source_marked: str
    source_plain: str
    tokens: tuple[InlineToken, ...] = ()
    translate: bool = True
    page_no: int | None = None
    box_hint: dict[str, float] | None = None
    codec_meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentProfile:
    """Zero-token preflight summary returned by :meth:`DocumentCodec.inspect`."""

    format_name: str
    page_count: int | None = None
    estimated_chars: int = 0
    is_scanned: bool = False
    is_encrypted: bool = False
    has_math: bool = False
    recommended_mode: DeliveryMode = DeliveryMode.MONO
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OverflowItem:
    """Measured layout overflow reported by fixed-layout typesetting (PDF)."""

    unit_id: str
    page_no: int
    overflow_ratio: float
    target_char_budget: int
    applied_font_scale: float
    clipped: bool


@dataclass(frozen=True, slots=True)
class ApplyReport:
    """Deterministic outcome returned by :meth:`DocumentCodec.apply`."""

    output_path: Path
    mode: DeliveryMode
    total_units: int
    patched_units: int
    overflows: tuple[OverflowItem, ...] = ()
    warnings: tuple[str, ...] = ()
    page_count_source: int | None = None
    page_count_output: int | None = None


@runtime_checkable
class DocumentCodec(Protocol):
    """Surgical extract/patch plugin contract for every supported format."""

    supported_suffixes: frozenset[str]
    supports_page_selection: bool

    def inspect(self, source_path: Path) -> DocumentProfile: ...

    def extract(
        self,
        source_path: Path,
        *,
        pages: set[int] | None = None,
    ) -> tuple[DocumentProfile, list[TranslationUnit]]: ...

    def apply(
        self,
        source_path: Path,
        units: list[TranslationUnit],
        translations: dict[str, str],
        *,
        target_lang: str,
        mode: DeliveryMode = DeliveryMode.MONO,
        output_path: Path,
        pages: set[int] | None = None,
    ) -> ApplyReport: ...

    def render_page_preview(
        self,
        source_path: Path,
        page_no: int,
        units: list[TranslationUnit],
        translations: dict[str, str],
        *,
        target_lang: str,
        dpi: int = 144,
    ) -> bytes: ...

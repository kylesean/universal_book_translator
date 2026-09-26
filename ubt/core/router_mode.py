"""Unified-entry router: adaptive execution for short vs long documents.

Single decision point for the "1 page to 1000 pages" generality goal.
The entry (probe + route + shared ledger/TM/report contracts) is unified;
the execution path is adaptive:

- SHORT (<= ``SHORT_CHAIN_MAX_PAGES`` born-digital pages or token budget):
  whole-chapter rewrite + reflow + full visual gate (Codex-style, 可读优先).
- LONG (everything else, always for scans and multi-chapter books): existing 6-stage block pipeline.

Only permissive libraries here (pdf_oxide via short_doc probe, page_profiler);
this module never imports adapters or the LLM router.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal

from ubt.core.ports import classify_pdf_content, probe_pdf_pages

logger = logging.getLogger(__name__)

RouteMode = Literal["short", "long"]


def _resolve_short_max_pages() -> int:
    """Resolve the short-chain page cap from the live config at call time.

    ``SHORT_CHAIN_MAX_PAGES`` in ``layout_policy`` is read from the environment
    at import time, so a programmatic override of ``UBTConfig.short_max_pages``
    never moved the decision, and its ``_env_int`` parse silently accepted
    0/garbage the validated config field rejects. Reading the config here makes
    the runtime decision follow the same validated value every caller uses.
    """
    from ubt.core.config import UBTConfig

    return UBTConfig.from_env().short_max_pages


@dataclass(frozen=True)
class RouteDecision:
    """Why a document takes the short or long chain."""

    mode: RouteMode
    pages: int
    chars: int
    has_scan: bool
    formula_heavy: bool
    reason: str
    chapters: int = 1
    estimated_tokens: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "pages": self.pages,
            "chars": self.chars,
            "has_scan": self.has_scan,
            "formula_heavy": self.formula_heavy,
            "chapters": self.chapters,
            "estimated_tokens": self.estimated_tokens,
            "reason": self.reason,
        }


class _TextExtractor(HTMLParser):
    """Collect visible text from an (X)HTML document, ignoring markup."""

    _SKIP_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        # Skip <script>/<style> bodies: they are not prose, and counting them
        # inflated the token/page estimate and forced short articles onto the
        # long chain.
        if self._skip_depth == 0:
            self._parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(self._parts)


def _strip_markup(raw: str) -> str:
    """Return visible text only.

    Token/page estimates must not count tags, attributes, scripts or styles as
    prose: a 1 KB article wrapped in 20 KB of HTML would estimate as ~5k
    tokens and get forced onto the long chain.
    """
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        # Malformed markup: fall back to a crude tag strip rather than
        # counting the raw bytes (which is the bug this exists to prevent).
        return re.sub(r"<[^>]+>", " ", raw)
    return parser.text


def _probe_non_pdf(path: Path) -> tuple[int, int, int]:
    """Probe a non-PDF document for (estimated_pages, char_count, chapter_count).

    Uses standard library only.
    """
    if not path.is_file():
        return 0, 0, 0
    ext = path.suffix.lower()
    if ext in (".md", ".markdown", ".txt"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            chars = len(text)
            # Any ATX heading level counts, matching the DOCX branch's "any
            # heading style": counting only "# " sent an H2-only multi-chapter
            # document to the short chain.
            headings = sum(
                1 for line in text.splitlines() if re.match(r"^#{1,6}\s", line.strip())
            )
            chapters = max(1, headings)
            estimated_pages = max(1, chars // 1500)
            return estimated_pages, chars, chapters
        except Exception:
            return 1, 0, 1

    if ext in (".html", ".htm"):
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            chars = len(_strip_markup(raw))
            # Any heading level counts (see the Markdown branch above).
            headings = len(re.findall(r"<h[1-6][\s>]", raw, re.IGNORECASE))
            chapters = max(1, headings)
            estimated_pages = max(1, chars // 1500)
            return estimated_pages, chars, chapters
        except Exception:
            return 1, 0, 1

    if ext == ".epub":
        try:
            with zipfile.ZipFile(path, "r") as zf:
                spine_count = 0
                total_chars = 0
                opf_names = [n for n in zf.namelist() if n.endswith(".opf")]
                if opf_names:
                    try:
                        tree = ET.fromstring(zf.read(opf_names[0]))
                        spine_count = len(tree.findall(".//{*}itemref"))
                    except Exception:
                        spine_count = 0
                html_files = [
                    n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))
                ]
                if spine_count <= 0:
                    spine_count = len(html_files)
                for hf in html_files:
                    total_chars += len(_strip_markup(zf.read(hf).decode("utf-8", errors="ignore")))
                chapters = max(1, spine_count)
                estimated_pages = max(1, total_chars // 1500)
                return estimated_pages, total_chars, chapters
        except Exception:
            return 1, 0, 1

    if ext == ".docx":
        try:
            with zipfile.ZipFile(path, "r") as zf:
                if "word/document.xml" in zf.namelist():
                    xml_content = zf.read("word/document.xml")
                    tree = ET.fromstring(xml_content)
                    text_parts = [
                        elem.text for elem in tree.iter() if elem.tag.endswith("}t") and elem.text
                    ]
                    chars = sum(len(t) for t in text_parts)
                    headings = 0
                    for elem in tree.iter():
                        if elem.tag.endswith("}pStyle"):
                            val = elem.attrib.get(
                                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val",
                                "",
                            )
                            if "heading" in val.lower() or "title" in val.lower():
                                headings += 1
                    chapters = max(1, headings)
                    estimated_pages = max(1, chars // 1500)
                    return estimated_pages, chars, chapters
        except Exception:
            return 1, 0, 1

    return 1, 0, 1


def decide(
    pdf_path: Path | str | None = None,
    *,
    doc_path: Path | str | None = None,
    short_max_pages: int | None = None,
    exec_mode: Literal["auto", "short", "long"] = "auto",
) -> RouteDecision:
    """Probe a document (PDF, EPUB, DOCX, MD, HTML) and decide short vs long chain.

    Adaptive short/long routing over whatever features each format exposes
    (not all three are available for every carrier):

    1. Capacity / token budget (estimated tokens <= short_max_pages * 800)
    2. Structural hierarchy (single chapter vs multi-chapter/TOC/spine)
    3. Carrier constraints (born-digital text vs scanned/unreadable pages)

    Flow formats (MD/HTML/EPUB/DOCX) expose chapters and text length; PDF
    exposes page count, text length and scan detection but no cheap chapter
    count. Forced modes win: 'short' forces short; 'long' always returns long.
    """
    raw_path = pdf_path if pdf_path is not None else doc_path
    if raw_path is None:
        raise ValueError("Document path must be provided to router decide()")
    path = Path(raw_path)
    if short_max_pages is None:
        short_max_pages = _resolve_short_max_pages()

    # 1. Non-PDF formats (flow documents: Markdown, HTML, EPUB, DOCX)
    if path.suffix.lower() != ".pdf":
        pages, chars, chapters = _probe_non_pdf(path)
        est_tokens = chars // 4

        if exec_mode == "short":
            return RouteDecision(
                mode="short",
                pages=pages,
                chars=chars,
                has_scan=False,
                formula_heavy=False,
                reason=f"forced short (exec_mode=short, ~{pages}pp)",
                chapters=chapters,
                estimated_tokens=est_tokens,
            )
        if exec_mode == "long":
            return RouteDecision(
                mode="long",
                pages=pages,
                chars=chars,
                has_scan=False,
                formula_heavy=False,
                reason=f"forced long (exec_mode=long, ~{pages}pp)",
                chapters=chapters,
                estimated_tokens=est_tokens,
            )

        # auto mode:
        if chars < 200:
            return RouteDecision(
                mode="long",
                pages=pages,
                chars=chars,
                has_scan=False,
                formula_heavy=False,
                reason=f"empty/unreadable non-PDF (~{chars}ch), long chain",
                chapters=chapters,
                estimated_tokens=est_tokens,
            )

        # Multi-chapter books take long chain (isolated ledger & progressive
        # translation). Only a single-chapter flow document may take the short
        # chain: chapters > 1, matching the guide's ">= 2 chapters => long book".
        if chapters > 1:
            return RouteDecision(
                mode="long",
                pages=pages,
                chars=chars,
                has_scan=False,
                formula_heavy=False,
                reason=f"multi-chapter structured document ({chapters} chapters), long chain book pipeline",
                chapters=chapters,
                estimated_tokens=est_tokens,
            )

        token_budget = short_max_pages * 800
        if pages <= short_max_pages and est_tokens <= token_budget:
            return RouteDecision(
                mode="short",
                pages=pages,
                chars=chars,
                has_scan=False,
                formula_heavy=False,
                reason=f"short born-digital document (~{pages}pp/~{est_tokens} tokens, {chapters} ch)",
                chapters=chapters,
                estimated_tokens=est_tokens,
            )

        return RouteDecision(
            mode="long",
            pages=pages,
            chars=chars,
            has_scan=False,
            formula_heavy=False,
            reason=f"long document (~{pages}pp/~{est_tokens} tokens > limit), long chain",
            chapters=chapters,
            estimated_tokens=est_tokens,
        )

    # 2. PDF documents
    pages, chars = probe_pdf_pages(path)
    est_tokens = chars // 4
    token_budget = short_max_pages * 800
    pdf_chapters = 1 if pages <= short_max_pages else max(1, pages // 20)

    if exec_mode == "short":
        return RouteDecision(
            mode="short",
            pages=pages,
            chars=chars,
            has_scan=False,
            formula_heavy=False,
            reason=f"forced short (exec_mode=short, {pages}pp)",
            chapters=pdf_chapters,
            estimated_tokens=est_tokens,
        )
    if exec_mode == "long":
        return RouteDecision(
            mode="long",
            pages=pages,
            chars=chars,
            has_scan=False,
            formula_heavy=False,
            reason=f"forced long (exec_mode=long, {pages}pp)",
            chapters=pdf_chapters,
            estimated_tokens=est_tokens,
        )

    # auto: unreadable / empty / scan-like always take the long VLM chain.
    if pages <= 0 or chars < 200:
        return RouteDecision(
            mode="long",
            pages=pages,
            chars=chars,
            has_scan=True,
            formula_heavy=False,
            reason=f"unreadable/scan-like ({pages}pp/{chars}ch), long chain VLM",
            chapters=pdf_chapters,
            estimated_tokens=est_tokens,
        )
    has_scan, formula_heavy = classify_pdf_content(path)
    if has_scan:
        return RouteDecision(
            mode="long",
            pages=pages,
            chars=chars,
            has_scan=True,
            formula_heavy=False,
            reason=f"scan page present ({pages}pp), long chain VLM",
            chapters=pdf_chapters,
            estimated_tokens=est_tokens,
        )
    # A PDF can have few pages but a huge text layer; the token budget must be
    # checked too or a 30-page 300k-char dump takes the short single-shot chain.
    if pages <= short_max_pages and est_tokens <= token_budget:
        return RouteDecision(
            mode="short",
            pages=pages,
            chars=chars,
            has_scan=False,
            formula_heavy=formula_heavy,
            reason=(
                f"short born-digital chapter ({pages}pp<={short_max_pages}, "
                f"~{est_tokens}<= {token_budget} tokens)"
            ),
            chapters=pdf_chapters,
            estimated_tokens=est_tokens,
        )
    return RouteDecision(
        mode="long",
        pages=pages,
        chars=chars,
        has_scan=False,
        formula_heavy=formula_heavy,
        reason=(f"long book ({pages}pp>{short_max_pages} or ~{est_tokens} tokens>{token_budget})"),
        chapters=pdf_chapters,
        estimated_tokens=est_tokens,
    )

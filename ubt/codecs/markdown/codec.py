"""AST line-range surgical codec for Markdown / plain text documents.

Uses ``markdown-it-py`` (MIT) to locate exact translatable inline blocks
(headings, paragraphs, list items, table cells, blockquotes) while keeping
YAML frontmatter, fenced code blocks, ``$$...$$`` display math blocks, and
inline code/math/links verbatim via UBT v2 inline anchors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt

from ubt.codecs.protocol import (
    ApplyReport,
    DeliveryMode,
    DocumentProfile,
    InlineToken,
    TranslationUnit,
    UnitRole,
)

_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)
_INLINE_RIGID_RE = re.compile(
    r"(`[^`\n]+`|\$\$[^$]+\$\$|\$(?!\s)[^$\n]+?(?<!\s)\$|!\[[^\]]*\]\([^)]+\))"
)
_INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_TAG_RESTORE_RE = re.compile(r"<(/?)([a-z]\d+)(\s*/)?\s*>")
_PREFIX_RE = re.compile(r"^(\s*(?:>\s*)*(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+)?)")


def _encode_markdown_inline(text: str) -> tuple[str, str, list[InlineToken]]:
    tokens: list[InlineToken] = []
    seq = 0

    def _sub_rigid(m: re.Match[str]) -> str:
        nonlocal seq
        seq += 1
        tid = f"x{seq}"
        tokens.append(InlineToken(tag_id=tid, kind="rigid", payload={"raw": m.group(0)}))
        return f"<{tid}/>"

    step1 = _INLINE_RIGID_RE.sub(_sub_rigid, text)

    def _sub_link(m: re.Match[str]) -> str:
        nonlocal seq
        seq += 1
        tid = f"a{seq}"
        tokens.append(InlineToken(tag_id=tid, kind="pair", payload={"href": m.group(2)}))
        return f"<{tid}>{m.group(1)}</{tid}>"

    marked = _INLINE_LINK_RE.sub(_sub_link, step1)
    plain = _TAG_RESTORE_RE.sub("", marked)
    return marked, plain, tokens


def _decode_markdown_inline(marked: str, tokens: tuple[InlineToken, ...]) -> str:
    tok_map = {t.tag_id: t for t in tokens}
    out: list[str] = []
    stack: list[tuple[str, str]] = []
    pos = 0
    for m in _TAG_RESTORE_RE.finditer(marked):
        out.append(marked[pos : m.start()])
        pos = m.end()
        is_close = bool(m.group(1))
        tid = m.group(2)
        is_self = bool(m.group(3))
        tok = tok_map.get(tid)
        if tok is None:
            continue
        if is_self or tok.kind == "rigid":
            out.append(str(tok.payload.get("raw", "")))
        elif not is_close:
            out.append("[")
            stack.append((tid, str(tok.payload.get("href", ""))))
        elif stack and stack[-1][0] == tid:
            _, href = stack.pop()
            out.append(f"]({href})")
    out.append(marked[pos:])
    while stack:
        _, href = stack.pop()
        out.append(f"]({href})")
    return "".join(out)


def _split_frontmatter(raw: str) -> tuple[str, str, int]:
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return "", raw, 0
    fm = m.group(0)
    return fm, raw[m.end() :], fm.count("\n")


def _extract_spans(body: str) -> list[tuple[int, int, UnitRole, str]]:
    """Return ``(start_line, end_line_exclusive, role, prefix)`` for translatable blocks."""
    md = MarkdownIt("commonmark").enable("table")
    lines = body.splitlines()
    tokens = md.parse(body)
    spans: list[tuple[int, int, UnitRole, str]] = []

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.type == "inline" and tok.map is not None:
            start, end = tok.map
            # Skip standalone $$...$$ display math blocks parsed as paragraphs.
            block_src = "\n".join(lines[start:end]).strip()
            if block_src.startswith("$$") and block_src.endswith("$$"):
                i += 1
                continue
            prev_type = tokens[i - 1].type if i > 0 else ""
            if prev_type == "heading_open":
                role = UnitRole.HEADING
            elif prev_type in {"td_open", "th_open"}:
                role = UnitRole.TABLE_CELL
            elif i >= 2 and tokens[i - 2].type == "list_item_open":
                role = UnitRole.LIST_ITEM
            else:
                role = UnitRole.PARAGRAPH
            first_line = lines[start] if start < len(lines) else ""
            pref_m = _PREFIX_RE.match(first_line)
            prefix = pref_m.group(1) if pref_m else ""
            if role != UnitRole.TABLE_CELL:
                spans.append((start, end, role, prefix))
        i += 1

    # Deduplicate overlapping maps (e.g. tight list items) preserving order.
    dedup: list[tuple[int, int, UnitRole, str]] = []
    seen_ranges: set[tuple[int, int]] = set()
    for sp in spans:
        key = (sp[0], sp[1])
        if key not in seen_ranges:
            seen_ranges.add(key)
            dedup.append(sp)
    return dedup


@dataclass(slots=True)
class MarkdownCodec:
    """AST line-range surgical codec for ``.md``, ``.markdown``, and ``.txt``."""

    supported_suffixes: frozenset[str] = frozenset({".md", ".markdown", ".txt"})
    supports_page_selection: bool = False

    def inspect(self, source_path: Path) -> DocumentProfile:
        raw = source_path.read_text(encoding="utf-8", errors="replace")
        return DocumentProfile(
            format_name="markdown",
            page_count=1,
            estimated_chars=len(raw),
            has_math="$$" in raw or "$" in raw,
            recommended_mode=DeliveryMode.MONO,
        )

    def extract(
        self,
        source_path: Path,
        *,
        pages: set[int] | None = None,
    ) -> tuple[DocumentProfile, list[TranslationUnit]]:
        del pages
        profile = self.inspect(source_path)
        raw = source_path.read_text(encoding="utf-8", errors="replace")
        _, body, fm_lines = _split_frontmatter(raw)
        lines = body.splitlines()
        spans = _extract_spans(body)

        units: list[TranslationUnit] = []
        chap_idx = 1
        for idx, (start, end, role, prefix) in enumerate(spans, start=1):
            raw_chunk = "\n".join(lines[start:end])
            # Strip leading markdown syntax prefix from first line before translation.
            stripped = (
                raw_chunk[len(prefix) :] if prefix and raw_chunk.startswith(prefix) else raw_chunk
            )
            marked, plain, tokens = _encode_markdown_inline(stripped.strip())
            if not plain.strip():
                continue
            if role == UnitRole.HEADING and prefix.lstrip().startswith("# ") and units:
                chap_idx += 1
            units.append(
                TranslationUnit(
                    unit_id=f"md_b{idx:05d}",
                    chapter_id=f"ch{chap_idx:03d}",
                    role=role,
                    source_marked=marked,
                    source_plain=plain,
                    tokens=tuple(tokens),
                    translate=True,
                    codec_meta={
                        "start_line": start,
                        "end_line": end,
                        "fm_lines": fm_lines,
                        "prefix": prefix,
                    },
                )
            )
        return profile, units

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
    ) -> ApplyReport:
        del target_lang, pages
        raw = source_path.read_text(encoding="utf-8", errors="replace")
        fm, body, _ = _split_frontmatter(raw)
        lines = body.splitlines()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply replacements bottom-to-top so line indices never shift.
        replacements: list[tuple[int, int, str]] = []
        patched = 0
        for unit in units:
            translated = translations.get(unit.unit_id)
            if translated is None or not translated.strip():
                continue
            start = int(unit.codec_meta["start_line"])
            end = int(unit.codec_meta["end_line"])
            prefix = str(unit.codec_meta.get("prefix", ""))
            restored = _decode_markdown_inline(translated.strip(), unit.tokens)
            rendered_line = (
                f"{prefix}{restored}"
                if prefix and not restored.startswith(prefix.strip())
                else restored
            )
            if mode == DeliveryMode.BILINGUAL:
                orig_block = "\n".join(lines[start:end])
                combined = f"{orig_block}\n\n{rendered_line}"
                replacements.append((start, end, combined))
            else:
                replacements.append((start, end, rendered_line))
            patched += 1

        for start, end, new_text in sorted(replacements, key=lambda r: r[0], reverse=True):
            lines[start:end] = new_text.splitlines()

        out_text = fm + "\n".join(lines) + ("\n" if raw.endswith("\n") else "")
        output_path.write_text(out_text, encoding="utf-8")

        return ApplyReport(
            output_path=output_path,
            mode=mode,
            total_units=len(units),
            patched_units=patched,
        )

    def render_page_preview(
        self,
        source_path: Path,
        page_no: int,
        units: list[TranslationUnit],
        translations: dict[str, str],
        *,
        target_lang: str,
        dpi: int = 144,
    ) -> bytes:
        raise NotImplementedError("MarkdownCodec outputs plain UTF-8 Markdown.")

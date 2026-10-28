"""Multi-page table stitcher for continuous cross-page tables.

Detects and stitches tables that span across page boundaries in fixed-layout
documents (e.g. PDF) or fragmented ASTs:
1. Column count and geometric alignment verification;
2. Fingerprint matching for repeated headers, continuation labels (e.g.
   "Table 1 (continued)", "表 1（续）"), or headless data continuations;
3. Fusion of fragmented data rows into a single continuous-stream Table element;
4. Provenance tracking with fused block IDs and multi-page bounding boxes so
   downstream renderers (Typst) emit a single `#table` with native
   `table.header(repeat: true)` across pages.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock
from ubt.model.ast import RegionKind
from ubt.render.table_compiler import _split_row, parse_markdown_table

logger = logging.getLogger(__name__)

#: Threshold for considering two header rows identical or near-identical.
HEADER_SIMILARITY_THRESHOLD: float = 0.75

#: Normalized continuation label patterns across English and CJK academic/book conventions.
CONTINUATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:table|tab\.|表)\s*[\w\.\-–—:]*\s*(?:continued|cont\.?|续|续表|接上页|接前页)",
        re.IGNORECASE,
    ),
    re.compile(r"[（\(\[\{]\s*(?:continued|cont\.?|续|续表|接上页)\s*[）\)\]\}]", re.IGNORECASE),
    re.compile(r"^(?:continued|cont\.?|续|续表|接上页|接前页)[\s\-–—:]*$", re.IGNORECASE),
    re.compile(r"^(?:table|tab\.|表)\s*[\w\.\-–—]+\s*\(cont(?:'d|inued|\.)?\)$", re.IGNORECASE),
)

#: Indicators that a cell contains numerical data, metrics, units, or codes rather than a column label.
_DATA_VALUE_RE = re.compile(
    r"^[\s$€¥£]*[-+]?\d+(?:[.,]\d+)*(?:\s*(?:%|ms|s|m|km|kg|g|kb|mb|gb|tb|hz|khz|mhz|ghz|db|pt|px|em|deg|°c|°f|k))?[\s$€¥£]*$",
    re.IGNORECASE,
)

_NUMERIC_CHARS_RE = re.compile(r"\d")


def is_continuation_label(text: str) -> bool:
    """Check if text matches standard table continuation annotations."""
    stripped = text.strip()
    if not stripped or len(stripped) > 100:
        return False
    return any(pat.search(stripped) for pat in CONTINUATION_PATTERNS)


def normalize_cell(text: str) -> str:
    """Normalize a table cell for robust comparison."""
    # Strip markdown bold, italics, backticks, math delimiters, whitespace
    cleaned = re.sub(r"[\*_`$]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def header_row_similarity(headers_a: Sequence[str], headers_b: Sequence[str]) -> float:
    """Compute structural similarity between two header rows (0.0 ~ 1.0)."""
    if not headers_a or not headers_b:
        return 0.0
    if len(headers_a) != len(headers_b):
        return 0.0

    scores: list[float] = []
    for a, b in zip(headers_a, headers_b, strict=True):
        na = normalize_cell(a)
        nb = normalize_cell(b)
        if (na == nb) or (not na and not nb):
            scores.append(1.0)
        elif not na or not nb:
            scores.append(0.0)
        else:
            matcher = difflib.SequenceMatcher(None, na, nb)
            scores.append(matcher.ratio())

    return sum(scores) / len(scores) if scores else 0.0


def is_data_like_row(cells: Sequence[str]) -> bool:
    """True if row cells appear to be data values (numbers, percentages, codes) rather than headers."""
    if not cells:
        return False
    data_count = 0
    total_non_empty = 0
    for cell in cells:
        stripped = cell.strip()
        if not stripped:
            continue
        total_non_empty += 1
        if _DATA_VALUE_RE.match(stripped) or len(_NUMERIC_CHARS_RE.findall(stripped)) >= 2:
            data_count += 1
    if total_non_empty == 0:
        return False
    return (data_count / total_non_empty) >= 0.5


@dataclass(frozen=True, slots=True)
class TableFingerprint:
    """Geometric and textual signature of a table block."""

    block_id: str
    page: int | None
    bbox: BoundingBox | None
    headers: tuple[str, ...]
    column_count: int
    row_count: int
    alignments: tuple[str, ...]
    has_continuation_label: bool
    continuation_label: str
    is_headless_candidate: bool


def parse_table_fingerprint(block: IRBlock) -> TableFingerprint | None:
    """Extract a structural fingerprint from an IRBlock."""
    if block.block_type is not BlockType.TABLE:
        return None
    source = (block.source_text or "").strip()
    if not source:
        return None

    parsed = parse_markdown_table(source, escape=False)
    if parsed is None:
        return None
    headers, rows = parsed
    cols = len(headers)
    if cols == 0:
        return None

    # Extract alignments from separator row
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    alignments: list[str] = ["left"] * cols
    for idx in range(1, min(3, len(table_lines))):
        sep_cells = [c.strip() for c in _split_row(table_lines[idx])]
        if sep_cells and all(re.match(r"^:?-+:?$", c) for c in sep_cells):
            alignments = []
            for c in sep_cells:
                if c.startswith(":") and c.endswith(":"):
                    alignments.append("center")
                elif c.endswith(":"):
                    alignments.append("right")
                else:
                    alignments.append("left")
            if len(alignments) < cols:
                alignments.extend(["left"] * (cols - len(alignments)))
            elif len(alignments) > cols:
                alignments = alignments[:cols]
            break

    # Check for continuation label
    has_cont = False
    cont_label = ""
    # Check table first line or header
    if is_continuation_label(lines[0]) or is_continuation_label(" ".join(headers)):
        has_cont = True
        cont_label = lines[0] if is_continuation_label(lines[0]) else " ".join(headers)

    is_headless = is_data_like_row(headers)

    page = block.bbox.page if block.bbox is not None else None
    if block.provenance and block.provenance.physical_boxes:
        recorded_pages = [
            b["page"]
            for b in block.provenance.physical_boxes
            if isinstance(b, dict) and "page" in b and b["page"] is not None
        ]
        if recorded_pages:
            page = max(recorded_pages)

    return TableFingerprint(
        block_id=block.id,
        page=page,
        bbox=block.bbox,
        headers=tuple(headers),
        column_count=cols,
        row_count=len(rows),
        alignments=tuple(alignments),
        has_continuation_label=has_cont,
        continuation_label=cont_label,
        is_headless_candidate=is_headless,
    )


def is_permissible_intervening_block(block: IRBlock) -> bool:
    """True if block between tables does not break multi-page table continuity."""
    # Running headers/footers/page numbers
    if block.region in (RegionKind.HEADER, RegionKind.FOOTER):
        return True
    # Headings and code blocks strictly break continuity
    if block.block_type in (BlockType.HEADING, BlockType.CODE, BlockType.TABLE):
        return False
    txt = (block.source_text or "").strip()
    if not txt:
        return True
    # Continuation captions are permissible
    if is_continuation_label(txt):
        return True
    # Very short debris / footnote reference (< 60 chars) is permissible
    return bool(len(txt) <= 60 and ("\n" not in txt))


def can_stitch_tables(
    table_a: IRBlock,
    table_b: IRBlock,
    intervening: Sequence[IRBlock] = (),
) -> tuple[bool, str]:
    """Determine whether table_b can be stitched onto the tail of table_a."""
    if table_a.block_type is not BlockType.TABLE or table_b.block_type is not BlockType.TABLE:
        return False, "not_both_tables"

    fp_a = parse_table_fingerprint(table_a)
    fp_b = parse_table_fingerprint(table_b)
    if fp_a is None or fp_b is None:
        return False, "unparseable_table_structure"

    # 1. Column count must agree
    if fp_a.column_count != fp_b.column_count:
        return False, f"column_count_mismatch: {fp_a.column_count} vs {fp_b.column_count}"

    # 2. Check intervening blocks
    for blk in intervening:
        if not is_permissible_intervening_block(blk):
            return False, f"blocking_intervening_element: {blk.block_type} ({blk.id})"

    # 3. Page adjacency check if bboxes are present
    if fp_a.page is not None and fp_b.page is not None:
        page_diff = fp_b.page - fp_a.page
        if page_diff < 0:
            return False, "reverse_page_order"
        if page_diff > 1:
            return False, f"non_adjacent_pages: {fp_a.page} to {fp_b.page}"

        # If on the same page, must be dual-column continuity (left column to right column)
        if page_diff == 0 and fp_a.bbox and fp_b.bbox and fp_b.bbox.x0 <= fp_a.bbox.x0:
            return False, "same_page_not_column_shift"

    # 4. Fingerprint matching
    # Case A: Explicit continuation label in B or intervening caption
    if fp_b.has_continuation_label:
        return True, "continuation_label"

    intervening_has_cont = any(
        is_continuation_label((b.source_text or "").strip()) for b in intervening
    )
    if intervening_has_cont:
        return True, "intervening_continuation_label"

    # Case B: Repeated header row across page boundary
    sim = header_row_similarity(fp_a.headers, fp_b.headers)
    if sim >= HEADER_SIMILARITY_THRESHOLD:
        return True, f"matched_header (sim={sim:.2f})"

    # Case C: Headless continuation (table_b starts directly with data rows)
    # If table_b's header row is data-like and table_a's header row is NOT data-like,
    # or pages are directly adjacent across boundary
    if fp_b.is_headless_candidate and not fp_a.is_headless_candidate:
        return True, "headless_continuation"

    # Case D: If pages are strictly adjacent (P -> P+1) and column count matches,
    # and no substantive text separates them, and table B has no distinct header
    if (
        fp_a.page is not None
        and fp_b.page is not None
        and (fp_b.page == fp_a.page + 1)
        and fp_a.alignments == fp_b.alignments
    ):
        return True, "adjacent_page_aligned_flow"

    return False, "no_continuity_fingerprint_match"


def format_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    alignments: Sequence[str] | None = None,
) -> str:
    """Format headers and rows into clean, well-delimited Markdown table text."""
    cols = len(headers)
    if cols == 0:
        return ""

    if alignments is None or len(alignments) != cols:
        align_specs = ["---"] * cols
    else:
        align_specs = []
        for a in alignments:
            if a == "center":
                align_specs.append(":---:")
            elif a == "right":
                align_specs.append("---:")
            else:
                align_specs.append(":---")

    lines: list[str] = []
    # Header row
    lines.append("| " + " | ".join(h.strip() for h in headers) + " |")
    # Separator row
    lines.append("| " + " | ".join(align_specs) + " |")
    # Data rows
    for row in rows:
        padded = list(row) + [""] * (cols - len(row))
        lines.append("| " + " | ".join(c.strip() for c in padded[:cols]) + " |")

    return "\n".join(lines)


def fuse_table_markups(
    markup_a: str,
    markup_b: str,
    match_type: str,
) -> str:
    """Fuse markup_b's rows into markup_a, eliminating duplicate headers."""
    parsed_a = parse_markdown_table(markup_a, escape=False)
    parsed_b = parse_markdown_table(markup_b, escape=False)
    if parsed_a is None:
        return markup_b
    if parsed_b is None:
        return markup_a

    headers_a, rows_a = parsed_a
    headers_b, rows_b = parsed_b
    cols = len(headers_a)

    new_rows: list[list[str]] = list(rows_a)

    # Determine whether headers_b is duplicate or data
    if match_type.startswith("matched_header") or match_type == "continuation_label":
        # Discard headers_b and separator, append rows_b
        for r in rows_b:
            new_rows.append(r[:cols] if len(r) >= cols else list(r) + [""] * (cols - len(r)))
    elif match_type == "headless_continuation":
        # headers_b is actually the first data row of the continuation fragment
        padded_head = (
            list(headers_b[:cols])
            if len(headers_b) >= cols
            else list(headers_b) + [""] * (cols - len(headers_b))
        )
        new_rows.append(padded_head)
        for r in rows_b:
            new_rows.append(r[:cols] if len(r) >= cols else list(r) + [""] * (cols - len(r)))
    else:
        # Default: check if headers_b looks like duplicate header
        if header_row_similarity(headers_a, headers_b) >= 0.5:
            for r in rows_b:
                new_rows.append(r[:cols] if len(r) >= cols else list(r) + [""] * (cols - len(r)))
        else:
            new_rows.append(list(headers_b) + [""] * (cols - len(headers_b)))
            for r in rows_b:
                new_rows.append(r[:cols] if len(r) >= cols else list(r) + [""] * (cols - len(r)))

    return format_markdown_table(headers_a, new_rows)


@dataclass(slots=True)
class TableStitchSummary:
    """Summary of table stitching operations across a document or chapter."""

    total_tables_inspected: int = 0
    tables_stitched_count: int = 0
    blocks_absorbed_count: int = 0
    stitched_pairs: list[dict[str, Any]] = field(default_factory=list)


def stitch_multipage_tables(
    blocks: Sequence[IRBlock],
    *,
    drop_absorbed: bool = True,
    summary_out: TableStitchSummary | None = None,
) -> list[IRBlock]:
    """Scan an IRBlock stream and fuse multi-page tables across page boundaries.

    When drop_absorbed=True (default for ingest), secondary table blocks and absorbed
    continuation labels are pruned from the returned sequence, while the primary table
    block receives the unified markup and records fused block IDs in its provenance.
    """
    if len(blocks) < 2:
        return list(blocks)

    out: list[IRBlock] = []
    absorbed_ids: set[str] = set()
    n = len(blocks)
    i = 0

    while i < n:
        cur = blocks[i]
        if cur.id in absorbed_ids:
            i += 1
            continue

        if cur.block_type is not BlockType.TABLE:
            out.append(cur)
            i += 1
            continue

        if summary_out is not None:
            summary_out.total_tables_inspected += 1

        # We have a candidate primary table `cur`. Search forward for continuation tables.
        j = i + 1
        intervening: list[IRBlock] = []

        while j < n:
            nxt = blocks[j]
            if nxt.id in absorbed_ids:
                j += 1
                continue

            # Non-table permissible intervening elements
            if nxt.block_type is not BlockType.TABLE:
                if is_permissible_intervening_block(nxt):
                    intervening.append(nxt)
                    j += 1
                    continue
                else:
                    # Broken continuity by non-permissible element (heading, body prose, etc.)
                    break

            # nxt is a TABLE! Check if it stitches to cur
            can_stitch, reason = can_stitch_tables(cur, nxt, intervening)
            if can_stitch:
                logger.info(
                    "Stitching table block %s on page %s with %s on page %s: %s",
                    cur.id,
                    cur.bbox.page if cur.bbox else "?",
                    nxt.id,
                    nxt.bbox.page if nxt.bbox else "?",
                    reason,
                )
                # 1. Fuse source markup
                fused_src = fuse_table_markups(
                    cur.source_text or "",
                    nxt.source_text or "",
                    reason,
                )
                cur.set_source_text(fused_src)

                # 2. Fuse target text if both are already present
                if cur.target_text and nxt.target_text:
                    fused_tgt = fuse_table_markups(
                        cur.target_text,
                        nxt.target_text,
                        reason,
                    )
                    cur.target_text = fused_tgt

                # 3. Update provenance
                cur.provenance.fused_block_ids.append(nxt.id)
                if nxt.provenance.fused_block_ids:
                    cur.provenance.fused_block_ids.extend(nxt.provenance.fused_block_ids)

                # Record multi-page physical boxes
                if not cur.provenance.physical_boxes and cur.bbox:
                    cur.provenance.physical_boxes.append(
                        {
                            "page": cur.bbox.page,
                            "x0": cur.bbox.x0,
                            "y0": cur.bbox.y0,
                            "x1": cur.bbox.x1,
                            "y1": cur.bbox.y1,
                        }
                    )
                if nxt.bbox:
                    cur.provenance.physical_boxes.append(
                        {
                            "page": nxt.bbox.page,
                            "x0": nxt.bbox.x0,
                            "y0": nxt.bbox.y0,
                            "x1": nxt.bbox.x1,
                            "y1": nxt.bbox.y1,
                        }
                    )
                if nxt.provenance.physical_boxes:
                    cur.provenance.physical_boxes.extend(nxt.provenance.physical_boxes)

                # 4. Mark nxt as absorbed
                nxt.skip_translate = True
                nxt.policy_translate = False
                nxt.policy_reason = f"fused_into_multipage_table:{cur.id}"
                nxt.status = BlockStatus.MTQE_PASSED
                absorbed_ids.add(nxt.id)

                # Mark any intervening continuation captions as absorbed
                for inv in intervening:
                    txt = (inv.source_text or "").strip()
                    if is_continuation_label(txt):
                        inv.skip_translate = True
                        inv.policy_translate = False
                        inv.policy_reason = f"fused_continuation_caption:{cur.id}"
                        inv.status = BlockStatus.MTQE_PASSED
                        absorbed_ids.add(inv.id)

                if summary_out is not None:
                    summary_out.tables_stitched_count += 1
                    summary_out.blocks_absorbed_count += 1
                    summary_out.stitched_pairs.append(
                        {
                            "primary_id": cur.id,
                            "secondary_id": nxt.id,
                            "reason": reason,
                            "primary_page": cur.bbox.page if cur.bbox else None,
                            "secondary_page": nxt.bbox.page if nxt.bbox else None,
                        }
                    )

                # Clear intervening list and continue forward in case of a 3rd table page!
                intervening = []
                j += 1
            else:
                # nxt is a table but cannot stitch to cur; stop searching for cur
                break

        out.append(cur)
        i += 1

    if drop_absorbed:
        return [b for b in out if b.id not in absorbed_ids]
    return out

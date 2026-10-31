"""Table parsing, cell-count verification, and the in-place vs. appendix decision.

A translated table is safe to typeset in the body only while its geometry still holds:
merged cells, six-plus columns, or prose-sized cells will not fit one column's measure
no matter what the renderer does to them, and a clipped table in the body is worse than
a clean one in the appendix. The complexity probe makes that call, and
``verify_table_reconstruction`` guards the other half — that the translation neither
gained nor lost a cell on the way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ubt.render.typst.escape import typst_escape


@dataclass(frozen=True, slots=True)
class TableComplexity:
    """Assessment of table geometry and structural complexity."""

    is_complex: bool
    reason: str
    rows: int
    cols: int
    suggests_landscape: bool = False
    contingency: str = "none"


def _split_row(line: str) -> list[str]:
    r"""Split one Markdown table row on its real field separators.

    A ``|`` is content — not a separator — when it is escaped (``\|``, the LaTeX
    double-bar norm ``\left\| x \right\|``) or when it sits inside inline math
    (``$a | b$``). A bare ``|`` elsewhere separates fields. The leading and
    trailing boundary ``|`` of the row are dropped, as before.

    Splitting naively on every ``|`` cut ``$\left\| \mu - \nu \right\|$`` into
    three cells, so the renderer emitted ``[\$\\left\\], [\\tilde{...}], [\$]``
    — LaTeX residue split across cells and a table that no longer compiles
    (measured on ``ddpm`` ``pdf_main#b0053``).
    """
    text = line.strip()
    cells: list[str] = []
    current: list[str] = []
    in_math = False
    index = 0
    while index < len(text):
        ch = text[index]
        if ch == "\\" and index + 1 < len(text):
            # Keep the escape and its target together; `\|` is a literal bar.
            current.append(text[index : index + 2])
            index += 2
            continue
        if ch == "$":
            in_math = not in_math
        elif ch == "|" and not in_math:
            cells.append("".join(current))
            current = []
            index += 1
            continue
        current.append(ch)
        index += 1
    cells.append("".join(current))
    # Drop the boundary fields the surrounding `|` produced.
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return cells


def parse_markdown_table(
    markup: str, *, escape: bool = True
) -> tuple[list[str], list[list[str]]] | None:
    """Parse a Markdown table into (headers, rows).

    Returns None if markup does not look like a recognizable Markdown table.

    ``escape=True`` (the default) Typst-escapes every cell, which is what a
    caller that interpolates cells **raw** into ``[...]`` needs. A caller that
    runs each cell through the prose renderer instead must pass
    ``escape=False``: that renderer owns escaping *and* math transpilation, so a
    pre-escaped ``\\$`` would be escaped a second time into ``\\\\$`` — inside a
    content block ``\\\\`` is a line break, so the cell's math delimiter never
    closes and the compile fails.

    Rows shorter than the header are padded with empty cells and rows longer
    than it are truncated, so the grid the renderer emits is always
    rectangular. That normalization is lossy ON PURPOSE for rendering, which
    is why :func:`verify_table_reconstruction` measures raw cell counts
    (:func:`count_table_cells`) instead of the padded result: a translated row
    that gained a cell must be a corrupt reconstruction, not silently
    shortened.
    """
    lines = [line.strip() for line in markup.strip().splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        return None

    header_line = table_lines[0]
    sep_idx = -1
    for idx in range(1, min(3, len(table_lines))):
        cells = [c.strip() for c in _split_row(table_lines[idx])]
        if cells and all(re.match(r"^:?-+:?$", c) for c in cells):
            sep_idx = idx
            break

    if sep_idx == -1:
        return None

    def _cell(value: str) -> str:
        stripped = value.strip()
        return typst_escape(stripped) if escape else stripped

    headers = [_cell(c) for c in _split_row(header_line)]
    cols = len(headers)
    if cols == 0:
        return None

    rows: list[list[str]] = []
    for line in table_lines[sep_idx + 1 :]:
        cells = [_cell(c) for c in _split_row(line)]
        if len(cells) < cols:
            cells.extend([""] * (cols - len(cells)))
        elif len(cells) > cols:
            cells = cells[:cols]
        rows.append(cells)

    return headers, rows


def count_table_cells(markup: str) -> tuple[int, list[int]] | None:
    """Raw cell counts of a Markdown grid: (header cell count, per-row counts).

    The verification oracle: :func:`parse_markdown_table` pads/truncates rows
    into a rectangular grid for rendering, so comparing its OUTPUT cannot see
    a translated row that gained or lost a cell — the old shape check compared
    two already-normalized grids and blessed a silently truncated row as
    ``verified``. Counting the raw split instead sees the true shape.
    """
    lines = [line.strip() for line in markup.strip().splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        return None

    header_cells = len(_split_row(table_lines[0]))
    sep_idx = -1
    for idx in range(1, min(3, len(table_lines))):
        cells = [c.strip() for c in _split_row(table_lines[idx])]
        if cells and all(re.match(r"^:?-+:?$", c) for c in cells):
            sep_idx = idx
            break
    if sep_idx == -1 or header_cells == 0:
        return None
    row_cells = [len(_split_row(line)) for line in table_lines[sep_idx + 1 :]]
    return header_cells, row_cells


def normalize_table_to_source(source_markup: str, target_markup: str) -> str:
    """Re-shape the translated grid to the source grid's row widths, content kept.

    A model translating a dense grid sometimes re-chunks one source cell into
    several target cells: the ATC'25 replay's API table carried each row's API
    signatures in one cell, and the translation came back with the descriptions
    split into their own cells (a 2-column row delivered as 6). The renderer
    then lays a ragged grid under a 2-column spec and the reconstruction check
    reports a shattered table.

    The repair keeps every character: cells beyond the source row's width are
    merged back into that row's last cell (CJK text joins with ``；``, Latin
    with ``; ``), and anything the source grid does not define -- a short row
    whose column the translation *dropped*, rows beyond the source's set --
    passes through untouched so the shape check still reports the loss. Applied
    to the delivered text itself, so the render, the HTML/EPUB companions and
    the shape verification all see one repaired grid.
    """
    source_counts = count_table_cells(source_markup)
    if source_counts is None:
        return target_markup
    source_header_cells, source_row_cells = source_counts
    lines = target_markup.splitlines()
    table_indexes = [
        i
        for i, line in enumerate(lines)
        if line.strip().startswith("|") and line.strip().endswith("|")
    ]
    if len(table_indexes) < 2:
        return target_markup

    def _is_separator(line: str) -> bool:
        cells = [c.strip() for c in _split_row(line.strip())]
        return bool(cells) and all(re.match(r"^:?-+:?$", c) for c in cells)

    def _merge(cells: list[str], width: int) -> list[str]:
        if width <= 0:
            return cells
        if len(cells) > width:
            head, tail = cells[: width - 1], cells[width - 1 :]
            pieces = [piece for piece in tail if piece.strip()]
            if not pieces:
                return head + [""]
            # The target's script picks the glue: a CJK deliverable reads
            # full-width semicolons, Latin reads "; ".
            joined_text = "".join(pieces)
            glue = (
                "；" if any("＀" <= ch <= "﹏" or "一" <= ch <= "鿿" for ch in joined_text) else "; "
            )
            return head + [glue.join(piece.strip() for piece in pieces)]
        return cells + [""] * (width - len(cells))

    body_position = -1  # -1 = header not yet passed
    repaired: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if idx not in table_indexes:
            repaired.append(line)
            continue
        if _is_separator(stripped):
            repaired.append(line)
            continue
        cells = _split_row(stripped)
        desired: int | None
        if body_position == -1:
            desired = source_header_cells
            body_position = 0
        else:
            desired = (
                source_row_cells[body_position] if body_position < len(source_row_cells) else None
            )
            body_position += 1
        # Only an OVERFILLED row is repaired, and the repair is lossless:
        # merging cannot drop content. A short row lost a column at
        # translation time -- padding it would bless real data loss, so it
        # ships unchanged and the shape check reports the shortfall.
        if desired is None or len(cells) <= desired:
            repaired.append(line)
            continue
        merged = _merge(cells, desired)
        lead = stripped[: len(stripped) - len(stripped.lstrip("|"))]
        trail = stripped[len(stripped.rstrip("|")) :]
        repaired.append(f"{lead} " + " | ".join(cell.strip() for cell in merged) + f" {trail}")
    return "\n".join(repaired)


@dataclass(frozen=True, slots=True)
class TableReconstruction:
    """The structural outcome of rebuilding a table grid for the page.

    ``ok`` is the *verified* verdict the delivery contract needs to record a
    non-text asset as ``RECONSTRUCTED``: the rebuild kept the source grid's
    shape (header count and row count) so the table is a real, interpreted
    reconstruction rather than an opaque crop. ``ok=False`` with a non-empty
    ``reason`` is a corrupt reconstruction (a shattered grid), which the
    contract reports as an ERROR.
    """

    ok: bool
    reason: str = ""
    rows: int = 0
    cols: int = 0


def verify_table_reconstruction(source_markup: str, target_markup: str) -> TableReconstruction:
    """Verify that ``target_markup`` rebuilt ``source_markup`` without shattering.

    The rendered table is a *reconstruction*: the source grid is parsed and
    re-emitted as a Typst ``#table``, so the only thing worth proving is that the
    shape survived — a translated grid that dropped or gained a column renders as
    a broken frame, and one that lost a row silently drops a line of data.

    The comparison uses raw cell counts (:func:`count_table_cells`), not the
    padded grids :func:`parse_markdown_table` renders from: both being
    rectangular by construction, their shapes always agreed and a row the
    model overfilled was silently truncated to the header width *and* blessed
    ``verified`` (measured: ``| A | B | C |`` rebuilt as ``| 一 | 二 | 三 | 四 |``
    reported ok=True while dropping the fourth cell). Shape now means the RAW
    shape.

    Returns ``ok=False`` (a *corrupt* reconstruction) when the source grid is a
    real table but the target's shape disagrees with it. A source grid that is
    not a parseable pipe table is not a reconstruction at all — the caller only
    reaches here for a target that parsed — so it is reported unverified, not
    corrupt, and must not raise a false ERROR.
    """
    source = count_table_cells(source_markup)
    target = parse_markdown_table(target_markup, escape=False)
    if target is None:
        # The caller renders the target only when it parsed; reaching here means
        # the grid is not a strict pipe table, so it cannot be verified.
        return TableReconstruction(ok=False, reason="target_not_a_pipe_table")
    target_headers, target_rows = target
    if source is None:
        return TableReconstruction(
            ok=False,
            reason="source_not_a_pipe_table",
            rows=len(target_rows) + 1,
            cols=len(target_headers),
        )
    source_header_cells, source_row_cells = source
    cols = source_header_cells
    rows = len(source_row_cells) + 1  # header row + body rows
    if len(target_headers) != cols:
        return TableReconstruction(
            ok=False,
            reason=f"column_count_mismatch:{cols}->{len(target_headers)}",
            rows=rows,
            cols=cols,
        )
    if len(target_rows) != len(source_row_cells):
        return TableReconstruction(
            ok=False,
            reason=f"row_count_mismatch:{len(source_row_cells)}->{len(target_rows)}",
            rows=rows,
            cols=cols,
        )
    # Per-row RAW width: a row that gained a cell was truncated before this
    # check ever saw it; compare what the target markup actually splits into.
    target_counts = count_table_cells(target_markup)
    if target_counts is not None:
        _, target_row_cells = target_counts
        # strict=False on purpose: a target with FEWER body rows is already a
        # row_count_mismatch above; the zip only widens the check to rows the
        # two grids share.
        for idx, (src_n, tgt_n) in enumerate(
            zip(source_row_cells, target_row_cells, strict=False), start=1
        ):
            if src_n != tgt_n:
                return TableReconstruction(
                    ok=False,
                    reason=f"row_{idx}_cell_count_mismatch:{src_n}->{tgt_n}",
                    rows=rows,
                    cols=cols,
                )
    return TableReconstruction(ok=True, rows=rows, cols=cols)


def probe_table_complexity(
    markup: str,
    bbox: Any | None = None,
    *,
    page_columns: int = 1,
    content_width_pt: float | None = None,
) -> TableComplexity:
    """Whether the table can be typeset in the body without clipping, or defers to the appendix."""
    text_lower = markup.lower()
    if (
        "colspan" in text_lower
        or "rowspan" in text_lower
        or r"\multicolumn" in markup
        or r"\multirow" in markup
    ):
        return TableComplexity(
            is_complex=True,
            reason="merged_cells",
            rows=0,
            cols=0,
            suggests_landscape=False,
            contingency="appendix",
        )

    parsed = parse_markdown_table(markup)
    if parsed is None:
        return TableComplexity(
            is_complex=True,
            reason="non_standard_grid",
            rows=0,
            cols=0,
            suggests_landscape=False,
            contingency="appendix",
        )

    headers, rows = parsed
    cols = len(headers)
    total_rows = len(rows) + 1  # Including header

    # A table this wide cannot get column space out of a single-column measure, so it
    # defers to the appendix; and when the measure is too narrow to hold it at all,
    # rotating the page buys the room back, so landscape is suggested instead.
    if cols >= 8:
        is_single_col_dense = page_columns == 1 and (
            content_width_pt is None or content_width_pt < 600.0
        )
        return TableComplexity(
            is_complex=True,
            reason=f"too_many_columns_{cols}",
            rows=total_rows,
            cols=cols,
            suggests_landscape=is_single_col_dense,
            contingency="landscape" if is_single_col_dense else "appendix",
        )
    if cols >= 6:
        is_narrow_dense = page_columns == 1 and (
            content_width_pt is not None and content_width_pt < 350.0
        )
        return TableComplexity(
            is_complex=True,
            reason=f"too_many_columns_{cols}",
            rows=total_rows,
            cols=cols,
            suggests_landscape=is_narrow_dense,
            contingency="landscape" if is_narrow_dense else "appendix",
        )

    if bbox is not None:
        height = getattr(bbox, "height", None)
        if height is None and hasattr(bbox, "y1") and hasattr(bbox, "y0"):
            height = bbox.y1 - bbox.y0
        width = getattr(bbox, "width", None)
        if width is None and hasattr(bbox, "x1") and hasattr(bbox, "x0"):
            width = bbox.x1 - bbox.x0

        if height is not None and height > 0 and total_rows > 0:
            row_height = height / total_rows
            if row_height < 12.0:
                return TableComplexity(
                    is_complex=True,
                    reason=f"dense_row_height_{row_height:.1f}pt",
                    rows=total_rows,
                    cols=cols,
                    suggests_landscape=False,
                    contingency="appendix",
                )

        if width is not None and 0 < width < 120.0 and cols >= 4:
            return TableComplexity(
                is_complex=True,
                reason=f"too_narrow_{width:.1f}pt",
                rows=total_rows,
                cols=cols,
                suggests_landscape=False,
                contingency="appendix",
            )

    all_cells = headers + [c for r in rows for c in r]
    if all_cells:
        avg_len = sum(len(c) for c in all_cells) / len(all_cells)
        if avg_len > 100 or any(len(c) > 250 for c in all_cells):
            return TableComplexity(
                is_complex=True,
                reason="prose_paragraphs_in_cells",
                rows=total_rows,
                cols=cols,
                suggests_landscape=False,
                contingency="appendix",
            )

    return TableComplexity(
        is_complex=False,
        reason="simple_grid",
        rows=total_rows,
        cols=cols,
        suggests_landscape=False,
        contingency="none",
    )


_LATIN_LOWER_EM = 0.557
_LATIN_UPPER_EM = 0.698
_DIGIT_EM = 0.548
_CJK_EM = 1.0
_CJK_PUNCT_EM = 0.5
_ASCII_PUNCT_EM = 0.364
_SPACE_EM = 0.256

_BREAKABLE_RANGES = (
    (0x2E80, 0x9FFF),  # CJK radicals .. CJK unified ideographs
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFF00, 0xFFEF),  # half/full-width forms (full-width punctuation)
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x1100, 0x11FF),  # Hangul Jamo
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7FF),  # Hangul syllables + Jamo Extended-B
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _BREAKABLE_RANGES)


def _is_hangul(char: str) -> bool:
    code = ord(char)
    return 0x1100 <= code <= 0x11FF or 0xA960 <= code <= 0xA97F or 0xAC00 <= code <= 0xD7FF


def _is_cjk_punct(char: str) -> bool:
    code = ord(char)
    return 0x3000 <= code <= 0x303F or 0xFF01 <= code <= 0xFF65


def _char_em(char: str) -> float:
    if char.isspace():
        return _SPACE_EM
    if _is_hangul(char):
        return _CJK_EM
    if _is_cjk(char):
        return _CJK_EM if ord(char) >= 0x3040 else _CJK_PUNCT_EM
    if char.isdigit():
        return _DIGIT_EM
    if "A" <= char <= "Z":
        return _LATIN_UPPER_EM
    if char.isalpha():
        return _LATIN_LOWER_EM
    return _ASCII_PUNCT_EM


def _text_em(text: str) -> float:
    return sum(_char_em(char) for char in text)


def _longest_unbreakable(text: str) -> float:
    widest = 0.0
    run = 0.0
    for char in text:
        if _is_cjk_punct(char):
            widest = max(widest, run)
            run = 0.0
        elif char.isspace() or _is_cjk(char):
            widest = max(widest, run)
            run = _char_em(char) if not char.isspace() else 0.0
        elif char in "-\u2010\u2011\u2013\u2014":
            run += _char_em(char)
            widest = max(widest, run)
            run = 0.0
        else:
            run += _char_em(char)
    return max(widest, run)


def compile_typst_booktabs(
    markup: str,
    *,
    content_width_pt: float | None = None,
    content_height_pt: float | None = None,
    base_font_size_pt: float | None = None,
    min_font_size_pt: float = 7.5,
    stretch_pct: float = 100.0,
    target_lang: str = "zh",
    is_math_ok: Any | None = None,
    enable_continuation: bool = False,
    table_uid: str = "",
) -> str:
    """Compile a Markdown table into an academic Typst booktabs (#table) structure.

    Features:
    - Standard 3-line academic booktabs formatting:
      - 1.2pt top rule
      - 0.5pt mid rule below header
      - 1.2pt bottom rule
      - stroke: none (no vertical rules)
    - Adaptive column widths based on font advance measurements and unbreakable runs.
    - Cell clipping and overlap defense:
      - Micro insets (x: 3.5pt, y: 2.0pt, adaptively reduced if height-constrained).
      - Elastic variable font stretch (stretch_pct: 90%~100%).
      - Dynamic font scale down to min_font_size_pt (floor 7.5pt).
    - Math formulas ($...$) and line breaks (<br>) cleanly transpiled.
    - Table header bolding ([*Header*]).
    - Continuation support when enabled.
    """
    parsed = parse_markdown_table(markup, escape=False)
    if parsed is None:
        return ""
    headers, rows = parsed
    columns = len(headers)
    if columns == 0:
        return ""

    try:
        from ubt.render.typst.document_renderer import render_prose
    except Exception:

        def render_prose(s: str, is_math_ok: Any = None) -> str:  # type: ignore[misc]
            return typst_escape(s)

    def _render_cell_content(val: str, *, is_hdr: bool) -> str:
        parts = re.split(r"<br\s*/?>", val.strip())
        lines = [render_prose(p.strip(), is_math_ok=is_math_ok) for p in parts]
        joined = "\\\n".join(lines)
        return f"[*{joined}*]" if is_hdr and joined else f"[{joined}]"

    base_pt = (
        float(base_font_size_pt) if base_font_size_pt is not None and base_font_size_pt > 0 else 9.0
    )
    width_pt = (
        float(content_width_pt) if content_width_pt is not None and content_width_pt > 0 else 468.0
    )

    weights: list[float] = []
    runs: list[float] = []
    for col in range(columns):
        col_longest = 0.0
        col_run = 0.0
        for values in [headers, *rows]:
            if col < len(values):
                for part in re.split(r"<br\s*/?>", values[col]):
                    p_txt = part.strip()
                    col_longest = max(col_longest, _text_em(p_txt))
                    col_run = max(col_run, _longest_unbreakable(p_txt))
        weights.append(max(col_longest, 1.0))
        runs.append(max(col_run, 1.0))

    total_weight = sum(weights) or 1.0
    gutter_pt = 0.7 * base_pt
    share_available = max(1.0, width_pt - (columns - 1) * gutter_pt)

    stretch_factor = max(0.85, min(1.0, stretch_pct / 100.0))
    inset_x_pt = 3.5
    inset_y_pt = 2.0

    total_run_em = sum(runs) * stretch_factor
    inset_budget = 2 * inset_x_pt * columns
    required_width = total_run_em * base_pt + inset_budget
    scale = 1.0
    if required_width > share_available:
        scale = max(
            min_font_size_pt / base_pt,
            (share_available - inset_budget) / max(1.0, total_run_em * base_pt),
        )

    if content_height_pt is not None and content_height_pt > 0.0:
        total_rows = len(rows) + 1
        est_height = total_rows * (base_pt * scale * 1.35 + 2 * inset_y_pt) + 2.9
        if est_height > content_height_pt:
            ratio = content_height_pt / est_height
            scale = max(min_font_size_pt / base_pt, scale * ratio)
            inset_y_pt = max(1.0, inset_y_pt * ratio)
            inset_x_pt = max(2.0, inset_x_pt * ratio)

    run_mins = [run * base_pt * scale * stretch_factor + 2 * inset_x_pt for run in runs]
    total_min = sum(run_mins)
    if total_min >= share_available:
        widths = [m * share_available / total_min for m in run_mins]
    else:
        slack = share_available - total_min
        widths = [m + slack * (w / total_weight) for m, w in zip(run_mins, weights, strict=True)]

    column_spec = "(" + ", ".join(f"{max(12.0, w * 0.998):.1f}pt" for w in widths) + ")"

    start_anchor = ""
    header_continuation_cell = ""
    footer_continuation_block = ""
    if enable_continuation and len(rows) >= 4:
        try:
            from ubt.core.tables.models import TableContinuationConfig
            from ubt.core.tables.styler import generate_typst_table_continuation_elements

            t_cfg = TableContinuationConfig()
            (
                start_anchor,
                header_continuation_cell,
                footer_continuation_block,
                _,
            ) = generate_typst_table_continuation_elements(
                columns=columns,
                table_uid=table_uid or f"tbl-{abs(hash(markup)) % 100000}",
                lang=target_lang,
                config=t_cfg,
                indent="  ",
            )
        except Exception:
            pass

    lines: list[str] = ["#["]
    stretch_arg = f", stretch: {stretch_pct:.0f}%" if stretch_pct < 100.0 else ""
    if scale < 1.0 or stretch_arg:
        lines.append(f"  #set text(size: {scale:.2f}em{stretch_arg})")
    lines.append("  #set par(justify: false, leading: 0.55em)")
    if start_anchor:
        lines.append(f"  {start_anchor}")
    lines.append("  #table(")
    lines.append(f"    columns: {column_spec},")
    lines.append(f"    inset: (x: {inset_x_pt:.1f}pt, y: {inset_y_pt:.1f}pt),")
    lines.append("    column-gutter: 0.7em,")
    lines.append(
        "    align: (col, row) => if row == 0 { center + horizon } else { left + horizon },"
    )
    lines.append("    stroke: none,")
    lines.append("    table.header(")
    lines.append("      repeat: true,")
    if header_continuation_cell:
        lines.append(header_continuation_cell)
    lines.append("      table.hline(stroke: 1.2pt),")
    hdr_cells_str = ", ".join(_render_cell_content(h, is_hdr=True) for h in headers)
    lines.append(f"      {hdr_cells_str},")
    lines.append("      table.hline(stroke: 0.5pt),")
    lines.append("    ),")
    for row in rows:
        padded_row = list(row) + [""] * (columns - len(row))
        row_str = ", ".join(_render_cell_content(c, is_hdr=False) for c in padded_row[:columns])
        lines.append(f"    {row_str},")
    if footer_continuation_block:
        lines.append(footer_continuation_block)
    lines.append("    table.hline(stroke: 1.2pt),")
    lines.append("  )")
    lines.append("]")
    return "\n".join(lines)


__all__ = [
    "TableComplexity",
    "TableReconstruction",
    "compile_typst_booktabs",
    "count_table_cells",
    "parse_markdown_table",
    "probe_table_complexity",
    "verify_table_reconstruction",
]

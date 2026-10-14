"""Table complexity assessment and Typst table compiler.

Determines whether a table block can be typeset in-place via Typst micro-grid
or should be deferred to a dedicated high-fidelity translated table appendix.
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

    # Check separator row
    header_line = table_lines[0]
    sep_idx = -1
    for idx in range(1, min(3, len(table_lines))):
        cells = [c.strip() for c in _split_row(table_lines[idx])]
        if cells and all(re.match(r"^:?-+:?$", c) for c in cells):
            sep_idx = idx
            break

    if sep_idx == -1:
        # No standard separator line found
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


def probe_table_complexity(markup: str, bbox: Any | None = None) -> TableComplexity:
    """Assess whether a table can be safely overlaid in-place without clipping or overflow."""
    text_lower = markup.lower()
    # 1. Merged cells or complex HTML markers
    if (
        "colspan" in text_lower
        or "rowspan" in text_lower
        or r"\multicolumn" in markup
        or r"\multirow" in markup
    ):
        return TableComplexity(is_complex=True, reason="merged_cells", rows=0, cols=0)

    parsed = parse_markdown_table(markup)
    if parsed is None:
        return TableComplexity(is_complex=True, reason="non_standard_grid", rows=0, cols=0)

    headers, rows = parsed
    cols = len(headers)
    total_rows = len(rows) + 1  # Including header

    # 2. Too many columns
    if cols >= 6:
        return TableComplexity(
            is_complex=True, reason=f"too_many_columns_{cols}", rows=total_rows, cols=cols
        )

    # 3. Geometry density check
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
                )

        if width is not None and 0 < width < 120.0 and cols >= 4:
            return TableComplexity(
                is_complex=True,
                reason=f"too_narrow_{width:.1f}pt",
                rows=total_rows,
                cols=cols,
            )

    # 4. Long prose in table cells
    all_cells = headers + [c for r in rows for c in r]
    if all_cells:
        avg_len = sum(len(c) for c in all_cells) / len(all_cells)
        if avg_len > 100 or any(len(c) > 250 for c in all_cells):
            return TableComplexity(
                is_complex=True, reason="prose_paragraphs_in_cells", rows=total_rows, cols=cols
            )

    return TableComplexity(is_complex=False, reason="simple_grid", rows=total_rows, cols=cols)


__all__ = [
    "TableComplexity",
    "TableReconstruction",
    "count_table_cells",
    "parse_markdown_table",
    "probe_table_complexity",
    "verify_table_reconstruction",
]

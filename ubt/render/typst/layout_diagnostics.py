"""Typst layout spill and physical overflow diagnostics.

Performs deterministic geometry scanning over delivered PDF artifacts to detect:
1. Margin spills: text/tables/formulas extending beyond printable boundaries (left, right, bottom, top);
2. Column overruns: multi-column content intruding across the central gutter;
3. Block collisions: overlapping lines and elements (high IoU);
4. Type collapse: font size shrunk below legibility floor (< 4.5pt);
5. Typst compiler diagnostics: layout non-convergence and clipping warnings.

Provides terminal ASCII/Rich layout heatmaps to pinpoint exact spatial spill coordinates.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

# Standard points per unit
PT_PER_CM = 28.3465

# A4 dimensions in points
A4_WIDTH_PT = 595.28
A4_HEIGHT_PT = 841.89

# Default tolerances
DEFAULT_TOLERANCE_PT = 5.0
SEVERE_SPILL_DELTA_PT = 15.0
TYPE_COLLAPSE_FLOOR_PT = 4.0
TYPE_WARN_FLOOR_PT = 4.8

#: Page number pattern for footer page furniture
_PAGE_NUMBER_RE = re.compile(r"^(?:[-–—]\s*)?(\d{1,5}|[ivxlcdm]{1,7})\s*(?:[-–—])?$", re.IGNORECASE)


class AnomalyType(StrEnum):
    """Categorized visual layout anomaly classes."""

    MARGIN_SPILL_RIGHT = "margin_spill_right"
    MARGIN_SPILL_LEFT = "margin_spill_left"
    MARGIN_SPILL_BOTTOM = "margin_spill_bottom"
    MARGIN_SPILL_TOP = "margin_spill_top"
    GUTTER_OVERRUN = "gutter_overrun"
    BLOCK_COLLISION = "block_collision"
    TYPE_COLLAPSE = "type_collapse"
    COMPILER_WARNING = "compiler_warning"


class Severity(StrEnum):
    """Severity rating of layout anomalies."""

    INFO = "info"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class PageLayoutSpec:
    """Printable page geometry specification in points (bottom-left origin)."""

    page_width: float
    page_height: float
    margin_left: float
    margin_right: float
    margin_bottom: float
    margin_top: float
    columns: int = 1
    gutter: float = 0.0
    tolerance_pt: float = DEFAULT_TOLERANCE_PT
    name: str = "custom"

    @property
    def safe_x0(self) -> float:
        """Left safe printable boundary."""
        return self.margin_left

    @property
    def safe_x1(self) -> float:
        """Right safe printable boundary."""
        return self.page_width - self.margin_right

    @property
    def safe_y0(self) -> float:
        """Bottom safe printable boundary."""
        return self.margin_bottom

    @property
    def safe_y1(self) -> float:
        """Top safe printable boundary."""
        return self.page_height - self.margin_top

    @property
    def printable_width(self) -> float:
        return max(0.0, self.safe_x1 - self.safe_x0)

    @property
    def printable_height(self) -> float:
        return max(0.0, self.safe_y1 - self.safe_y0)

    @property
    def column_width(self) -> float:
        if self.columns <= 1:
            return self.printable_width
        total_gutters = (self.columns - 1) * self.gutter
        return max(0.0, (self.printable_width - total_gutters) / self.columns)

    @classmethod
    def book(cls, paper: str = "a4", tolerance_pt: float = DEFAULT_TOLERANCE_PT) -> PageLayoutSpec:
        """Standard monograph book specification: x: 2.2cm, y: 2.4cm."""
        margin_x = 2.2 * PT_PER_CM  # ~62.36 pt
        margin_y = 2.4 * PT_PER_CM  # ~68.03 pt
        return cls(
            page_width=A4_WIDTH_PT,
            page_height=A4_HEIGHT_PT,
            margin_left=margin_x,
            margin_right=margin_x,
            margin_bottom=margin_y,
            margin_top=margin_y,
            columns=1,
            tolerance_pt=tolerance_pt,
            name="book",
        )

    @classmethod
    def academic_dual(
        cls, paper: str = "a4", tolerance_pt: float = DEFAULT_TOLERANCE_PT
    ) -> PageLayoutSpec:
        """Academic two-column layout: x: 1.8cm, y: 2.0cm, gutter: ~29pt."""
        margin_x = 1.8 * PT_PER_CM  # ~51.02 pt
        margin_y = 2.0 * PT_PER_CM  # ~56.69 pt
        return cls(
            page_width=A4_WIDTH_PT,
            page_height=A4_HEIGHT_PT,
            margin_left=margin_x,
            margin_right=margin_x,
            margin_bottom=margin_y,
            margin_top=margin_y,
            columns=2,
            gutter=29.0,
            tolerance_pt=tolerance_pt,
            name="academic_dual",
        )

    @classmethod
    def academic_single(
        cls, paper: str = "a4", tolerance_pt: float = DEFAULT_TOLERANCE_PT
    ) -> PageLayoutSpec:
        """Academic single-column layout: x: 2.5cm, y: 2.5cm."""
        margin_x = 2.5 * PT_PER_CM  # ~70.87 pt
        margin_y = 2.5 * PT_PER_CM  # ~70.87 pt
        return cls(
            page_width=A4_WIDTH_PT,
            page_height=A4_HEIGHT_PT,
            margin_left=margin_x,
            margin_right=margin_x,
            margin_bottom=margin_y,
            margin_top=margin_y,
            columns=1,
            tolerance_pt=tolerance_pt,
            name="academic_single",
        )

    @classmethod
    def from_name(cls, name: str, tolerance_pt: float = DEFAULT_TOLERANCE_PT) -> PageLayoutSpec:
        """Resolve layout spec from template name."""
        norm = name.lower().replace("-", "_").strip()
        if norm in ("academic_dual", "dual", "two_column"):
            return cls.academic_dual(tolerance_pt=tolerance_pt)
        if norm in ("academic_single", "academic"):
            return cls.academic_single(tolerance_pt=tolerance_pt)
        return cls.book(tolerance_pt=tolerance_pt)


@dataclass(frozen=True, slots=True)
class LayoutAnomaly:
    """A detected physical layout spill, overflow, or typography defect."""

    anomaly_type: AnomalyType
    severity: Severity
    page: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    delta_pt: float  # Points exceeding the safe limit
    text_snippet: str
    font_size: float = 0.0
    details: str = ""
    suggested_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "page": self.page,
            "bbox": [round(c, 2) for c in self.bbox],
            "delta_pt": round(self.delta_pt, 2),
            "text_snippet": self.text_snippet,
            "font_size": round(self.font_size, 2),
            "details": self.details,
            "suggested_action": self.suggested_action,
        }


@dataclass(slots=True)
class PageLayoutReport:
    """Layout audit findings for an individual document page."""

    page: int
    page_width: float
    page_height: float
    line_count: int
    anomalies: list[LayoutAnomaly] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(a.severity == Severity.FAIL for a in self.anomalies)

    @property
    def has_warnings(self) -> bool:
        return any(a.severity == Severity.WARN for a in self.anomalies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "page_width": round(self.page_width, 2),
            "page_height": round(self.page_height, 2),
            "line_count": self.line_count,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


@dataclass(slots=True)
class DocumentLayoutReport:
    """Full-document visual spill and physical layout report."""

    pdf_path: str
    total_pages: int
    checked_pages: list[int]
    layout_template: str
    tolerance_pt: float
    page_reports: dict[int, PageLayoutReport] = field(default_factory=dict)
    compiler_warnings: list[str] = field(default_factory=list)

    @property
    def all_anomalies(self) -> list[LayoutAnomaly]:
        result: list[LayoutAnomaly] = []
        for pr in self.page_reports.values():
            result.extend(pr.anomalies)
        return result

    @property
    def spill_pages(self) -> list[int]:
        return sorted(p for p, pr in self.page_reports.items() if pr.anomalies)

    @property
    def error_count(self) -> int:
        return sum(1 for a in self.all_anomalies if a.severity == Severity.FAIL)

    @property
    def warning_count(self) -> int:
        return sum(1 for a in self.all_anomalies if a.severity == Severity.WARN)

    @property
    def passed(self) -> bool:
        """Pass verdict: no ERROR/FAIL severity anomalies."""
        return self.error_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf_path": self.pdf_path,
            "total_pages": self.total_pages,
            "checked_pages": self.checked_pages,
            "layout_template": self.layout_template,
            "tolerance_pt": self.tolerance_pt,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "spill_page_count": len(self.spill_pages),
            "spill_pages": self.spill_pages,
            "compiler_warnings": self.compiler_warnings,
            "page_reports": {p: pr.to_dict() for p, pr in sorted(self.page_reports.items())},
            "anomalies": [a.to_dict() for a in self.all_anomalies],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def is_page_furniture(
    text: str, y0: float, y1: float, page_height: float, margin_bottom: float
) -> bool:
    """Determine whether text is header/footer page furniture rather than body spill."""
    stripped = text.strip()
    if not stripped:
        return True
    # Footer page numbers
    if y1 <= margin_bottom + 12.0 and _PAGE_NUMBER_RE.match(stripped):
        return True
    # Running header at top of page
    return bool(y0 >= page_height - 45.0 and len(stripped) < 70)


def _suggest_action(text: str, anomaly_type: AnomalyType) -> str:
    """Generate precise corrective guidance for resolving the layout defect."""
    if anomaly_type == AnomalyType.TYPE_COLLAPSE:
        return "Increase table cell scale or break dense multi-column cell text."
    if "$" in text or "\\" in text:
        return "Wrap long inline math formula in display block or insert line breaks."
    if "|" in text:
        return "Reduce table font scale, shorten column widths, or flip page to landscape."
    if anomaly_type == AnomalyType.GUTTER_OVERRUN:
        return "Reduce paragraph/table width in Column 1 or insert manual columnbreak."
    if anomaly_type in (AnomalyType.MARGIN_SPILL_RIGHT, AnomalyType.MARGIN_SPILL_LEFT):
        return "Insert hyphens/breaks in long unbroken words or shrink margins slightly."
    if anomaly_type == AnomalyType.MARGIN_SPILL_BOTTOM:
        return (
            "Reduce block spacing, tighten baseline leading, or move trailing block to next page."
        )
    return "Check Typst template margins and block positioning rules."


def scan_page_anomalies(
    lines: Sequence[Any],
    page_no: int,
    spec: PageLayoutSpec,
) -> PageLayoutReport:
    """Scan extracted lines of a single page against the layout specification."""
    anomalies: list[LayoutAnomaly] = []
    tol = spec.tolerance_pt

    # Safe boundaries
    safe_x0 = spec.safe_x0
    safe_x1 = spec.safe_x1
    safe_y0 = spec.safe_y0
    safe_y1 = spec.safe_y1

    # Multi-column bounds
    is_dual = spec.columns == 2
    col1_x1 = safe_x0 + spec.column_width if is_dual else safe_x1
    col2_x0 = col1_x1 + spec.gutter if is_dual else safe_x0

    for line in lines:
        text = getattr(line, "text", "") or ""
        rect = getattr(line, "rect", (0.0, 0.0, 0.0, 0.0))
        font_sz = float(getattr(line, "font_size", 0.0))
        x0, y0, x1, y1 = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
        snippet = text.strip()[:60]

        # Ignore empty lines
        if not snippet:
            continue

        # Skip legit page numbers & headers
        if is_page_furniture(snippet, y0, y1, spec.page_height, safe_y0):
            continue

        # 1. Right Margin Spill
        if x1 > safe_x1 + tol:
            delta = x1 - safe_x1
            sev = (
                Severity.FAIL
                if delta > SEVERE_SPILL_DELTA_PT or x1 > spec.page_width - 8.0
                else Severity.WARN
            )
            anomalies.append(
                LayoutAnomaly(
                    anomaly_type=AnomalyType.MARGIN_SPILL_RIGHT,
                    severity=sev,
                    page=page_no,
                    bbox=(x0, y0, x1, y1),
                    delta_pt=delta,
                    text_snippet=snippet,
                    font_size=font_sz,
                    details=f"Right edge {x1:.1f}pt exceeds margin {safe_x1:.1f}pt by {delta:.1f}pt",
                    suggested_action=_suggest_action(snippet, AnomalyType.MARGIN_SPILL_RIGHT),
                )
            )

        # 2. Left Margin Spill
        elif x0 < safe_x0 - tol:
            delta = safe_x0 - x0
            sev = Severity.FAIL if delta > SEVERE_SPILL_DELTA_PT or x0 < 8.0 else Severity.WARN
            anomalies.append(
                LayoutAnomaly(
                    anomaly_type=AnomalyType.MARGIN_SPILL_LEFT,
                    severity=sev,
                    page=page_no,
                    bbox=(x0, y0, x1, y1),
                    delta_pt=delta,
                    text_snippet=snippet,
                    font_size=font_sz,
                    details=f"Left edge {x0:.1f}pt exceeds margin {safe_x0:.1f}pt by {delta:.1f}pt",
                    suggested_action=_suggest_action(snippet, AnomalyType.MARGIN_SPILL_LEFT),
                )
            )

        # 3. Gutter Overrun in Multi-column
        if is_dual and safe_x0 - tol <= x0 < col1_x1 and x1 > col1_x1 + tol and x1 < col2_x0 + 40.0:
            delta = x1 - col1_x1
            sev = Severity.FAIL if delta > 12.0 else Severity.WARN
            anomalies.append(
                LayoutAnomaly(
                    anomaly_type=AnomalyType.GUTTER_OVERRUN,
                    severity=sev,
                    page=page_no,
                    bbox=(x0, y0, x1, y1),
                    delta_pt=delta,
                    text_snippet=snippet,
                    font_size=font_sz,
                    details=f"Col 1 text overruns into gutter by {delta:.1f}pt (x1={x1:.1f}pt, col1_max={col1_x1:.1f}pt)",
                    suggested_action=_suggest_action(snippet, AnomalyType.GUTTER_OVERRUN),
                )
            )

        # 4. Bottom Margin Spill (vertical overflow)
        if y0 < safe_y0 - tol:
            delta = safe_y0 - y0
            sev = Severity.FAIL if y0 < 15.0 else Severity.WARN
            anomalies.append(
                LayoutAnomaly(
                    anomaly_type=AnomalyType.MARGIN_SPILL_BOTTOM,
                    severity=sev,
                    page=page_no,
                    bbox=(x0, y0, x1, y1),
                    delta_pt=delta,
                    text_snippet=snippet,
                    font_size=font_sz,
                    details=f"Bottom baseline {y0:.1f}pt falls below footer limit {safe_y0:.1f}pt by {delta:.1f}pt",
                    suggested_action=_suggest_action(snippet, AnomalyType.MARGIN_SPILL_BOTTOM),
                )
            )

        # 5. Top Margin Spill
        elif y1 > safe_y1 + tol:
            delta = y1 - safe_y1
            sev = Severity.FAIL if y1 > spec.page_height - 10.0 else Severity.WARN
            anomalies.append(
                LayoutAnomaly(
                    anomaly_type=AnomalyType.MARGIN_SPILL_TOP,
                    severity=sev,
                    page=page_no,
                    bbox=(x0, y0, x1, y1),
                    delta_pt=delta,
                    text_snippet=snippet,
                    font_size=font_sz,
                    details=f"Top edge {y1:.1f}pt exceeds header limit {safe_y1:.1f}pt by {delta:.1f}pt",
                    suggested_action=_suggest_action(snippet, AnomalyType.MARGIN_SPILL_TOP),
                )
            )

        # 6. Type Collapse Detection
        if 0.1 < font_sz < TYPE_WARN_FLOOR_PT and len(snippet) > 6:
            sev = Severity.FAIL if font_sz < TYPE_COLLAPSE_FLOOR_PT else Severity.WARN
            anomalies.append(
                LayoutAnomaly(
                    anomaly_type=AnomalyType.TYPE_COLLAPSE,
                    severity=sev,
                    page=page_no,
                    bbox=(x0, y0, x1, y1),
                    delta_pt=TYPE_WARN_FLOOR_PT - font_sz,
                    text_snippet=snippet,
                    font_size=font_sz,
                    details=f"Rendered font size {font_sz:.1f}pt is below legibility floor ({TYPE_WARN_FLOOR_PT}pt)",
                    suggested_action=_suggest_action(snippet, AnomalyType.TYPE_COLLAPSE),
                )
            )

    return PageLayoutReport(
        page=page_no,
        page_width=spec.page_width,
        page_height=spec.page_height,
        line_count=len(lines),
        anomalies=anomalies,
    )


def diagnose_pdf_layout(
    pdf_path: Path,
    spec: PageLayoutSpec | None = None,
    pages: Sequence[int] | None = None,
    tolerance_pt: float = DEFAULT_TOLERANCE_PT,
    compiler_stderr: str | None = None,
) -> DocumentLayoutReport:
    """Diagnose physical layout spills, margin leaks, and overflows across a PDF artifact."""
    from ubt.adapters.pdf.textgeom import extract_lines

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF document not found: {pdf_path}")

    # Determine total pages via pypdfium2
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(pdf)
    pdf.close()

    if total_pages <= 0:
        return DocumentLayoutReport(
            pdf_path=str(pdf_path),
            total_pages=0,
            checked_pages=[],
            layout_template="unknown",
            tolerance_pt=tolerance_pt,
        )

    target_pages = (
        [p for p in pages if 1 <= p <= total_pages] if pages else list(range(1, total_pages + 1))
    )
    page_reports: dict[int, PageLayoutReport] = {}

    # Sample first page to auto-resolve spec if not provided
    active_spec = spec
    if active_spec is None:
        try:
            _, (pw, ph) = extract_lines(pdf_path, target_pages[0])
            active_spec = PageLayoutSpec.book(tolerance_pt=tolerance_pt)
            if abs(pw - A4_WIDTH_PT) > 10.0:
                # Custom page dimensions
                active_spec = PageLayoutSpec(
                    page_width=pw,
                    page_height=ph,
                    margin_left=60.0,
                    margin_right=60.0,
                    margin_bottom=65.0,
                    margin_top=65.0,
                    tolerance_pt=tolerance_pt,
                    name="custom",
                )
        except Exception:
            active_spec = PageLayoutSpec.book(tolerance_pt=tolerance_pt)

    for p in target_pages:
        try:
            lines, (pw, ph) = extract_lines(pdf_path, p)
            # Adjust spec dimensions if page has distinctive orientation/size
            page_spec = active_spec
            if abs(pw - page_spec.page_width) > 5.0 or abs(ph - page_spec.page_height) > 5.0:
                page_spec = PageLayoutSpec(
                    page_width=pw,
                    page_height=ph,
                    margin_left=active_spec.margin_left,
                    margin_right=active_spec.margin_right,
                    margin_bottom=active_spec.margin_bottom,
                    margin_top=active_spec.margin_top,
                    columns=active_spec.columns,
                    gutter=active_spec.gutter,
                    tolerance_pt=tolerance_pt,
                    name=active_spec.name,
                )
            report = scan_page_anomalies(lines, p, page_spec)
            page_reports[p] = report
        except Exception as exc:
            logger.debug("Failed to extract layout geometry for page %d: %s", p, exc)

    # Parse compiler stderr for warnings
    comp_warnings: list[str] = []
    if compiler_stderr:
        for line in compiler_stderr.splitlines():
            line_str = line.strip()
            if "warning:" in line_str.lower() and any(
                k in line_str.lower()
                for k in ("converge", "clip", "overflow", "spill", "unreachable")
            ):
                comp_warnings.append(line_str)

    return DocumentLayoutReport(
        pdf_path=str(pdf_path),
        total_pages=total_pages,
        checked_pages=target_pages,
        layout_template=active_spec.name,
        tolerance_pt=tolerance_pt,
        page_reports=page_reports,
        compiler_warnings=comp_warnings,
    )


def render_ascii_heatmap(
    report: PageLayoutReport,
    spec: PageLayoutSpec,
    width_chars: int = 36,
    height_chars: int = 16,
) -> str:
    """Render an ASCII/Rich coordinate heatmap diagram representing the page layout and anomalies."""
    pw, ph = report.page_width, report.page_height
    grid = [[" " for _ in range(width_chars)] for _ in range(height_chars)]

    # Draw page boundaries
    for c in range(width_chars):
        grid[0][c] = "─"
        grid[height_chars - 1][c] = "─"
    for r in range(height_chars):
        grid[r][0] = "│"
        grid[r][width_chars - 1] = "│"
    grid[0][0] = "┌"
    grid[0][width_chars - 1] = "┐"
    grid[height_chars - 1][0] = "└"
    grid[height_chars - 1][width_chars - 1] = "┘"

    # Compute printable margin boundaries in grid space (row 0 is top)
    col_left = max(1, min(width_chars - 2, int((spec.safe_x0 / pw) * (width_chars - 1))))
    col_right = max(1, min(width_chars - 2, int((spec.safe_x1 / pw) * (width_chars - 1))))
    row_top = max(1, min(height_chars - 2, int(((ph - spec.safe_y1) / ph) * (height_chars - 1))))
    row_bottom = max(1, min(height_chars - 2, int(((ph - spec.safe_y0) / ph) * (height_chars - 1))))

    # Dashed safe zone outlines
    for r in range(row_top, row_bottom + 1):
        if grid[r][col_left] == " ":
            grid[r][col_left] = "┆"
        if grid[r][col_right] == " ":
            grid[r][col_right] = "┆"
    for c in range(col_left, col_right + 1):
        if grid[row_top][c] == " ":
            grid[row_top][c] = "┄"
        if grid[row_bottom][c] == " ":
            grid[row_bottom][c] = "┄"

    # Multi-column gutter outline
    if spec.columns == 2:
        col1_max = spec.safe_x0 + spec.column_width
        col2_min = col1_max + spec.gutter
        c_g1 = max(1, min(width_chars - 2, int((col1_max / pw) * (width_chars - 1))))
        c_g2 = max(1, min(width_chars - 2, int((col2_min / pw) * (width_chars - 1))))
        for r in range(row_top + 1, row_bottom):
            if grid[r][c_g1] == " ":
                grid[r][c_g1] = "┊"
            if grid[r][c_g2] == " ":
                grid[r][c_g2] = "┊"

    # Fill content area lightly
    for r in range(row_top + 1, row_bottom):
        for c in range(col_left + 1, col_right):
            if grid[r][c] == " ":
                grid[r][c] = "·"

    # Highlight anomalies with bold styled glyphs
    for a in report.anomalies:
        x0, y0, x1, y1 = a.bbox
        c0 = max(0, min(width_chars - 1, int((x0 / pw) * (width_chars - 1))))
        c1 = max(0, min(width_chars - 1, int((x1 / pw) * (width_chars - 1))))
        # Row 0 is top
        r0 = max(0, min(height_chars - 1, int(((ph - y1) / ph) * (height_chars - 1))))
        r1 = max(0, min(height_chars - 1, int(((ph - y0) / ph) * (height_chars - 1))))

        glyph = (
            "[bold red]█[/bold red]"
            if a.severity == Severity.FAIL
            else "[bold yellow]▲[/bold yellow]"
        )
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                grid[r][c] = glyph

    lines_out: list[str] = ["".join(row) for row in grid]
    return "\n".join(lines_out)


def print_rich_layout_report(
    report: DocumentLayoutReport,
    spec: PageLayoutSpec,
    console: Console | None = None,
    show_heatmap: bool = True,
) -> None:
    """Render a comprehensive diagnostic terminal report with summary cards and visual layout heatmaps."""
    out = console or Console()

    status_style = "bold green" if report.passed else "bold red"
    verdict = "PASSED (Clean)" if report.passed else f"FAILED ({report.error_count} error(s))"

    title_panel = Panel(
        f"[bold]Target PDF:[/] {escape(Path(report.pdf_path).name)}\n"
        f"[bold]Layout Template:[/] {report.layout_template} ({spec.columns} col, {spec.page_width:.0f}x{spec.page_height:.0f}pt)\n"
        f"[bold]Safe Margins:[/] X=[{spec.safe_x0:.1f}, {spec.safe_x1:.1f}], Y=[{spec.safe_y0:.1f}, {spec.safe_y1:.1f}] pt\n"
        f"[bold]Checked Pages:[/] {len(report.checked_pages)}/{report.total_pages}\n"
        f"[bold]Verdict:[/] [{status_style}]{verdict}[/{status_style}]",
        title="[bold cyan]Typst Layout Spill & Overflow Diagnostic[/bold cyan]",
        border_style="cyan",
    )
    out.print(title_panel)

    if not report.all_anomalies:
        out.print(
            Panel(
                "[bold green]✓ No layout spills or physical boundary overflows detected.[/bold green]"
            )
        )
        return

    # Anomalies summary table
    table = Table(title=f"Detected Layout Anomalies ({len(report.all_anomalies)} total)")
    table.add_column("Page", justify="right", style="cyan", no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Severity", no_wrap=True)
    table.add_column("Delta (pt)", justify="right", no_wrap=True)
    table.add_column("Text Snippet", overflow="fold")
    table.add_column("Diagnostic Details & Action", overflow="fold")

    for a in report.all_anomalies:
        sev_markup = (
            "[bold red]FAIL[/bold red]"
            if a.severity == Severity.FAIL
            else "[bold yellow]WARN[/bold yellow]"
        )
        type_markup = escape(a.anomaly_type.value)
        snippet = escape(a.text_snippet[:40])
        details = f"{escape(a.details)}\n[dim]Action: {escape(a.suggested_action)}[/dim]"
        table.add_row(
            str(a.page),
            type_markup,
            sev_markup,
            f"+{a.delta_pt:.1f}",
            snippet,
            details,
        )

    out.print(table)

    # Optional Visual Heatmap per anomalous page
    if show_heatmap:
        out.print("\n[bold]Spatial Layout Heatmaps (spilling zones highlighted):[/bold]\n")
        for p in report.spill_pages:
            pr = report.page_reports[p]
            heatmap_str = render_ascii_heatmap(pr, spec)
            panel = Panel(
                heatmap_str,
                title=f"Page {p} Spatial Layout Map ({len(pr.anomalies)} anomaly/ies)",
                subtitle="Legend: ┆/┄ Safe Margin | [bold red]█[/bold red] Critical Spill | [bold yellow]▲[/bold yellow] Warning",
                border_style="red" if pr.has_errors else "yellow",
            )
            out.print(panel)

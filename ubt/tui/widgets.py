"""Reusable Textual widgets for TUI v2. Thin views over SessionState.

Visual rules (tui-design skill, visual-patterns):
- No nested boxes: these widgets render unbordered content; separation comes
  from whitespace and one-line labels. The only bordered chrome lives in
  modal boxes and the command input (focus signal).
- Bold is reserved for titles and the active step; body text is plain.
- Status is always words + numbers, never color alone.
- Block characters (sparkline/QE bar) are data, not decoration.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import RichLog, Static

from ubt.core.job_options import default_output_path
from ubt.tui.events import stage_label
from ubt.tui.state import SessionState
from ubt.tui.theme import ACCENT, ERROR, MUTED_RICH, STEP_JOINT

STAGES: list[tuple[str, str]] = [
    ("PRE", "预处理"),
    ("DISC", "章节发现"),
    ("BIBLE", "术语圣经"),
    ("DRAFT", "初译"),
    ("QE", "质检"),
    ("REPAIR", "修复"),
    ("RENDER", "排版"),
]

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 20) -> str:
    """Render a tiny sparkline from 0..1 values. Pure, tested."""
    if not values:
        return "─" * width
    tail = values[-width:]
    out: list[str] = []
    for v in tail:
        v = max(0.0, min(1.0, v))
        out.append(_SPARK_CHARS[min(len(_SPARK_CHARS) - 1, int(v * len(_SPARK_CHARS)))])
    return "".join(out)


def progress_bar(frac: float, width: int = 20) -> str:
    """Determinate █░ bar (skill: percent + counts + ETA travel together)."""
    filled = max(0, min(width, int(round(max(0.0, min(1.0, frac)) * width))))
    return "█" * filled + "░" * (width - filled)


def qe_bar(qe: float, width: int = 10) -> str:
    """Render a █░ quality bar."""
    return progress_bar(qe, width)


class StageStepper(Static):
    """Seven-stage pipeline stepper. Active step is reverse video only."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self._stage = "IDLE"

    def update_stage(self, stage: str) -> None:
        self._stage = (stage or "IDLE").upper()
        joint = f"[{MUTED_RICH}]{STEP_JOINT}[/]"
        parts: list[str] = []
        is_done = self._stage in ("EXPORT_COMPLETED", "DONE", "COMPLETED")
        for code, label in STAGES:
            if is_done:
                parts.append(f"[bold green]{label}[/]")
            elif code in self._stage or self._stage.startswith(code):
                parts.append(f"[reverse]{label}[/]")
            else:
                parts.append(f"[{MUTED_RICH}]{label}[/]")
        if is_done:
            parts.append("[bold green](全部完成)[/]")
        self.update(Text.from_markup(joint.join(parts)))


class TelemetryPanel(Static):
    """Slim rail: one hero fact per row. Words and numbers carry meaning.

    Run-watcher contract: the rail answers 质量/进度/花费/速度 and nothing
    else. Cache-hit and QE-trend live in /report -- power-user levers, not
    glanceable state, and early-run trend lines are pure noise.
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)

    def update_state(self, s: SessionState) -> None:
        qe_c = SessionState.qe_color(s.avg_qe)
        t = Table.grid(padding=(0, 1), expand=True)
        t.add_column(width=6)
        t.add_column()
        t.add_row("形态", f"[{ACCENT}]{s.dual_label()}[/]")
        t.add_row("质量", f"[{qe_c}]{s.avg_qe:.3f}[/] {qe_bar(s.avg_qe)}")
        t.add_row("", f"[{MUTED_RICH}]末位15% {s.bottom15_qe:.3f}[/]")
        t.add_row("完成", f"{s.completed_blocks}/{s.total_blocks} ({s.progress_pct()}%)")
        t.add_row("修复", f"{s.repaired_blocks} 块")
        if s.needs_human_blocks:
            t.add_row("审校", f"[bold magenta]{s.needs_human_blocks} 块（按 p 审校）[/]")
        if s.failed_blocks:
            t.add_row("失败", f"[{ERROR}]{s.failed_blocks} 块（按 r 重试）[/]")
        t.add_row("", "")
        cost_text = f"{s.cost_label()} USD" if s.cost_usd is not None else "未知"
        t.add_row("花费", cost_text)
        if s.is_completed:
            t.add_row("状态", "[bold green]已完成[/]")
            t.add_row("耗时", s.fmt_duration(s.elapsed_secs(time.monotonic())))
            rate = s.rate_per_min(time.monotonic())
            if rate > 0:
                t.add_row("均速", f"{rate:.1f} 块/分")
        else:
            t.add_row("速度", s.pace_text(time.monotonic()))
        self.update(t)


class BilingualCard(Static):
    """Active-block source/target preview with clean code-agent diff styling."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)

    def update_state(self, s: SessionState) -> None:
        t = Table.grid(padding=(0, 1), expand=True)
        t.add_column()
        if s.is_completed:
            t.add_row(
                f"[bold green]全书翻译与导出完成[/]  [{MUTED_RICH}]· 状态: 导出完成 · 形态: {s.dual_label()}[/]"
            )
            t.add_row("")
            out_file = s.output_path or (
                default_output_path(s.input_path) if s.input_path else None
            )
            if out_file:
                t.add_row(
                    Panel(
                        f"[bold white]{escape(str(out_file))}[/]",
                        title="[bold]交付产物[/]",
                        title_align="left",
                        border_style=MUTED_RICH,
                        expand=True,
                    )
                )
                t.add_row("")
            if s.active_target:
                t.add_row(
                    Panel(
                        f"[bold]{escape(s.active_target[:400])}[/]",
                        title="[bold]尾块译文预览[/]",
                        title_align="left",
                        border_style=ACCENT,
                        expand=True,
                    )
                )
                t.add_row("")
            t.add_row(
                f"[{MUTED_RICH}]全书共 {s.completed_blocks}/{s.total_blocks} 块全部处理完毕（平均质量 QE: {s.avg_qe:.3f}）[/]"
            )
            t.add_row(
                f"[{MUTED_RICH}]快捷操作：输入 [bold white]/open[/] 调用系统查看器 · 输入 [bold white]/report[/] 查看质检报告 · 按 [bold white]w[/] 返回主页[/]"
            )
            self.update(t)
            return

        label = s.active_block_id or "就绪等待"
        t.add_row(
            f"[bold cyan]● 活跃块 {escape(label)}[/]  [{MUTED_RICH}]· 阶段: {escape(stage_label(s.stage))} · 形态: {s.dual_label()}[/]"
        )
        t.add_row("")
        if s.active_source:
            t.add_row(
                Panel(
                    escape(s.active_source[:500]),
                    title="[bold]原文[/]",
                    title_align="left",
                    border_style=MUTED_RICH,
                    expand=True,
                )
            )
            t.add_row("")
        if s.active_target:
            t.add_row(
                Panel(
                    f"[bold]{escape(s.active_target[:500])}[/]",
                    title=f"[bold]译文 ({s.dual_label()})[/]",
                    title_align="left",
                    border_style=ACCENT,
                    expand=True,
                )
            )
        if not s.active_source and not s.active_target:
            if s.completed_blocks == 0 and (
                "INGEST" in s.stage
                or "STARTED" in s.stage
                or "PREPROCESSING" in s.stage
                or "IDLE" in s.stage
            ):
                t.add_row(
                    f"[{MUTED_RICH}]正在进行文档版面、公式与文本结构解析 (Docling VLM / GPU)...[/]"
                )
                t.add_row(f"[{MUTED_RICH}]解析完成后将自动开始段落草稿翻译[/]")
            else:
                t.add_row(f"[{MUTED_RICH}]等待引擎分配文本任务…[/]")
        self.update(t)


class EventLogView(RichLog):
    """Scrollable engine event log. Time-only stamps; stage is plain text."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(highlight=False, markup=True, max_lines=300, auto_scroll=True, id=id)

    def push_event(self, ts: str, line: str) -> None:
        self.write(f"[{MUTED_RICH}]{escape(ts)}[/] {escape(line[:160])}")


@dataclass(frozen=True)
class ChapterHeatmapData:
    """Chapter structure and block state breakdown for cockpit heatmap."""

    chapter_index: int
    title: str
    total_blocks: int
    completed_blocks: int
    repaired_blocks: int
    failed_blocks: int
    needs_human_blocks: int
    avg_qe: float | None
    block_statuses: tuple[tuple[str, float | None], ...] = ()


class ChapterTreeHeatmap(Static):
    """Cockpit studio chapter tree and block-level heatmap visualization."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self._chapters: list[ChapterHeatmapData] = []

    def on_mount(self) -> None:
        self.refresh_display()

    def update_chapters(self, chapters: list[ChapterHeatmapData]) -> None:
        self._chapters = chapters
        self.refresh_display()

    def format_heatmap_text(self) -> str:
        if not self._chapters:
            return f"[{MUTED_RICH}]等待章节发现与任务分块…[/]"

        lines: list[str] = []
        for ch in self._chapters:
            status_tag = ""
            if ch.total_blocks > 0 and ch.completed_blocks == ch.total_blocks:
                status_tag = " [green][已完成][/]"
            elif ch.failed_blocks > 0:
                status_tag = f" [red][{ch.failed_blocks}失败][/]"
            elif ch.needs_human_blocks > 0:
                status_tag = f" [magenta][{ch.needs_human_blocks}待审][/]"

            qe_tag = f" · QE: {ch.avg_qe:.2f}" if ch.avg_qe is not None else ""
            lines.append(
                f"[bold cyan]第 {ch.chapter_index} 章[/] [bold white]{escape(ch.title[:20])}[/]{status_tag}"
            )
            lines.append(f"[{MUTED_RICH}]{ch.completed_blocks}/{ch.total_blocks} 块{qe_tag}[/]")

            glyphs: list[str] = []
            for status, qe in ch.block_statuses:
                s_lower = (status or "").lower()
                if s_lower in ("completed", "translated", "mtqe_passed"):
                    if qe is not None and qe >= 0.85:
                        glyphs.append("[green]■[/]")
                    elif qe is not None and qe >= 0.70:
                        glyphs.append("[yellow]■[/]")
                    else:
                        glyphs.append("[dark_orange]■[/]")
                elif s_lower == "repaired":
                    glyphs.append("[cyan]■[/]")
                elif s_lower in ("needs_human", "blocked_human"):
                    glyphs.append("[bold magenta]■[/]")
                elif s_lower == "failed":
                    glyphs.append("[bold red]■[/]")
                elif s_lower in ("translating", "in_flight", "drafted"):
                    glyphs.append("[bold blue]■[/]")
                else:
                    glyphs.append(f"[{MUTED_RICH}]░[/]")

            if glyphs:
                chunk_size = 20
                for i in range(0, len(glyphs), chunk_size):
                    lines.append("".join(glyphs[i : i + chunk_size]))
            lines.append("")

        lines.append(
            f"[{MUTED_RICH}]图例: [green]■[/]通过 [cyan]■[/]修复 [magenta]■[/]待审 [red]■[/]失败 [dim]░[/]等待[/]"
        )
        return "\n".join(lines)

    def refresh_display(self) -> None:
        with contextlib.suppress(Exception):
            self.update(Text.from_markup(self.format_heatmap_text()))


__all__ = [
    "BilingualCard",
    "ChapterHeatmapData",
    "ChapterTreeHeatmap",
    "EventLogView",
    "StageStepper",
    "STAGES",
    "TelemetryPanel",
    "qe_bar",
    "sparkline",
]

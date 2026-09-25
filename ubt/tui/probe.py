"""Probe presentation for TUI v2: read-only document card + local discovery.

Pure Rich/plain helpers (no prompts, no Live): the fullscreen screens own
all interaction. Moved here so the TUI has a single implementation.

Deliberately borderless: the wizard section label frames this card,
so an additional border is unnecessary. This also avoids
CJK character width-vs-border misalignment issues.
"""

from __future__ import annotations

from pathlib import Path

from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ubt.tui.advisor import AdvisoryReport, MathDensity
from ubt.tui.presets import PRESETS


def render_probe_card(report: AdvisoryReport) -> Table:
    """Read-only probe facts: what the engine detected, never questions.

    environment facts are displayed, never asked. No technical parameter
    names; the tier suggestion is a single user-facing line.
    """
    page_unit = "页" if report.format_ext == "pdf" else "章"
    math_label = {
        MathDensity.HIGH: "[bold red]密集（微分方程 / 复杂下标）[/]",
        MathDensity.LOW: "[yellow]少量数学符号[/]",
        MathDensity.NONE: "[green]无[/]",
    }.get(report.math_density, "未知")
    text_type = "[yellow]扫描件（走视觉识别）[/]" if report.is_scanned else "[green]原生文本[/]"

    env_bits = []
    env_bits.append("[green]GPU 就绪[/]" if report.has_gpu else "[dim]CPU 模式[/]")
    env_bits.append("[green]Typst 就绪[/]" if report.has_typst else "[yellow]Typst 缺失[/]")
    env_bits.append(
        "[green]API 密钥就绪 (真实翻译)[/]"
        if report.api_ready
        else "[bold yellow]未配 API 密钥 (演练模式)[/]"
    )

    route_label = {
        "long": "长链（完整术语圣经）",
        "short": "短链（快速整章重写）",
    }.get(report.route_mode, "自动路由")

    glossary_label = (
        f"内置 `{report.recommended_glossary.name}`"
        if report.recommended_glossary
        else "无内置词典（可用高级设置挂载）"
    )

    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(justify="right", width=12)
    grid.add_column()
    grid.add_column(justify="right", width=12)
    grid.add_column()

    grid.add_row(
        "文档:",
        f"[bold]{escape(report.file_name)}[/] ({report.file_size_mb} MB)",
        "规模:",
        f"{report.page_or_ch_count} {page_unit}",
    )
    grid.add_row("文本类型:", text_type, "公式密度:", math_label)
    grid.add_row(
        "领域:",
        f"[magenta]{report.detected_domain}[/] (置信度 {int(report.domain_confidence * 100)}%)",
        "术语表:",
        glossary_label,
    )
    grid.add_row("运行环境:", " · ".join(env_bits), "预计路由:", f"[bold cyan]{route_label}[/]")

    suggestion = PRESETS[report.recommended_preset]
    suggest_line = f"[bold]{suggestion.label}[/] —— {suggestion.tagline}"

    content = Table.grid(expand=True)
    content.add_row(Text.from_markup(f"[bold]文档预检 · {escape(report.file_name)}[/]"))
    content.add_row(Text(""))
    content.add_row(grid)
    content.add_row(Text(""))
    content.add_row(Text.from_markup(f"建议档位: {suggest_line}"))

    if report.assessment is not None:
        cost = report.assessment.cost
        cost_grid = Table.grid(padding=(0, 2), expand=True)
        cost_grid.add_column(justify="right", width=12)
        cost_grid.add_column()
        cost_grid.add_column(justify="right", width=12)
        cost_grid.add_column()

        def _money_str(val: float | None) -> str:
            return "未知" if val is None else f"${val:.4f}"

        draft_range = (
            f"{_money_str(cost.draft_cost_usd_cached)} ~ {_money_str(cost.draft_cost_usd_uncached)}"
        )
        repair_str = (
            f"≤{cost.repair_blocks} 块 ({_money_str(cost.repair_cost_usd)})"
            if cost.repair_cost_usd is not None
            else "无需"
        )
        total_str = (
            f"[bold green]{_money_str(cost.total_cost_usd)}[/]"
            if cost.total_cost_usd is not None
            else "[dim]未知[/]"
        )

        cost_grid.add_row(
            "计费分块:",
            f"{cost.billable_blocks} 块" + ("" if cost.billable_blocks_is_exact else " (估算)"),
            "预期总成本:",
            total_str,
        )
        cost_grid.add_row(
            "预估 Token:",
            f"{cost.prompt_tokens} in / {cost.completion_tokens} out",
            "草稿报价:",
            draft_range,
        )
        cost_grid.add_row(
            "靶向修复:",
            repair_str,
            "QE 与视觉:",
            f"QE: {cost.qe_calls} 次 · 视觉: {cost.vlm_page_calls + cost.ocr_page_calls} 页",
        )

        content.add_row(Text(""))
        content.add_row(Text.from_markup("[bold cyan]── 译前报价与成本预估 (Assess Quote) ──[/]"))
        content.add_row(cost_grid)

        if report.assessment.warnings:
            content.add_row(Text(""))
            content.add_row(Text.from_markup("[bold yellow]── 风险与风控雷达 ──[/]"))
            for aw in report.assessment.warnings:
                content.add_row(Text.from_markup(f"[yellow]▲ [/]{escape(aw.detail_zh)}"))

    if not report.api_ready:
        content.add_row(Text(""))
        content.add_row(
            Text.from_markup(
                "[yellow]▲ 提示：未检测到有效 API 密钥，管线将自动进入零消耗演练模式"
                "（真实翻译请配置 UBT_LLM_API_KEY 或 OPENAI_API_KEY）[/]"
            )
        )

    if report.conflict_warnings:
        content.add_row(Text(""))
        for w in report.conflict_warnings:
            content.add_row(Text.from_markup(f"[yellow]! [/]{w}"))

    return content


def scan_local_books(search_dir: Path | None = None) -> list[Path]:
    """Find candidate book files (.pdf, .epub, .docx, .md) in local project directories."""
    cwd = search_dir or Path.cwd()
    candidates: list[Path] = []
    extensions = {".pdf", ".epub", ".docx", ".md"}

    # Search current directory and common docs dirs
    search_dirs = [cwd, cwd / "docs", cwd / "books", cwd / "input", cwd / "tests/fixtures"]
    seen: set[Path] = set()

    for d in search_dirs:
        if d.exists() and d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in extensions:
                    resolved = p.resolve()
                    if resolved not in seen and not p.name.startswith("."):
                        candidates.append(p)
                        seen.add(resolved)

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates

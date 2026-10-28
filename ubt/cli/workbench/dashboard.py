"""Rich terminal UI dashboard renderer for the interactive console workbench."""

from __future__ import annotations

from rich.console import RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ubt.cli.workbench.models import JobControlStatus, WorkbenchState


def _make_header(state: WorkbenchState) -> Panel:
    """Build workbench top status header."""
    header_table = Table.grid(expand=True)
    header_table.add_column(justify="left", ratio=3)
    header_table.add_column(justify="center", ratio=4)
    header_table.add_column(justify="right", ratio=3)

    # Status pill styling
    if state.status == JobControlStatus.RUNNING:
        status_text = "[bold green]● RUNNING[/]"
    elif state.status == JobControlStatus.PAUSED:
        status_text = "[bold yellow]❚❚ PAUSED[/]"
    elif state.status == JobControlStatus.CANCELLING:
        status_text = "[bold red]✖ CANCELLING[/]"
    elif state.status == JobControlStatus.COMPLETED:
        status_text = "[bold green]✔ COMPLETED[/]"
    else:
        status_text = f"[dim]{state.status.value.upper()}[/]"

    title_part = f"[bold cyan]UBT Workbench[/] — [bold]{state.active_job_title or state.active_job_id or 'Translation Engine'}[/]"
    center_part = f"{status_text}  [dim]|[/]  Stage: [bold cyan]{state.current_stage}[/]"
    right_part = (
        f"Slots: [bold magenta]{state.concurrency}[/]  [dim]|[/]  "
        f"Time: [bold]{int(state.elapsed_s)}s[/]"
    )

    header_table.add_row(title_part, center_part, right_part)
    return Panel(header_table, style="blue", height=3)


def _make_tabs(state: WorkbenchState) -> Text:
    """Render horizontal tab strip."""
    tabs = [
        ("0", "Pipeline Overview"),
        ("1", "Multi-Job Queue"),
        ("2", "Resources & Costs"),
        ("3", "Diagnostics"),
    ]
    txt = Text("  Views: ", style="dim")
    for idx, (num, name) in enumerate(tabs):
        if idx == state.selected_tab:
            txt.append(f" [{num}] {name} ", style="bold black on cyan")
        else:
            txt.append(f" [{num}] {name} ", style="dim white")
        txt.append("  ")
    return txt


def _render_tab_overview(state: WorkbenchState) -> RenderableType:
    """Render Tab 0: Stage progression and block lifecycle breakdown."""
    content = Table.grid(expand=True, padding=(0, 1))
    content.add_column(ratio=6)
    content.add_column(ratio=4)

    # Left: Progress and Block Metrics
    left_table = Table(title="Document Pipeline Progress", expand=True, show_edge=False)
    left_table.add_column("Metric", style="cyan")
    left_table.add_column("Value", justify="right")

    left_table.add_row(
        "Overall Progress",
        f"{state.progress_pct:.1f}%",
    )
    left_table.add_row(
        "Blocks Delivered",
        f"{state.blocks_done} / {max(1, state.blocks_total)}",
    )
    left_table.add_row(
        "Repaired Blocks",
        f"[green]{state.repaired_count}[/]",
    )
    left_table.add_row(
        "Quarantined / Human Review",
        f"[bold red]{state.quarantined_count}[/]" if state.quarantined_count > 0 else "[green]0[/]",
    )

    # Right: Live Event Ticker
    events_table = Table(title="Live Event Stream", expand=True, show_edge=False)
    events_table.add_column("Timestamp & Event", style="dim")
    if not state.recent_events:
        events_table.add_row("[dim]Awaiting pipeline events...[/]")
    else:
        for ev in state.recent_events[-6:]:
            events_table.add_row(ev)

    content.add_row(left_table, events_table)
    return Panel(content, title="Stage Progression & Real-Time Events", border_style="cyan")


def _render_tab_queue(state: WorkbenchState) -> RenderableType:
    """Render Tab 1: Multi-job Queue Table."""
    table = Table(title="Job Queue & Concurrency Slots", expand=True)
    table.add_column("Job ID", style="bold cyan")
    table.add_column("Title / Input", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Progress", justify="right")
    table.add_column("Blocks", justify="right")
    table.add_column("Lease", justify="center")
    table.add_column("Errors", style="red")

    if not state.jobs:
        # Show at least the active job if no external jobs discovered
        active_id = state.active_job_id or "job_active"
        table.add_row(
            active_id,
            state.active_job_title or "Current Translation Run",
            f"[bold green]{state.status.value}[/]",
            f"{state.progress_pct:.1f}%",
            f"{state.blocks_done}/{state.blocks_total}",
            "Active",
            "—",
        )
    else:
        for j in state.jobs[:8]:
            status_style = "green" if j.status == "running" else "yellow"
            table.add_row(
                j.job_id[:16],
                j.title[:24] or "Document",
                f"[{status_style}]{j.status}[/]",
                f"{j.progress_pct:.1f}%",
                f"{j.blocks_done}/{j.blocks_total}",
                f"{int(j.lease_expires_in or 0)}s" if j.lease_expires_in else "—",
                j.error[:20] if j.error else "—",
            )
    return Panel(table, title="Multi-Job Execution Matrix", border_style="cyan")


def _render_tab_resources(state: WorkbenchState) -> RenderableType:
    """Render Tab 2: Resource and financial telemetry."""
    res = state.resource

    table = Table(title="System Utilization & Token Telemetry", expand=True)
    table.add_column("Telemetry Metric", style="cyan")
    table.add_column("Current Measurement", justify="right")
    table.add_column("Operational Context", style="dim")

    table.add_row(
        "Process Memory (RSS)",
        f"[bold]{res.rss_mb:.1f} MB[/]",
        "Bounded in-process memory footprint",
    )
    table.add_row(
        "CPU Utilization Time",
        f"{res.cpu_time_s:.2f}s",
        "Cumulative process CPU compute time",
    )
    table.add_row(
        "Active Concurrency Slots",
        f"[bold magenta]{state.concurrency}[/] slots",
        "Adjustable with [+] and [-] keys",
    )
    table.add_row(
        "Prompt Tokens",
        f"{res.prompt_tokens:,}",
        "Total input tokens billed across stages",
    )
    table.add_row(
        "Completion Tokens",
        f"{res.completion_tokens:,}",
        "Generated translation & repair tokens",
    )
    table.add_row(
        "Cache Hit Tokens",
        f"{res.cached_tokens:,} ({res.cache_hit_rate:.1f}%)",
        "[bold green]Cloud context caching savings[/]",
    )
    table.add_row(
        "Estimated Expenditure",
        f"[bold green]${res.cost_usd:.4f}[/]",
        "Real-time cost projection from prices.toml",
    )

    return Panel(table, title="Resource Telemetry & Financial Radar", border_style="cyan")


def _render_tab_diagnostics(state: WorkbenchState) -> RenderableType:
    """Render Tab 3: Quarantined & needs_human block inspection."""
    table = Table(title="Quarantined & High-Risk Blocks (MQM Major/Critical)", expand=True)
    table.add_column("Block ID", style="cyan")
    table.add_column("Reason / Gate Rule", style="yellow")
    table.add_column("Source Snippet", style="dim")

    if not state.quarantined_details:
        table.add_row(
            "[dim]None[/]",
            "[green]All drafted blocks conform to quality and structural invariants[/]",
            "—",
        )
    else:
        for q in state.quarantined_details[:6]:
            table.add_row(
                str(q.get("block_id", "—")),
                str(q.get("reason", "Violation detected"))[:40],
                str(q.get("source", ""))[:30],
            )

    return Panel(table, title="Safety Gate & Quarantined Block Inspector", border_style="cyan")


def _make_footer(state: WorkbenchState) -> Panel:
    """Render bottom hotkey help bar."""
    status_msg = f"[bold]{state.status_message}[/]"
    key_hints = (
        "[bold cyan][P][/] Pause/Resume  "
        "[bold cyan][+][/]/[bold cyan][-][/] Concurrency  "
        "[bold cyan][Tab][/] Switch Tab  "
        "[bold cyan][R][/] Retry  "
        "[bold cyan][Q][/] Abort  "
        "[bold cyan][?][/] Help"
    )

    table = Table.grid(expand=True)
    table.add_column(justify="left", ratio=4)
    table.add_column(justify="right", ratio=6)
    table.add_row(status_msg, key_hints)

    return Panel(table, style="blue", height=3)


def _make_help_overlay() -> Panel:
    """Render modal help panel when '?' is pressed."""
    table = Table(title="Interactive Console Workbench Keybindings", expand=True)
    table.add_column("Key", style="bold cyan", justify="center")
    table.add_column("Action", style="white")
    table.add_column("Description", style="dim")

    table.add_row(
        "P / Space",
        "Toggle Pause/Resume",
        "Suspends dispatch at next block checkpoint without killing the job",
    )
    table.add_row(
        "+ / =",
        "Increase Concurrency",
        "Dynamically raises parallel worker slots (up to 16)",
    )
    table.add_row(
        "- / _",
        "Decrease Concurrency",
        "Lowers parallel worker slots to mitigate 429 rate-limiting (down to 1)",
    )
    table.add_row(
        "Tab / l",
        "Next Tab",
        "Cycle views: [0] Pipeline -> [1] Queue -> [2] Resources -> [3] Diagnostics",
    )
    table.add_row(
        "h",
        "Previous Tab",
        "Cycle views backward",
    )
    table.add_row(
        "R",
        "Retry Quarantined",
        "Re-queues blocked/quarantined blocks for targeted repair",
    )
    table.add_row(
        "Q / Ctrl-C",
        "Graceful Abort",
        "Flushes active checkpoints to SQLite ledger and exits cleanly",
    )
    table.add_row(
        "?",
        "Toggle Help",
        "Shows / hides this interactive keybinding guide",
    )

    return Panel(table, title="Help & Controls Overlay", border_style="yellow")


def render_dashboard(state: WorkbenchState) -> Layout:
    """Assemble the complete full-screen Rich Layout for the workbench."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="tabs", size=1),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3),
    )

    layout["header"].update(_make_header(state))
    layout["tabs"].update(_make_tabs(state))

    if state.show_help:
        layout["body"].update(_make_help_overlay())
    elif state.selected_tab == 0:
        layout["body"].update(_render_tab_overview(state))
    elif state.selected_tab == 1:
        layout["body"].update(_render_tab_queue(state))
    elif state.selected_tab == 2:
        layout["body"].update(_render_tab_resources(state))
    elif state.selected_tab == 3:
        layout["body"].update(_render_tab_diagnostics(state))

    layout["footer"].update(_make_footer(state))
    return layout

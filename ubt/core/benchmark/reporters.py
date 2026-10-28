"""Multi-format benchmark reporting: Rich terminal dashboard, Markdown, HTML and JSON."""

from __future__ import annotations

import html
import json
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ubt.core.benchmark.models import (
    BenchmarkDriftReport,
    CaseDrift,
    CorpusBenchmarkReport,
)


def _format_diff_str(val: float, is_percentage: bool = False, higher_is_better: bool = True) -> str:
    """Format a metric delta with direction arrow and sign."""
    if abs(val) < 1e-4:
        return "±0.0%" if is_percentage else "±0.0"
    sign = "+" if val > 0 else ""
    val_str = f"{sign}{val:.1%}" if is_percentage else f"{sign}{val:.2f}"
    arrow = "▲" if val > 0 else "▼"
    return f"{val_str} {arrow}"


def render_terminal_dashboard(
    report: CorpusBenchmarkReport,
    drift: BenchmarkDriftReport | None = None,
    console: Console | None = None,
) -> None:
    """Display comprehensive Rich terminal benchmark dashboard with KPI cards and drift analysis."""
    con = console or Console()

    # 1. Header Panel
    overall_color = "green" if report.overall_passed else "red"
    overall_badge = f"[bold {overall_color}]{'PASSED' if report.overall_passed else 'FAILED'}[/]"
    header_text = (
        f"Mode: [bold cyan]{report.mode}[/] | Run ID: [dim]{report.run_id}[/] | "
        f"Cases: [bold]{report.total_cases}[/] (Pass: [green]{report.passed_cases}[/], "
        f"Fail: [red]{report.failed_cases}[/], Skip: [yellow]{report.skipped_cases}[/]) | "
        f"Verdict: {overall_badge}"
    )
    con.print(
        Panel(
            header_text,
            title="[bold]UBT Corpus Automated Benchmark Dashboard[/]",
            border_style="cyan",
        )
    )

    # 2. KPI Summary Grid
    kpi_table = Table(
        title="Aggregate Benchmark KPIs", show_header=True, header_style="bold magenta"
    )
    kpi_table.add_column("Metric")
    kpi_table.add_column("Value", justify="right")
    kpi_table.add_column("Metric")
    kpi_table.add_column("Value", justify="right")

    cost_str = f"${report.total_cost_usd:.4f}" if report.total_cost_usd is not None else "Unpriced"
    kpi_table.add_row(
        "Avg Delivered Ratio",
        f"{report.avg_delivered_ratio:.1%}",
        "Avg Accounted Ratio",
        f"{report.avg_accounted_ratio:.1%}",
    )
    kpi_table.add_row(
        "Contract Errors",
        str(report.total_contract_errors),
        "Missing Assets",
        str(report.total_missing_assets),
    )
    kpi_table.add_row(
        "Avg Layout Spill Rate",
        f"{report.avg_spill_rate:.1%}",
        "Spill Pages / Collapses",
        f"{report.total_spill_pages} / {report.total_type_collapse}",
    )
    kpi_table.add_row(
        "Total Tokens",
        f"{report.total_tokens:,}",
        "Cache Hit Rate",
        f"{report.overall_cache_hit_rate:.1%}",
    )
    kpi_table.add_row(
        "Total Runtime",
        f"{report.total_runtime_sec:.1f}s",
        "Total Model Cost",
        cost_str,
    )
    con.print(kpi_table)

    # 3. Drift Summary (if baseline provided)
    drift_by_case: dict[str, CaseDrift] = {}
    if drift:
        drift_by_case = {cd.case_id: cd for cd in drift.case_drifts}
        drift_color = "green" if drift.drift_passed else "red"
        drift_title = f"[bold {drift_color}]Baseline Drift Summary — {'NO REGRESSION' if drift.drift_passed else 'REGRESSION DETECTED'}[/]"
        drift_info = (
            f"Baseline: [dim]{drift.baseline_run_id}[/] | Candidate: [dim]{drift.candidate_run_id}[/] | "
            f"Regressed: [bold red]{drift.regressed_count}[/] | Improved: [bold green]{drift.improved_count}[/] | "
            f"Stable: [blue]{drift.stable_count}[/] | New: [yellow]{drift.new_count}[/]"
        )
        con.print(Panel(drift_info, title=drift_title, border_style=drift_color))

    # 4. Detailed Case Matrix Table
    case_table = Table(title="Corpus Cases Breakdown", show_header=True, header_style="bold cyan")
    case_table.add_column("Case ID", style="bold")
    case_table.add_column("Status")
    case_table.add_column("Delivered", justify="right")
    case_table.add_column("Accounted", justify="right")
    case_table.add_column("Errors", justify="right")
    case_table.add_column("Spill Rate", justify="right")
    case_table.add_column("Max Over", justify="right")
    case_table.add_column("Time", justify="right")
    case_table.add_column("Cost", justify="right")
    if drift:
        case_table.add_column("Drift Verdict")

    for c in report.cases:
        stat_color = "green" if c.status == "pass" else ("yellow" if c.status == "skip" else "red")
        cost_val = (
            f"${c.cost.estimated_cost_usd:.3f}" if c.cost.estimated_cost_usd is not None else "-"
        )

        row = [
            escape(c.id),
            f"[{stat_color}]{c.status.upper()}[/]",
            f"{c.contract.delivered_ratio:.1%}",
            f"{c.contract.accounted_ratio:.1%}",
            str(c.contract.errors_count),
            f"{c.layout.spill_rate:.1%}",
            f"{c.layout.max_overflow_pt:.1f}pt",
            f"{c.runtime_sec:.1f}s",
            cost_val,
        ]

        if drift:
            cd = drift_by_case.get(c.id)
            if cd:
                dv_color = {
                    "REGRESSED": "bold red",
                    "IMPROVED": "bold green",
                    "STABLE": "dim cyan",
                    "NEW": "yellow",
                    "MISSING": "dim",
                }.get(cd.verdict, "white")
                diff_tag = (
                    f" ({_format_diff_str(cd.delta_delivered_ratio, is_percentage=True)})"
                    if cd.verdict in ("REGRESSED", "IMPROVED")
                    else ""
                )
                row.append(f"[{dv_color}]{cd.verdict}{diff_tag}[/]")
            else:
                row.append("-")

        case_table.add_row(*row)

    con.print(case_table)

    # 5. Failures and Regressions Details
    failures = [
        c for c in report.cases if c.status == "fail" or (c.status == "pass" and not c.passed)
    ]
    if failures:
        con.print("\n[bold red]Failures and Defect Details:[/]")
        for f in failures:
            con.print(
                f"  • [bold]{escape(f.id)}[/]: {escape(f.failure_reason or 'Validation failed')}"
            )
            for ef in f.contract.expectation_failures:
                con.print(f"      [red]↳ Contract violation:[/] {escape(ef)}")
            if not f.layout.layout_passed:
                con.print(
                    f"      [red]↳ Layout spill:[/] {f.layout.spill_count} anomalies, max overflow {f.layout.max_overflow_pt:.1f}pt"
                )

    if drift and drift.regressed_count > 0:
        con.print("\n[bold red]Baseline Drift Regressions:[/]")
        for cd in drift.case_drifts:
            if cd.verdict == "REGRESSED":
                con.print(f"  • [bold]{escape(cd.case_id)}[/]:")
                for r in cd.reasons:
                    con.print(f"      [red]↳ Regression:[/] {escape(r)}")


def render_markdown_report(
    report: CorpusBenchmarkReport,
    drift: BenchmarkDriftReport | None = None,
) -> str:
    """Generate GitHub Flavored Markdown benchmark report."""
    lines: list[str] = []
    lines.append("# UBT Corpus Benchmark Report")
    lines.append("")

    overall_label = "PASSED" if report.overall_passed else "FAILED"
    alert_type = "NOTE" if report.overall_passed else "WARNING"
    lines.append(f"> [!{alert_type}]")
    lines.append(
        f"> **Benchmark Status: {overall_label}** — {report.passed_cases}/{report.total_cases} cases passed."
    )
    if drift:
        drift_label = (
            "NO REGRESSION"
            if drift.drift_passed
            else f"{drift.regressed_count} REGRESSION(S) DETECTED"
        )
        lines.append(f"> Baseline Drift: **{drift_label}** (against `{drift.baseline_run_id}`).")
    lines.append("")

    # Metadata Table
    lines.append("### Execution Metadata")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Run ID** | `{report.run_id}` |")
    lines.append(f"| **Timestamp** | {report.timestamp} |")
    lines.append(f"| **Corpus Directory** | `{report.corpus_dir}` |")
    lines.append(f"| **Execution Mode** | `{report.mode}` |")
    lines.append(f"| **Total Runtime** | {report.total_runtime_sec:.2f} seconds |")
    lines.append("")

    # KPI Table
    cost_str = f"${report.total_cost_usd:.4f}" if report.total_cost_usd is not None else "Unpriced"
    lines.append("### Aggregate KPIs")
    lines.append("")
    lines.append("| Metric | Result | Metric | Result |")
    lines.append("| :--- | :---: | :--- | :---: |")
    lines.append(
        f"| **Avg Delivered Ratio** | {report.avg_delivered_ratio:.1%} | **Avg Accounted Ratio** | {report.avg_accounted_ratio:.1%} |"
    )
    lines.append(
        f"| **Contract Errors** | {report.total_contract_errors} | **Missing Assets** | {report.total_missing_assets} |"
    )
    lines.append(
        f"| **Avg Layout Spill Rate** | {report.avg_spill_rate:.1%} | **Spill Pages** | {report.total_spill_pages} |"
    )
    lines.append(
        f"| **Total Tokens** | {report.total_tokens:,} | **Prompt Cache Hit Rate** | {report.overall_cache_hit_rate:.1%} |"
    )
    lines.append(
        f"| **Total Runtime** | {report.total_runtime_sec:.1f}s | **Total Estimated Cost** | {cost_str} |"
    )
    lines.append("")

    # Drift Summary Table
    if drift:
        lines.append("### Baseline Drift Summary")
        lines.append("")
        lines.append("| Baseline ID | Regressed | Improved | Stable | New | Verdict |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        verdict_str = "**PASS**" if drift.drift_passed else "**FAIL**"
        lines.append(
            f"| `{drift.baseline_run_id}` | {drift.regressed_count} | {drift.improved_count} | {drift.stable_count} | {drift.new_count} | {verdict_str} |"
        )
        lines.append("")

    # Case Breakdown Table
    lines.append("### Case Performance Matrix")
    lines.append("")
    drift_header = " Drift |" if drift else ""
    lines.append(
        f"| Case ID | Status | Delivered % | Accounted % | Errors | Spill Rate % | Max Overflow | Runtime | Cost |{drift_header}"
    )
    drift_sep = " :---: |" if drift else ""
    lines.append(
        f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |{drift_sep}"
    )

    drift_by_case = {cd.case_id: cd for cd in drift.case_drifts} if drift else {}

    for c in report.cases:
        cost_val = (
            f"${c.cost.estimated_cost_usd:.3f}" if c.cost.estimated_cost_usd is not None else "-"
        )
        drift_col = ""
        if drift:
            cd = drift_by_case.get(c.id)
            diff_str = (
                f" ({_format_diff_str(cd.delta_delivered_ratio, is_percentage=True)})"
                if cd and cd.verdict in ("REGRESSED", "IMPROVED")
                else ""
            )
            drift_col = f" {cd.verdict}{diff_str} |" if cd else " - |"
        lines.append(
            f"| `{c.id}` | {c.status.upper()} | {c.contract.delivered_ratio:.1%} | "
            f"{c.contract.accounted_ratio:.1%} | {c.contract.errors_count} | "
            f"{c.layout.spill_rate:.1%} | {c.layout.max_overflow_pt:.1f}pt | "
            f"{c.runtime_sec:.1f}s | {cost_val} |{drift_col}"
        )
    lines.append("")

    # Detailed Regressions
    if drift and drift.regressed_count > 0:
        lines.append("### Baseline Regressions Detail")
        lines.append("")
        for cd in drift.case_drifts:
            if cd.verdict == "REGRESSED":
                lines.append(f"#### Case: `{cd.case_id}`")
                for r in cd.reasons:
                    lines.append(f"- ❌ {r}")
                lines.append("")

    # Case Failures Detail
    failures = [
        c for c in report.cases if c.status == "fail" or (c.status == "pass" and not c.passed)
    ]
    if failures:
        lines.append("### Failures and Defect Details")
        lines.append("")
        for f in failures:
            lines.append(f"#### `{f.id}` — {f.status.upper()}")
            if f.failure_reason:
                lines.append(f"- **Reason**: {f.failure_reason}")
            for ef in f.contract.expectation_failures:
                lines.append(f"- **Contract violation**: {ef}")
            if not f.layout.layout_passed:
                lines.append(
                    f"- **Layout spill**: {f.layout.spill_count} anomalies, max overflow {f.layout.max_overflow_pt:.1f}pt"
                )
            lines.append("")

    return "\n".join(lines)


def render_html_dashboard(
    report: CorpusBenchmarkReport,
    drift: BenchmarkDriftReport | None = None,
    title: str = "UBT Corpus Benchmark Dashboard",
) -> str:
    """Generate a self-contained, responsive HTML5 visual dashboard."""
    overall_color = "#10b981" if report.overall_passed else "#ef4444"
    status_label = "PASSED" if report.overall_passed else "FAILED"
    drift_by_case = {cd.case_id: cd for cd in drift.case_drifts} if drift else {}
    cost_str = f"${report.total_cost_usd:.4f}" if report.total_cost_usd is not None else "Unpriced"

    case_rows_html = []
    for c in report.cases:
        badge_cls = (
            "badge-pass"
            if c.status == "pass"
            else ("badge-skip" if c.status == "skip" else "badge-fail")
        )
        deliv_pct = round(c.contract.delivered_ratio * 100, 1)
        spill_pct = round(c.layout.spill_rate * 100, 1)
        cost_val = (
            f"${c.cost.estimated_cost_usd:.3f}" if c.cost.estimated_cost_usd is not None else "—"
        )

        drift_html = ""
        if drift:
            cd = drift_by_case.get(c.id)
            if cd:
                d_cls = {
                    "REGRESSED": "badge-fail",
                    "IMPROVED": "badge-pass",
                    "STABLE": "badge-stable",
                    "NEW": "badge-skip",
                }.get(cd.verdict, "badge-stable")
                drift_html = (
                    f'<td><span class="badge {d_cls}">{html.escape(cd.verdict)}</span></td>'
                )
            else:
                drift_html = "<td>—</td>"

        detail_notes = []
        if c.failure_reason:
            detail_notes.append(f"<strong>Failure:</strong> {html.escape(c.failure_reason)}")
        for ef in c.contract.expectation_failures:
            detail_notes.append(f"<strong>Contract:</strong> {html.escape(ef)}")
        if not c.layout.layout_passed:
            detail_notes.append(
                f"<strong>Layout:</strong> {c.layout.spill_count} spills, max {c.layout.max_overflow_pt}pt"
            )

        detail_html = ""
        if detail_notes:
            detail_html = (
                f'<tr><td colspan="{10 if drift else 9}" class="details-cell">'
                f"<details><summary>Issues & Violations ({len(detail_notes)})</summary>"
                f"<ul>{''.join(f'<li>{note}</li>' for note in detail_notes)}</ul>"
                f"</details></td></tr>"
            )

        row = f"""
        <tr>
            <td class="font-mono"><strong>{html.escape(c.id)}</strong></td>
            <td><span class="badge {badge_cls}">{html.escape(c.status.upper())}</span></td>
            <td>
                <div class="meter-bar">
                    <div class="meter-fill fill-deliv" style="width: {deliv_pct}%;"></div>
                </div>
                <span class="meter-label">{deliv_pct}%</span>
            </td>
            <td>{round(c.contract.accounted_ratio * 100, 1)}%</td>
            <td class="{"text-danger" if c.contract.errors_count > 0 else ""}">{c.contract.errors_count}</td>
            <td>
                <div class="meter-bar">
                    <div class="meter-fill fill-spill" style="width: {min(100.0, spill_pct * 2)}%;"></div>
                </div>
                <span class="meter-label">{spill_pct}%</span>
            </td>
            <td>{c.layout.max_overflow_pt:.1f} pt</td>
            <td>{c.runtime_sec:.1f}s</td>
            <td>{cost_val}</td>
            {drift_html}
        </tr>
        {detail_html}
        """
        case_rows_html.append(row)

    drift_section_html = ""
    if drift:
        d_verdict_cls = "badge-pass" if drift.drift_passed else "badge-fail"
        d_verdict_text = (
            "PASS (NO REGRESSIONS)"
            if drift.drift_passed
            else f"FAIL ({drift.regressed_count} REGRESSED)"
        )
        drift_section_html = f"""
        <div class="card drift-card">
            <h3>Historical Baseline Drift Comparison</h3>
            <div class="drift-stats">
                <div>Baseline Run: <code>{html.escape(drift.baseline_run_id or "unknown")}</code></div>
                <div>Verdict: <span class="badge {d_verdict_cls}">{d_verdict_text}</span></div>
                <div>Regressed: <strong class="text-danger">{drift.regressed_count}</strong></div>
                <div>Improved: <strong class="text-success">{drift.improved_count}</strong></div>
                <div>Stable: <strong>{drift.stable_count}</strong></div>
            </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
    --bg-primary: #0f172a;
    --bg-card: #1e293b;
    --border: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --accent-sky: #0ea5e9;
    --accent-amber: #f59e0b;
}}
@media (prefers-color-scheme: light) {{
    :root {{
        --bg-primary: #f8fafc;
        --bg-card: #ffffff;
        --border: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #64748b;
    }}
}}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    margin: 0;
    padding: 24px;
    line-height: 1.5;
}}
.container {{
    max-width: 1200px;
    margin: 0 auto;
}}
header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
}}
h1, h2, h3 {{ margin: 0 0 8px 0; }}
.badge {{
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
.badge-pass {{ background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); border: 1px solid var(--accent-emerald); }}
.badge-fail {{ background: rgba(244, 63, 94, 0.2); color: var(--accent-rose); border: 1px solid var(--accent-rose); }}
.badge-skip {{ background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); border: 1px solid var(--accent-amber); }}
.badge-stable {{ background: rgba(14, 165, 233, 0.15); color: var(--accent-sky); border: 1px solid var(--accent-sky); }}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}}
.card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}}
.card-label {{
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
    font-weight: 600;
}}
.card-value {{
    font-size: 24px;
    font-weight: 700;
    margin-top: 4px;
}}
.drift-card {{
    border-left: 4px solid var(--accent-sky);
    margin-bottom: 24px;
}}
.drift-stats {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    font-size: 13px;
    margin-top: 8px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    font-size: 13px;
}}
th, td {{
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}}
th {{
    background: rgba(0, 0, 0, 0.05);
    color: var(--text-secondary);
    font-weight: 600;
}}
.meter-bar {{
    background: var(--border);
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    display: inline-block;
    width: 60px;
    vertical-align: middle;
    margin-right: 6px;
}}
.meter-fill {{ height: 100%; }}
.fill-deliv {{ background: var(--accent-emerald); }}
.fill-spill {{ background: var(--accent-rose); }}
.meter-label {{ font-size: 12px; font-variant-numeric: tabular-nums; }}
.details-cell {{ background: rgba(0,0,0,0.02); font-size: 12px; }}
details summary {{ cursor: pointer; color: var(--accent-sky); font-weight: 600; }}
.text-danger {{ color: var(--accent-rose); font-weight: 700; }}
.text-success {{ color: var(--accent-emerald); font-weight: 700; }}
.font-mono {{ font-family: monospace; }}
footer {{
    margin-top: 32px;
    text-align: center;
    font-size: 12px;
    color: var(--text-secondary);
}}
</style>
</head>
<body>
<div class="container">
    <header>
        <div>
            <h1>{html.escape(title)}</h1>
            <div style="font-size: 13px; color: var(--text-secondary);">
                Run ID: <code>{html.escape(report.run_id)}</code> &bull; Mode: <code>{html.escape(report.mode)}</code> &bull; {html.escape(report.timestamp)}
            </div>
        </div>
        <div>
            <span class="badge {"badge-pass" if report.overall_passed else "badge-fail"}" style="font-size: 14px; padding: 6px 12px;">
                {status_label}
            </span>
        </div>
    </header>

    <div class="grid">
        <div class="card">
            <div class="card-label">Pass Rate</div>
            <div class="card-value" style="color: {overall_color};">{report.passed_cases}/{report.total_cases}</div>
        </div>
        <div class="card">
            <div class="card-label">Avg Delivered</div>
            <div class="card-value">{report.avg_delivered_ratio:.1%}</div>
        </div>
        <div class="card">
            <div class="card-label">Contract Errors</div>
            <div class="card-value {"text-danger" if report.total_contract_errors > 0 else ""}">{report.total_contract_errors}</div>
        </div>
        <div class="card">
            <div class="card-label">Avg Spill Rate</div>
            <div class="card-value {"text-danger" if report.avg_spill_rate > 0.05 else ""}">{report.avg_spill_rate:.1%}</div>
        </div>
        <div class="card">
            <div class="card-label">Total Runtime</div>
            <div class="card-value">{report.total_runtime_sec:.1f}s</div>
        </div>
        <div class="card">
            <div class="card-label">Model Cost</div>
            <div class="card-value">{cost_str}</div>
        </div>
    </div>

    {drift_section_html}

    <div class="card" style="padding: 0; overflow-x: auto;">
        <table>
            <thead>
                <tr>
                    <th>Case ID</th>
                    <th>Status</th>
                    <th>Delivered</th>
                    <th>Accounted</th>
                    <th>Errors</th>
                    <th>Spill Rate</th>
                    <th>Max Overflow</th>
                    <th>Time</th>
                    <th>Cost</th>
                    {"<th>Drift</th>" if drift else ""}
                </tr>
            </thead>
            <tbody>
                {"".join(case_rows_html)}
            </tbody>
        </table>
    </div>

    <footer>
        Generated by UBT Corpus Benchmark Runner &bull; Universal Book Translator Engine
    </footer>
</div>
</body>
</html>
"""


def render_json_report(
    report: CorpusBenchmarkReport,
    drift: BenchmarkDriftReport | None = None,
) -> str:
    """Format benchmark report and optional drift into structured JSON."""
    payload: dict[str, Any] = {
        "report": json.loads(report.to_json()),
    }
    if drift:
        payload["drift"] = json.loads(drift.to_json())
    return json.dumps(payload, ensure_ascii=False, indent=2)

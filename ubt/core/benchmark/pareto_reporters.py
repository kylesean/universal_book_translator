"""Reporters for multi-model quality/cost Pareto analysis.

Provides:
- Rich terminal console table with ASCII scatter visualization.
- Standalone zero-CDN HTML5 visual interactive dashboard with embedded SVG Pareto curve.
- GitHub-Flavored Markdown report.
- Structured JSON export.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from ubt.core.benchmark.pareto import (
    ModelPoint,
    ParetoAnalysisReport,
    RecommendationCategory,
)

_default_console = Console()


def render_pareto_terminal(report: ParetoAnalysisReport, console: Console | None = None) -> None:
    """Print an interactive Rich terminal summary of the Pareto analysis."""
    c = console or _default_console
    frontier = report.frontier

    c.print(
        "\n[bold cyan]════════════════════════════════════════════════════════════════════════════════[/]"
    )
    c.print(
        f"[bold white]  UBT Multi-Model Pareto Frontier Analysis[/] [dim]({report.corpus_id})[/]"
    )
    c.print(
        f"[dim]  Generated at: {report.timestamp} | Minimum QE Threshold: {report.min_qe_threshold:.2f}[/]"
    )
    c.print(
        "[bold cyan]════════════════════════════════════════════════════════════════════════════════[/]\n"
    )

    # Key recommendation callout cards
    recs_table = Table(
        title="[bold yellow]★ Recommended Model Configurations[/]", box=None, padding=(0, 2)
    )
    recs_table.add_column("Category", style="bold", justify="left")
    recs_table.add_column("Model", style="bold cyan", justify="left")
    recs_table.add_column("Quality (QE)", style="green", justify="right")
    recs_table.add_column("Cost / 10k words", style="yellow", justify="right")
    recs_table.add_column("Throughput", style="magenta", justify="right")
    recs_table.add_column("Strategic Rationale", style="dim", justify="left")

    if frontier.sweet_spot:
        sp = frontier.sweet_spot
        recs_table.add_row(
            "[bold green]🎯 SWEET SPOT (Best Value)[/]",
            sp.model_name,
            f"{sp.quality_score:.3f}",
            f"${sp.cost_per_10k_words:.4f}",
            f"{sp.throughput_cps:.0f} cps",
            "Optimal marginal quality per dollar before diminishing returns",
        )
    if frontier.budget_pick:
        bp = frontier.budget_pick
        recs_table.add_row(
            "[bold blue]💰 ULTRA-BUDGET[/]",
            bp.model_name,
            f"{bp.quality_score:.3f}",
            f"${bp.cost_per_10k_words:.4f}",
            f"{bp.throughput_cps:.0f} cps",
            f"Lowest cost clearing minimum readability gate (≥{report.min_qe_threshold:.2f})",
        )
    if frontier.flagship_pick:
        fp = frontier.flagship_pick
        recs_table.add_row(
            "[bold magenta]👑 FLAGSHIP (High Fidelity)[/]",
            fp.model_name,
            f"{fp.quality_score:.3f}",
            f"${fp.cost_per_10k_words:.4f}",
            f"{fp.throughput_cps:.0f} cps",
            "Peak translation quality, domain terminology adherence & nuance",
        )
    if frontier.speed_pick:
        sp = frontier.speed_pick
        recs_table.add_row(
            "[bold cyan]⚡ SPEED CHAMPION[/]",
            sp.model_name,
            f"{sp.quality_score:.3f}",
            f"${sp.cost_per_10k_words:.4f}",
            f"{sp.throughput_cps:.0f} cps",
            "Fastest turnaround velocity while sustaining readability",
        )

    c.print(recs_table)
    c.print()

    # Detailed Comparison Matrix
    table = Table(
        title="[bold]Model Evaluation & Frontier Status Matrix[/]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Model Name", justify="left")
    table.add_column("Provider", justify="left")
    table.add_column("Quality", justify="right")
    table.add_column("Cost / 10k w", justify="right")
    table.add_column("Speed (cps)", justify="right")
    table.add_column("Frontier Status", justify="center")
    table.add_column("Marginal Return", justify="right")
    table.add_column("Recommendation", justify="left")

    # Sort all points: Pareto frontier first (by cost), then dominated points
    sorted_all = sorted(
        frontier.all_points,
        key=lambda p: (not p.is_pareto, p.cost_per_10k_words, -p.quality_score),
    )

    for p in sorted_all:
        status_badge = "[bold green]✔ PARETO[/]" if p.is_pareto else "[dim red]✘ Dominated[/]"
        rec_label = ""
        if p.recommendation == RecommendationCategory.SWEET_SPOT:
            rec_label = "[bold green]★ Sweet Spot[/]"
        elif p.recommendation == RecommendationCategory.BUDGET:
            rec_label = "[bold blue]● Budget[/]"
        elif p.recommendation == RecommendationCategory.FLAGSHIP:
            rec_label = "[bold magenta]▲ Flagship[/]"
        elif p.recommendation == RecommendationCategory.SPEED:
            rec_label = "[bold cyan]⚡ Speed[/]"

        marginal_str = f"+{p.marginal_return:.2f} ΔQ/$" if p.marginal_return is not None else "-"

        table.add_row(
            p.model_name,
            p.provider or "-",
            f"{p.quality_score:.3f}",
            f"${p.cost_per_10k_words:.4f}",
            f"{p.throughput_cps:.0f}",
            status_badge,
            marginal_str,
            rec_label,
        )

    c.print(table)
    c.print()


def render_pareto_markdown(report: ParetoAnalysisReport) -> str:
    """Render a GitHub Flavored Markdown report with callout boxes and matrices."""
    frontier = report.frontier
    lines: list[str] = [
        "# Multi-Model Quality vs. Cost Pareto Frontier Analysis",
        "",
        f"- **Corpus**: `{report.corpus_id}`",
        f"- **Evaluated At**: `{report.timestamp}`",
        f"- **Quality Gate Floor**: `{report.min_qe_threshold:.2f}` MTQE",
        "",
        "## Strategic Recommendations",
        "",
    ]

    if frontier.sweet_spot:
        sp = frontier.sweet_spot
        lines.extend(
            [
                "> [!TIP]",
                f"> **Sweet Spot (Best Value): `{sp.model_name}`**",
                f"> - Quality: `{sp.quality_score:.3f}` | Cost: `${sp.cost_per_10k_words:.4f}` per 10k words | Throughput: `{sp.throughput_cps:.0f}` cps",
                "> - *Optimal inflection point on the Pareto curve maximizing quality gain per dollar expended.*",
                "",
            ]
        )

    if frontier.budget_pick:
        bp = frontier.budget_pick
        lines.extend(
            [
                "> [!NOTE]",
                f"> **Ultra-Budget Configuration: `{bp.model_name}`**",
                f"> - Quality: `{bp.quality_score:.3f}` | Cost: `${bp.cost_per_10k_words:.4f}` per 10k words",
                f"> - *Lowest unit cost clearing the minimum readability threshold ({report.min_qe_threshold:.2f}).*",
                "",
            ]
        )

    if frontier.flagship_pick:
        fp = frontier.flagship_pick
        lines.extend(
            [
                "> [!IMPORTANT]",
                f"> **Flagship High-Fidelity Configuration: `{fp.model_name}`**",
                f"> - Quality: `{fp.quality_score:.3f}` | Cost: `${fp.cost_per_10k_words:.4f}` per 10k words",
                "> - *Highest translation fidelity, mathematical terminology adherence, and nuance preservation.*",
                "",
            ]
        )

    lines.extend(
        [
            "## Evaluated Model Options Matrix",
            "",
            "| Model Name | Provider | Quality (MTQE) | Cost / 10k Words | Speed (cps) | Pareto Status | Recommendation |",
            "| :--- | :--- | ---: | ---: | ---: | :---: | :--- |",
        ]
    )

    sorted_all = sorted(
        frontier.all_points,
        key=lambda p: (not p.is_pareto, p.cost_per_10k_words, -p.quality_score),
    )
    for p in sorted_all:
        status = "**Pareto Optimal**" if p.is_pareto else "Dominated"
        rec = p.recommendation.value.upper() if p.recommendation else "-"
        lines.append(
            f"| `{p.model_name}` | {p.provider} | `{p.quality_score:.3f}` | `${p.cost_per_10k_words:.4f}` | `{p.throughput_cps:.0f}` | {status} | `{rec}` |"
        )

    lines.append("")
    return "\n".join(lines)


def render_pareto_html_dashboard(report: ParetoAnalysisReport) -> str:
    """Render a standalone, self-contained HTML5 visual dashboard with embedded SVG Pareto chart.

    100% zero external CDN dependencies.
    """
    frontier = report.frontier

    # Calculate chart bounding coordinates
    all_points = frontier.all_points or []
    max_cost = max([p.cost_per_10k_words for p in all_points] + [0.10]) * 1.15
    min_cost = 0.0
    min_quality = 0.70
    max_quality = 1.00

    svg_width = 800
    svg_height = 420
    pad_left = 70
    pad_right = 40
    pad_top = 40
    pad_bottom = 50

    chart_w = svg_width - pad_left - pad_right
    chart_h = svg_height - pad_top - pad_bottom

    def x_scale(cost: float) -> float:
        ratio = (cost - min_cost) / max(0.0001, max_cost - min_cost)
        return pad_left + ratio * chart_w

    def y_scale(qe: float) -> float:
        ratio = (qe - min_quality) / max(0.0001, max_quality - min_quality)
        return (svg_height - pad_bottom) - ratio * chart_h

    # Build SVG path for the Pareto frontier curve
    frontier_pts = frontier.frontier_points
    curve_elements: list[str] = []
    if len(frontier_pts) > 1:
        path_data = []
        for i, pt in enumerate(frontier_pts):
            cx = x_scale(pt.cost_per_10k_words)
            cy = y_scale(pt.quality_score)
            if i == 0:
                path_data.append(f"M {cx:.1f} {cy:.1f}")
            else:
                path_data.append(f"L {cx:.1f} {cy:.1f}")
        curve_elements.append(
            f'<path d="{" ".join(path_data)}" fill="none" stroke="#10b981" stroke-width="3" stroke-dasharray="6,4" />'
        )

    # Threshold horizontal reference line
    threshold_y = y_scale(report.min_qe_threshold)
    curve_elements.append(
        f'<line x1="{pad_left}" y1="{threshold_y:.1f}" x2="{svg_width - pad_right}" y2="{threshold_y:.1f}" '
        f'stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4,4" />'
    )
    curve_elements.append(
        f'<text x="{pad_left + 10}" y="{threshold_y - 6:.1f}" fill="#ef4444" font-size="11" font-weight="bold">Min Quality Threshold ({report.min_qe_threshold:.2f})</text>'
    )

    # Plot points
    for p in all_points:
        cx = x_scale(p.cost_per_10k_words)
        cy = y_scale(p.quality_score)
        if p.is_pareto:
            color = "#10b981"
            radius = 7
            stroke = "#ffffff"
            stroke_w = 2
            if p.recommendation == RecommendationCategory.SWEET_SPOT:
                color = "#3b82f6"
                radius = 9
                stroke = "#fbbf24"
                stroke_w = 3
        else:
            color = "#94a3b8"
            radius = 5
            stroke = "#64748b"
            stroke_w = 1

        curve_elements.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_w}">'
            f"<title>{p.model_name} (QE: {p.quality_score:.3f}, Cost: ${p.cost_per_10k_words:.4f})</title>"
            f"</circle>"
        )
        # Model name label
        curve_elements.append(
            f'<text x="{cx + 9:.1f}" y="{cy + 4:.1f}" fill="#1e293b" font-size="11" font-weight="{"bold" if p.is_pareto else "normal"}">{p.model_name}</text>'
        )

    # Grid axes
    grid_svg = f"""
    <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{svg_height - pad_bottom}" stroke="#cbd5e1" stroke-width="1.5" />
    <line x1="{pad_left}" y1="{svg_height - pad_bottom}" x2="{svg_width - pad_right}" y2="{svg_height - pad_bottom}" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="{svg_width / 2}" y="{svg_height - 15}" text-anchor="middle" font-size="12" fill="#475569" font-weight="600">Cost (USD per 10,000 words)</text>
    <text x="-{svg_height / 2}" y="25" transform="rotate(-90)" text-anchor="middle" font-size="12" fill="#475569" font-weight="600">Translation Quality (MTQE Score)</text>
    """

    svg_content = f"""
    <svg viewBox="0 0 {svg_width} {svg_height}" class="chart-svg" xmlns="http://www.w3.org/2000/svg">
      {grid_svg}
      {"".join(curve_elements)}
    </svg>
    """

    # Recommendation cards HTML
    def _card(title: str, tag_class: str, pt: ModelPoint | None, desc: str) -> str:
        if not pt:
            return ""
        return f"""
        <div class="card {tag_class}">
          <div class="card-badge">{title}</div>
          <div class="card-model">{pt.model_name}</div>
          <div class="card-metrics">
            <span><strong>Quality:</strong> {pt.quality_score:.3f}</span>
            <span><strong>Cost / 10k:</strong> ${pt.cost_per_10k_words:.4f}</span>
            <span><strong>Speed:</strong> {pt.throughput_cps:.0f} cps</span>
          </div>
          <p class="card-desc">{desc}</p>
        </div>
        """

    c_sweet = _card(
        "🎯 Sweet Spot",
        "card-sweet",
        frontier.sweet_spot,
        "Optimal inflection point with highest marginal quality gain per dollar.",
    )
    c_budget = _card(
        "💰 Ultra Budget",
        "card-budget",
        frontier.budget_pick,
        f"Lowest cost configuration clearing quality threshold (≥{report.min_qe_threshold:.2f}).",
    )
    c_flag = _card(
        "👑 Flagship",
        "card-flagship",
        frontier.flagship_pick,
        "Uncompromised top-tier quality for demanding publications.",
    )
    c_speed = _card(
        "⚡ Speed",
        "card-speed",
        frontier.speed_pick,
        "Fastest turnaround velocity while sustaining readability.",
    )

    # Table rows
    sorted_all = sorted(
        all_points,
        key=lambda p: (not p.is_pareto, p.cost_per_10k_words, -p.quality_score),
    )
    table_rows = []
    for p in sorted_all:
        pareto_badge = (
            '<span class="badge badge-pareto">Pareto</span>'
            if p.is_pareto
            else '<span class="badge badge-dom">Dominated</span>'
        )
        rec_badge = ""
        if p.recommendation == RecommendationCategory.SWEET_SPOT:
            rec_badge = '<span class="badge badge-sweet">Sweet Spot</span>'
        elif p.recommendation == RecommendationCategory.BUDGET:
            rec_badge = '<span class="badge badge-budget">Budget</span>'
        elif p.recommendation == RecommendationCategory.FLAGSHIP:
            rec_badge = '<span class="badge badge-flag">Flagship</span>'
        elif p.recommendation == RecommendationCategory.SPEED:
            rec_badge = '<span class="badge badge-spd">Speed</span>'

        table_rows.append(f"""
        <tr>
          <td><strong>{p.model_name}</strong></td>
          <td>{p.provider}</td>
          <td>{p.quality_score:.3f}</td>
          <td>${p.cost_per_10k_words:.4f}</td>
          <td>{p.throughput_cps:.0f} cps</td>
          <td>{pareto_badge}</td>
          <td>{rec_badge}</td>
        </tr>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UBT Multi-Model Pareto Frontier Analysis</title>
<style>
  :root {{
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --text: #0f172a;
    --border: #e2e8f0;
    --primary: #2563eb;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  header {{ margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }}
  h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
  .subtitle {{ color: #64748b; font-size: 14px; margin: 0; }}
  .cards-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }}
  .card {{
    background: var(--card-bg);
    border-radius: 10px;
    padding: 16px;
    border: 1px solid var(--border);
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    position: relative;
  }}
  .card-sweet {{ border-left: 4px solid var(--primary); }}
  .card-budget {{ border-left: 4px solid var(--success); }}
  .card-flagship {{ border-left: 4px solid var(--warning); }}
  .card-speed {{ border-left: 4px solid #06b6d4; }}
  .card-badge {{ font-size: 12px; font-weight: bold; text-transform: uppercase; color: #64748b; margin-bottom: 6px; }}
  .card-model {{ font-size: 18px; font-weight: bold; color: var(--text); margin-bottom: 8px; }}
  .card-metrics {{ display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: #334155; margin-bottom: 8px; }}
  .card-desc {{ font-size: 12px; color: #64748b; margin: 0; line-height: 1.4; }}
  .chart-box {{
    background: var(--card-bg);
    border-radius: 10px;
    border: 1px solid var(--border);
    padding: 20px;
    margin-bottom: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .chart-svg {{ width: 100%; height: auto; max-height: 440px; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card-bg);
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
    font-size: 14px;
  }}
  th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: #f1f5f9; font-weight: 600; color: #475569; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
  .badge-pareto {{ background: #dcfce7; color: #15803d; }}
  .badge-dom {{ background: #f1f5f9; color: #94a3b8; }}
  .badge-sweet {{ background: #dbeafe; color: #1d4ed8; }}
  .badge-budget {{ background: #d1fae5; color: #047857; }}
  .badge-flag {{ background: #fef3c7; color: #b45309; }}
  .badge-spd {{ background: #cffafe; color: #0e7490; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>UBT Multi-Model Quality vs. Cost Pareto Frontier Analysis</h1>
    <p class="subtitle">Corpus: {report.corpus_id} | Timestamp: {report.timestamp} | Minimum Quality Floor: {report.min_qe_threshold:.2f}</p>
  </header>

  <div class="cards-grid">
    {c_sweet}
    {c_budget}
    {c_flag}
    {c_speed}
  </div>

  <div class="chart-box">
    <h3 style="margin-top: 0; margin-bottom: 12px;">Quality vs. Cost Trade-off Curve (Pareto Frontier)</h3>
    {svg_content}
  </div>

  <div>
    <h3 style="margin-top: 0; margin-bottom: 12px;">Candidate Model Evaluation & Frontier Roster</h3>
    <table>
      <thead>
        <tr>
          <th>Model Name</th>
          <th>Provider</th>
          <th>Quality (MTQE)</th>
          <th>Cost / 10k Words</th>
          <th>Throughput</th>
          <th>Frontier Status</th>
          <th>Recommendation</th>
        </tr>
      </thead>
      <tbody>
        {"".join(table_rows)}
      </tbody>
    </table>
  </div>
</div>
</body>
</html>
"""
    return html


def render_pareto_json(report: ParetoAnalysisReport) -> str:
    """Serialize the report to JSON format."""
    return report.to_json(indent=2)


__all__ = [
    "render_pareto_html_dashboard",
    "render_pareto_json",
    "render_pareto_markdown",
    "render_pareto_terminal",
]

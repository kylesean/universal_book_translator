"""Multi-Language Cross-Lingual Quality Matrix Auditor.

Aggregates bidirectional back-translation results across multiple language pairs
(e.g., en-zh, zh-en, en-ja, zh-ja) to produce comprehensive MQM alignment reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.core.qe.roundtrip import RoundTripResult, RoundTripVerifier


@dataclass
class MatrixAuditSummary:
    """Aggregated quality metrics for a single language direction or pair."""

    direction: str  # e.g., "en <-> zh"
    sample_count: int = 0
    passed_count: int = 0
    pass_rate: float = 0.0
    avg_composite_score: float = 0.0
    avg_bleu: float = 0.0
    avg_fuzzy: float = 0.0
    entity_conservation_rate: float = 0.0
    polarity_accuracy_rate: float = 0.0
    mqm_critical_count: int = 0
    mqm_major_count: int = 0
    mqm_minor_count: int = 0
    results: list[RoundTripResult] = field(default_factory=list)

    @classmethod
    def from_results(cls, direction: str, results: Sequence[RoundTripResult]) -> MatrixAuditSummary:
        if not results:
            return cls(direction=direction)

        n = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_comp = sum(r.composite_score for r in results) / float(n)
        avg_bleu = sum(r.bleu_score for r in results) / float(n)
        avg_fuzzy = sum(r.fuzzy_score for r in results) / float(n)
        avg_ent = sum(r.entity_conservation for r in results) / float(n)
        avg_pol = sum(1.0 if r.polarity_conserved else 0.0 for r in results) / float(n)

        crit = sum(1 for r in results if r.mqm_severity == "critical")
        major = sum(1 for r in results if r.mqm_severity == "major")
        minor = sum(1 for r in results if r.mqm_severity == "minor")

        return cls(
            direction=direction,
            sample_count=n,
            passed_count=passed,
            pass_rate=round(float(passed) / float(n) * 100.0, 1),
            avg_composite_score=round(avg_comp, 4),
            avg_bleu=round(avg_bleu, 4),
            avg_fuzzy=round(avg_fuzzy, 4),
            entity_conservation_rate=round(avg_ent * 100.0, 1),
            polarity_accuracy_rate=round(avg_pol * 100.0, 1),
            mqm_critical_count=crit,
            mqm_major_count=major,
            mqm_minor_count=minor,
            results=list(results),
        )


class MultiLanguageMatrixAuditor:
    """Orchestrates bidirectional roundtrip evaluation across a matrix of language pairs."""

    def __init__(self, verifier: RoundTripVerifier | None = None) -> None:
        self.verifier = verifier or RoundTripVerifier()

    async def audit_pairs(
        self,
        pairs: Sequence[tuple[str, str, str | None]],  # (src, mt, back)
        src_lang: str = "en",
        tgt_lang: str = "zh",
    ) -> MatrixAuditSummary:
        """Audit a collection of translation pairs for one language pair."""
        results: list[RoundTripResult] = []
        for src, mt, back in pairs:
            res = await self.verifier.verify_pair(
                src=src,
                mt=mt,
                back_translated=back,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
            results.append(res)

        direction = f"{src_lang} <-> {tgt_lang}"
        return MatrixAuditSummary.from_results(direction, results)


def render_terminal_matrix(
    summaries: Sequence[MatrixAuditSummary],
    console: Console | None = None,
) -> None:
    """Print an interactive Rich terminal report card for the quality matrix."""
    con = console or Console()

    table = Table(title="UBT Multi-Language Bidirectional Quality Matrix", expand=True)
    table.add_column("Language Pair", style="bold cyan", no_wrap=True)
    table.add_column("Samples", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("BLEU", justify="right")
    table.add_column("Entity Retention", justify="right")
    table.add_column("Polarity Acc", justify="right")
    table.add_column("MQM Critical", justify="center", style="bold red")
    table.add_column("MQM Major", justify="center", style="yellow")
    table.add_column("MQM Minor", justify="center", style="dim")

    for s in summaries:
        pass_color = "green" if s.pass_rate >= 80.0 else "yellow" if s.pass_rate >= 60.0 else "red"
        crit_color = "bold red" if s.mqm_critical_count > 0 else "dim"

        table.add_row(
            s.direction,
            str(s.sample_count),
            f"[{pass_color}]{s.pass_rate:.1f}%[/]",
            f"{s.avg_composite_score:.3f}",
            f"{s.avg_bleu:.3f}",
            f"{s.entity_conservation_rate:.1f}%",
            f"{s.polarity_accuracy_rate:.1f}%",
            f"[{crit_color}]{s.mqm_critical_count}[/]",
            str(s.mqm_major_count),
            str(s.mqm_minor_count),
        )

    con.print(Panel(table, border_style="cyan"))


def render_markdown_matrix(summaries: Sequence[MatrixAuditSummary]) -> str:
    """Format the matrix summary as GitHub Flavored Markdown."""
    lines: list[str] = [
        "# UBT Multi-Language Bidirectional Quality Matrix & MQM Audit",
        "",
        "> [!NOTE]",
        "> Evaluated using bidirectional round-trip back-translation, token-smoothed BLEU,",
        "> numeric/entity conservation tracking, and polarity inversion guards.",
        "",
        "| Language Pair | Samples | Pass Rate | Composite Score | BLEU | Entity Retention | Polarity Accuracy | Critical Defects | Major Defects |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in summaries:
        lines.append(
            f"| `{s.direction}` | {s.sample_count} | **{s.pass_rate:.1f}%** | {s.avg_composite_score:.3f} | {s.avg_bleu:.3f} | {s.entity_conservation_rate:.1f}% | {s.polarity_accuracy_rate:.1f}% | {s.mqm_critical_count} | {s.mqm_major_count} |"
        )

    lines.append("")
    return "\n".join(lines)


def render_json_matrix(summaries: Sequence[MatrixAuditSummary]) -> dict[str, Any]:
    """Serialize the matrix audit summaries into structured JSON."""
    return {
        "matrix_summaries": [
            {
                "direction": s.direction,
                "sample_count": s.sample_count,
                "passed_count": s.passed_count,
                "pass_rate": s.pass_rate,
                "avg_composite_score": s.avg_composite_score,
                "avg_bleu": s.avg_bleu,
                "avg_fuzzy": s.avg_fuzzy,
                "entity_conservation_rate": s.entity_conservation_rate,
                "polarity_accuracy_rate": s.polarity_accuracy_rate,
                "mqm_critical_count": s.mqm_critical_count,
                "mqm_major_count": s.mqm_major_count,
                "mqm_minor_count": s.mqm_minor_count,
                "results": [asdict(r) for r in s.results],
            }
            for s in summaries
        ]
    }

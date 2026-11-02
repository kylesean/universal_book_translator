"""Typst adaptive layout auto-healing.

A page that compiles can still ship ink outside the margins, and the human fix is
to shorten what overflowed. :meth:`LayoutAutoHealer.run_healing_loop` does the
mechanical version: it reads the layout audit of the delivered PDF, rebuilds the
document with tighter tables, formulas and paragraph spacing, recompiles, and
promotes a candidate **only** when the measured spill strictly shrinks — a pass
that made the page worse is thrown away rather than delivered.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ubt.core.ir.models import IRBlock
from ubt.render.typst.layout_diagnostics import (
    AnomalyType,
    PageLayoutSpec,
    diagnose_pdf_layout,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AutoHealingAction:
    """A discrete layout remediation action applied during auto-healing."""

    action_type: str
    description: str
    pass_num: int
    delta_errors_resolved: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "description": self.description,
            "pass_num": self.pass_num,
            "delta_errors_resolved": self.delta_errors_resolved,
        }


@dataclass(slots=True)
class AutoHealingReport:
    """Summary record of layout auto-healing passes and outcomes."""

    initial_passed: bool
    final_passed: bool
    initial_error_count: int
    final_error_count: int
    initial_max_overflow_pt: float
    final_max_overflow_pt: float
    passes_attempted: int = 0
    healed: bool = False
    actions: list[AutoHealingAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_passed": self.initial_passed,
            "final_passed": self.final_passed,
            "initial_error_count": self.initial_error_count,
            "final_error_count": self.final_error_count,
            "initial_max_overflow_pt": round(self.initial_max_overflow_pt, 2),
            "final_max_overflow_pt": round(self.final_max_overflow_pt, 2),
            "passes_attempted": self.passes_attempted,
            "healed": self.healed,
            "actions": [a.to_dict() for a in self.actions],
        }


class LayoutAutoHealer:
    def __init__(
        self,
        spec: PageLayoutSpec,
        max_passes: int = 2,
    ) -> None:
        self.spec = spec
        self.max_passes = max(1, max_passes)

    def run_healing_loop(
        self,
        output_path: Path,
        build_dir: Path,
        blocks: list[IRBlock],
        target_lang: str,
        renderer_builder_fn: Callable[..., Any],
        compile_fn: Callable[[str, str], tuple[bool, str]],
    ) -> tuple[bool, AutoHealingReport]:
        initial_rep = diagnose_pdf_layout(output_path, spec=self.spec)
        initial_max_ov = max((a.delta_pt for a in initial_rep.all_anomalies), default=0.0)

        report = AutoHealingReport(
            initial_passed=initial_rep.passed,
            final_passed=initial_rep.passed,
            initial_error_count=initial_rep.error_count,
            final_error_count=initial_rep.error_count,
            initial_max_overflow_pt=initial_max_ov,
            final_max_overflow_pt=initial_max_ov,
        )

        if initial_rep.passed:
            return True, report

        logger.info(
            "Physical layout spills detected in %s (%d errors, max %.1fpt). Initiating auto-healing loop...",
            output_path.name,
            initial_rep.error_count,
            initial_max_ov,
        )

        current_rep = initial_rep
        current_max_ov = initial_max_ov
        actions: list[AutoHealingAction] = []

        for pass_num in range(1, self.max_passes + 1):
            report.passes_attempted = pass_num

            has_horizontal_spill = any(
                a.anomaly_type
                in (
                    AnomalyType.MARGIN_SPILL_RIGHT,
                    AnomalyType.MARGIN_SPILL_LEFT,
                    AnomalyType.GUTTER_OVERRUN,
                )
                for a in current_rep.all_anomalies
            )
            has_vertical_spill = any(
                a.anomaly_type
                in (
                    AnomalyType.MARGIN_SPILL_BOTTOM,
                    AnomalyType.MARGIN_SPILL_TOP,
                )
                for a in current_rep.all_anomalies
            )

            if pass_num == 1:
                table_scale_factor = 0.88 if has_horizontal_spill else 1.0
                formula_scale = 0.90 if has_horizontal_spill else 1.0
                break_formulas = has_horizontal_spill
                par_spacing_em = 0.82 if has_vertical_spill else 0.9
            else:
                table_scale_factor = 0.80 if has_horizontal_spill else 0.88
                formula_scale = 0.82 if has_horizontal_spill else 0.88
                break_formulas = True
                par_spacing_em = 0.75 if has_vertical_spill else 0.82

            healed_renderer = renderer_builder_fn(
                table_scale_factor=table_scale_factor,
                formula_scale=formula_scale,
                break_long_formulas=break_formulas,
                par_spacing_em=par_spacing_em,
            )

            healed_source, _ = healed_renderer.build_source(blocks, build_dir, target_lang)
            healed_typ = build_dir / f"main_healed_p{pass_num}.typ"
            healed_pdf = build_dir / f"healed_p{pass_num}.pdf"
            healed_typ.write_text(healed_source, encoding="utf-8")

            ok, stderr = compile_fn(str(healed_typ), str(healed_pdf))
            if not ok or not healed_pdf.exists():
                logger.warning(
                    "Auto-healing pass %d compile failed: %s",
                    pass_num,
                    stderr.strip(),
                )
                continue

            try:
                healed_rep = diagnose_pdf_layout(healed_pdf, spec=self.spec)
            except Exception as exc:
                # A candidate the auditor cannot even open is worse than the
                # spill it was meant to fix. Discard it and keep the original
                # delivery: the healer must never turn a readable PDF into a
                # failure.
                logger.warning(
                    "Auto-healing pass %d produced an unreadable PDF; discarding it: %s",
                    pass_num,
                    exc,
                )
                continue
            healed_max_ov = max((a.delta_pt for a in healed_rep.all_anomalies), default=0.0)

            improved = healed_rep.error_count < current_rep.error_count or (
                healed_rep.error_count == current_rep.error_count
                and healed_max_ov < current_max_ov - 1.0
            )

            if improved:
                delta_resolved = current_rep.error_count - healed_rep.error_count
                action = AutoHealingAction(
                    action_type="adaptive_layout_scaling",
                    description=(
                        f"Pass {pass_num}: table_scale={table_scale_factor}, "
                        f"formula_scale={formula_scale}, break_math={break_formulas}, "
                        f"par_spacing={par_spacing_em}em"
                    ),
                    pass_num=pass_num,
                    delta_errors_resolved=delta_resolved,
                )
                actions.append(action)

                shutil.copy2(healed_pdf, output_path)
                current_rep = healed_rep
                current_max_ov = healed_max_ov

                logger.info(
                    "Auto-healing pass %d succeeded: errors %d -> %d, max overflow %.1fpt -> %.1fpt",
                    pass_num,
                    report.final_error_count,
                    healed_rep.error_count,
                    report.final_max_overflow_pt,
                    healed_max_ov,
                )

                report.final_error_count = healed_rep.error_count
                report.final_max_overflow_pt = healed_max_ov
                report.final_passed = healed_rep.passed
                report.healed = True

                if healed_rep.passed:
                    break

        report.actions = actions
        return report.final_passed, report


__all__ = [
    "AutoHealingAction",
    "AutoHealingReport",
    "LayoutAutoHealer",
]

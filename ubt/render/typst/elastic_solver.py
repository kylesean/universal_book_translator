"""Three-tier elasticity solver for variable font micro-typography compaction.

Absorbs positive text expansion (deficit_ratio = measured_h / allowed_h - 1.0) through:
- Tier 1: Micro-leading reduction (leading_em: 0.62 -> 0.55 -> 0.48)
- Tier 2: Variable font horizontal stretch (stretch_pct: 100% -> 95% -> 90%)
- Tier 3: Font scale envelope safety floor (scale: 1.00 -> 0.92, never below 0.92)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ElasticTypeStyle:
    """Calculated styling parameters under bounding box constraints."""

    font_size_pt: float
    leading_em: float
    stretch_pct: float
    scale: float = 1.0


class TypeElasticitySolver:
    """Calculates optimal typography parameters (font_size, leading, stretch) under spatial constraints."""

    # Leading steps (in em gap above font base)
    LEADING_STEPS = (0.62, 0.55, 0.48)
    # Variable font horizontal stretch percentages
    STRETCH_STEPS = (100.0, 95.0, 90.0)
    # Hard floor: font size scale never drops below 92% of base
    SCALE_MIN = 0.92

    @classmethod
    def resolve_params(
        cls,
        base_font_size: float,
        deficit_ratio: float,
    ) -> ElasticTypeStyle:
        """Resolve optimal ElasticTypeStyle given measured text deficit ratio.

        Args:
            base_font_size: Base font size in points.
            deficit_ratio: (measured_height / allowed_height) - 1.0. Negative or 0 means fits within budget.
        """
        if deficit_ratio <= 0.0:
            return ElasticTypeStyle(
                font_size_pt=base_font_size,
                leading_em=cls.LEADING_STEPS[0],
                stretch_pct=cls.STRETCH_STEPS[0],
                scale=1.0,
            )

        # Tier 1: adjust leading only (absorbs up to ~12% vertical height deficit)
        if deficit_ratio <= 0.12:
            return ElasticTypeStyle(
                font_size_pt=base_font_size,
                leading_em=cls.LEADING_STEPS[1],
                stretch_pct=cls.STRETCH_STEPS[0],
                scale=1.0,
            )

        # Tier 2: compact leading + 95% variable font stretch (absorbs up to ~22% height deficit, 0 font scale reduction)
        if deficit_ratio <= 0.22:
            return ElasticTypeStyle(
                font_size_pt=base_font_size,
                leading_em=cls.LEADING_STEPS[2],
                stretch_pct=cls.STRETCH_STEPS[1],
                scale=1.0,
            )

        # Tier 3: compact leading + 90% variable font stretch + gentle font scaling (floor >= 0.92)
        scale = max(cls.SCALE_MIN, 1.0 - (deficit_ratio - 0.22) * 0.5)
        return ElasticTypeStyle(
            font_size_pt=round(base_font_size * scale, 2),
            leading_em=cls.LEADING_STEPS[2],
            stretch_pct=cls.STRETCH_STEPS[2],
            scale=round(scale, 3),
        )

"""The render decision and the renderer's outcome (compiler render plan protocol).

``RenderPlan`` is what the advisories decide about *this run's* render: the mode,
the chrome/cover treatment, and whether a companion artifact was requested. It
used to be written onto ``manifest.run`` -- the run manifest used as an
inter-stage bus -- and read back by the renderer, the export stage and the
visual gate. It is a value the plan owns and hands to the stage that acts on it.

There is no render-engine field: every PDF composes through the single
source-canvas ``overlay`` engine (``ubt.core.config.RENDER_ENGINE``).

``RenderOutcome`` is the other half: what the renderer *actually did* -- today
only the effective bilingual mode it shipped. The adapter records it after
``render_blocks`` returns, so the renderer never writes a decision back onto the
manifest.

This module lives in ``ubt.core.ir`` because both the adapters (which read the
plan and produce the outcome) and ``ubt.pipeline`` (which owns the plan) depend on
it, and (per ``ubt.core.ports``) neither layer may depend on the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RenderPlan:
    """What the advisories decided this run should render.

    ``bilingual_mode`` is the adapter-domain mode (``RENDER_MODE_VALUE``), while
    ``effective_dual_mode`` is the advisor ``DualMode``; both are carried because
    the renderer reads the former and the report names the latter.
    """

    bilingual_mode: str | None = None
    effective_dual_mode: str | None = None
    dual_mode_downgraded: str | None = None
    facing_spread: bool = False
    translate_chrome: bool | None = None
    cover_mode: str | None = None
    bilingual_advisory: dict[str, Any] | None = None
    emit_secondary_mode: str | None = None


@dataclass
class RenderOutcome:
    """What the renderer actually used.

    ``dual_mode_downgraded`` is set only when the renderer replaced a requested
    bilingual mode with monolingual.
    """

    bilingual_mode: str | None = None
    effective_dual_mode: str | None = None
    dual_mode_downgraded: str | None = None


__all__ = ["RenderOutcome", "RenderPlan"]

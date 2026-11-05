"""The render decision and the renderer's outcome.

``RenderPlan`` is what the advisories decide about *this run's* render: the mode,
the chrome/cover treatment, and whether a companion artifact was requested. It
is a value the plan owns and hands to the stage that acts on it; the renderer,
the export stage and the visual gate read it as an argument rather than from the
run manifest.

There is no render-engine field: every PDF is rebuilt from the semantic IR by
the Typst publication renderer.

``RenderOutcome`` is the other half: what the renderer *actually did* -- today
only the effective bilingual mode it shipped. The adapter records it after
``render_blocks`` returns, so the renderer never writes a decision back onto the
manifest.

This module lives in ``ubt.core.ir`` because both the adapters (which read the
plan and produce the outcome) and the export stage (which owns the plan) depend
on it, and (per ``ubt.core.ports``) neither layer may depend on the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Advisor ``DualMode`` -> adapter ``bilingual_mode``. Kept as a plain mapping
#: here rather than imported from ``ubt.core.policy``: ``ubt.core.ir`` is the
#: shared leaf every layer reads, and a ``core.policy -> core.ir.models`` edge
#: already exists, so importing the other way would close a package cycle.
_DUAL_TO_ADAPTER_MODE: dict[str, str] = {
    "dual": "bilingual",
    "mono": "monolingual",
}


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
    translate_chrome: bool | None = None
    bilingual_advisory: dict[str, Any] | None = None
    emit_secondary_mode: str | None = None

    def adapter_mode(self, *, fallback: str | None = None) -> str | None:
        """The one value every caller passes to ``render_blocks``.

        The adapter-domain ``bilingual_mode`` wins when present (it is what the
        advisories wrote and what the export stage persists). A plan carrying
        only the advisor's ``effective_dual_mode`` is translated here, once, so
        preflight and export cannot disagree about which vocabulary they were
        handed: preflight used to map the advisor value itself while export
        passed the adapter value straight through, and the rehearsal took the
        bilingual branch when the real render did not.
        """
        if self.bilingual_mode:
            return self.bilingual_mode
        if self.effective_dual_mode:
            return _DUAL_TO_ADAPTER_MODE.get(self.effective_dual_mode, self.effective_dual_mode)
        return fallback


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

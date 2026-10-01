"""Shadow comparison for the renderer handshake (ADR-0001 final cut).

The render decision is moving off ``manifest.run`` and onto ``RenderPlan`` (the
advisory's decision) and ``RenderOutcome`` (what the renderer actually used).
While both channels exist, the legacy ``manifest.run`` fields are kept populated
and this comparator asserts the two agree. It is the acceptance evidence that the
new channel is *equivalent* before the manifest fields are deleted -- and it is
deleted together with them.

After a render the ``manifest.run`` mode fields reflect the renderer's outcome
(it mirrors the rigid downgrade), so the mode fields are compared against the
outcome and the policy fields against the plan.
"""

from __future__ import annotations

from typing import Any

from ubt.core.ir.render_plan import RenderOutcome, RenderPlan


def render_handshake_drift(
    plan: RenderPlan,
    outcome: RenderOutcome | None,
    manifest: Any,
) -> list[str]:
    """Return one line per value where the new channel and ``manifest.run`` differ.

    Empty means the two channels agree (the migration is safe to finish).
    """
    run = getattr(manifest, "run", None)
    if run is None:
        return []

    def mismatch(name: str, new: Any, old: Any) -> str | None:
        if new != old:
            return f"{name}: plan/outcome={new!r} manifest.run={old!r}"
        return None

    checks: list[str | None] = [
        # Policy fields the advisory owns (the renderer does not touch them).
        mismatch("render_engine", plan.render_engine, getattr(run, "render_engine", None)),
        mismatch("translate_chrome", plan.translate_chrome, getattr(run, "translate_chrome", None)),
        mismatch("cover_mode", plan.cover_mode, getattr(run, "cover_mode", None)),
        mismatch(
            "facing_spread",
            bool(plan.facing_spread),
            bool(getattr(run, "facing_spread", None)),
        ),
    ]
    # Mode fields: after a render the run channel holds the outcome (the
    # renderer mirrors its rigid downgrade there), so compare against it; before
    # one it holds the plan.
    if outcome is not None:
        checks += [
            mismatch(
                "bilingual_mode",
                outcome.bilingual_mode,
                getattr(run, "bilingual_mode", None),
            ),
            mismatch(
                "effective_dual_mode",
                outcome.effective_dual_mode,
                getattr(run, "effective_dual_mode", None),
            ),
            mismatch(
                "dual_mode_downgraded",
                outcome.dual_mode_downgraded,
                getattr(run, "dual_mode_downgraded", None),
            ),
        ]
    else:
        checks += [
            mismatch("bilingual_mode", plan.bilingual_mode, getattr(run, "bilingual_mode", None)),
            mismatch(
                "effective_dual_mode",
                plan.effective_dual_mode,
                getattr(run, "effective_dual_mode", None),
            ),
            mismatch(
                "dual_mode_downgraded",
                plan.dual_mode_downgraded,
                getattr(run, "dual_mode_downgraded", None),
            ),
        ]
    return [line for line in checks if line]


__all__ = ["render_handshake_drift"]

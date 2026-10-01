#!/usr/bin/env python
"""Renderer-handshake acceptance: the new channel matches the legacy manifest.run.

While the render decision moves off ``manifest.run`` and onto ``RenderPlan`` (the
advisory's decision) and ``RenderOutcome`` (what the renderer actually used), the
export stage keeps the legacy channel populated and asserts the two agree
(ADR-0001 final cut). This harness exercises that comparator directly:

- an agreeing plan/outcome/manifest triple yields no drift;
- each drifting field is reported;
- after a render the mode fields are judged against the *outcome* (the renderer
  mirrors its rigid downgrade onto the manifest), so a rigid render whose run
  channel shows monolingual agrees even though the plan asked for bilingual.
"""

from __future__ import annotations

from typing import Any

from ubt.core.ir.render_plan import RenderOutcome, RenderPlan
from ubt.core.ir.run_metadata import RunMetadata
from ubt.pipeline.render_shadow import render_handshake_drift


class _Manifest:
    """Just the ``run`` attribute the comparator reads."""

    def __init__(self, run: RunMetadata) -> None:
        self.run = run


def _plan(**kw: Any) -> RenderPlan:
    base: dict[str, Any] = {
        "bilingual_mode": "bilingual",
        "effective_dual_mode": "inline",
        "dual_mode_downgraded": None,
        "facing_spread": False,
        "render_engine": "auto",
        "translate_chrome": False,
        "cover_mode": "auto",
    }
    base.update(kw)
    return RenderPlan(**base)


def _run(**kw: Any) -> RunMetadata:
    base: dict[str, Any] = {
        "bilingual_mode": "bilingual",
        "effective_dual_mode": "inline",
        "dual_mode_downgraded": None,
        "facing_spread": False,
        "render_engine": "auto",
        "translate_chrome": False,
        "cover_mode": "auto",
    }
    base.update(kw)
    return RunMetadata(**base)


def main() -> int:
    problems: list[str] = []

    def expect(
        name: str, plan: RenderPlan, outcome: RenderOutcome | None, run: RunMetadata, drift: bool
    ) -> None:
        found = render_handshake_drift(plan, outcome, _Manifest(run))
        if bool(found) != drift:
            problems.append(f"{name}: expected {'drift' if drift else 'agreement'}, got {found!r}")

    outcome = RenderOutcome(
        bilingual_mode="bilingual", effective_dual_mode="inline", dual_mode_downgraded=None
    )

    # Agreement: plan == outcome == manifest.run.
    expect("agree", _plan(), outcome, _run(), drift=False)

    # Every policy field drifting is caught.
    expect("drift render_engine", _plan(), outcome, _run(render_engine="rigid"), drift=True)
    expect("drift translate_chrome", _plan(), outcome, _run(translate_chrome=True), drift=True)
    expect("drift cover_mode", _plan(), outcome, _run(cover_mode="first_page"), drift=True)
    expect("drift facing_spread", _plan(), outcome, _run(facing_spread=True), drift=True)

    # The mode fields are judged against the outcome after a render.
    rigid_outcome = RenderOutcome(
        bilingual_mode="monolingual",
        effective_dual_mode="monolingual",
        dual_mode_downgraded="inline",
    )
    rigid_plan = _plan(bilingual_mode="bilingual", effective_dual_mode="inline")
    expect(
        "rigid downgrade mirrored",
        rigid_plan,
        rigid_outcome,
        _run(
            bilingual_mode="monolingual",
            effective_dual_mode="monolingual",
            dual_mode_downgraded="inline",
        ),
        drift=False,
    )
    # The manifest not mirroring the outcome is the drift the shadow exists to catch.
    expect(
        "rigid downgrade not mirrored",
        rigid_plan,
        rigid_outcome,
        _run(bilingual_mode="bilingual", effective_dual_mode="inline"),
        drift=True,
    )
    # A manifest whose mode matches the plan but not the outcome still drifts.
    expect(
        "outcome differs from manifest",
        _plan(),
        rigid_outcome,
        _run(),
        drift=True,
    )

    print("\nRenderer-handshake shadow (RenderPlan/RenderOutcome vs manifest.run)")
    print(f"  checks=9 problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"    {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

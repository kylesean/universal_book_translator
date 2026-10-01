#!/usr/bin/env python
"""Renderer-handshake acceptance: where the render decision lives (ADR-0001).

The bilingual/engine mode cluster was the last inter-stage bus on
``manifest.run``: the PDF renderer read and rewrote it across the adapter
boundary. It now lives in typed values --

- ``ubt.core.ir.render_plan.RenderPlan`` -- the advisories' decision, threaded to
  the renderer as an explicit ``render_blocks`` argument;
- ``RenderOutcome`` -- what the renderer actually used, recorded on the adapter
  (``last_render_outcome``), never written back onto the manifest.

During the migration an "old manifest.run.X vs RenderPlan.X" shadow assertion
proved the two channels equivalent on real runs; the fields were deleted only
after the corpus gate passed 4/4 with that assertion active. This harness is the
permanent guard that they do not come back: the mode cluster must be on the
plan/outcome, and ``RunMetadata`` must not carry it again.
"""

from __future__ import annotations

import dataclasses

from ubt.core.ir.render_plan import RenderOutcome, RenderPlan
from ubt.core.ir.run_metadata import RunMetadata

#: The cluster that used to be a ``manifest.run`` bus.
_MODE_CLUSTER = (
    "bilingual_mode",
    "effective_dual_mode",
    "dual_mode_downgraded",
    "facing_spread",
    "render_engine",
    "translate_chrome",
    "cover_mode",
)
#: The purely-advisory outputs that moved to the plan in the earlier cut.
_ADVISORY_OUTPUTS = (
    "bilingual_advisory",
    "emit_secondary_mode",
    "emit_secondary_engine",
)


def main() -> int:
    problems: list[str] = []

    plan_fields = {f.name for f in dataclasses.fields(RenderPlan)}
    outcome_fields = {f.name for f in dataclasses.fields(RenderOutcome)}
    run_fields = set(RunMetadata.model_fields)

    for name in _MODE_CLUSTER + _ADVISORY_OUTPUTS:
        if name not in plan_fields:
            problems.append(f"RenderPlan is missing {name!r}")
        if name in run_fields:
            problems.append(f"RunMetadata still carries the bus field {name!r}")

    # The renderer's result channel: the mode it actually used.
    for name in ("bilingual_mode", "effective_dual_mode", "dual_mode_downgraded"):
        if name not in outcome_fields:
            problems.append(f"RenderOutcome is missing {name!r}")

    print("\nRenderer-handshake field homes (RenderPlan/RenderOutcome vs manifest.run)")
    print(f"  plan={len(plan_fields)} outcome={len(outcome_fields)} run={len(run_fields)}")
    print(f"  problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"    {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

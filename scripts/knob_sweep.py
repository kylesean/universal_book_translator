#!/usr/bin/env python3
"""K-1: measure each sweepable knob's tolerance band against the test corpus.

The registry says what a knob is *worth*; nothing said what it is *sensitive to*.
The 2026-09-20 hardcode audit (folded into ``docs/knob-calibration-protocol.md``
附录 B) named the gap: the only defence against a knob change is an end-to-end
golden baseline, so when a baseline goes
red there is no mapping from the failure back to the number that moved — and a
knob no test can see is unmeasured no matter what its rationale claims.

This is the mapping. For every knob that has a ``_env_*`` override point (see
``docs/knob-calibration-protocol.md`` §4 for why that set is small), run the
corpus at the centre and at ``×factor`` / ``÷factor``, and report which tests
moved. A cell with no red means the corpus cannot measure that knob at that
distance — which is a result to record in the registry, not a passing grade.

Usage::

    uv run python scripts/knob_sweep.py                 # ±1.5x, the protocol band
    uv run python scripts/knob_sweep.py --factor 1e6    # detectability probe
    uv run python scripts/knob_sweep.py --json band.json

Exit status is always 0: this is a measurement, not a gate. Insensitivity is
normal for most knobs today, and failing CI on it would only train people to
ignore the run.
"""

# No `from __future__ import annotations` here on purpose: stringified
# annotations make @dataclass resolve types through sys.modules[__name__], which a
# script loaded by path has no entry in — and patching sys.modules from a test to
# paper over it is the pattern behind the historical exit-139 SIGSEGV. Real
# annotations cost nothing on the py312 floor and remove the need.
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where a knob change could plausibly show up: the render/layout consumers plus
#: the KPI goldens. Chosen by consumer, not by name (the by-name blind spot is
#: the gap this harness exists to close — see 附录 B of the protocol doc).
DEFAULT_TARGETS = (
    "tests/baselines",
    "tests/unit/test_rigid_zones.py",
    "tests/unit/test_rigid_overlay_golden.py",
    "tests/unit/test_text_fit.py",
    "tests/unit/test_router_mode.py",
    "tests/unit/test_short_doc_render_routing.py",
    "tests/unit/test_reflow_control_loop.py",
)


@dataclass(frozen=True)
class Sweepable:
    """A knob with an env override point, and the value that override replaces."""

    name: str
    env_var: str
    default: float
    is_int: bool = False

    def value_at(self, factor: float) -> float | int:
        scaled = self.default * factor
        return int(round(scaled)) if self.is_int else scaled


def sweepable_knobs() -> tuple[Sweepable, ...]:
    """Every knob with a ``_env_float`` / ``_env_int`` override, read from the source.

    Parsed rather than imported so an ``int`` knob is not coerced to float by the
    override helper, and so the defaults come from the file a reader can open.
    """
    import ast
    import re

    source = (REPO_ROOT / "ubt" / "core" / "policy" / "layout_policy.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    found: list[Sweepable] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        call = node.value
        if not (isinstance(target, ast.Name) and isinstance(call, ast.Call)):
            continue
        func = call.func
        helper = getattr(func, "id", "")
        if helper not in {"_env_float", "_env_int"} or len(call.args) < 2:
            continue
        env_var, default = call.args[0], call.args[1]
        if not (isinstance(env_var, ast.Constant) and isinstance(default.value, (int, float))):
            continue
        if not re.fullmatch(r"UBT_[A-Z0-9_]+", str(env_var.value)):
            continue
        found.append(
            Sweepable(
                name=target.id,
                env_var=str(env_var.value),
                default=float(default.value),
                is_int=helper == "_env_int",
            )
        )
    return tuple(found)


@dataclass
class Cell:
    """One (knob, factor) run."""

    label: str
    failures: set[str] = field(default_factory=set)
    summary: str = ""
    error: str = ""


def parse_failures(pytest_stdout: str) -> set[str]:
    """Nodeids from pytest's short summary, e.g. FAILED path::test - Assert..."""
    found: set[str] = set()
    for line in pytest_stdout.splitlines():
        parts = line.split()
        # ERROR too: a collection/import/setup error previously read as
        # "no reaction", a false measurement.
        if line.startswith(("FAILED ", "ERROR ")) and len(parts) > 1:
            found.add(parts[1])
    return found


def run_cell(label: str, env: dict[str, str], targets: tuple[str, ...], timeout: int) -> Cell:
    kwargs: dict[str, Any] = {
        "cwd": REPO_ROOT,
        "capture_output": True,
        "text": True,
        "timeout": timeout,
        "env": {**os.environ, **env},
    }
    result = Cell(label=label)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-o",
                "addopts=-m 'not network'",
                "-q",
                "-p",
                "no:cacheprovider",
                *targets,
            ],
            **kwargs,
        )
    except subprocess.TimeoutExpired:
        result.error = f"timeout after {timeout}s"
        return result
    lines = (completed.stdout or "").splitlines()
    result.failures = parse_failures(completed.stdout or "")
    result.summary = lines[-1].strip() if lines else "no output"
    if completed.returncode not in (0, 1):
        # 0 = pass, 1 = test failures; 2+ = collection/internal error or "no
        # tests ran". A knob that breaks collection must not read as "no reaction".
        result.error = f"pytest exited {completed.returncode}: {result.summary}"
    return result


def verdict(baseline: Cell, low: Cell, high: Cell) -> str:
    """What the corpus can see about this knob."""
    if baseline.error or low.error or high.error:
        return "ERROR"
    low_delta = low.failures - baseline.failures
    high_delta = high.failures - baseline.failures
    if low_delta and high_delta:
        return "MEASURED both directions"
    if low_delta or high_delta:
        return f"MEASURED one-sided (only {'low' if low_delta else 'high'} moves)"
    return "INVISIBLE — no test reacts at this distance"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--factor", type=float, default=1.5, help="band multiplier (default 1.5)")
    parser.add_argument("--timeout", type=int, default=900, help="per-cell seconds")
    parser.add_argument("--json", type=Path, default=None, help="also write the raw result here")
    parser.add_argument("--targets", nargs="+", default=None, help="pytest targets to run")
    args = parser.parse_args(argv)
    targets: tuple[str, ...] = tuple(args.targets) if args.targets else DEFAULT_TARGETS

    knobs = sweepable_knobs()
    if not knobs:
        print("no _env_* override points found — nothing is sweepable", file=sys.stderr)
        return 1

    print(f"control run over {len(targets)} target(s)...")
    control = run_cell("control", {}, targets, args.timeout)
    if control.error:
        print(f"control failed: {control.error}", file=sys.stderr)
        return 1
    print(f"  {control.summary}  ({len(control.failures)} pre-existing red)\n")

    rows: list[dict[str, Any]] = []
    width = max(len(k.name) for k in knobs)
    for knob in knobs:
        low = run_cell(
            f"{knob.name}/{knob.value_at(1 / args.factor)}",
            {knob.env_var: str(knob.value_at(1 / args.factor))},
            targets,
            args.timeout,
        )
        high = run_cell(
            f"{knob.name}/{knob.value_at(args.factor)}",
            {knob.env_var: str(knob.value_at(args.factor))},
            targets,
            args.timeout,
        )
        outcome = verdict(control, low, high)
        print(f"{knob.name:<{width}}  {knob.default:>10.4g}  ±{args.factor:g}  {outcome}")
        for cell, direction in ((low, "low"), (high, "high")):
            for test in sorted(cell.failures - control.failures):
                print(f"    {direction} {cell.label.split('/')[-1]}: {test}")
            if cell.error:
                print(f"    {direction}: {cell.error}")
        rows.append(
            {
                "knob": knob.name,
                "env": knob.env_var,
                "default": knob.default,
                "factor": args.factor,
                "low": {"value": knob.value_at(1 / args.factor), "failures": sorted(low.failures)},
                "high": {"value": knob.value_at(args.factor), "failures": sorted(high.failures)},
                "verdict": outcome,
            }
        )
        print()

    print(f"control failures (unchanged by any cell): {sorted(control.failures) or 'none'}")
    if args.json:
        args.json.write_text(
            json.dumps(
                {"factor": args.factor, "control": sorted(control.failures), "rows": rows}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

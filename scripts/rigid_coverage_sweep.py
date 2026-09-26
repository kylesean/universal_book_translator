#!/usr/bin/env python3
"""M1: rigid render coverage as a function of target length, per font floor.

Why this exists
---------------
``docs/design/LAYOUT_PRESERVATION_MASTERPLAN_2026-09.md`` §4 M2 named "reclaim more
blank margin" as the lever for rigid's coverage. Measuring it showed that lever
is worth single digits, while the value that dominates every rendered page --
``RIGID_MIN_FONT_PT``, the floor ``RigidTypesetter`` shrinks text against -- is
recorded in ``ubt.core.policy.layout_policy`` as "never swept". A knob in force
on every rigid render had no measurement behind it. This is that measurement.

Why not ``ubt translate --dry-run``
-----------------------------------
The rehearsal provider (``ubt.core.router.provider.MockModelProvider``) returns a
fixed short string, so its targets are far SHORTER than their sources and every
length-driven skip is systematically understated -- the leak this measures is
exactly the one a dry run hides. This drives the pure decision function
(``RigidTypesetter._plan_blocks``: no Typst compile, no PDF write) with target
text rescaled to a controlled ratio instead.

Geometry is real: blocks come from the ledger of an actual run, so bboxes, block
types and zones are the ones the pipeline produces. Produce one cheaply with a
page-limited rehearsal (zero token spend)::

    uv run ubt translate docs/synthetic-duo.pdf \\
        --render-engine rigid --dual-mode monolingual --pages 1-3 --dry-run -y \\
        --job-id sweep_seed -o /tmp/sweep/out.pdf --db-dir /tmp/sweep/ledgers

Usage::

    uv run python scripts/rigid_coverage_sweep.py \\
        --ledger /tmp/sweep/ledgers/sweep_seed.sqlite --job-id sweep_seed

    # sweep the floor itself -- the calibration RIGID_MIN_FONT_PT never had
    uv run python scripts/rigid_coverage_sweep.py --floors 5 6.5 7.5 9

    uv run python scripts/rigid_coverage_sweep.py --json /tmp/sweep/sweep.json

Exit status is always 0: this is a measurement, not a gate (same contract as
``scripts/knob_sweep.py``).

Two fill modes, because the two directions leak differently:

* ``latin`` -- an EXPANDING translation (en->de/fr): more characters, same glyph
  widths.
* ``cjk`` -- the en->zh case, UBT's main battlefield: FEWER characters, each
  roughly twice as wide, so a char ratio near 0.5 already costs the source's
  line width.

Skip classification uses the engine's own list
(``ubt.core.qe.defect_taxonomy.INTENTIONAL_PRESERVED_SKIP_PREFIXES``) rather
than a copy, so "kept by design" here cannot drift from the report's verdict.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pikepdf

from ubt.adapters.pdf.rigid.extract import extract_pages
from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter
from ubt.adapters.pdf.rigid.zones import build_zones
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.policy.layout_policy import RIGID_MIN_FONT_PT
from ubt.core.qe.defect_taxonomy import INTENTIONAL_PRESERVED_SKIP_PREFIXES

#: Raw skip reasons are the suffix of the flags the engine classifies, so the
#: engine's prefixes -- not a hand-rolled set -- own "kept by design" vs "lost".
PRESERVED = {
    prefix.split(":", 1)[1] for prefix in INTENTIONAL_PRESERVED_SKIP_PREFIXES if ":" in prefix
}

#: A representative Chinese sentence, repeated/truncated to a character budget.
CJK_FILLER = "这是一个用于版面测量的中文句子，内容本身并不重要。"

LATIN_RATIOS = (0.9, 1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.6)
CJK_RATIOS = (0.35, 0.45, 0.5, 0.55, 0.6, 0.7, 0.85, 1.0)


def rescale(text: str, ratio: float, filler: str) -> str:
    """Target text at ``ratio`` x the source character count.

    A result equal to the source is nudged one character: ``skip_reason`` in
    ``ubt.adapters.pdf.rigid.gate`` returns ``verbatim`` for exactly that case
    (deliberately keeping a verbatim pair source-visible, since painting it
    would strip pristine text to no benefit). That guard is correct; letting the
    sweep land on it would hide the fit question instead of measuring it.
    """
    if not text.strip():
        return text
    body = filler or text
    budget = max(1, round(len(text) * ratio))
    out = (body * (budget // len(body) + 1))[:budget]
    if out == text:
        budget += 1
        out = (body * (budget // len(body) + 1))[:budget]
    return out


def measure_ratio(
    base: list[object],
    page_numbers: list[int],
    corpus: Path,
    ratio: float,
    filler: str,
    floor: float,
) -> dict[str, object]:
    """Plan one (floor, ratio) cell and return its coverage row."""
    blocks = [
        b.model_copy(update={"target_text": rescale(b.source_text, ratio, filler)}) for b in base
    ]
    pages = extract_pages(corpus, page_numbers, blocks)
    zones = build_zones(pages, blocks)
    typesetter = RigidTypesetter(min_font_pt=floor)
    _paints, report = typesetter._plan_blocks(
        blocks,
        zones,
        {p: f.height for p, f in pages.items()},
        {p: list(f.images) for p, f in pages.items()},
    )
    families = collections.Counter(reason for _bid, reason in report.skipped)
    kept = {r: n for r, n in families.items() if r in PRESERVED}
    lost = {r: n for r, n in families.items() if r not in PRESERVED}
    return {
        "floor_pt": floor,
        "ratio": ratio,
        "rendered": len(report.rendered_blocks),
        "reclaimed": len(report.reclaimed_blocks),
        "preserved": sum(kept.values()),
        "preserved_families": dict(kept),
        "fail_closed": dict(lost),
        "fail_closed_total": sum(lost.values()),
        "coverage_pct": round(100.0 * len(report.rendered_blocks) / len(blocks), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="rigid coverage vs target length, per font floor")
    parser.add_argument("--ledger", type=Path, required=True, help="SQLite job ledger")
    parser.add_argument("--job-id", required=True, help="job id inside that ledger")
    parser.add_argument(
        "--corpus", type=Path, default=Path("docs/synthetic-mono.pdf"), help="source PDF"
    )
    parser.add_argument(
        "--floors",
        type=float,
        nargs="+",
        default=[RIGID_MIN_FONT_PT],
        help=f"min_font_pt values to sweep (default: {RIGID_MIN_FONT_PT})",
    )
    parser.add_argument("--json", type=Path, default=None, help="write every row here")
    args = parser.parse_args()

    ledger = SQLiteJobLedger(args.ledger, read_only=True)
    try:
        base = ledger.get_all_blocks(args.job_id)
    finally:
        # Close the sqlite connection/handle; the ledger is used no further.
        ledger.close()
    if not base:
        print(f"no blocks for {args.job_id!r} in {args.ledger}", file=sys.stderr)
        return 2

    with pikepdf.open(str(args.corpus)) as pdf:
        total_pages = len(pdf.pages)
    page_numbers = list(range(1, total_pages + 1))

    print(f"corpus={args.corpus} pages={total_pages} blocks={len(base)}")
    print(f"types: {dict(collections.Counter(str(b.block_type) for b in base))}")
    print(f"preserved-by-design reasons (engine list): {sorted(PRESERVED)}")

    results: list[dict[str, object]] = []
    for floor in args.floors:
        for mode, ratios in (("latin", LATIN_RATIOS), ("cjk", CJK_RATIOS)):
            filler = "" if mode == "latin" else CJK_FILLER
            print(f"\n===== min_font_pt={floor} fill={mode} =====")
            print(
                f"{'ratio':>6} {'rendered':>9} {'reclaim':>8} {'preserved':>10} "
                f"{'fail-close':>11} {'cov%':>6}  kept / lost families"
            )
            for ratio in ratios:
                row = measure_ratio(base, page_numbers, args.corpus, ratio, filler, floor)
                row["mode"] = mode
                results.append(row)
                print(
                    f"{ratio:6.2f} {row['rendered']:9d} {row['reclaimed']:8d} "
                    f"{row['preserved']:10d} {row['fail_closed_total']:11d} "
                    f"{row['coverage_pct']:6.1f}  kept={row['preserved_families'] or '-'} "
                    f"lost={row['fail_closed'] or '-'}"
                )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

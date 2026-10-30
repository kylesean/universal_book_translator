"""Read-only probe for the G4 Stage 3A document body group.

Answers one question before any renderer changes: **which single body size can
this document's translated text actually hold**, and what does the micro-typography
ladder need to buy to get there?

It reads the job's ledger (translated text, `page_geometry` from the last
render) and the source PDF (per-page body size/pitch), builds the same per-page
fit models the renderer will use, and runs
:func:`ubt.adapters.pdf.facsimile_plan.plan_facsimile_body_group`.

Nothing is written except the two evidence files; no model is called and no
artifact is rendered. The point is to size the problem honestly — if the source
size needs a dozen exceptions, the plan should say so *before* the renderer is
rewired, not after.

Run::

    uv run python scripts/style_group_probe.py \
        --db .ubt/ledgers/web_mvufvchd_4ea52924bdbe.sqlite \
        --job-id web_mvufvchd_4ea52924bdbe \
        --out-dir output/adaptive_delivery/g4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pikepdf

from ubt.adapters.pdf.docling_render import _page_metrics
from ubt.adapters.pdf.facsimile_plan import (
    build_page_fit_models,
    fit_page_model,
    plan_facsimile_body_group,
)
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.layout.style_group import LADDER, body_band, derive_document_body_metrics


def _page_dims(source: Path) -> tuple[int, list[tuple[float, float]]]:
    dims: list[tuple[float, float]] = []
    with pikepdf.open(source) as pdf:
        for page in pdf.pages:
            box = page.CropBox if page.get("/CropBox") is not None else page.MediaBox
            dims.append((float(box[2]) - float(box[0]), float(box[3]) - float(box[1])))
        return len(pdf.pages), dims


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("output/adaptive_delivery/g4"))
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=float,
        default=None,
        help="Candidate body sizes to compare against the planned one (e.g. 8.9 8.5).",
    )
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"no ledger at {args.db}", file=sys.stderr)
        return 1
    ledger = SQLiteJobLedger(args.db)
    try:
        blocks = ledger.get_all_blocks(args.job_id)
        source_path = Path(ledger.get_job_source_path(args.job_id) or "")
        geometry = ledger.retrieve_page_geometry(args.job_id)
    finally:
        ledger.close()

    if not blocks:
        print(f"job {args.job_id} has no blocks", file=sys.stderr)
        return 1
    if not source_path.exists():
        print(f"source PDF missing at {source_path}", file=sys.stderr)
        return 1

    total_pages, page_dims = _page_dims(source_path)
    blocks_by_page: dict[int, list[Any]] = {}
    for block in blocks:
        pages: set[int] = set()
        if block.bbox is not None and block.bbox.page > 0:
            pages.add(int(block.bbox.page))
        if block.provenance and block.provenance.physical_boxes:
            for entry in block.provenance.physical_boxes:
                page = int(getattr(entry, "page", 0) or 0)
                if page > 0:
                    pages.add(page)
        for page in pages:
            blocks_by_page.setdefault(page, []).append(block)

    metrics: dict[int, tuple[float, float]] = {}
    for page_no in range(1, total_pages + 1):
        measured = _page_metrics(source_path, page_no)
        if measured is None:
            continue
        pitch = measured.leading_pt / measured.size_pt if measured.size_pt > 0 else 1.348
        metrics[page_no] = (round(measured.size_pt, 3), round(pitch, 3))

    models = build_page_fit_models(blocks_by_page, page_dims, metrics, total_pages=total_pages)
    plan = plan_facsimile_body_group(models)
    group = plan.group
    source_size, source_pitch = derive_document_body_metrics(
        {
            page: (model.source_size_pt, model.source_pitch, model.text_weight)
            for page, model in models.items()
        }
    )
    floor, ceiling = body_band(source_size)

    rows: list[dict[str, Any]] = []
    for page in sorted(models):
        model = models[page]
        at_group = fit_page_model(
            model,
            size_pt=group.size_pt,
            pitch=group.pitch,
            tracking_em=group.tracking_em,
            width_borrow_pt=group.width_borrow_pt,
        )
        at_band_ceiling = fit_page_model(
            model, size_pt=ceiling, pitch=source_pitch, tracking_em=0.0, width_borrow_pt=0.0
        )
        delivered = geometry.get(page, {}) if isinstance(geometry, dict) else {}
        rows.append(
            {
                "page": page,
                "source_size_pt": model.source_size_pt,
                "last_delivered_pt": delivered.get("delivered_size_pt"),
                "fits_at_group": at_group.fits,
                "deficit_pt": at_group.deficit_pt,
                "blocks_over": list(at_group.block_ids),
                "fits_at_ceiling": at_band_ceiling.fits,
                "exception": page in plan.exception_pages,
            }
        )

    candidates = args.sizes or [
        source_size,
        group.size_pt,
        floor,
        round((source_size + floor) / 2, 2),
    ]
    sweep: list[dict[str, Any]] = []
    for size in sorted(set(candidates), reverse=True):
        fits = sum(
            1
            for model in models.values()
            if fit_page_model(
                model,
                size_pt=size,
                pitch=group.pitch,
                tracking_em=group.tracking_em,
                width_borrow_pt=group.width_borrow_pt,
            ).fits
        )
        sweep.append({"size_pt": size, "pages_fitting": fits, "pages": len(models)})

    last_delivered = [
        float(row["last_delivered_pt"])
        for row in rows
        if isinstance(row["last_delivered_pt"], (int, float))
    ]
    drift_last = round(max(last_delivered) - min(last_delivered), 3) if last_delivered else None

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "job_id": args.job_id,
        "source_pdf": str(source_path),
        "pages": total_pages,
        "document_source_size_pt": source_size,
        "document_source_pitch": source_pitch,
        "band": {"floor_pt": floor, "ceiling_pt": ceiling},
        "group": group.to_payload(),
        "exception_pages": list(plan.exception_pages),
        "exceptions": [exception.to_payload() for exception in plan.exceptions],
        "rounds": [dict(entry) for entry in plan.rounds],
        "ladder": [asdict(rung) for rung in LADDER],
        "last_delivered_drift_pt": drift_last,
        "sweep": sweep,
        "pages_detail": rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "style_group_probe.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out_dir / "STYLE_GROUP_PROBE.md").write_text(_markdown(payload), encoding="utf-8")

    print(f"source body {source_size}pt @ pitch {source_pitch} (band {floor}-{ceiling})")
    print(f"group      {group.size_pt}pt @ pitch {group.pitch} rung {group.rung}")
    print(f"exceptions {len(plan.exception_pages)}: {list(plan.exception_pages)}")
    for entry in sweep:
        print(f"  at {entry['size_pt']:>5}pt: {entry['pages_fitting']}/{entry['pages']} pages fit")
    print(f"evidence   {args.out_dir / 'STYLE_GROUP_PROBE.md'}")
    return 0


def _markdown(payload: dict[str, Any]) -> str:
    group = payload["group"]
    return "\n".join(
        [
            f"# Style group probe — {payload['job_id']}",
            "",
            f"- Source: `{payload['source_pdf']}` ({payload['pages']} pages)",
            f"- Document body: **{payload['document_source_size_pt']}pt** @ pitch "
            f"{payload['document_source_pitch']}",
            f"- Band: floor {payload['band']['floor_pt']}pt / ceiling {payload['band']['ceiling_pt']}pt",
            f"- Planned group: **{group['size_pt']}pt** @ pitch {group['pitch']}, "
            f"tracking {group['tracking_em']}em, borrow {group['width_borrow_pt']}pt, rung `{group['rung']}`",
            f"- Exceptions: {len(payload['exception_pages'])} — {payload['exception_pages']}",
            f"- Last delivered drift: {payload['last_delivered_drift_pt']}pt",
            "",
            "## Size sweep",
            "",
            "| size | pages fitting |",
            "|---|---|",
            *[
                f"| {entry['size_pt']}pt | {entry['pages_fitting']}/{entry['pages']} |"
                for entry in payload["sweep"]
            ],
            "",
            "## Per page",
            "",
            "| page | source | last delivered | fits | deficit | exception | over blocks |",
            "|---|---|---|---|---|---|---|",
            *[
                f"| p{row['page']} | {row['source_size_pt']} | {row['last_delivered_pt']} | "
                f"{'yes' if row['fits_at_group'] else 'NO'} | {row['deficit_pt']} | "
                f"{'EXCEPTION' if row['exception'] else ''} | "
                f"{', '.join(row['blocks_over']) or ''} |"
                for row in payload["pages_detail"]
            ],
            "",
            "## Ladder rounds",
            "",
            "| size | rung | fits | failing pages | worst deficit |",
            "|---|---|---|---|---|",
            *[
                f"| {entry['size_pt']}pt | {entry['rung']} | {entry['fits']} | "
                f"{entry['failing_pages']} | {entry['worst_deficit_pt']} |"
                for entry in payload["rounds"]
            ],
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

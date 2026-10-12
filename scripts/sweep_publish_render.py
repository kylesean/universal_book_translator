#!/usr/bin/env python3
"""Renderer robustness sweep: extract → publish-render → measure, no LLM calls.

Answers the question the RFC cannot answer from one paper: *does the semantic
reflow renderer survive documents it has never seen?* It runs the real Docling
extraction and the real ``publish`` renderer, but skips translation entirely, so
it needs no provider and costs nothing. What it validates is everything
downstream of the translation:

- every block type reaches the renderer (tables, formulas, figures, lists);
- every figure asset resolves (a silent drop is the failure this catches);
- the generated Typst compiles;
- the delivered PDF's font-size spread and untranslated-marker count.

A document whose **source** renders cleanly proves the renderer is sound; the
translation quality is a separate axis the gold corpus measures with a provider.

Usage:
    uv run python scripts/sweep_publish_render.py [--limit N] [--json PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "corpus" / "semantic_gold" / "manifest.json"
DOCS = ROOT / "corpus" / "semantic_gold" / "documents"


async def _extract(pdf: Path) -> list:
    """Run the real Docling extraction, returning IRBlocks."""
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter

    adapter = DoclingPDFAdapter()
    blocks = []
    async for chapter in adapter.parse_stream(pdf):
        blocks.extend(chapter.blocks)
    return blocks


async def _render(blocks: list, pdf: Path, out: Path) -> tuple[bool, str]:
    """Render with ``publish``; return (compiled, detail)."""
    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
    from ubt.core.ir.models import BookManifest

    strategy = DoclingRenderStrategy()
    strategy.render_mode = "publish"
    manifest = BookManifest(doc_id=pdf.stem, title=pdf.stem, source_path=str(pdf), chapters=[])
    try:
        await strategy.render_blocks(
            manifest=manifest,
            blocks=blocks,
            target_lang="zh",
            output_path=out,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    skips = list(strategy.last_render_skips)
    detail = f"{len(skips)} render skip(s)" if skips else ""
    return out.exists(), detail


def _measure(pdf: Path) -> dict:
    from scripts.eval_semantic import evaluate

    return evaluate(pdf, None, None, None)


async def _one(case: dict, workdir: Path) -> dict:
    from dataclasses import asdict

    from ubt.render.typst.document_renderer import TypstDocumentRenderer

    pdf = DOCS / f"{case['id']}.pdf"
    if not pdf.exists():
        return {"id": case["id"], "status": "missing_source"}

    row: dict = {"id": case["id"], "category": case["category"]}
    try:
        blocks = await _extract(pdf)
    except Exception as exc:
        return {**row, "status": "extract_failed", "detail": f"{type(exc).__name__}: {exc}"}
    row["blocks"] = len(blocks)

    build = workdir / case["id"]
    build.mkdir(parents=True, exist_ok=True)
    renderer = TypstDocumentRenderer(assets_root=None)
    try:
        _source, stats = renderer.build_source(blocks, build)
    except Exception as exc:
        return {**row, "status": "build_failed", "detail": f"{type(exc).__name__}: {exc}"}
    row["stats"] = {k: (len(v) if isinstance(v, list) else v) for k, v in asdict(stats).items()}

    out = build / "out.pdf"
    ok, detail = await _render(blocks, pdf, out)
    if not ok:
        return {**row, "status": "compile_failed", "detail": detail}

    row["status"] = "ok"
    row["detail"] = detail
    try:
        row["metrics"] = _measure(out)
    except Exception as exc:
        row["metrics"] = {"error": str(exc)}
    return row


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--keep", action="store_true", help="keep the build directory")
    args = ap.parse_args()

    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    if args.limit:
        cases = cases[: args.limit]

    workdir = Path(tempfile.mkdtemp(prefix="ubt_sweep_"))
    rows: list[dict] = []
    try:
        for case in cases:
            row = await _one(case, workdir)
            rows.append(row)
            stats = row.get("stats") or {}
            print(
                f"{row['id']:<28} {row['status']:<16} "
                f"blocks={row.get('blocks', 0):<5} "
                f"tbl={stats.get('tables', 0):<3} fig={stats.get('figures', 0):<3} "
                f"fml={stats.get('formulas', 0):<3} "
                f"missing={len(row.get('stats', {}).get('assets_missing', []) or [])}"
                f"  {row.get('detail', '')}"
            )
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"\nbuild dir kept: {workdir}")

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n{len(ok)}/{len(rows)} rendered and compiled")
    failed = [r for r in rows if r["status"] not in ("ok",)]
    for row in failed:
        print(f"  {row['status']:<16} {row['id']}: {row.get('detail', '')[:120]}")

    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

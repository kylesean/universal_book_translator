#!/usr/bin/env python3
"""Convert an academic PDF to section-partitioned Markdown via DoclingPDFAdapter.

This produces a clean Markdown file where section headings are '# ' headers,
formulas are '$$ ... $$' math blocks, and tables/code are preserved, ready
for end-to-end LLM benchmarking with scripts/cost_benchmark.py.

Usage:
    .venv/bin/python scripts/export_pdf_to_markdown.py [input.pdf] [output.md]
"""

import asyncio
import sys
from pathlib import Path

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.core.ir.models import BlockType


async def export_pdf_to_markdown(pdf_path: Path, md_path: Path) -> None:
    adapter = DoclingPDFAdapter()
    if not adapter.is_docling_installed():
        sys.exit("error: docling is not installed in the current environment")

    if not pdf_path.exists():
        sys.exit(f"error: input PDF not found: {pdf_path}")

    print(f"[pdf->md] Parsing {pdf_path} via DoclingPDFAdapter...")
    blocks = []
    async for ch in adapter.parse_stream(pdf_path):
        blocks.extend(ch.blocks)

    print(f"[pdf->md] Extracted {len(blocks)} blocks. Writing Markdown to {md_path}...")
    lines: list[str] = []
    for b in blocks:
        text = b.source_text.strip()
        if not text:
            continue
        if b.block_type == BlockType.HEADING:
            lines.append(f"# {text}\n\n")
        elif b.block_type == BlockType.FORMULA:
            lines.append(f"$$\n{text}\n$$\n\n")
        elif b.block_type == BlockType.CODE:
            lines.append(f"```\n{text}\n```\n\n")
        elif b.block_type == BlockType.TABLE:
            lines.append(f"{text}\n\n")
        else:
            lines.append(f"{text}\n\n")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"[pdf->md] Done. Output written to {md_path} ({md_path.stat().st_size} bytes)")


def main() -> None:
    input_pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/ubt_fixtures/doclaynet.pdf")
    output_md = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/ubt_bench/doclaynet.md")
    asyncio.run(export_pdf_to_markdown(input_pdf, output_md))


if __name__ == "__main__":
    main()

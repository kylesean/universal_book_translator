"""The publish renderer across the gold corpus (no provider, no LLM).

The RFC's product claims rest on one paper. A renderer that only survives the
document it was written against is not a renderer, so this test extracts a
spread of real documents with the real Docling pipeline and asserts the
*publish* path turns each into a compilable PDF.

Extraction is the slow part (GPU, ~10-40s per document), so the sweep is
``slow``-marked and skipped by default. The assertions are deliberately about
*robustness*, not quality: every block type survives, no figure asset is
silently dropped, the Typst compiles, and the body type comes out uniform. The
translation axis needs a provider and is measured by ``scripts/eval_gold_corpus``
instead.

Run explicitly:
    uv run pytest tests/integration/test_publish_corpus_sweep.py -q -m slow
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "corpus" / "semantic_gold" / "manifest.json"
DOCS = ROOT / "corpus" / "semantic_gold" / "documents"

#: A spread by shape rather than by name: the four categories the manifest
#: defines, one document each, so a failure names a *kind* of input.
SAMPLE = (
    "switch-transformers",  # formula-dense + tables + outline
    "gan",  # short, minimal math
    "neural-ode",  # pure-math, the hardest case for the transpiler
    "gpt3",  # long, table-heavy (811 blocks)
    "resnet",  # two-column CV layout
    "dropout",  # older style, different font/table conventions
)


def _manifest() -> dict[str, object]:
    parsed: dict[str, object] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return parsed


def _present(ids: tuple[str, ...]) -> list[str]:
    """Only the cases whose source is actually on disk (corpus is not committed)."""
    return [i for i in ids if (DOCS / f"{i}.pdf").exists()]


@pytest.mark.slow
@pytest.mark.parametrize("case_id", _present(SAMPLE))
def test_a_real_document_extracts_and_publishes(case_id: str, tmp_path: Path) -> None:
    """Extract with Docling, render with publish, and require a real PDF.

    Asserts the four things a renderer must not get wrong on unseen input:

    1. extraction produced blocks (an empty document would render "successfully");
    2. every figure asset resolved (a silent drop is the failure this catches);
    3. the generated Typst compiled;
    4. the body type is uniform — the property the overlay path cannot have.
    """
    asyncio = pytest.importorskip("asyncio")

    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
    from ubt.core.ir.models import BookManifest, IRBlock
    from ubt.render.typst.document_renderer import TypstDocumentRenderer

    pdf = DOCS / f"{case_id}.pdf"

    async def _extract() -> list[IRBlock]:
        adapter = DoclingPDFAdapter()
        blocks: list[IRBlock] = []
        async for chapter in adapter.parse_stream(pdf):
            blocks.extend(chapter.blocks)
        return blocks

    blocks = asyncio.run(_extract())
    assert blocks, "extraction produced no blocks"

    build = tmp_path / case_id
    build.mkdir(parents=True, exist_ok=True)
    renderer = TypstDocumentRenderer()
    _source, stats = renderer.build_source(blocks, build)

    assert not stats.assets_missing, f"unresolved figure asset(s): {stats.assets_missing}"
    assert stats.blocks_rendered > 0

    strategy = DoclingRenderStrategy()
    manifest = BookManifest(doc_id=case_id, title=case_id, source_path=str(pdf), chapters=[])
    out = build / "out.pdf"
    asyncio.run(
        strategy.render_blocks(manifest=manifest, blocks=blocks, target_lang="zh", output_path=out)
    )
    assert out.exists() and out.stat().st_size > 10_000, "no usable PDF produced"

    from scripts.eval_semantic import evaluate

    report = evaluate(out, None, None, None)
    # The whole point of the semantic path: one body size, not 44.
    assert report["p95_font_size_pt"] == pytest.approx(10.5, abs=0.5)
    assert report["distinct_font_sizes"] <= 8

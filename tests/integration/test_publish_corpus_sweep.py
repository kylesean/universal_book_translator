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


_AVAILABLE = _present(SAMPLE)


@pytest.mark.slow
def test_the_gold_corpus_is_present() -> None:
    """Fail loudly when the corpus is missing instead of collecting zero cases.

    ``parametrize("case_id", _present(SAMPLE))`` with an empty list collects
    *zero* tests and pytest reports success — the sweep silently stopped
    covering anything the moment the (gitignored) PDFs were absent, and CI
    runs it in that state. An explicit failure turns "not measured" into a
    visible red instead of a green that means nothing.
    """
    if not _AVAILABLE:
        pytest.fail(
            "the gold corpus is missing, so the publish sweep measured nothing. "
            f"Expected PDFs under {DOCS}: {', '.join(SAMPLE)}. "
            "Fetch them with `python scripts/fetch_semantic_gold.py`, or deselect "
            "this test explicitly if you are building a corpus-free environment."
        )


@pytest.mark.slow
@pytest.mark.parametrize("case_id", _AVAILABLE)
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
    # The bound counts TABULAR sizes too, which are legitimately per-table now:
    # a table whose measured minimum does not fit its column is set at its own
    # computed scale (``emitted_scale``), so a paper with three or four
    # differently-shaped wide tables contributes that many extra sizes (this one
    # measures 10: body 10.5, headings 12/14/15, table bodies 6.30/6.51/6.82/8.40
    # and 8.82/9.00). The property that matters -- the BODY is uniform, not the
    # 44-size overlay -- is the ``p95`` assertion above. A ceiling low enough to
    # forbid per-table scales would force every wide table back to one constant
    # size, which is the ideograph-per-line defect the scale exists to fix.
    assert report["distinct_font_sizes"] <= 12

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "corpus" / "semantic_gold" / "manifest.json"
DOCS = ROOT / "corpus" / "semantic_gold" / "documents"

SAMPLE = (
    "switch-transformers",
    "gan",
    "neural-ode",
    "gpt3",
    "resnet",
    "dropout",
)


def _manifest() -> dict[str, object]:
    parsed: dict[str, object] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return parsed


def _present(ids: tuple[str, ...]) -> list[str]:
    return [i for i in ids if (DOCS / f"{i}.pdf").exists()]


_AVAILABLE = _present(SAMPLE)


@pytest.mark.slow
def test_the_gold_corpus_is_present() -> None:
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
    assert report["p95_font_size_pt"] == pytest.approx(10.5, abs=0.5)
    assert report["distinct_font_sizes"] <= 12

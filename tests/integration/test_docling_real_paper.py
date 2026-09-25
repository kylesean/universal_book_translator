"""Real-Docling integration tests against downloaded arXiv fixtures.

These tests are SKIPPED when docling or the fixture PDF is unavailable, so CI
stays green without network/model downloads. Locally run with:

    uv run pytest tests/integration/test_docling_real_paper.py -v

Fixtures are not committed (they are multi-MB PDFs). To enable these tests
locally, place the source PDFs at these exact paths:

    mkdir -p /tmp/ubt_fixtures
    # /tmp/ubt_fixtures/attention.pdf
    # /tmp/ubt_fixtures/doclaynet.pdf   # two-column ACM layout sample

Point ``UBT_FIXTURE_DIR`` somewhere else to keep a durable fixture cache (and to
hand one to CI: the path used to be a literal, so no job could ever enable these
ten cases and they skipped everywhere without anyone noticing).
"""

import asyncio
import os
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import BlockType, FlowID, IRBlock

pytestmark = pytest.mark.slow  # real Docling model on a full paper

FIXTURES = Path(os.environ.get("UBT_FIXTURE_DIR", "/tmp/ubt_fixtures"))
ATTENTION_PDF = FIXTURES / "attention.pdf"
DOCLAYNET_PDF = FIXTURES / "doclaynet.pdf"  # two-column ACM layout stress test
# NOTE: arXiv 2206.01028 resolved to "Impact of Sampling on Locally
# Differentially Private Data Collection" (two-column ACM format) — kept as
# the two-column stress fixture regardless of the original intent.

requires_docling = pytest.mark.skipif(
    not DoclingPDFAdapter().is_docling_installed(),
    reason="docling not installed (uv sync --extra docling)",
)
requires_fixture = pytest.mark.skipif(
    not ATTENTION_PDF.exists(),
    reason=(
        f"fixture missing: {ATTENTION_PDF} "
        "(drop the PDF at this path to enable; see module docstring)"
    ),
)
requires_twocolumn_fixture = pytest.mark.skipif(
    not DOCLAYNET_PDF.exists(),
    reason=(
        f"fixture missing: {DOCLAYNET_PDF} "
        "(drop the PDF at this path to enable; see module docstring)"
    ),
)


def _collect_blocks() -> list[IRBlock]:
    adapter = DoclingPDFAdapter()

    async def _run() -> list[IRBlock]:
        result: list[IRBlock] = []
        async for ch in adapter.parse_stream(ATTENTION_PDF):
            result.extend(ch.blocks)
        return result

    return asyncio.run(_run())


@pytest.fixture(scope="module")
def blocks() -> list[IRBlock]:
    return _collect_blocks()


class _FakeLedger:
    """Minimal ledger stub exposing only what render_output reads."""

    def __init__(self, blocks: list[IRBlock]) -> None:
        self._blocks = blocks

    def get_all_blocks(self, doc_id: str) -> list[IRBlock]:
        return self._blocks

    def assemble_document_ir(self, doc_id: str) -> None:  # pragma: no cover - not used
        raise NotImplementedError("not needed for render smoke tests")


@requires_docling
@requires_fixture
def test_blocks_extracted_nonempty(blocks: list[IRBlock]) -> None:
    """A real paper must produce a substantial, type-diverse block stream."""
    assert len(blocks) > 50
    types = Counter(b.block_type for b in blocks)
    assert types[BlockType.NARRATIVE] > 20
    assert types[BlockType.HEADING] >= 5
    assert types[BlockType.TABLE] >= 1


@requires_docling
@requires_fixture
def test_no_page_chrome_leaked_as_heading(blocks: list[IRBlock]) -> None:
    """#3 验收：arXiv 水印/页眉页脚不得混入 HEADING（如 'arXiv:1706...'）。"""
    leaked = [
        b.source_text
        for b in blocks
        if b.block_type == BlockType.HEADING
        and (
            "arXiv:" in b.source_text
            or "et al" in b.source_text
            or b.source_text.strip().startswith("31st Conference")
        )
    ]
    assert not leaked, f"page chrome leaked as headings: {leaked}"


@requires_docling
@requires_fixture
def test_tables_are_in_reading_order(blocks: list[IRBlock]) -> None:
    """#1 验收：表格必须出现在其阅读序位置，不得全部堆在文档末尾。"""
    tables = [b for b in blocks if b.flow_id == FlowID.TABLE_GRID]
    assert tables, "no tables extracted at all"
    narr = [b for b in blocks if b.flow_id == FlowID.MAIN_STORY]
    mid_text_spine = narr[len(narr) // 2].spine_index
    first_table_spine = min(t.spine_index for t in tables)
    assert first_table_spine < mid_text_spine, (
        f"tables out of reading order: first table at spine {first_table_spine} "
        f"of {len(blocks)} (mid text at {mid_text_spine})"
    )


@requires_docling
@requires_fixture
def test_table_text_is_structured_not_raw_dict(blocks: list[IRBlock]) -> None:
    """#1b 验收：表格内容必须是 markdown/grid 形态，不能是原始 dict 的 str。"""
    tables = [b for b in blocks if b.flow_id == FlowID.TABLE_GRID]
    bad = [t for t in tables if t.source_text.lstrip().startswith("{")]
    assert not bad, (
        "table source_text is a raw dict repr — grid→markdown mapping missing: "
        f"{[t.source_text[:60] for t in bad]}"
    )
    # 至少一张真实表格应包含 markdown 分隔行或竖线单元格
    assert any("|" in t.source_text for t in tables), (
        "table text contains no markdown cell separators"
    )


@requires_docling
@requires_fixture
def test_no_figure_internal_text_leaked(blocks: list[IRBlock]) -> None:
    """图内标注（Figure 1 的 'Softmax'/'Add & Norm' 等）不得进入 narrative。"""
    figure_labels = {"Softmax", "Add & Norm", "Ouput", "Probabilities", "Linear"}
    leaked = [b.source_text for b in blocks if b.source_text.strip() in figure_labels]
    assert not leaked, f"in-figure labels leaked as narrative blocks: {leaked}"


@requires_docling
@requires_fixture
def test_formulas_and_code_are_skipped(blocks: list[IRBlock]) -> None:
    """公式/代码必须 skip_translate，防止 LLM 搅坏数学符号。"""
    for b in blocks:
        if b.block_type in (BlockType.FORMULA, BlockType.CODE):
            assert b.skip_translate, f"block {b.id} type={b.block_type} not skipped"


@requires_docling
@requires_fixture
def test_extraction_is_deterministic() -> None:
    """同输入两次解析必须产出相同块数（幂等断点恢复的前提）。"""
    first = _collect_blocks()
    second = _collect_blocks()
    assert len(first) == len(second)
    assert [b.source_text for b in first] == [b.source_text for b in second]


@requires_docling
@requires_fixture
async def test_render_markdown_and_typst(tmp_path: Path, blocks: list[IRBlock]) -> None:
    """渲染产物存在且非空；.typ 若能编译则必须零错误（转义验证）。"""
    adapter = DoclingPDFAdapter()
    manifest = await adapter.extract_manifest(ATTENTION_PDF)
    fake_ledger = cast(SQLiteJobLedger, _FakeLedger(blocks))

    md = await adapter.render_output(manifest, fake_ledger, "zh", tmp_path / "out.md")
    assert md.exists() and md.stat().st_size > 1000

    typ = await adapter.render_output(manifest, fake_ledger, "zh", tmp_path / "out.typ")
    assert typ.exists() and typ.stat().st_size > 1000

    if adapter.reconstructor.is_compiler_available():
        pdf_out = await adapter.render_output(manifest, fake_ledger, "zh", tmp_path / "out.pdf")
        assert pdf_out.exists() and pdf_out.stat().st_size > 10000


@requires_docling
@requires_twocolumn_fixture
def test_twocolumn_reading_order_preserved() -> None:
    """双栏论文（DocLayNet, ACM 版式）阅读序不得被物理切栏打乱。

    判定：标题后紧跟摘要句，正文段落之间无跨栏断句（段落以句号收尾为主），
    且第一张表格不堆在文档末尾。
    """
    adapter = DoclingPDFAdapter()

    async def _run() -> list[IRBlock]:
        result: list[IRBlock] = []
        async for ch in adapter.parse_stream(DOCLAYNET_PDF):
            result.extend(ch.blocks)
        return result

    blocks = asyncio.run(_run())
    assert len(blocks) > 100

    # Title should be among the first blocks, abstract shortly after
    first_headings = [
        b.source_text.lower() for b in blocks[:8] if b.block_type == BlockType.HEADING
    ]
    assert any("differentially private" in h or "sampling" in h for h in first_headings), (
        f"expected paper title near the top, got: {first_headings}"
    )

    # Paragraph integrity: multi-column parsing must not split mid-sentence —
    # most long narrative blocks should end with sentence-final punctuation.
    long_narr = [
        b for b in blocks if b.block_type == BlockType.NARRATIVE and len(b.source_text) > 200
    ]
    assert long_narr, "no long narrative paragraphs extracted"
    punctuated = sum(1 for b in long_narr if b.source_text.rstrip().endswith((".", "!", "?", '"')))
    ratio = punctuated / len(long_narr)
    assert ratio >= 0.7, (
        f"only {ratio:.0%} of long paragraphs end with sentence punctuation — "
        "likely column-boundary splits"
    )

    # Table position sanity (same invariant as single-column paper)
    tables = [b for b in blocks if b.flow_id == FlowID.TABLE_GRID]
    if tables:
        narr = [b for b in blocks if b.flow_id == FlowID.MAIN_STORY]
        mid_text_spine = narr[len(narr) // 2].spine_index
        first_table_spine = min(t.spine_index for t in tables)
        assert first_table_spine < mid_text_spine


@requires_docling
@requires_twocolumn_fixture
def test_twocolumn_abbreviations_mineable() -> None:
    """双栏论文应能挖掘出至少一个合法缩写对（端到端学术策略验证）。"""
    from ubt.core.memory.abbreviation_miner import mine_abbreviations

    adapter = DoclingPDFAdapter()

    async def _run() -> str:
        text: list[str] = []
        async for ch in adapter.parse_stream(DOCLAYNET_PDF):
            text.extend(b.source_text for b in ch.blocks)
        return "\n".join(text)

    entries = mine_abbreviations(asyncio.run(_run()))
    assert len(entries) >= 1, "no abbreviation pairs mined from a real two-column paper"

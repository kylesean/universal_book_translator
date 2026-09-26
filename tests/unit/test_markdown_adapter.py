"""Unit tests for Markdown and flat text document adapter."""

import tempfile
from pathlib import Path

import pytest

from ubt.adapters.markdown.adapter import MarkdownAdapter
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import BlockStatus, BlockType, BookManifest, FlowID, IRBlock


@pytest.mark.asyncio
async def test_plaintext_mode_ignores_markdown_syntax(tmp_path: Path) -> None:
    """Regression M10: a .txt line starting with ``#`` is a comment, not a chapter.

    Logs and configs route through this adapter in plain-text mode; without the
    mode, ``# TODO`` split the document into chapters and turned the line into a
    heading block. Here the whole file is one chapter of narrative paragraphs,
    fenced-code markers are literal text, and nothing is skipped as pre-formula.
    """
    txt = tmp_path / "app.log"
    txt.write_text("# comment header line\nvalue = 1\n\nsecond paragraph\n", encoding="utf-8")

    md = MarkdownAdapter()
    plain = MarkdownAdapter(plain_text=True)

    # Markdown mode fragments on the ``#`` line (the behaviour M10 removes for .txt).
    assert len((await md.extract_manifest(txt)).chapters) == 1  # only one h1 -> no preface split
    md_chapters = [ch async for ch in md.parse_stream(txt)]
    md_types = [b.block_type for ch in md_chapters for b in ch.blocks]
    assert BlockType.HEADING in md_types

    # Plain-text mode: no heading block, everything is narrative.
    plain_chapters = [ch async for ch in plain.parse_stream(txt)]
    assert len(plain_chapters) == 1
    plain_types = [b.block_type for ch in plain_chapters for b in ch.blocks]
    assert BlockType.HEADING not in plain_types
    assert all(t == BlockType.NARRATIVE for t in plain_types)
    first_text = plain_chapters[0].blocks[0].source_text
    assert first_text.startswith("# comment header line")


@pytest.mark.asyncio
async def test_markdown_adapter_parse_and_render_roundtrip(tmp_path: Path) -> None:
    """Validate Markdown parsing, chapter partitioning, and bilingual interleaved rendering."""
    md_file = tmp_path / "sample.md"
    md_content = """# The Great Adventure

Once upon a time in a far away land.

A brave hero began a journey.

```python
# Code must be protected
def hero_action():
    return "quest"
```

# Chapter Two: The Forest

The forest was dark and silent.
"""
    md_file.write_text(md_content, encoding="utf-8")

    adapter = MarkdownAdapter()

    # 1. Manifest extraction
    manifest = await adapter.extract_manifest(md_file)
    assert manifest.title == "The Great Adventure"
    assert len(manifest.chapters) == 2
    assert manifest.chapters[0].title == "The Great Adventure"
    assert manifest.chapters[1].title == "Chapter Two: The Forest"

    # 2. Streaming parse
    chapters = []
    async for ch in adapter.parse_stream(md_file):
        chapters.append(ch)

    assert len(chapters) == 2
    ch1 = chapters[0]
    assert ch1.title == "The Great Adventure"
    assert ch1.total_blocks == 4  # heading, p1, p2, code block
    assert ch1.blocks[0].block_type == BlockType.HEADING
    assert ch1.blocks[3].block_type == BlockType.CODE
    assert ch1.blocks[3].skip_translate is True

    ch2 = chapters[1]
    assert ch2.title == "Chapter Two: The Forest"
    assert ch2.total_blocks == 2  # heading, p1

    # 3. Store in ledger and simulate translations
    db_path = tmp_path / "md_ledger.db"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest(manifest.doc_id, manifest)
        ledger.append_chapter(manifest.doc_id, ch1)
        ledger.append_chapter(manifest.doc_id, ch2)

        # Checkpoint some translations
        ledger.save_checkpoint(
            block_id=ch1.blocks[0].id,
            status=BlockStatus.MTQE_PASSED,
            target_text="# 伟大的冒险",
        )
        ledger.save_checkpoint(
            block_id=ch1.blocks[1].id,
            status=BlockStatus.MTQE_PASSED,
            target_text="从前在遥远的国度里。",
        )
        ledger.save_checkpoint(
            block_id=ch1.blocks[2].id,
            status=BlockStatus.MTQE_PASSED,
            target_text="勇敢的英雄踏上了征程。",
        )

        # 4. Render bilingual output
        out_file = tmp_path / "bilingual.md"
        rendered = await adapter.render_output(
            manifest, ledger, target_lang="zh", output_path=out_file
        )
        assert rendered.exists()

        result_text = out_file.read_text(encoding="utf-8")
        assert "# The Great Adventure\n\n# 伟大的冒险" in result_text
        assert "Once upon a time in a far away land.\n\n从前在遥远的国度里。" in result_text
        assert "def hero_action():" in result_text  # Code unchanged


@pytest.mark.asyncio
async def test_gutenberg_markers_skip_translation(tmp_path: Path) -> None:
    """[Illustration]/[Footnote] blocks must be 0-token skipped (benchmark finding)."""
    md = tmp_path / "gutenberg.md"
    md.write_text(
        "# CHAPTER I.\n\n"
        "[Illustration: List of Illustrations.]\n\n"
        "Real narrative text that should be translated normally.\n\n"
        "[Footnote 1: A translator note worth keeping verbatim.]\n",
        encoding="utf-8",
    )
    adapter = MarkdownAdapter()
    blocks: list[IRBlock] = []
    async for ch in adapter.parse_stream(md):
        blocks.extend(ch.blocks)

    markers = [b for b in blocks if b.source_text.startswith("[")]
    assert len(markers) == 2
    assert all(b.skip_translate for b in markers)
    narrative = [b for b in blocks if "Real narrative" in b.source_text]
    assert len(narrative) == 1 and not narrative[0].skip_translate


@pytest.mark.asyncio
async def test_math_formulas_skip_translation(tmp_path: Path) -> None:
    """Math blocks ($$...$$) must be identified as BlockType.FORMULA and skip_translate=True."""
    md = tmp_path / "paper.md"
    md.write_text(
        "# Section 1\n\n"
        "Here is the famous formula:\n\n"
        "$$\n"
        "\\mathbb{P}[\\mathcal{R}(x) = y] \\leq e^\\epsilon\n"
        "$$\n\n"
        "And inline display:\n\n"
        "$$ E = mc^2 $$\n\n"
        "Concluding remarks.\n",
        encoding="utf-8",
    )
    adapter = MarkdownAdapter()
    blocks: list[IRBlock] = []
    async for ch in adapter.parse_stream(md):
        blocks.extend(ch.blocks)

    formulas = [b for b in blocks if b.block_type == BlockType.FORMULA]
    assert len(formulas) == 2
    assert all(b.skip_translate for b in formulas)
    assert "\\mathbb{P}" in formulas[0].source_text
    assert "E = mc^2" in formulas[1].source_text
    assert all(b.block_type != BlockType.NARRATIVE for b in formulas)


@pytest.mark.asyncio
async def test_markdown_adapter_preface_partition(tmp_path: Path) -> None:
    """Validate that pre-heading text creates an aligned ch_000 Preface chapter."""
    md = tmp_path / "with_preface.md"
    md.write_text(
        "This is an introductory paragraph before any header.\n\n"
        "# Chapter 1: The Beginning\n\n"
        "Narrative for chapter one.\n\n"
        "# Chapter 2: The End\n\n"
        "Narrative for chapter two.\n",
        encoding="utf-8",
    )
    adapter = MarkdownAdapter()
    manifest = await adapter.extract_manifest(md)

    assert len(manifest.chapters) == 3
    assert manifest.chapters[0].chapter_id == "ch_000"
    assert manifest.chapters[0].title == "Preface"
    assert manifest.chapters[1].chapter_id == "ch_001"
    assert manifest.chapters[1].title == "Chapter 1: The Beginning"
    assert manifest.chapters[2].chapter_id == "ch_002"
    assert manifest.chapters[2].title == "Chapter 2: The End"

    streamed_chapters = []
    async for ch in adapter.parse_stream(md):
        streamed_chapters.append(ch)

    assert len(streamed_chapters) == 3
    assert streamed_chapters[0].chapter_id == "ch_000"
    assert streamed_chapters[0].title == "Preface"
    assert any("introductory paragraph" in b.source_text for b in streamed_chapters[0].blocks)

    assert streamed_chapters[1].chapter_id == "ch_001"
    assert streamed_chapters[1].title == "Chapter 1: The Beginning"

    assert streamed_chapters[2].chapter_id == "ch_002"
    assert streamed_chapters[2].title == "Chapter 2: The End"


@pytest.mark.asyncio
async def test_unresolved_blocks_stay_marked_in_output(tmp_path: Path) -> None:
    """FAILED/NEEDS_HUMAN/BLOCKED_HUMAN blocks must stay visible.

    The renderer used to strip the ``<mark class="ubt-failed-draft">`` wrapper
    and ship the machine draft as if it were a final translation (or silently
    fall back to source). Every unresolved block now carries an explicit
    warning note; blocked blocks never render their machine draft at all.
    """
    md_file = tmp_path / "gates.md"
    md_file.write_text("Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph.\n", "utf-8")
    adapter = MarkdownAdapter()
    manifest = await adapter.extract_manifest(md_file)

    chapters = [ch async for ch in adapter.parse_stream(md_file)]
    blocks = [b for ch in chapters for b in ch.blocks]
    assert len(blocks) >= 3

    db_path = tmp_path / "gates.db"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest(manifest.doc_id, manifest)
        for ch in chapters:
            ledger.append_chapter(manifest.doc_id, ch)
        ledger.save_checkpoint(
            block_id=blocks[0].id,
            status=BlockStatus.FAILED,
            target_text='<mark class="ubt-failed-draft">失败草稿</mark>',
        )
        ledger.save_checkpoint(
            block_id=blocks[1].id,
            status=BlockStatus.NEEDS_HUMAN,
            target_text="待人工草稿",
            error_flags=["needs_human_review"],
        )
        ledger.save_checkpoint(
            block_id=blocks[2].id,
            status=BlockStatus.BLOCKED_HUMAN,
            target_text=None,
            mqm_severity="critical",
        )

        out_file = tmp_path / "bilingual.md"
        await adapter.render_output(manifest, ledger, target_lang="zh", output_path=out_file)
        result = out_file.read_text(encoding="utf-8")

    # FAILED: source kept, machine draft visible but explicitly marked.
    assert "[UBT] Translation FAILED quality gates" in result
    assert "> 失败草稿" in result
    # NEEDS_HUMAN: draft preserved for post-editing under a visible note.
    assert "[UBT] Flagged NEEDS_HUMAN" in result
    assert "> 待人工草稿" in result
    # BLOCKED_HUMAN: never ships machine text, source kept with a note.
    assert "[UBT] Blocked from machine translation" in result
    assert "Gamma paragraph." in result
    # The raw <mark> wrapper itself must never leak into markdown output.
    assert "<mark" not in result


async def test_markdown_adapter_monolingual_mode() -> None:
    """Verify that MarkdownAdapter supports monolingual target mode."""
    adapter = MarkdownAdapter()
    manifest = BookManifest(doc_id="test_md", title="Test", source_path="test.md")
    blocks = [
        IRBlock(
            id="p1",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            source_text="Hello world.",
            target_text="你好，世界。",
        )
    ]
    tmp = Path(tempfile.mkdtemp())

    # Monolingual target
    out_mono = tmp / "mono.md"
    await adapter.render_blocks(manifest, blocks, "zh", out_mono, bilingual_mode="target")
    content_mono = out_mono.read_text(encoding="utf-8")
    assert "你好，世界。" in content_mono
    assert "Hello world." not in content_mono

    # Alternating
    out_alter = tmp / "alter.md"
    await adapter.render_blocks(manifest, blocks, "zh", out_alter, bilingual_mode="alternating")
    content_alter = out_alter.read_text(encoding="utf-8")
    assert "Hello world." in content_alter
    assert "你好，世界。" in content_alter


async def test_monolingual_markdown_keeps_heading_structure(tmp_path: Path) -> None:
    """A chapter book exported target-only must still be an outline.

    The ``#`` lives in the source line, so the monolingual branch emitted bare
    text for every heading: zero structure, no TOC, every section a paragraph —
    while the other adapters keep a heading a heading.
    """
    adapter = MarkdownAdapter()
    manifest = BookManifest(doc_id="md_head", title="Test", source_path="book.md")
    blocks = [
        IRBlock(
            id="h1",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            block_type=BlockType.HEADING,
            source_text="# Chapter One",
            target_text="第一章",
        ),
        IRBlock(
            id="h2",
            spine_index=2,
            flow_id=FlowID.MAIN_STORY,
            block_type=BlockType.HEADING,
            source_text="## Section A",
            target_text="A 节",
        ),
        IRBlock(
            id="p1",
            spine_index=3,
            flow_id=FlowID.MAIN_STORY,
            source_text="Body prose here.",
            target_text="正文段落。",
        ),
    ]

    out = tmp_path / "mono.md"
    await adapter.render_blocks(manifest, blocks, "zh", out, bilingual_mode="monolingual")
    content = out.read_text(encoding="utf-8")

    assert "# 第一章" in content
    assert "## A 节" in content
    assert "正文段落。" in content
    assert not content.lstrip().startswith("第一章"), "heading lost its marker"


@pytest.mark.asyncio
async def test_markdown_render_removes_temp_file_when_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed write must not leave an orphan temp file in the output directory."""
    adapter = MarkdownAdapter()
    manifest = BookManifest(doc_id="md_fail", title="Test", source_path="book.md")
    blocks = [
        IRBlock(
            id="b1",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            source_text="Hello.",
            target_text="你好。",
        )
    ]
    out = tmp_path / "out.md"

    def _boom(_fd: int) -> None:
        raise OSError("disk full")

    # String target: ``md_adapter.os`` is the stdlib ``os`` module, which the
    # adapter imports directly, so patching the global is the same object.
    monkeypatch.setattr("os.fsync", _boom)
    with pytest.raises(OSError):
        await adapter.render_blocks(manifest, blocks, "zh", out, bilingual_mode="target")
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "out.md"]
    assert leftovers == [], leftovers


@pytest.mark.asyncio
async def test_markdown_adapter_heading_not_merged_with_prose(tmp_path: Path) -> None:
    md_content = "# Title\nFirst line of body prose without blank line.\nSecond line.\n"
    f_path = tmp_path / "heading_prose.md"
    f_path.write_text(md_content, encoding="utf-8")

    adapter = MarkdownAdapter()
    manifest = await adapter.extract_manifest(f_path)
    assert manifest is not None

    blocks: list[IRBlock] = []
    async for ch in adapter.parse_stream(f_path):
        blocks.extend(ch.blocks)

    assert len(blocks) >= 2, f"Expected at least 2 blocks (heading + body), got {len(blocks)}"
    assert blocks[0].block_type == BlockType.HEADING
    assert blocks[0].source_text == "# Title"
    assert blocks[1].block_type == BlockType.NARRATIVE
    assert not blocks[1].source_text.startswith("#")


@pytest.mark.fast
@pytest.mark.asyncio
async def test_markdown_adapter_does_not_html_escape_math_and_code(tmp_path: Path) -> None:
    from ubt.core.ir.models import ChapterMeta

    adapter = MarkdownAdapter()
    manifest = BookManifest(
        doc_id="doc1",
        title="Title",
        source_path="test.md",
        chapters=[
            ChapterMeta(chapter_id="ch_001", title="Title", spine_index=1, source_file="test.md")
        ],
    )
    blocks = [
        IRBlock(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="Formula $x < y$ and code `List<T>` with a & b.",
            target_text="公式 $x < y$ 与代码 `List<T>` 以及 a & b。",
        )
    ]
    out_path = tmp_path / "out.md"
    await adapter.render_blocks(manifest, blocks, target_lang="zh", output_path=out_path)

    content = out_path.read_text(encoding="utf-8")
    assert "&lt;" not in content
    assert "&amp;" not in content
    assert "$x < y$" in content
    assert "`List<T>`" in content
    assert "a & b" in content

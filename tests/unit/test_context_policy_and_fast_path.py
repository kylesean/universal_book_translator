"""Unit tests for Context Policy Resolver and Fast-Path execution."""

from pathlib import Path

import pytest

from ubt.adapters.markdown.adapter import MarkdownAdapter
from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.ir.models import BookManifest
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


@pytest.mark.asyncio
async def test_academic_profile_auto_disables_rolling_summary(tmp_path: Path) -> None:
    """Academic profile should automatically disable rolling summary to protect prompt focus."""
    md = tmp_path / "academic_chapters.md"
    md.write_text(
        "# Chapter 1: Cognitive Foundations\n\nWorking memory has limited capacity.\n\n"
        "# Chapter 2: Attention Mechanisms\n\nSelective attention filters perceptual stimuli.\n",
        encoding="utf-8",
    )
    provider = MockModelProvider(default_response="这是学术翻译。")
    router = ModelRouter(provider=provider)
    orchestrator = PipelineOrchestrator(
        config=UBTConfig(db_dir=tmp_path / "db", rate_limit_rpm=600),
        router=router,
        qe_runner=MockQERunner(default_score=0.9),
    )

    async for _ in orchestrator.run(
        input_path=md,
        output_path=tmp_path / "out.md",
        target_lang="zh",
        profile_name="academic",
        job_id="job_academic_policy",
    ):
        pass

    # No continuation summary should be called for academic textbooks
    assert not any("continuation summary" in c["prompt"] for c in provider.call_history)


@pytest.mark.asyncio
async def test_page_slice_metadata_auto_disables_rolling_summary(tmp_path: Path) -> None:
    """Manifests flagged with is_page_slice_epub should auto-disable rolling summary."""
    md = tmp_path / "scanned_slices.md"
    md.write_text(
        "# Page 1\n\nContent of scanned page 1.\n\n# Page 2\n\nContent of scanned page 2.\n",
        encoding="utf-8",
    )
    # The source carries page numbers ("Page 1"/"Page 2"); the reply must keep
    # them, or the numeric fidelity gate quarantines every block and the export
    # completion floor refuses the job before this test's assertion is reached.
    provider = MockModelProvider(default_response="第1页与第2页的扫描翻译内容。")
    router = ModelRouter(provider=provider)

    class MockSliceAdapter(MarkdownAdapter):
        async def extract_manifest(self, input_path: Path) -> BookManifest:
            manifest = await super().extract_manifest(input_path)
            manifest.metadata["is_page_slice_epub"] = True
            return manifest

    orchestrator = PipelineOrchestrator(
        config=UBTConfig(db_dir=tmp_path / "db", rate_limit_rpm=600),
        router=router,
        qe_runner=MockQERunner(default_score=0.9),
        adapter=MockSliceAdapter(),
    )

    async for _ in orchestrator.run(
        input_path=md,
        output_path=tmp_path / "out.md",
        target_lang="zh",
        profile_name="general",
        job_id="job_page_slice_policy",
    ):
        pass

    # No continuation summary calls on page slices
    assert not any("continuation summary" in c["prompt"] for c in provider.call_history)


@pytest.mark.asyncio
async def test_fast_path_small_doc_single_batch(tmp_path: Path) -> None:
    """Small single-chapter document (<= 15 blocks) runs in fast-path without overhead."""
    md = tmp_path / "short_note.md"
    md.write_text(
        "# Abstract\n\n"
        "This is a concise abstract of the paper.\n\n"
        "It contains only two narrative paragraphs.\n",
        encoding="utf-8",
    )
    provider = MockModelProvider(default_response="这是简短摘要。")
    router = ModelRouter(provider=provider)
    orchestrator = PipelineOrchestrator(
        config=UBTConfig(db_dir=tmp_path / "db", rate_limit_rpm=600),
        router=router,
        qe_runner=MockQERunner(default_score=0.9),
    )

    events = []
    async for event in orchestrator.run(
        input_path=md,
        output_path=tmp_path / "out.md",
        target_lang="zh",
        job_id="job_fast_path",
    ):
        events.append(event)

    out_file = tmp_path / "out.md"
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "这是简短摘要。" in content

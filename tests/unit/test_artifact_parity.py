"""Tests for artifact parity — physical evidence against the source.

The pure parsers get fabricated poppler output; the aggregate check runs
against real repo PDFs when poppler is installed (CI pins the binaries).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.corpus_markers import requires_synthetic_duo, requires_synthetic_mono
from ubt.adapters.pdf.artifact_parity import (
    ParityFinding,
    check_artifact_parity,
    parse_image_objects,
    read_page_sizes,
    target_script_counts,
    target_script_ratio,
)
from ubt.core.job_options import artifact_and_report_paths, sidecar_path

HAS_POPPLER = shutil.which("pdftotext") is not None and shutil.which("pdfinfo") is not None

REPO = Path(__file__).resolve().parents[2]


class TestPureParsers:
    def test_target_script_counts_zh(self) -> None:
        hits, total = target_script_counts("这是中文 content 混排", "zh")
        assert hits == 6  # 这是中文混排
        assert total == 13

    def test_ratio_zero_when_none(self) -> None:
        assert target_script_ratio("pure english text", "zh") == 0.0

    def test_japanese_kana_counts(self) -> None:
        hits, _ = target_script_counts("カタカナと漢字", "ja")
        assert hits >= 5

    def test_korean_counts(self) -> None:
        hits, _ = target_script_counts("한국어 text", "ko")
        assert hits == 3

    def test_latin_target_has_no_ranges(self) -> None:
        hits, total = target_script_counts("plain text", "en")
        assert hits == 0 and total == 9

    @requires_synthetic_duo
    def test_page_sizes_read_from_mediabox(self) -> None:
        sizes = read_page_sizes(REPO / "tests/fixtures/synthetic-duo.pdf")
        assert len(sizes) == 26
        assert all(abs(w - 540.0) < 1.0 and abs(h - 665.972) < 1.0 for w, h in sizes)

    def test_image_objects_dedupes_smask_and_ignores_header(self) -> None:
        listing = (
            "page   num  type   width height color comp bpc  enc interp  object ID\n"
            "   3     0 image     948   542  icc     3   8  jpeg   yes       74  0\n"
            "   3     1 smask     948   542  gray    1   8  jpeg   yes       75  0\n"
            "   7     2 image     888   693  cmyk    4   8  jpeg   yes      144  0\n"
            "   7     3 image     919   724  cmyk    4   8  jpeg   yes      145  0\n"
        )
        assert parse_image_objects(listing) == {3: 1, 7: 2}


@pytest.mark.skipif(not HAS_POPPLER, reason="poppler not installed")
@requires_synthetic_mono
@requires_synthetic_duo
class TestAgainstRealRepoPdfs:
    def test_clean_render_has_no_fail_closed_findings(self) -> None:
        findings = check_artifact_parity(
            source_pdf=REPO / "tests/fixtures/synthetic-mono.pdf",
            artifact_pdf=REPO / "tests/fixtures/synthetic-mono.pdf",
            target_lang="en",
            keeps_source_geometry=True,
        )
        assert not [f for f in findings if f.severity in ("major", "critical")]

    def test_latin_target_reports_unmeasurable_script_not_critical(self) -> None:
        """en/fr/de have no script range, so presence is unmeasurable.

        The gate must say so (info) instead of silently skipping — the old
        code never evaluated its own fail-closed condition for Latin targets.
        """
        findings = check_artifact_parity(
            source_pdf=REPO / "tests/fixtures/synthetic-mono.pdf",
            artifact_pdf=REPO / "tests/fixtures/synthetic-mono.pdf",
            target_lang="en",
            keeps_source_geometry=True,
        )
        info = [f for f in findings if f.code == "target_script_unmeasurable"]
        assert len(info) == 1 and info[0].severity == "info"
        assert "target_language_absent" not in {f.code for f in findings}

    def test_absent_target_language_fails_closed(self) -> None:
        findings = check_artifact_parity(
            source_pdf=REPO / "tests/fixtures/synthetic-mono.pdf",
            artifact_pdf=REPO / "tests/fixtures/synthetic-mono.pdf",  # English artifact
            target_lang="zh",
            keeps_source_geometry=True,
        )
        codes = {f.code for f in findings}
        assert "target_language_absent" in codes
        # CJK behaviour is unchanged: the unmeasurable-script info is Latin-only.
        assert "target_script_unmeasurable" not in codes
        assert any(
            f.severity == "critical" and f.code == "target_language_absent" for f in findings
        )

    def test_geometry_parity_fires_for_wrong_size(self) -> None:
        # chapter-1 against chapter-3 (different page size): source and
        # artifact swapped so sizes must mismatch under keeps-source-geometry.
        findings = check_artifact_parity(
            source_pdf=REPO / "tests/fixtures/synthetic-duo.pdf",
            artifact_pdf=REPO / "tests/fixtures/synthetic-mono.pdf",
            target_lang="en",
            keeps_source_geometry=True,
        )
        codes = {f.code for f in findings}
        assert "page_count_changed" in codes or "page_geometry_changed" in codes

    def test_geometry_parity_skipped_for_reflow(self) -> None:
        findings = check_artifact_parity(
            source_pdf=REPO / "tests/fixtures/synthetic-duo.pdf",
            artifact_pdf=REPO / "tests/fixtures/synthetic-mono.pdf",
            target_lang="en",
            keeps_source_geometry=False,
        )
        codes = {f.code for f in findings}
        assert "page_count_changed" not in codes
        assert "image_count_changed" not in codes


class TestFindingShape:
    def test_parity_finding_has_ducktype_contract(self) -> None:
        f = ParityFinding("critical", "target_language_absent", "msg")
        assert (f.severity, f.code, f.message) == ("critical", "target_language_absent", "msg")
        assert getattr(f, "page", None) is None


def test_deliverables_sharing_a_stem_do_not_share_a_report(tmp_path: Path) -> None:
    """``book.epub`` and ``book.md`` are two documents, not two names for one.

    Sidecar names used to hang off the bare stem, so both defaults
    (``book_bilingual.epub`` / ``book_bilingual.md``, and the same trap for a
    zh-then-ja re-run of one file) resolved to a single
    ``book_bilingual_quality_report.json``. Whichever run finished last owned
    the name: the other's report was overwritten by a document whose
    ``output_path`` pointed at a file no reader finds beside it, while
    ``artifact_and_report_paths`` kept handing both jobs the same JSON.
    """
    epub = sidecar_path(tmp_path / "book_bilingual.epub", "quality_report.json")
    markdown = sidecar_path(tmp_path / "book_bilingual.md", "quality_report.json")
    assert epub != markdown, f"two deliverables share {epub.name}"

    # The reader resolves the same names the writers use.
    epub.touch()
    _, report, _ = artifact_and_report_paths(tmp_path / "book_bilingual.epub")
    assert report == epub
    _, missing, _ = artifact_and_report_paths(tmp_path / "book_bilingual.md")
    assert missing is None, "the reader crossed over into the other document's report"

    metrics = sidecar_path(tmp_path / "book_bilingual.epub", "metrics.json")
    assert metrics.name == "book_bilingual_epub_metrics.json"

"""Tests for output path normalization and artifact parity with page slice."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.job_options import default_output_path, resolve_target_output


@pytest.mark.fast
def test_resolve_target_output_variants(tmp_path: Path) -> None:
    input_pdf = Path("docs/paper.pdf")

    # None -> default_output_path
    assert resolve_target_output(None, input_pdf) == default_output_path(input_pdf)

    # Bare path without extension -> inherits input extension
    assert resolve_target_output(Path("out_fast"), input_pdf) == Path("out_fast.pdf")
    assert resolve_target_output("out_fast", input_pdf) == Path("out_fast.pdf")

    # Existing directory -> puts default filename inside directory
    sub_dir = tmp_path / "deliverables"
    sub_dir.mkdir()
    assert resolve_target_output(sub_dir, input_pdf) == sub_dir / "paper_bilingual.pdf"

    # Trailing slash path (treated as directory)
    assert resolve_target_output("out_fast/", input_pdf) == Path("out_fast/paper_bilingual.pdf")

    # Explicit filename with extension stays as-is
    assert resolve_target_output(Path("custom/result.pdf"), input_pdf) == Path("custom/result.pdf")


@pytest.mark.fast
def test_artifact_parity_selected_pages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When selected_pages is passed, target script ratio only inspects selected pages."""
    from ubt.adapters.pdf import artifact_parity

    calls: list[list[str]] = []

    def mock_run(cmd: list[str]) -> str | None:
        calls.append(cmd)
        # If examining page 1-2, return Chinese
        if "-f" in cmd and "1" in cmd:
            return "这是中文页面内容"
        # Otherwise return English
        return "This is completely English text without any target script characters at all."

    monkeypatch.setattr(artifact_parity, "_run", mock_run)

    src = tmp_path / "src.pdf"
    art = tmp_path / "art.pdf"
    src.write_bytes(b"%PDF-1.4 mock")
    art.write_bytes(b"%PDF-1.4 mock")

    # With selected_pages=[1, 2], calls pdftotext with page constraints
    findings = artifact_parity.check_artifact_parity(
        source_pdf=src,
        artifact_pdf=art,
        target_lang="zh",
        keeps_source_geometry=False,
        selected_pages=[1, 2],
    )

    # Should not report target_language_sparse or target_language_absent
    codes = [f.code for f in findings]
    assert "target_language_sparse" not in codes
    assert "target_language_absent" not in codes
    assert any("-f" in c and "-l" in c for c in calls)

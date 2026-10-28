"""Unit tests for the 'ubt layout-audit' CLI command and diagnostic tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_layout_audit_missing_file_exits_2() -> None:
    """Non-existent file exits with code 2."""
    result = runner.invoke(app, ["layout-audit", "non_existent_doc_abc.pdf"])
    assert result.exit_code == 2
    assert "not found" in result.output.lower() or "error" in result.output.lower()


def test_layout_audit_non_pdf_file_exits_2(tmp_path: Path) -> None:
    """Non-PDF file format is rejected with exit code 2."""
    fake_txt = tmp_path / "notes.txt"
    fake_txt.write_text("Hello world", encoding="utf-8")
    result = runner.invoke(app, ["layout-audit", str(fake_txt)])
    assert result.exit_code == 2
    assert "not a pdf" in result.output.lower()


def test_layout_audit_json_mode_missing_file() -> None:
    """JSON output on non-existent file produces valid structured JSON error."""
    result = runner.invoke(app, ["layout-audit", "missing.pdf", "--json"])
    assert result.exit_code == 2
    data = json.loads(result.output)
    assert data["passed"] is False
    assert "error" in data


def test_layout_audit_real_pdf_human_mode() -> None:
    """Audit over real PDF emits stylized table and spatial heatmap."""
    doc_path = Path("corpus/documents/chapter-3.pdf")
    if not doc_path.is_file():
        pytest.skip("chapter-3.pdf not available")

    result = runner.invoke(app, ["layout-audit", str(doc_path), "--pages", "1-2"])
    assert result.exit_code == 0
    assert "Typst Layout Spill & Overflow Diagnostic" in result.output
    assert "chapter-3.pdf" in result.output


def test_layout_audit_real_pdf_json_mode() -> None:
    """JSON mode emits clean machine-readable report."""
    doc_path = Path("corpus/documents/chapter-3.pdf")
    if not doc_path.is_file():
        pytest.skip("chapter-3.pdf not available")

    result = runner.invoke(app, ["layout-audit", str(doc_path), "--pages", "1-2", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "pdf_path" in data
    assert "total_pages" in data
    assert data["total_pages"] == 26
    assert data["checked_pages"] == [1, 2]
    assert "anomalies" in data
    assert isinstance(data["passed"], bool)


def test_verify_command_with_layout_flag(tmp_path: Path) -> None:
    """Verify command honors --layout option against real PDF artifact."""
    doc_path = Path("corpus/documents/chapter-3.pdf")
    if not doc_path.is_file():
        pytest.skip("chapter-3.pdf not available")

    from ubt.core.content.verify import contract_path_for_artifact

    # Write a dummy contract sidecar next to the PDF in a temp location
    test_pdf = tmp_path / "book.pdf"
    test_pdf.write_bytes(doc_path.read_bytes())
    contract_file = contract_path_for_artifact(test_pdf)
    contract_file.write_text(
        json.dumps(
            {
                "doc_id": "test_doc",
                "total_text": 10,
                "delivered_text": 10,
                "violations": [],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["verify", str(test_pdf), "--layout"])
    assert result.exit_code == 0
    assert "Typst Layout Spill & Overflow Diagnostic" in result.output

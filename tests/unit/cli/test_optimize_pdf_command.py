"""Unit tests for the 'ubt optimize-pdf' CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pikepdf
import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def _create_sample_pdf(tmp_path: Path) -> Path:
    pdf = pikepdf.Pdf.new()
    p1 = pdf.add_blank_page(page_size=(200, 200))
    p2 = pdf.add_blank_page(page_size=(200, 200))

    raw_data = b"STREAM_DATA_FOR_CLI_TEST_" * 20
    s1 = pdf.make_stream(raw_data)
    s1["/Subtype"] = pikepdf.Name("/Image")
    s2 = pdf.make_stream(raw_data)
    s2["/Subtype"] = pikepdf.Name("/Image")

    p1.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im1=s1))
    p2.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im2=s2))
    p1.Contents = pdf.make_stream(b"q /Im1 Do Q")
    p2.Contents = pdf.make_stream(b"q /Im2 Do Q")

    target = tmp_path / "cli_sample.pdf"
    pdf.save(target)
    return target


def test_optimize_pdf_cli_help() -> None:
    res = runner.invoke(app, ["optimize-pdf", "--help"])
    assert res.exit_code == 0
    assert "Miniaturize, deduplicate, and compress a PDF deliverable." in res.stdout
    assert "--profile" in res.stdout
    assert "--engine" in res.stdout


def test_optimize_pdf_cli_success(tmp_path: Path) -> None:
    pdf_path = _create_sample_pdf(tmp_path)
    out_path = tmp_path / "out.pdf"

    res = runner.invoke(
        app,
        ["optimize-pdf", str(pdf_path), "-o", str(out_path), "--profile", "lossless"],
    )
    assert res.exit_code == 0
    assert "PDF Miniaturization Report" in res.stdout
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_optimize_pdf_cli_json(tmp_path: Path) -> None:
    pdf_path = _create_sample_pdf(tmp_path)
    out_path = tmp_path / "out_json.pdf"

    res = runner.invoke(
        app,
        ["optimize-pdf", str(pdf_path), "-o", str(out_path), "--json"],
    )
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["status"] in ("optimized", "skipped_already_optimal")
    assert "reduction_bytes" in data
    assert "pages_processed" in data


def test_optimize_pdf_cli_quiet(tmp_path: Path) -> None:
    pdf_path = _create_sample_pdf(tmp_path)
    out_path = tmp_path / "out_quiet.pdf"

    res = runner.invoke(
        app,
        ["optimize-pdf", str(pdf_path), "-o", str(out_path), "--quiet"],
    )
    assert res.exit_code == 0
    assert res.stdout.strip() == ""
    assert out_path.exists()


def test_optimize_pdf_cli_rejects_unknown_profile(tmp_path: Path) -> None:
    pdf_path = _create_sample_pdf(tmp_path)

    res = runner.invoke(
        app,
        ["optimize-pdf", str(pdf_path), "--profile", "nonexistent_profile"],
    )
    assert res.exit_code == 2
    assert "Unknown profile" in res.stdout


def test_optimize_pdf_cli_rejects_unknown_engine(tmp_path: Path) -> None:
    pdf_path = _create_sample_pdf(tmp_path)

    res = runner.invoke(
        app,
        ["optimize-pdf", str(pdf_path), "--engine", "nonexistent_engine"],
    )
    assert res.exit_code == 2
    assert "Unknown engine" in res.stdout


def test_optimize_pdf_cli_missing_file() -> None:
    res = runner.invoke(app, ["optimize-pdf", "/nonexistent/fake.pdf"])
    assert res.exit_code != 0

"""Unit tests for the 'ubt figure' CLI command suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

runner = CliRunner()
pytestmark = pytest.mark.fast


@pytest.fixture
def sample_figures_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "figures_sample.md"
    content = (
        "# 实验与评估\n\n"
        "![结构A](asset_a.png)\n\n"
        "![结构B](asset_b.png)\n\n"
        "图 4: 实验架构与对比。（a）基线模型 （b）本工作提出的双流架构。\n\n"
        "后续分析正文内容。\n"
    )
    doc.write_text(content, encoding="utf-8")
    return doc


def test_figure_inspect_command(sample_figures_doc: Path) -> None:
    # Text mode
    result = runner.invoke(app, ["figure", "inspect", str(sample_figures_doc)])
    assert result.exit_code == 0
    assert "Total Figure Panels: 1" in result.output
    assert "Composite/Grid Panels: 1" in result.output
    assert "GRID" in result.output

    # JSON mode
    json_res = runner.invoke(app, ["figure", "inspect", str(sample_figures_doc), "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.output)
    assert data["total_panels"] == 1
    assert data["composite_panels"] == 1
    panel = data["panels"][0]
    assert panel["columns"] == 2
    assert len(panel["subfigures"]) == 2
    assert panel["subfigures"][0]["label"] == "(a)"
    assert panel["subfigures"][1]["label"] == "(b)"


def test_figure_generate_command_typst(tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        [
            "figure",
            "generate",
            "model_a.png",
            "model_b.png",
            "-c",
            "Figure 1: Architecture comparison. (a) Generator. (b) Discriminator.",
            "--format",
            "typst",
        ],
    )
    assert res.exit_code == 0
    assert "#figure(" in res.output
    assert "grid(" in res.output
    assert "columns: (1fr, 1fr)" in res.output
    assert 'image("model_a.png", width: 100%)' in res.output
    assert "*(a)* Generator." in res.output
    assert 'image("model_b.png", width: 100%)' in res.output
    assert "*(b)* Discriminator." in res.output


def test_figure_generate_command_html(tmp_path: Path) -> None:
    out_file = tmp_path / "panel.html"
    res = runner.invoke(
        app,
        [
            "figure",
            "generate",
            "fig1.png",
            "fig2.png",
            "-c",
            "Performance charts",
            "--subcaptions",
            "(a) Accuracy, (b) Latency",
            "--format",
            "html",
            "-o",
            str(out_file),
        ],
    )
    assert res.exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert '<figure class="ubt-figure-grid"' in content
    assert "<strong>(a)</strong> Accuracy" in content
    assert "<strong>(b)</strong> Latency" in content
    assert '<figcaption class="ubt-main-caption"' in content

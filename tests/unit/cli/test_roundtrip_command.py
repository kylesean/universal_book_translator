from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_roundtrip_cli_single_pair() -> None:
    res = runner.invoke(
        app,
        [
            "roundtrip",
            "--source",
            "This model is valid.",
            "--target",
            "该模型是有效的。",
            "--back",
            "This model is valid.",
        ],
    )
    assert res.exit_code == 0
    assert "UBT Multi-Language Bidirectional Quality Matrix" in res.stdout
    assert "en <-> zh" in res.stdout


def test_roundtrip_cli_json_export() -> None:
    res = runner.invoke(
        app,
        [
            "roundtrip",
            "--source",
            "Total speed is 500 iter/sec.",
            "--target",
            "总速度非常快。",
            "--back",
            "Total speed is very fast.",
            "--json",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert len(data["matrix_summaries"]) == 1
    item = data["matrix_summaries"][0]["results"][0]
    assert "500" in item["missing_entities"]
    assert item["mqm_severity"] == "critical"


def test_roundtrip_cli_pairs_file(tmp_path: Path) -> None:
    pairs_file = tmp_path / "pairs.json"
    pairs = [
        {
            "src": "First sentence.",
            "mt": "第一句。",
            "back": "First sentence.",
        },
        {
            "src": "Second sentence.",
            "mt": "第二句。",
            "back": "Second sentence.",
        },
    ]
    pairs_file.write_text(json.dumps(pairs), encoding="utf-8")
    md_out = tmp_path / "report.md"

    res = runner.invoke(
        app,
        [
            "roundtrip",
            "--pairs-file",
            str(pairs_file),
            "--markdown-out",
            str(md_out),
        ],
    )
    assert res.exit_code == 0
    assert md_out.exists()
    content = md_out.read_text(encoding="utf-8")
    assert "en <-> zh" in content
    assert "100.0%" in content


def test_roundtrip_cli_missing_input() -> None:
    res = runner.invoke(app, ["roundtrip"])
    assert res.exit_code != 0
    assert "No inputs specified" in res.stdout

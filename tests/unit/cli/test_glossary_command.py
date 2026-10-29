"""Unit tests for the 'ubt glossary' CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_glossary_audit_command_json_and_table(tmp_path: Path) -> None:
    """Verify ubt glossary audit outputs rich table and clean JSON."""
    glossary_file = tmp_path / "glossary.csv"
    glossary_file.write_text(
        "source,target,scope,context_hints,negative_hints,priority\n"
        "driver,驱动程序,chapter:2,kernel;device,golf,10\n"
        "driver,心理驱力,chapter:7,freud;motivation,hardware,10\n"
        "compiler,编译器,*,code;ast,,5\n",
        encoding="utf-8",
    )

    # 1. Human table format
    result = runner.invoke(app, ["glossary", "audit", str(glossary_file)])
    assert result.exit_code == 0
    assert "Glossary Scope & Conflict Audit" in result.stdout
    assert "Total Entries" in result.stdout
    assert "driver" in result.stdout

    # 2. JSON format
    result_json = runner.invoke(app, ["glossary", "audit", str(glossary_file), "--json"])
    assert result_json.exit_code == 0
    data = json.loads(result_json.stdout)
    assert data["total_terms"] == 3
    assert data["unique_sources"] == 2
    assert data["conflicting_sources"] == 1


def test_glossary_audit_strict_fails_on_unresolved(tmp_path: Path) -> None:
    """Verify ubt glossary audit --strict exits with code 1 if ambiguities exist."""
    glossary_file = tmp_path / "ambiguous.csv"
    glossary_file.write_text(
        "source,target,scope\ntransformer,变压器,*\ntransformer,变换器架构,*\n",
        encoding="utf-8",
    )

    # Without strict: exit 0
    result_normal = runner.invoke(app, ["glossary", "audit", str(glossary_file)])
    assert result_normal.exit_code == 0
    assert "UNRESOLVED" in result_normal.stdout

    # With strict: exit 1
    result_strict = runner.invoke(app, ["glossary", "audit", str(glossary_file), "--strict"])
    assert result_strict.exit_code == 1

    # With strict + json
    result_strict_json = runner.invoke(
        app, ["glossary", "audit", str(glossary_file), "--strict", "--json"]
    )
    assert result_strict_json.exit_code == 1


def test_glossary_resolve_command(tmp_path: Path) -> None:
    """Verify ubt glossary resolve calculates chapter/section scope inheritance."""
    glossary_file = tmp_path / "terms.json"
    glossary_data = [
        {"source": "driver", "target": "驾驶员", "scope": "*"},
        {"source": "driver", "target": "驱动程序", "scope": "chapter:2"},
        {"source": "driver", "target": "心理驱力", "scope": "chapter:7"},
        {"source": "buffer", "target": "缓冲器", "scope": "*"},
    ]
    glossary_file.write_text(json.dumps(glossary_data), encoding="utf-8")

    # Resolve chapter 2
    res_ch2 = runner.invoke(
        app, ["glossary", "resolve", str(glossary_file), "--chapter", "2", "--json"]
    )
    assert res_ch2.exit_code == 0
    items_ch2 = json.loads(res_ch2.stdout)
    lookup_ch2 = {
        item["source"]: item.get("translation") or item.get("target") for item in items_ch2
    }
    assert lookup_ch2["driver"] == "驱动程序"
    assert lookup_ch2["buffer"] == "缓冲器"

    # Resolve chapter 7
    res_ch7 = runner.invoke(app, ["glossary", "resolve", str(glossary_file), "-c", "7", "--json"])
    assert res_ch7.exit_code == 0
    items_ch7 = json.loads(res_ch7.stdout)
    lookup_ch7 = {
        item["source"]: item.get("translation") or item.get("target") for item in items_ch7
    }
    assert lookup_ch7["driver"] == "心理驱力"

    # Human table output
    res_table = runner.invoke(app, ["glossary", "resolve", str(glossary_file), "-c", "2"])
    assert res_table.exit_code == 0
    assert "Resolved Effective Glossary" in res_table.stdout
    assert "驱动程序" in res_table.stdout

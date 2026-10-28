"""Unit tests for scripts/diagnose_spills.py diagnostic script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.diagnose_spills import _parse_pages, main

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOC_PATH = _REPO_ROOT / "corpus" / "documents" / "chapter-3.pdf"


def test_diagnose_spills_parse_pages() -> None:
    """Verify page range parsing."""
    assert _parse_pages(None) is None
    assert _parse_pages("") is None
    assert _parse_pages("1,3,5") == [1, 3, 5]
    assert _parse_pages("1-3,5") == [1, 2, 3, 5]
    assert _parse_pages("4-4") == [4]


def test_diagnose_spills_script_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Invoking scripts/diagnose_spills.py with --help exits cleanly."""
    monkeypatch.setattr(sys, "argv", ["diagnose_spills.py", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Diagnose Typst layout spills" in captured.out


def test_diagnose_spills_script_summary_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Summary mode reports concise one-line pass status."""
    if not _DOC_PATH.is_file():
        pytest.skip("chapter-3.pdf not available")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_spills.py",
            str(_DOC_PATH),
            "--pages",
            "1-2",
            "--summary",
        ],
    )
    code = main()
    assert code == 0
    captured = capsys.readouterr()
    assert "chapter-3.pdf" in captured.out
    assert "across" in captured.out


def test_diagnose_spills_script_json_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """JSON mode outputs machine-readable layout diagnosis payload."""
    if not _DOC_PATH.is_file():
        pytest.skip("chapter-3.pdf not available")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_spills.py",
            str(_DOC_PATH),
            "--pages",
            "1-2",
            "--json",
        ],
    )
    code = main()
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "pdf_path" in data
    assert "checked_pages" in data
    assert data["checked_pages"] == [1, 2]
    assert "anomalies" in data

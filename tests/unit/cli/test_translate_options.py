"""`ubt translate` rejects impossible chapter windows and malformed page ranges.

Before this guard a non-positive ``--start-chapter`` / ``--max-chapters`` was
clamped downstream, so the run could deliver a single-chapter book under a
"Translation Completed Successfully!" banner, and a malformed ``--pages``
string only failed deep inside the reader after ingestion had begun. Both are
usage errors and must exit 2 before any work starts.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def _output(result: object) -> str:
    """stdout+stderr: typer.echo(err=True) lands on stderr under click >= 8.2."""
    text = getattr(result, "output", "") or ""
    return text + (getattr(result, "stderr", "") or "")


@pytest.mark.parametrize("flag", ["--start-chapter", "--max-chapters"])
def test_non_positive_chapter_window_is_rejected(flag: str) -> None:
    result = runner.invoke(app, ["translate", "book.pdf", flag, "0"])
    assert result.exit_code == 2
    assert f"{flag} must be >= 1" in _output(result)


def test_negative_start_chapter_is_rejected() -> None:
    result = runner.invoke(app, ["translate", "book.pdf", "--start-chapter", "-3"])
    assert result.exit_code == 2
    assert "--start-chapter must be >= 1" in _output(result)


@pytest.mark.parametrize("pages", ["abc", "5-1", "0-3", "1-"])
def test_malformed_page_range_is_rejected(pages: str) -> None:
    result = runner.invoke(app, ["translate", "book.pdf", "--pages", pages])
    assert result.exit_code == 2
    assert "invalid --pages" in _output(result)


def test_a_valid_window_passes_validation_and_reaches_the_input_check() -> None:
    # A well-formed window must clear validation and fall through to the next
    # guard (a missing input file), proving the new check does not reject it.
    result = runner.invoke(
        app,
        [
            "translate",
            "definitely-missing.pdf",
            "--start-chapter",
            "1",
            "--max-chapters",
            "3",
            "--pages",
            "1-2,5",
        ],
    )
    assert result.exit_code == 2
    assert "Input file not found" in _output(result)
    assert "invalid --pages" not in _output(result)

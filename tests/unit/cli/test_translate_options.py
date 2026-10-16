"""`ubt translate` rejects impossible chapter windows and malformed page ranges.

Before this guard a non-positive ``--start-chapter`` / ``--max-chapters`` was
clamped downstream, so the run could deliver a single-chapter book under a
"Translation Completed Successfully!" banner, and a malformed ``--pages``
string only failed deep inside the reader after ingestion had begun. Both are
usage errors and must exit 2 before any work starts.
"""

from __future__ import annotations

from pathlib import Path

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


@pytest.mark.parametrize("value", ["0", "3", "-1"])
def test_invalid_columns_is_rejected(value: str) -> None:
    result = runner.invoke(app, ["translate", "book.pdf", "--columns", value])
    assert result.exit_code == 2
    assert "--columns must be 1 or 2" in _output(result)


def test_valid_columns_passes_validation_and_reaches_the_input_check() -> None:
    # --columns 1/2 must clear the guard and fall through to the missing-input check.
    result = runner.invoke(app, ["translate", "definitely-missing.pdf", "--columns", "2"])
    assert result.exit_code == 2
    assert "Input file not found" in _output(result)
    assert "--columns must be" not in _output(result)


def test_refuse_existing_output_catches_suffixless_and_mono_dir(tmp_path: Path) -> None:
    """The overwrite guard must see every name the run could write.

    Reported defect: a suffix-less ``-o out/delivery`` resolves to
    ``out/delivery.md`` at delivery time, but the guard checked ``out/delivery``
    (which never exists) and let a second run silently replace the first
    delivery. A directory's *monolingual* derived name was missed for the same
    reason (only the bilingual name was checked).
    """
    import typer

    from ubt.cli.commands.translate import _refuse_existing_output

    root = Path(str(tmp_path))
    inp = root / "book.md"
    inp.write_text("x", encoding="utf-8")

    # Suffix-less explicit output: the .md file is the real deliverable.
    (root / "delivery.md").write_text("previous run", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="already exists"):
        _refuse_existing_output(root / "delivery", fresh=False, input_path=inp)

    # A directory whose MONOLINGUAL name exists must also be refused.
    out_dir = root / "out"
    out_dir.mkdir()
    (out_dir / "book_mono.md").write_text("previous mono", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="already exists"):
        _refuse_existing_output(out_dir, fresh=False, input_path=inp)

    # --fresh still authorizes it.
    assert (
        _refuse_existing_output(root / "delivery", fresh=True, input_path=inp) == root / "delivery"
    )

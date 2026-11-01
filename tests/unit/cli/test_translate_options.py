from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def _output(result: object) -> str:
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
    result = runner.invoke(app, ["translate", "definitely-missing.pdf", "--columns", "2"])
    assert result.exit_code == 2
    assert "Input file not found" in _output(result)
    assert "--columns must be" not in _output(result)


def test_refuse_existing_output_catches_suffixless_and_mono_dir(tmp_path: Path) -> None:
    import typer

    from ubt.cli.commands.translate import _refuse_existing_output

    root = Path(str(tmp_path))
    inp = root / "book.md"
    inp.write_text("x", encoding="utf-8")

    (root / "delivery.md").write_text("previous run", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="already exists"):
        _refuse_existing_output(root / "delivery", fresh=False, input_path=inp)

    out_dir = root / "out"
    out_dir.mkdir()
    (out_dir / "book_mono.md").write_text("previous mono", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="already exists"):
        _refuse_existing_output(out_dir, fresh=False, input_path=inp)

    assert (
        _refuse_existing_output(root / "delivery", fresh=True, input_path=inp) == root / "delivery"
    )

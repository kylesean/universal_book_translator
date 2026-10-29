"""Machine-readable output must survive any payload and any terminal width.

Every ``--json`` flag is a contract: a caller runs ``json.loads(process
.stdout)`` and trusts the result. Two independent ways a Rich ``Console`` used
to break that contract are pinned here, because both were live in the CLI
before ``ubt.cli.ui.emit_json`` existed:

* a single token wider than the console was cut, emitting a bare newline
  inside a JSON string literal (``Invalid control character``);
* ``[bold]``-shaped text inside a payload was consumed as style markup, so the
  document still parsed but the data had silently lost characters.

The assertions below are written against a deliberately hostile payload, so a
regression recovers the bug rather than just failing a formatting check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ubt.cli.ui import emit_json

pytestmark = pytest.mark.fast

# Long enough to exceed the 80-column default that Rich used to chop at, and
# shaped so that Rich's markup detector would eat the bracketed segments.
LONG_PATH = "/tmp/" + ("very-long-directory-name/" * 4) + "pytest-10/popen-gw0/doc.md"
HOSTILE = {
    "source": LONG_PATH,
    "title": "算法[导论] 第2版",
    "note": "see [bold]appendix[/bold] and [red]chapter 3[/red]",
    "nested": {"path": LONG_PATH * 2, "items": [LONG_PATH, "x" * 400]},
}


def test_a_payload_round_trips_through_the_shared_emitter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The emitted document parses back to exactly the payload that went in."""
    emit_json(HOSTILE)
    out = capsys.readouterr().out

    assert json.loads(out) == HOSTILE


def test_no_control_character_is_injected_into_a_long_token(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A token wider than the terminal must never acquire a raw newline.

    This is the exact corruption that blocked release: ``source`` held a path
    of 79 characters, the console was 80 columns wide, and Rich broke the line
    inside the string.
    """
    emit_json({"source": LONG_PATH})
    out = capsys.readouterr().out

    stray = [(i, repr(c)) for i, c in enumerate(out) if ord(c) < 32 and c not in "\n\r\t"]
    assert stray == []
    assert LONG_PATH in json.loads(out)["source"]


@pytest.mark.parametrize("width", [20, 40, 80, 200])
def test_the_payload_is_independent_of_the_terminal_width(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    """Narrow terminals change nothing: the emitter does not consult a width."""
    monkeypatch.setenv("COLUMNS", str(width))
    emit_json(HOSTILE)

    assert json.loads(capsys.readouterr().out) == HOSTILE


def test_bracketed_text_is_data_not_rich_markup(capsys: pytest.CaptureFixture[str]) -> None:
    """Square brackets in a book title survive verbatim.

    Rich would have rendered ``算法[导论]`` as ``算法``: the JSON still parses,
    so nothing catches it downstream, which makes silent data loss worse than a
    crash.
    """
    emit_json({"title": "算法[导论] 第2版", "style": "[bold red]x[/bold red]"})
    parsed = json.loads(capsys.readouterr().out)

    assert parsed["title"] == "算法[导论] 第2版"
    assert parsed["style"] == "[bold red]x[/bold red]"


def test_no_terminal_escape_sequence_reaches_a_pipe(capsys: pytest.CaptureFixture[str]) -> None:
    """Structured output carries no ANSI codes even on a colour-capable tty."""
    emit_json({"status": "ok"})
    out = capsys.readouterr().out

    assert "\x1b" not in out


def test_non_ascii_is_emitted_literally(capsys: pytest.CaptureFixture[str]) -> None:
    """CJK stays readable; ``ensure_ascii`` must not be switched back on."""
    emit_json({"书名": "论法的精神"})
    out = capsys.readouterr().out

    assert "论法的精神" in out
    assert "\\u" not in out


def test_the_emitter_ends_with_a_single_newline(capsys: pytest.CaptureFixture[str]) -> None:
    """Callers that split on lines must not see a trailing blank record."""
    emit_json({"a": 1})
    out = capsys.readouterr().out

    assert out.endswith("}\n")
    assert not out.endswith("\n\n")


# --- the table command end to end, at the width that used to break it --------


def _write_split_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "split_doc.md"
    doc.write_text(
        "| Architecture | Top-1 | Latency |\n"
        "| :--- | :---: | ---: |\n"
        "| ResNet-18 | 69.8% | 8ms |\n\n"
        "| Table 1 (continued) | | |\n"
        "| :--- | :---: | ---: |\n"
        "| ResNet-101 | 77.4% | 24ms |\n",
        encoding="utf-8",
    )
    return doc


def test_table_stitch_json_survives_a_long_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release blocker, reproduced end to end.

    ``pytest-xdist`` puts the worker name (``popen-gw0``) into the temporary
    path, which pushed ``source`` past the console width and broke the JSON --
    so the failure only appeared under ``-n auto``. A deep directory here
    reproduces it deterministically, without depending on the runner's naming.
    """
    from typer.testing import CliRunner

    from ubt.cli.main import app

    deep = tmp_path / ("nested-segment/" * 4)
    deep.mkdir(parents=True)
    doc = _write_split_doc(deep)
    monkeypatch.setenv("COLUMNS", "80")

    res = CliRunner().invoke(app, ["table", "stitch", str(doc), "--json"])

    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["tables_stitched"] == 1
    assert data["source"] == str(doc)


def test_table_inspect_json_survives_a_long_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same guarantee for the sibling command that shared the defect."""
    from typer.testing import CliRunner

    from ubt.cli.main import app

    deep = tmp_path / ("nested-segment/" * 4)
    deep.mkdir(parents=True)
    doc = _write_split_doc(deep)
    monkeypatch.setenv("COLUMNS", "80")

    res = CliRunner().invoke(app, ["table", "inspect", str(doc), "--json"])

    assert res.exit_code == 0
    assert json.loads(res.stdout)["source"] == str(doc)


def test_the_hostile_payload_really_would_have_broken_a_console() -> None:
    """Guard the guard: prove the payload is genuinely hostile.

    If someone shortens ``LONG_PATH`` or the console default widens, the tests
    above could start passing for the wrong reason. This asserts the payload
    still trips the old mechanism, so the suite keeps its teeth.
    """
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    Console(file=buf, width=80, force_terminal=False).print(json.dumps({"source": LONG_PATH}))

    with pytest.raises(json.JSONDecodeError):
        json.loads(buf.getvalue())

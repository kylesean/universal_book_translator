"""A failed corpus case must report the pipeline's own error, not a warning.

``ubt verify --corpus --run`` shells out to ``ubt translate`` and, on a nonzero
exit, reported the *last six lines* of its output. A refused book therefore read
as::

    translation failed (exit 1):  / Using a slow image processor as `use_fast`
    is unset and a slow processor was saved with this model.

which is the HuggingFace startup notice. The actual refusal -- "1 of 92 page(s)
have no extractable text layer" -- was above it. The reporter now prefers the
``Pipeline Execution Failed:`` line the CLI prints, and keeps the tail as the
fallback for a run that died before reaching that handler.
"""

from __future__ import annotations

import subprocess

import pytest

from ubt.cli.commands.verify import _failure_reason

pytestmark = pytest.mark.fast


def _proc(*, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 1, stdout=stdout, stderr=stderr)


def test_the_pipeline_error_line_wins_over_a_trailing_warning() -> None:
    proc = _proc(
        stderr=(
            "Using a slow image processor as `use_fast` is unset and a slow "
            "processor was saved with this model.\n"
            "\x1b[1m\x1b[31mPipeline Execution Failed:\x1b[0m Only part of 'x.pdf' "
            "could be read: 1 of 92 page(s) have no extractable text layer\n"
        )
    )

    reason = _failure_reason(proc)

    assert reason == (
        "Only part of 'x.pdf' could be read: 1 of 92 page(s) have no extractable text layer"
    )
    assert "use_fast" not in reason


def test_rich_tags_are_stripped_when_colour_is_disabled() -> None:
    proc = _proc(stdout="\n[bold red]Pipeline Execution Failed:[/] refused the book\n")

    assert _failure_reason(proc) == "refused the book"


def test_a_run_that_died_before_the_handler_still_reports_its_tail() -> None:
    proc = _proc(stderr="Traceback (most recent call last):\nImportError: no module named typer\n")

    assert _failure_reason(proc) == (
        "Traceback (most recent call last): / ImportError: no module named typer"
    )


def test_an_empty_stream_does_not_crash() -> None:
    assert _failure_reason(_proc()) == ""

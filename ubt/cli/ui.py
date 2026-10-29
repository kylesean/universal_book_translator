"""Shared CLI presentation helpers.

One owner for the things every command needs and no command should
re-implement: the UI language selector, the warning-copy catalogue reader, and
the machine-readable JSON writer.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def ui_msg(en: str, zh: str) -> str:
    """Return English CLI copy by default, or Chinese when ``UBT_UI_LANG=zh``.

    The operator-facing language is an explicit switch (``UBT_UI_LANG``), not a
    property of the terminal: a Chinese-locale shell on an English team is
    common, and the reverse. English is the default so logs and console output
    match the code and the docs.
    """
    return zh if os.environ.get("UBT_UI_LANG", "en").strip().lower().startswith("zh") else en


def emit_json(payload: Any, *, indent: int | None = 2) -> None:
    """Write ``payload`` to stdout as JSON that is guaranteed to parse back.

    Every ``--json`` flag in this CLI is a machine-readable contract: a caller
    does ``json.loads(process.stdout)`` and trusts the result. A Rich
    ``Console.print()`` cannot honour that contract, for two independent
    reasons:

    1. **Hard wrapping.** When a single token is wider than the console, Rich
       cuts it and emits a bare newline *inside* what was a JSON string
       literal. ``{"source": "/a/very/long/path"}`` becomes a document with an
       unescaped control character and ``json.loads`` raises
       ``Invalid control character``. The token width that triggers this is
       the temp-directory name, the book title, or the output path -- all of
       which the user chooses.
    2. **Markup interpretation.** Rich reads ``[bold]`` in the payload as
       style markup and deletes it. A title of ``算法[导论]`` silently loses
       ``[导论]``: the JSON still parses, so the corruption is invisible.

    ``soft_wrap=True, markup=False`` suppresses both, but only if every call
    site remembers both flags -- and 13 of the 31 JSON-emitting sites had
    drifted to no flags at all, with 18 more carrying only ``soft_wrap``.
    Writing to the plain stdout stream removes the possibility instead of
    remembering to avoid it.

    This intentionally emits no colour. Structured output is consumed by
    pipes and parsers, not eyes; a ``--json`` run is already opting out of
    presentation. Terminal runs keep their Rich panels and tables.
    """
    sys.stdout.write(json.dumps(payload, indent=indent, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


__all__ = ["emit_json", "ui_msg"]

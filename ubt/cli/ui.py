"""Shared CLI presentation helpers.

One owner for the two things every command needs and no command should
re-implement: the UI language selector and the warning-copy catalogue reader.
"""

from __future__ import annotations

import os


def ui_msg(en: str, zh: str) -> str:
    """Return English CLI copy by default, or Chinese when ``UBT_UI_LANG=zh``.

    The operator-facing language is an explicit switch (``UBT_UI_LANG``), not a
    property of the terminal: a Chinese-locale shell on an English team is
    common, and the reverse. English is the default so logs and console output
    match the code and the docs.
    """
    return zh if os.environ.get("UBT_UI_LANG", "en").strip().lower().startswith("zh") else en


__all__ = ["ui_msg"]

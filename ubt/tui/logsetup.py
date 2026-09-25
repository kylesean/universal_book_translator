"""Stdout/stderr hygiene for the fullscreen TUI.

tui-design skill, build discipline: never print diagnostics into an active
raw-mode or alternate-screen UI. Engine and third-party loggers (docling,
transformers, Textual itself) all propagate to the root logger, so entering
the TUI re-points the root logger at a rotating file and detaches every
console handler (including the stderr handler ``setup_logging`` installs
for the CLI). Full fidelity stays in the file; the screen only ever shows
curated Chinese summaries (see ``ubt.tui.events``).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path

from ubt.core.fs_perms import restrict_dir_to_owner, restrict_file_to_owner
from ubt.core.log_config import DEFAULT_DATE_FORMAT, DEFAULT_LOG_FORMAT

FILE_MAX_BYTES = 5 * 1024 * 1024
FILE_BACKUPS = 3


def _console_streams() -> tuple[object, ...]:
    return (sys.stdout, sys.stderr)


def _writes_to_console(handler: logging.Handler) -> bool:
    """Whether ``handler`` targets the terminal (stdout or stderr).

    ``RichHandler`` carries no ``.stream``: its destination is
    ``handler.console.file``. Scanning only ``.stream`` therefore left the
    CLI's RichHandler attached when the TUI started, and every record it
    received was drawn straight into the alternate screen.
    """
    streams = _console_streams()
    if getattr(handler, "stream", None) in streams:
        return True
    console = getattr(handler, "console", None)
    if console is not None:
        try:
            return console.file in streams
        except Exception:
            return False
    return False


def default_log_dir() -> Path:
    """Log directory, overridable for tests via ``UBT_TUI_LOG_DIR``."""
    override = os.environ.get("UBT_TUI_LOG_DIR")
    return Path(override).expanduser() if override else Path(".ubt/logs")


def _resolve_level() -> int:
    env_level = os.environ.get("UBT_LOG_LEVEL", "").upper()
    if env_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        return int(getattr(logging, env_level))
    return logging.INFO


def route_logs_to_file(log_dir: Path | None = None) -> Path:
    """Detach console handlers from the root logger; attach a file handler.

    Idempotent enough for one TUI session: repeated calls keep a single
    UBT file handler (same path) instead of stacking duplicates.
    Returns the log file path for the doctor screen.
    """
    target_dir = log_dir or default_log_dir()
    # Session logs quote translated text, so the directory is owner-only: that
    # also covers the rotated copies RotatingFileHandler writes later.
    restrict_dir_to_owner(target_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = target_dir / f"ubt-tui-{stamp}.log"

    # Progress bars from model libraries also write to the tty; the file
    # keeps their records while the screen stays clean.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from ubt.adapters.pdf.docling_parser import configure_hf_environment

        configure_hf_environment(enrich=True)
    except Exception:
        pass

    root = logging.getLogger()
    root.setLevel(_resolve_level())
    for handler in list(root.handlers):
        if _writes_to_console(handler):
            root.removeHandler(handler)
        try:  # noqa: SIM105
            handler.flush()
        except Exception:
            pass

    for handler in root.handlers:
        if (
            isinstance(handler, logging.handlers.RotatingFileHandler)
            and Path(getattr(handler, "baseFilename", "")) == log_path.resolve()
        ):
            return log_path

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=FILE_MAX_BYTES, backupCount=FILE_BACKUPS, encoding="utf-8"
    )
    # The handler opened the file with the process umask (0644 by default).
    restrict_file_to_owner(log_path)
    file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
    root.addHandler(file_handler)
    logging.getLogger("ubt.tui").info("TUI session log: %s", log_path)
    return log_path

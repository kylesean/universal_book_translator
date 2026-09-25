"""Logging configuration and setup for UBT applications and services."""

import logging
import os
import sys
from typing import Any, TextIO

from rich.logging import RichHandler

from ubt.core.log_aggregate import install_noise_aggregators

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Marks a handler this module installed. A later call may replace *its own*
#: handler when the routing request changes, while leaving handlers a foreign
#: owner installed (pytest, an embedding app) untouched. Without the marker the
#: CLI callback's stdout RichHandler was treated as foreign on the second call
#: and a `--json` run kept writing records onto stdout.
_MANAGED_HANDLER_ATTR = "_ubt_managed_handler"

# Third-party loggers that emit high-volume logs during regular operation:
# ``logger name -> (floor level, --verbose restores it)``.
#
# pdf_oxide emits one WARNING per tolerated malformed object (a plain
# Dictionary where ISO 32000 expects a Stream -- see its src/object.rs
# "treating as empty stream" path). The tolerance is correct and not
# actionable for UBT users, but the per-object repetition drowns pipeline
# progress, so it is floored at ERROR. Cheap to restore, hence True: a debug
# run should still see the tolerated objects.
#
# docling logs one PIPELINE_PROFILING DEBUG line per stage per page-batch --
# its own tuning probe, carrying nothing for a UBT run. The table-structure
# warnings are a different animal and are NOT in this table: they are a
# correctness signal, so ``ubt.core.log_aggregate`` counts and buckets them
# rather than muting them. Restoring docling's timings needs an explicit
# UBT_LOG_LEVEL=DEBUG, not --verbose.
_NOISY_LOGGERS: dict[str, tuple[int, bool]] = {
    "pdf_oxide": (logging.ERROR, True),
    "tiny_skia": (logging.ERROR, True),
    "tiny_skia.painter": (logging.ERROR, True),
    "pikepdf": (logging.ERROR, True),
    "pikepdf._core": (logging.ERROR, True),
    "docling": (logging.INFO, False),
    # httpcore/httpx emit 6-8 DEBUG lines per HTTP call (send_request_headers,
    # receive_response_body, ...). With 727 draft calls at concurrency 8 those
    # wire traces buried every line of UBT's own output. The floor is a
    # transport-level detail, not UBT behaviour, so --verbose stays silent;
    # UBT_LOG_LEVEL=DEBUG is the explicit way to ask for it.
    "httpcore": (logging.INFO, False),
    "httpx": (logging.INFO, False),
}


def setup_logging(
    level: int | str | None = None,
    verbose: bool = False,
    stream: TextIO = sys.stderr,
    console: Any = None,
) -> None:
    """Configure root logging for CLI and server processes.

    Outputs to sys.stderr so stdout remains clean for piping and machine-readable
    JSON. Pass ``console`` (a ``rich.console.Console``) to route records through
    Rich instead: a Rich ``Live`` progress bar redraws in place and knows
    nothing about a foreign writer on the same terminal, so a stderr handler
    and the bar end up fighting over the same rows and log lines land *inside*
    the bar. Sharing one Console is the only way the two cooperate.
    """
    if level is None:
        env_level = os.environ.get("UBT_LOG_LEVEL", "").upper()
        if env_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            resolved_level: int = int(getattr(logging, env_level))
            # An explicit level is the deliberate ask; DEBUG lifts every floor.
            lift_all = resolved_level <= logging.DEBUG
        else:
            resolved_level = logging.DEBUG if verbose else logging.INFO
            lift_all = False
    elif isinstance(level, str):
        resolved_level = int(getattr(logging, level.upper(), logging.INFO))
        lift_all = resolved_level <= logging.DEBUG
    else:
        resolved_level = int(level)
        lift_all = resolved_level <= logging.DEBUG

    root = logging.getLogger()
    root.setLevel(resolved_level)

    managed = [h for h in root.handlers if getattr(h, _MANAGED_HANDLER_ATTR, False)]
    foreign = [h for h in root.handlers if not getattr(h, _MANAGED_HANDLER_ATTR, False)]
    # Replace our handler whenever an explicit routing was requested or we
    # already own one; skip only when a foreign owner has the root to itself
    # (pytest, an embedding app) and no console was passed — then we adjust
    # levels and leave their routing intact.
    if console is not None or managed or not foreign:
        if console is not None:
            # RichHandler defers to whatever Live is currently displaying, so
            # records print above the live region instead of tearing it.
            handler: logging.Handler = RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False,
                markup=False,
            )
            handler.setFormatter(logging.Formatter("%(message)s", datefmt=DEFAULT_DATE_FORMAT))
        else:
            handler = logging.StreamHandler(stream)
            handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
        setattr(handler, _MANAGED_HANDLER_ATTR, True)
        for stale in managed:
            root.removeHandler(stale)
            stale.close()
        root.addHandler(handler)
    else:
        for h in root.handlers:
            h.setLevel(resolved_level)

    # NOTSET restores inheritance from root, so a previous non-debug run in the
    # same process cannot leave a floor stuck.
    for name, (floor, restore_on_verbose) in _NOISY_LOGGERS.items():
        restore = lift_all or (verbose and restore_on_verbose)
        logging.getLogger(name).setLevel(logging.NOTSET if restore else floor)

    # Count-and-summarize, never mute: a third party that warns once per
    # recoverable object still has to be countable for visual_report.json.
    install_noise_aggregators()

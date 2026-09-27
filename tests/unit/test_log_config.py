"""Unit tests for UBT logging configuration."""

import io
import logging
from collections.abc import Iterator

import pytest
from rich.console import Console

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.log_aggregate import install_noise_aggregators, noise_aggregators
from ubt.core.log_config import setup_logging


@pytest.fixture(autouse=True)
def _restore_logging_globals() -> Iterator[None]:
    """setup_logging mutates the root and pdf_oxide loggers in place.

    Without this, a test that leaves root at ERROR makes every later test in the
    same process capture nothing, and which one breaks depends on file order.
    """
    root = logging.getLogger()
    quieted = logging.getLogger("pdf_oxide")
    saved_level, saved_handlers = root.level, list(root.handlers)
    saved_quieted = quieted.level
    saved_docling = logging.getLogger("docling").level
    yield
    root.setLevel(saved_level)
    for handler in list(root.handlers):
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()
    root.handlers[:] = saved_handlers
    quieted.setLevel(saved_quieted)
    logging.getLogger("docling").setLevel(saved_docling)
    noise_aggregators().clear()


def test_setup_logging_defaults() -> None:
    buf = io.StringIO()
    setup_logging(stream=buf)
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_installs_a_working_stream_handler() -> None:
    """A handler count is satisfied by pytest's own root handler, so assert on
    bytes: with root cleared, the record has to land in the stream we passed in.
    """
    root = logging.getLogger()
    root.handlers.clear()
    buf = io.StringIO()
    setup_logging(stream=buf, level="WARNING")

    logging.getLogger("ubt.some.module").warning("stream wiring")

    assert "stream wiring" in buf.getvalue()


def test_setup_logging_verbose() -> None:
    buf = io.StringIO()
    setup_logging(verbose=True, stream=buf)
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_LOG_LEVEL", "WARNING")
    buf = io.StringIO()
    setup_logging(stream=buf)
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_explicit_level() -> None:
    buf = io.StringIO()
    setup_logging(level="ERROR", stream=buf)
    assert logging.getLogger().level == logging.ERROR


def test_pdf_oxide_tolerated_object_warnings_are_quieted() -> None:
    """pdf_oxide logs one WARNING per tolerated malformed object; the
    pipeline default must floor that logger at ERROR so progress output
    survives, while a debug run restores full visibility."""
    buf = io.StringIO()
    quieted = logging.getLogger("pdf_oxide")

    setup_logging(stream=buf)  # default INFO
    assert quieted.level == logging.ERROR

    setup_logging(verbose=True, stream=buf)
    assert quieted.level == logging.NOTSET

    setup_logging(stream=buf)
    assert quieted.level == logging.ERROR


# ---------------------------------------------------------------------------
# Noise aggregation: a third-party logger that emits one WARNING
# per recoverable object must collapse into a tiered summary, never into silence.
# ---------------------------------------------------------------------------

_ORPHAN = (
    "Orphan pdf_cell {cid} recovered to col={col} by nearest-column fallback "
    "(row={row}, x={x:.1f}, dist={dist:.1f})"
)


def _emit_orphans(count: int, dist: float) -> None:
    log = logging.getLogger("docling_ibm_models.tableformer.data_management")
    for cid in range(count):
        log.warning(_ORPHAN.format(cid=cid, col=cid % 3, row=cid % 40, x=612.0, dist=dist))


def test_repeated_orphan_warnings_collapse_to_a_tiered_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """1492 per-cell WARNINGs must become a handful of lines, not 1492.

    Measured on a real 92-page paper: 1492 cells were placed by nearest-column
    geometry, 55% of them more than 200pt (7cm) from the column they were
    snapped to. That is a correctness signal, so it is aggregated — never muted.
    """
    install_noise_aggregators()
    with caplog.at_level(logging.WARNING):
        _emit_orphans(1492, dist=241.1)

    orphan_lines = [r.getMessage() for r in caplog.records if "Orphan" in r.getMessage()]
    assert len(orphan_lines) <= 6, f"{len(orphan_lines)} lines — not aggregated"
    # The cascade reports its snapshot, not the final total; the exact count
    # reaches visual_report.json via noise_report().
    assert install_noise_aggregators()["table_structure_guess"].stats()["total"] == 1492


def test_aggregation_reports_the_risk_distribution_not_merely_a_count() -> None:
    """A bare total hides the only actionable part: how bad were the guesses."""
    agg = install_noise_aggregators()["table_structure_guess"]
    _emit_orphans(30, dist=10.0)  # <30pt  reasonable
    _emit_orphans(820, dist=241.0)  # >200pt almost certainly wrong

    stats = agg.stats()
    assert stats["total"] == 850
    assert stats["buckets"]["<=30pt"] == 30
    assert stats["buckets"][">200pt"] == 820
    assert agg.summary().count(">200pt") == 1


def test_aggregation_never_swallows_unrelated_records() -> None:
    """Only the matched pattern is aggregated; the same logger's other warnings
    must still reach the stream, or a real defect would be hidden."""
    logging.getLogger().handlers.clear()  # pytest's own root handler would
    buf = io.StringIO()  # otherwise swallow the stream
    setup_logging(stream=buf)
    install_noise_aggregators()
    _emit_orphans(5, dist=241.0)
    logging.getLogger("docling_ibm_models.tableformer.data_management").warning(
        "TableFormer failed to load a page"
    )
    assert "TableFormer failed to load a page" in buf.getvalue()


def test_setup_logging_installs_the_docling_aggregator() -> None:
    setup_logging(stream=io.StringIO())
    assert "table_structure_guess" in noise_aggregators()


def test_docling_stage_timings_stay_out_of_a_verbose_run() -> None:
    """PIPELINE_PROFILING is docling's own tuning probe: one DEBUG line per
    page-batch per stage. It drowns UBT progress even under --verbose, so the
    docling logger is floored at INFO and only an explicit UBT_LOG_LEVEL=DEBUG
    (not --verbose) restores it."""
    setup_logging(verbose=True, stream=io.StringIO())
    assert logging.getLogger("docling").level == logging.INFO


def test_http_stack_does_not_flood_a_verbose_run() -> None:
    """httpcore/httpx emit 6-8 DEBUG lines per HTTP call.

    Measured on a real 92-page run: 727 draft calls under --concurrency 8
    buried every line of UBT's own output, including the progress bar. The
    sub-phase traces (send_request_headers / receive_response_body / ...)
    carry no information a UBT operator can act on, so --verbose must not
    surface them. An explicit UBT_LOG_LEVEL=DEBUG still restores them.
    """
    setup_logging(verbose=True, stream=io.StringIO())
    assert logging.getLogger("httpcore").level == logging.INFO
    assert logging.getLogger("httpx").level == logging.INFO


def test_explicit_debug_level_still_lifts_the_http_floor() -> None:
    """The floor must not become a one-way door: someone debugging a transport
    problem sets UBT_LOG_LEVEL=DEBUG on purpose and must get the wire trace."""
    setup_logging(level="DEBUG", stream=io.StringIO())
    assert logging.getLogger("httpcore").level == logging.NOTSET


def test_logs_route_through_a_rich_console_when_one_is_given() -> None:
    """A Rich Live progress bar and a stderr StreamHandler fight over the same
    terminal rows, so log lines land *inside* the bar's line.

    Routing both through one Rich Console lets Live own the region and print
    records above it, which is the only way the two can share a terminal.
    """
    import rich.logging

    console = Console(file=io.StringIO())
    setup_logging(console=console, stream=io.StringIO())
    root = logging.getLogger()
    handlers = [h for h in root.handlers if isinstance(h, rich.logging.RichHandler)]
    assert handlers, "expected a RichHandler so Live and log records share a console"


def test_progress_bar_does_not_read_100_before_the_job_finishes() -> None:
    """The bar's numerator is blocks-with-a-terminal-status, which is reached
    the moment drafting ends — while repair, render and export are still
    running. Reporting 100% then is a lie the operator acts on (they stop
    watching a run that is still doing work), so the bar parks at 99% until
    EXPORT_COMPLETED proves the job is actually done."""
    from rich.progress import Progress

    from ubt.cli.main import _note_progress

    buf = io.StringIO()
    progress = Progress(console=Console(file=buf))
    with progress:
        task = progress.add_task("init", total=938)
        draft_done = TranslationProgressEvent(
            job_id="job_t",
            event_type=EventType.DRAFT_BATCH_COMPLETED,
            total_blocks=938,
            completed_blocks=938,
        )
        _note_progress(progress, task, draft_done)
        assert progress.tasks[0].completed < 938, "claimed 100% at draft end"
        assert progress.tasks[0].percentage < 100.0

        _note_progress(
            progress,
            task,
            TranslationProgressEvent(
                job_id="job_t",
                event_type=EventType.EXPORT_COMPLETED,
                total_blocks=938,
                completed_blocks=938,
            ),
        )
        assert progress.tasks[0].percentage == 100.0


def test_json_mode_repoints_logs_from_stdout_to_stderr() -> None:
    import io
    import logging

    from rich.console import Console

    from ubt.core.log_config import setup_logging

    root = logging.getLogger()
    saved = list(root.handlers)
    root.handlers.clear()
    try:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        # Human-mode CLI callback: records go through a stdout Rich console.
        setup_logging(console=Console(file=stdout_buf))
        # `--json` then requests stderr routing at WARNING.
        setup_logging(level="WARNING", stream=stderr_buf)

        logging.getLogger("ubt.some.module").warning("json-mode warning")

        assert "json-mode warning" in stderr_buf.getvalue()
        assert "json-mode warning" not in stdout_buf.getvalue()
    finally:
        for handler in list(root.handlers):
            if handler not in saved:
                root.removeHandler(handler)
        root.handlers[:] = saved


@pytest.mark.fast
def test_noise_aggregator_intercepts_child_logger_orphan_pdf_cell() -> None:
    """Child logger 'docling_ibm_models.tableformer.data_management.matching_post_processor'
    must not bypass NoiseAggregator during propagation."""
    import logging

    from ubt.core.log_aggregate import install_noise_aggregators, noise_aggregators
    from ubt.core.log_config import setup_logging

    stream = io.StringIO()
    setup_logging(level="INFO", stream=stream)
    install_noise_aggregators()
    agg = noise_aggregators()["table_structure_guess"]
    before_seen = agg.seen

    child_logger = logging.getLogger(
        "docling_ibm_models.tableformer.data_management.matching_post_processor"
    )
    for cell_id in range(2277, 2282):
        child_logger.warning(
            "Orphan pdf_cell %d recovered to col=1 by nearest-column fallback (row=34, x=877.6, dist=156.1)",
            cell_id,
        )

    assert agg.seen == before_seen + 5
    assert "Orphan pdf_cell 2278" not in stream.getvalue()


@pytest.mark.fast
def test_configure_logging_quiets_pikepdf_and_pdf_oxide_bridge() -> None:
    """pikepdf, pdf_oxide, and tiny_skia C++/Rust logger bridges must be quieted to ERROR level."""
    import logging

    from ubt.core.log_config import setup_logging

    setup_logging()
    assert logging.getLogger("pikepdf").level >= logging.ERROR
    assert logging.getLogger("pikepdf._core").level >= logging.ERROR
    assert logging.getLogger("pdf_oxide").level >= logging.ERROR
    assert logging.getLogger("tiny_skia").level >= logging.ERROR
    assert logging.getLogger("tiny_skia.painter").level >= logging.ERROR

"""Unit tests for the UBT fullscreen TUI: commands, state, widgets, probe."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest
from rich.console import Console
from typer.testing import CliRunner

from ubt.cli.main import app
from ubt.tui.advisor import DocCategory, DocumentAdvisor, MathDensity
from ubt.tui.commands import help_text, parse_command
from ubt.tui.presets import PRESETS, Preset
from ubt.tui.probe import render_probe_card, scan_local_books
from ubt.tui.state import SessionState
from ubt.tui.widgets import qe_bar, sparkline

runner = CliRunner()


def _render_text(renderable: object) -> str:
    console = Console(record=True, width=140, force_terminal=False)
    console.print(renderable)
    return console.export_text()


def _dual_choice(state: SessionState) -> str:
    """Read the session's dual choice through a value mypy cannot stale-narrow.

    These tests mutate ``state`` by pressing keys, which mypy cannot see; its
    attribute narrowing from the previous assert then reports the next (correct)
    comparison as non-overlapping.
    """
    return state.dual_choice


def test_parse_preset_aliases() -> None:
    assert parse_command("/preset pub").args["preset"] == "publication"
    assert parse_command("/p s").args["preset"] == "standard"
    assert parse_command("/preset preview").ok


def test_parse_translate_full() -> None:
    c = parse_command("/translate book.pdf --preset standard --pages 1-3 --dry-run")
    assert c.ok and c.action == "translate"
    assert c.args["path"] == "book.pdf"
    assert c.args["preset"] == "standard"
    assert c.args["pages"] == "1-3"
    assert c.args["dry_run"] is True


def test_parse_absolute_path_wins_over_command() -> None:
    c = parse_command("/tmp/a.pdf")
    assert c.action == "select_file"


def test_parse_unknown_and_empty() -> None:
    assert parse_command("/nope").error is not None
    assert parse_command("").action == "noop"
    assert "translate" in help_text()


def test_pages_glossary_model_dual() -> None:
    assert parse_command("/pages none").args["pages"] is None
    assert parse_command("/pages 1-5").args["pages"] == "1-5"
    assert parse_command("/dual facing").args["dual"] == "facing"
    assert parse_command("/glossary auto").args["mode"] == "auto"
    assert parse_command("/glossary none").args["mode"] == "none"
    assert parse_command("/model none").args["model"] is None
    assert parse_command("/model gpt-x").args["model"] == "gpt-x"


def test_state_overrides_single_source() -> None:
    from ubt.core.config import UBTConfig
    from ubt.core.job_options import apply_config_overrides

    s = SessionState()
    ov = s.to_overrides()
    # Untouched wizard state injects nothing: env/ubt.toml keep control of
    # UBT_DUAL_MODE / UBT_FACING_SPREAD / UBT_FRESH (R11).
    assert "dual_mode" not in ov
    assert "fresh" not in ov
    cfg = apply_config_overrides(UBTConfig.from_env(), ov)
    assert cfg is not None
    s.choose_dual("monolingual")
    assert s.to_overrides()["dual_mode"] == "monolingual"


def test_state_untouched_knobs_do_not_clobber_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An untouched wizard must not rewrite UBT_DUAL_MODE / FACING_SPREAD / FRESH.

    The old to_overrides() hard-injected all three from defaults, so a TUI run
    silently reset an operator's environment (2026-09 review R11)."""
    from ubt.core.config import UBTConfig
    from ubt.core.job_options import apply_config_overrides

    monkeypatch.setenv("UBT_DUAL_MODE", "alternating")
    monkeypatch.setenv("UBT_FACING_SPREAD", "true")
    monkeypatch.setenv("UBT_FRESH", "true")
    base = UBTConfig.from_env()
    assert base.dual_mode == "alternating" and base.facing_spread and base.fresh

    cfg = apply_config_overrides(base, SessionState().to_overrides())
    assert cfg.dual_mode == "alternating"
    assert cfg.facing_spread is True
    assert cfg.fresh is True

    # The moment the user chooses, the explicit pick wins over env.
    s = SessionState()
    s.choose_fresh(False)
    s.choose_dual("facing")
    cfg2 = apply_config_overrides(base, s.to_overrides())
    assert cfg2.fresh is False
    assert cfg2.dual_mode == "facing" and cfg2.facing_spread is True


def test_state_job_id_rule() -> None:
    s = SessionState()
    assert s.validate_job_id("job_abc-123") == "job_abc-123"
    with pytest.raises(ValueError):
        s.validate_job_id("bad id!")


def test_state_event_fold_and_cost() -> None:
    from ubt.core.engine.events import EventType, TranslationProgressEvent

    s = SessionState()
    assert s.cost_label() == "未知"
    e = TranslationProgressEvent(
        event_type=EventType.DRAFT_BATCH_COMPLETED,
        job_id="job_x",
        total_blocks=10,
        completed_blocks=4,
        current_avg_qe=0.9,
        bottom_15_avg_qe=0.8,
        estimated_cost_usd=0.12,
        cache_hit_rate=0.5,
        message="hello",
        active_block_id="b1",
    )
    s.apply_event(e)
    assert s.completed_blocks == 4
    assert s.progress_pct() == 40.0
    assert s.cost_label() == "$0.1200"
    assert SessionState.qe_color(0.9) == "green"
    assert SessionState.qe_color(0.75) == "yellow"
    assert SessionState.qe_color(0.5) == "red"
    # unknown cost never fabricates zero
    e2 = TranslationProgressEvent(
        event_type=EventType.MTQE_EVALUATED,
        job_id="job_x",
        total_blocks=10,
        completed_blocks=5,
        current_avg_qe=0.8,
        bottom_15_avg_qe=0.7,
        estimated_cost_usd=None,
        message="m",
    )
    s.apply_event(e2)
    assert s.cost_label() == "未知"


def test_sparkline_and_bar() -> None:
    assert len(sparkline([0.1, 0.5, 0.9])) == 3
    assert sparkline([]) != ""
    assert qe_bar(1.0) == "█" * 10
    assert qe_bar(0.0) == "░" * 10


@pytest.mark.asyncio
async def test_v2_app_headless_mount() -> None:
    """Textual headless smoke: wizard mounts without pipeline."""
    from ubt.tui.app import UBTApp
    from ubt.tui.screens import WizardScreen

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)


@pytest.fixture
def sample_math_md(tmp_path: Path) -> Path:
    f = tmp_path / "poisson_test.md"
    f.write_text(
        "# Core Models for FinFETs\n\n"
        "The Poisson equation describes the electrostatic potential in the channel:\n"
        "$$\\frac{\\partial^2 \\psi}{\\partial x^2} = \\frac{q N_{ch}}{\\epsilon_{si}}$$\n"
        "where $t_{ox} = 1$ nm, $T_{FIN} = 20$ nm, and $V_{ch} = 0$ V for MOSFET subthreshold slope.\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_literature_md(tmp_path: Path) -> Path:
    f = tmp_path / "alice_story.md"
    f.write_text(
        "# Chapter 1: Down the Rabbit-Hole\n\n"
        "Alice was beginning to get very tired of sitting by her sister on the bank, "
        "and of having nothing to do: once or twice she had peeped into the book her sister was reading.\n",
        encoding="utf-8",
    )
    return f


def test_advisor_math_document_analysis(sample_math_md: Path) -> None:
    """DocumentAdvisor should detect high math density and semiconductor domain."""
    report = DocumentAdvisor.analyze(sample_math_md)

    assert report.file_name == "poisson_test.md"
    assert report.math_density == MathDensity.HIGH
    assert report.detected_domain == "semiconductor"
    assert report.recommended_render_engine == "reflow"
    assert report.recommended_dual_mode == "inline"
    assert report.recommended_preset == Preset.PUBLICATION
    assert report.recommended_profile == "textbook"
    assert len(report.reasons) >= 2


def test_advisor_literature_analysis(sample_literature_md: Path) -> None:
    """DocumentAdvisor should detect literature and low/none math density."""
    report = DocumentAdvisor.analyze(sample_literature_md)

    assert report.file_name == "alice_story.md"
    assert report.math_density == MathDensity.NONE
    assert report.detected_domain == "general"
    assert report.category == DocCategory.LITERATURE
    assert report.recommended_preset == Preset.STANDARD
    assert report.recommended_profile == "general"
    assert report.route_mode == "auto"


def test_advisor_real_synthetic_duo_pdf_if_exists() -> None:
    """If docs/synthetic-duo.pdf exists in the workspace, verify end-to-end diagnosis."""
    pdf_path = Path("docs/synthetic-duo.pdf")
    if not pdf_path.exists():
        pytest.skip("docs/synthetic-duo.pdf not present")

    report = DocumentAdvisor.analyze(pdf_path)
    assert report.format_ext == "pdf"
    assert report.page_or_ch_count >= 20
    assert report.detected_domain == "semiconductor"
    assert report.math_density == MathDensity.HIGH
    assert report.recommended_render_engine == "rigid"
    assert report.recommended_dual_mode == "monolingual"
    assert report.recommended_glossary is not None
    assert report.recommended_glossary.name == "en-zh.json"
    # The synthetic duo corpus is denser per page than the retired Elsevier
    # sample (tagged equations in every variant paragraph), so its token
    # estimate exceeds the short-chain budget: long-chain routing is the
    # correct verdict here, not a regression.
    assert report.route_mode == "long"
    assert "long" in report.route_reason


def test_probe_card_shows_facts_not_parameter_names(sample_math_md: Path) -> None:
    """The probe card is read-only user-facing copy."""
    report = DocumentAdvisor.analyze(sample_math_md)
    text = _render_text(render_probe_card(report))

    assert "文档预检" in text
    assert report.file_name in text
    assert "建议档位" in text
    for forbidden in ("render_engine", "dual_mode", "math_backend", "formula_enrichment"):
        assert forbidden not in text


def test_presets_map_to_engine_policies() -> None:
    """Preset bundles, correctness gates never disabled."""
    pub = PRESETS[Preset.PUBLICATION].engine_overrides()
    assert pub["exec_mode"] == "auto"
    assert pub["prompt_strategy"] == "rich"
    assert pub["math_backend"] == "mathjax"
    assert pub["formula_render"] == "witness"
    assert "render_engine" not in pub
    assert pub["emit_both"] is False

    std = PRESETS[Preset.STANDARD].engine_overrides()
    assert std["exec_mode"] == "auto"
    assert std["prompt_strategy"] == "auto"
    assert std["formula_render"] == "witness"
    assert "render_engine" not in std

    preview = PRESETS[Preset.PREVIEW].engine_overrides()
    assert preview["formula_enrichment"] == "off"
    assert preview["math_backend"] == "image"
    assert preview["prompt_strategy"] == "minimal"


def test_scan_local_books(tmp_path: Path) -> None:
    """scan_local_books should find files with supported extensions."""
    (tmp_path / "book1.pdf").touch()
    (tmp_path / "book2.epub").touch()
    (tmp_path / "ignore.bin").touch()

    found = scan_local_books(tmp_path)
    names = [f.name for f in found]
    assert "book1.pdf" in names
    assert "book2.epub" in names
    assert "ignore.bin" not in names


def test_cli_tui_help() -> None:
    """CLI should expose 'ubt tui' subcommand with help text."""
    result = runner.invoke(app, ["tui", "--help"])
    assert result.exit_code == 0
    assert "Terminal User Interface" in result.stdout
    assert "--classic" not in result.stdout


def test_cli_translate_interactive_flag() -> None:
    """CLI translate command should support -i/--interactive."""
    result = runner.invoke(app, ["translate", "--help"])
    assert result.exit_code == 0
    assert "--interactive" in result.stdout
    assert "-i" in result.stdout


def test_no_emoji_in_tui_sources() -> None:
    """Skill contract: no emoji / Nerd Font glyphs; ASCII + box + CJK only.

    There is no font detection, only opt-in -- and we do not opt in.
    Block/shade characters (sparkline, QE bar) and box drawing are data,
    not icons, and stay allowed.
    """
    import ubt.core.presets
    import ubt.tui

    # NOTE: preset labels render inside measured boxes (probe card, select
    # options). One emoji there once corrupted panel width math and broke the
    # right border, so engine copy is scanned too -- not just TUI sources.
    targets = sorted(Path(ubt.tui.__file__).parent.glob("*.py"))
    targets.append(Path(ubt.core.presets.__file__))
    allowed_ranges = (
        (0x0000, 0x24FF),  # ASCII, latin, symbols, box drawing
        (0x2500, 0x25FF),  # box drawing + block elements + geometric shapes
        (0x3000, 0x303F),  # CJK punctuation
        (0x4E00, 0x9FFF),  # CJK unified
        (0xFF00, 0xFFEF),  # fullwidth forms
    )
    offenders: list[str] = []
    for py in targets:
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            for ch in line:
                if not any(lo <= ord(ch) <= hi for lo, hi in allowed_ranges):
                    offenders.append(f"{py.name}:{i}: U+{ord(ch):04X} {ch}")
    assert offenders == []


def test_footer_hints_never_rebind_interrupt() -> None:
    """Ctrl+C belongs to the terminal: it quits cleanly, never aborts a job."""
    from ubt.tui.screens import RUN_HINTS

    assert "Ctrl+C 退出" in RUN_HINTS
    assert "中止" not in RUN_HINTS


def test_all_buttons_use_default_variant() -> None:
    """One calm button style; color must not shout on every action."""
    from textual.widgets import Button

    from ubt.tui.app import UBTApp

    async def _check() -> None:
        app = UBTApp(dry_run_override=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for button in app.screen.query(Button):
                assert button.variant == "default", button.id

    import asyncio

    asyncio.run(_check())


@pytest.mark.asyncio
async def test_help_modal_never_stacks() -> None:
    """? while help is open must not stack a second modal."""
    from textual.screen import ModalScreen

    from ubt.tui.app import UBTApp

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.action_show_help()
        await pilot.pause()
        app.action_show_help()
        await pilot.pause()
        modals = [s for s in app.screen_stack if isinstance(s, ModalScreen)]
        assert len(modals) == 1


@pytest.mark.asyncio
async def test_confirm_defaults_to_no() -> None:
    """Destructive confirms focus Cancel: Enter alone must not confirm."""
    from textual.widgets import Button

    from ubt.tui.app import UBTApp
    from ubt.tui.screens import ConfirmModal

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(ConfirmModal("退出确认", "确认退出？"))
        await pilot.pause()
        focused = app.screen.focused
        assert isinstance(focused, Button) and focused.id == "cf-no"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [(140, 40, "full"), (100, 30, "compact"), (70, 24, "minimal"), (50, 20, "too-small")],
)
async def test_breakpoint_ladder(width: int, height: int, expected: str) -> None:
    """Breakpoint ladder: full -> compact -> minimal -> honest too-small."""
    from ubt.tui.app import UBTApp
    from ubt.tui.screens import RunScreen
    from ubt.tui.state import SessionState

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        await app.push_screen(RunScreen(SessionState()))
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RunScreen)
        assert screen.layout_mode == expected, (width, height, screen.layout_mode)


@pytest.mark.asyncio
async def test_finish_probe_ok_updates_state(tmp_path: Path) -> None:
    """Probe completion (UI-thread half) folds the report into session state."""
    from textual.widgets import Static

    from ubt.tui.app import UBTApp
    from ubt.tui.screens import WizardScreen

    doc = tmp_path / "note.md"
    doc.write_text("# Hello\n\nSome plain English prose without math.\n", encoding="utf-8")
    report = DocumentAdvisor.analyze(doc)

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, WizardScreen)
        screen._finish_probe_ok(str(doc), report)
        await pilot.pause()
        assert app.session.input_path == doc.resolve()
        assert app.session.preset == report.recommended_preset
        card = screen.query_one("#probe-card", Static)
        assert "文档预检" in _render_text(card.renderable)


def _max_cell_width(text: str) -> int:
    from rich.cells import cell_len

    return max((cell_len(line) for line in text.splitlines()), default=0)


def test_probe_card_fits_width(sample_math_md: Path) -> None:
    """Probe card lines must fit the console: overflow breaks the border."""
    report = DocumentAdvisor.analyze(sample_math_md)
    assert _max_cell_width(_render_text(render_probe_card(report))) <= 140


def test_probe_card_fits_width_real_pdf() -> None:
    """User-reported case: docs/chapter-1-zh.pdf once broke the right border."""
    from rich.console import Console

    pdf_path = Path("docs/chapter-1-zh.pdf")
    if not pdf_path.exists():
        pytest.skip("docs/chapter-1-zh.pdf not present")
    report = DocumentAdvisor.analyze(pdf_path)
    console = Console(record=True, width=140, force_terminal=False)
    console.print(render_probe_card(report))
    assert _max_cell_width(console.export_text()) <= 140


def test_log_routing_detaches_console_and_files_records(tmp_path: Path) -> None:
    """Entering the TUI must move ALL logger output off stdout/stderr."""
    import logging
    import sys

    from ubt.tui.logsetup import route_logs_to_file

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    probe = logging.StreamHandler(sys.stderr)
    root.addHandler(probe)
    try:
        log_path = route_logs_to_file(tmp_path)
        assert log_path.parent == tmp_path
        assert log_path.exists()
        for handler in root.handlers:
            assert getattr(handler, "stream", None) not in (sys.stdout, sys.stderr)
        logging.getLogger("docling.models.factories").warning("第三方噪音")
        logging.getLogger("ubt.core.engine").warning("引擎记录")
        text = log_path.read_text(encoding="utf-8")
        assert "第三方噪音" in text
        assert "引擎记录" in text
    finally:
        for handler in list(root.handlers):
            if handler not in saved_handlers:
                root.removeHandler(handler)
                try:  # noqa: SIM105
                    handler.close()
                except Exception:
                    pass
        root.setLevel(saved_level)


def test_describe_event_speaks_chinese_only() -> None:
    """Screen ticker lines: Chinese skeleton, raw English stays in the file."""
    from ubt.core.engine.events import EventType, TranslationProgressEvent
    from ubt.tui.events import describe_event, stage_label

    def _ev(kind: EventType, **kw: object) -> TranslationProgressEvent:
        base: dict[str, object] = {
            "event_type": kind,
            "job_id": "job_x",
            "total_blocks": 10,
            "completed_blocks": 4,
            "message": "",
        }
        base.update(kw)
        return TranslationProgressEvent(**base)  # type: ignore[arg-type]

    assert stage_label("BIBLE_EXTRACTED") == "术语圣经就绪"
    assert stage_label("nope") == "NOPE"
    assert "任务开始" in describe_event(_ev(EventType.JOB_STARTED))
    assert "初译 4/10" in describe_event(_ev(EventType.DRAFT_BATCH_COMPLETED))
    bible = describe_event(_ev(EventType.BIBLE_EXTRACTED, message="Translation Bible ok"))
    assert "术语圣经" in bible and "Bible" not in bible
    advised = describe_event(
        _ev(EventType.MODE_ADVISED, message="Render-mode advisory: requested 'inline'")
    )
    assert "引擎建议" in advised and "Render-mode" not in advised
    failed = describe_event(_ev(EventType.PIPELINE_FAILED))
    assert "管线失败" in failed
    done = describe_event(_ev(EventType.EXPORT_COMPLETED, artifact_path="/tmp/a_bilingual.pdf"))
    assert "导出完成" in done and "a_bilingual.pdf" in done


@pytest.mark.asyncio
async def test_run_header_and_cost_speak_chinese() -> None:
    """Header stage and unknown cost render in user language."""
    from rich.console import Console
    from textual.widgets import Static

    from ubt.tui.app import UBTApp
    from ubt.tui.screens import RunScreen
    from ubt.tui.state import SessionState
    from ubt.tui.widgets import TelemetryPanel

    panel = TelemetryPanel()
    state = SessionState()
    state.stage = "BIBLE_EXTRACTED"
    panel.update_state(state)
    console = Console(record=True, width=100, force_terminal=False)
    console.print(panel.renderable)
    text = console.export_text()
    assert "未知" in text and "USD" not in text

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        state.input_path = Path("book.pdf")
        await app.push_screen(RunScreen(state))
        await pilot.pause()
        header = app.screen.query_one("#run-header", Static)
        rendered = header.renderable
        assert isinstance(rendered, str)
        assert "术语圣经就绪" in rendered and "BIBLE_EXTRACTED" not in rendered


def test_doctor_shows_log_path(tmp_path: Path) -> None:
    """Doctor screen names the session log file for post-mortem."""
    from ubt.tui.app import UBTApp

    app = UBTApp(dry_run_override=True, log_path=tmp_path / "ubt-tui-x.log")
    assert str(tmp_path) in app._doctor_body()


def test_should_refresh_policy() -> None:
    """Bursts coalesce to ~7Hz; stage transitions always repaint."""
    from ubt.tui.app import UI_REFRESH_INTERVAL, should_refresh

    assert should_refresh("JOB_STARTED", 100.0, 100.0)
    assert should_refresh("EXPORT_COMPLETED", 100.0, 100.0)
    assert should_refresh("PIPELINE_FAILED", 100.0, 100.0)
    assert not should_refresh("DRAFT_BATCH_COMPLETED", 100.05, 100.0)
    assert should_refresh("DRAFT_BATCH_COMPLETED", 100.0 + UI_REFRESH_INTERVAL, 100.0)


@pytest.mark.asyncio
async def test_single_run_screen_reused() -> None:
    """Repeated runs reuse one RunScreen instead of piling screens."""
    from ubt.tui.app import UBTApp
    from ubt.tui.screens import RunScreen

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._show_run_screen()
        await pilot.pause()
        app._show_run_screen()
        await pilot.pause()
        runs = [s for s in app.screen_stack if isinstance(s, RunScreen)]
        assert len(runs) == 1


@pytest.mark.asyncio
async def test_log_goes_to_visible_run_screen() -> None:
    """With two RunScreens (legacy stacks), the newest wins updates."""
    from ubt.tui.app import UBTApp
    from ubt.tui.screens import RunScreen
    from ubt.tui.state import SessionState
    from ubt.tui.widgets import EventLogView

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        state = SessionState()
        app.push_screen(RunScreen(state))
        await pilot.pause()
        app.push_screen(RunScreen(state))
        await pilot.pause()
        app._emit_log("JOB", "hello-top", refresh=False)
        await pilot.pause()
        logs = [
            s.query_one("#event-log", EventLogView)
            for s in app.screen_stack
            if isinstance(s, RunScreen)
        ]
        assert len(logs) == 2
        assert len(logs[-1].lines) == 1
        assert len(logs[0].lines) == 0


def test_progress_math() -> None:
    """Rate / ETA / stall math is pure and clamped."""
    import time

    from ubt.tui.state import STALL_AFTER_SECS, SessionState
    from ubt.tui.widgets import progress_bar

    assert progress_bar(0.0, 8) == "░" * 8
    assert progress_bar(1.0, 4) == "█" * 4
    assert progress_bar(0.5, 4) == "██░░"
    assert progress_bar(9.9, 4) == "█" * 4
    assert progress_bar(-1.0, 4) == "░" * 4

    s = SessionState(total_blocks=100, completed_blocks=25)
    assert s.progress_frac() == 0.25
    now = time.monotonic()
    assert s.rate_per_min(now) == 0.0
    assert s.eta_text(now) == "—"
    assert s.pace_text(now) == "启动中"

    s.run_started_at = now - 120.0
    s.last_event_at = now
    assert s.rate_per_min(now) == 12.5
    assert s.eta_text(now) == "剩余约6分"
    assert not s.is_stalled(now)
    pace = s.pace_text(now)
    assert "块/分" in pace and "已跑2分" in pace

    s.last_event_at = now - (STALL_AFTER_SECS + 5)
    assert s.is_stalled(now)
    assert "等待响应" in s.pace_text(now)

    assert SessionState.fmt_duration(45) == "45秒"
    assert SessionState.fmt_duration(180) == "3分"
    assert "时" in SessionState.fmt_duration(3900)


@pytest.mark.asyncio
async def test_header_shows_bar_pace_and_stall() -> None:
    """Header: bar + counts + liveness; stale runs say so honestly."""
    import time

    from textual.widgets import Static

    from ubt.tui.app import UBTApp
    from ubt.tui.screens import RunScreen
    from ubt.tui.state import SessionState

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        state = SessionState(total_blocks=84, completed_blocks=20)
        state.input_path = Path("book.pdf")
        state.run_started_at = time.monotonic() - 120.0
        state.last_event_at = time.monotonic()
        run_screen = RunScreen(state)
        await app.push_screen(run_screen)
        await pilot.pause()
        header = run_screen.query_one("#run-header", Static)
        rendered = header.renderable
        assert isinstance(rendered, str)
        assert "█" in rendered and "20/84" in rendered and "块/分" in rendered

        state.last_event_at = time.monotonic() - 500.0
        run_screen.refresh_all()
        await pilot.pause()
        rendered = run_screen.query_one("#run-header", Static).renderable
        assert isinstance(rendered, str)
        assert "等待响应" in rendered


def _render_panel_text(panel: object) -> str:
    from rich.console import Console

    console = Console(record=True, width=100, force_terminal=False)
    console.print(panel)
    return console.export_text()


def test_rail_answers_four_questions_only() -> None:
    """Run-watcher rail: 质量/进度/花费/速度; power levers live in /report."""
    from ubt.tui.state import SessionState
    from ubt.tui.widgets import TelemetryPanel

    panel = TelemetryPanel()
    panel.update_state(SessionState())
    text = _render_panel_text(panel.renderable)
    for wanted in ("质量", "完成", "花费", "速度"):
        assert wanted in text
    for moved in ("缓存命中", "QE 趋势"):
        assert moved not in text
    assert "失败" not in text


def test_rail_failure_row_is_exception_only() -> None:
    """Failure row (with retry hint) appears solely when failures exist."""
    from ubt.tui.state import SessionState
    from ubt.tui.widgets import TelemetryPanel

    panel = TelemetryPanel()
    state = SessionState(failed_blocks=2)
    panel.update_state(state)
    text = _render_panel_text(panel.renderable)
    assert "失败" in text and "r" in text


@pytest.mark.asyncio
async def test_header_failure_marker_is_exception_only() -> None:
    """Header names failures solely when nonzero."""
    from textual.widgets import Static

    from ubt.tui.app import UBTApp
    from ubt.tui.screens import RunScreen
    from ubt.tui.state import SessionState

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        state = SessionState(total_blocks=10, completed_blocks=3)
        state.input_path = Path("book.pdf")
        run_screen = RunScreen(state)
        await app.push_screen(run_screen)
        await pilot.pause()
        clean = run_screen.query_one("#run-header", Static).renderable
        assert isinstance(clean, str) and "失败" not in clean
        state.failed_blocks = 2
        run_screen.refresh_all()
        await pilot.pause()
        marked = run_screen.query_one("#run-header", Static).renderable
        assert isinstance(marked, str) and "失败 2" in marked


def test_launch_detached_never_blocks_or_shares_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    """External viewer spawn: no waiting, no inherited fds, new session."""
    import subprocess
    import sys

    from ubt.tui.app import launch_detached

    calls: list[dict[str, Any]] = []

    class _Popen:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(subprocess, "Popen", _Popen)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not block"))
    )
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("UBT_API_KEY", "sk-secret")
    launch_detached("/tmp/a.pdf")
    assert len(calls) == 1
    assert calls[0]["args"][0] == ["xdg-open", "/tmp/a.pdf"]
    kwargs = calls[0]["kwargs"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    # The launcher must not inherit LLM credentials (defense in depth).
    assert "UBT_API_KEY" not in kwargs["env"]


def test_open_artifact_notifies_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No artifact yet: warn in-app instead of spawning anything."""
    import subprocess

    from ubt.tui.app import UBTApp

    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not spawn"))
    )
    app = UBTApp(dry_run_override=True)
    app._artifact = tmp_path / "nope.pdf"
    notes: list[str] = []
    monkeypatch.setattr(app, "notify", lambda msg, **k: notes.append(str(msg)))
    app.open_artifact()
    assert notes and "尚未生成" in notes[0]


def test_open_artifact_honours_an_explicit_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``/open <path>`` must open the named file, not the last artifact (X35).

    The parser passed the argument through as ``target`` while the handler
    dropped it, so ``/open /tmp/other.pdf`` silently opened an unrelated
    artifact.
    """
    from ubt.tui.app import UBTApp

    opened: list[str] = []
    monkeypatch.setattr("ubt.tui.app.launch_detached", lambda path: opened.append(path))

    named = tmp_path / "named.pdf"
    named.write_bytes(b"%PDF-1.4\n")
    other = tmp_path / "last_artifact.pdf"
    other.write_bytes(b"%PDF-1.4\n")

    app = UBTApp(dry_run_override=True)
    app._artifact = other
    app.open_artifact(str(named))

    assert opened == [str(named.resolve())]


def test_open_artifact_warns_for_a_missing_explicit_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ubt.tui.app import UBTApp

    monkeypatch.setattr(
        "ubt.tui.app.launch_detached",
        lambda path: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )
    app = UBTApp(dry_run_override=True)
    notes: list[str] = []
    monkeypatch.setattr(app, "notify", lambda msg, **k: notes.append(str(msg)))

    app.open_artifact(str(tmp_path / "absent.pdf"))
    assert notes and "路径不存在" in notes[0]


@pytest.mark.asyncio
async def test_cmd_input_submits_to_handle_command() -> None:
    """Typing a command in #cmd-input and pressing Enter must execute it."""
    from textual.widgets import Input

    from ubt.tui.app import UBTApp
    from ubt.tui.screens import DoctorModal, RunScreen
    from ubt.tui.state import SessionState

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = RunScreen(SessionState())
        await app.push_screen(screen)
        await pilot.pause()
        inp = screen.query_one("#cmd-input", Input)
        inp.focus()
        inp.value = "/doctor"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert inp.value == ""
        assert any(isinstance(s, DoctorModal) for s in app.screen_stack)


@pytest.mark.asyncio
async def test_heartbeat_tick_updates_stalled_pace() -> None:
    """1Hz heartbeat tick must update pace/stall without requiring new events."""
    import time

    from textual.widgets import Static

    from ubt.tui.app import UBTApp
    from ubt.tui.screens import RunScreen
    from ubt.tui.state import SessionState

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        state = SessionState(total_blocks=50, completed_blocks=10)
        state.run_started_at = time.monotonic() - 100.0
        state.last_event_at = time.monotonic() - 70.0
        screen = RunScreen(state)
        await app.push_screen(screen)
        await pilot.pause()

        # Simulate heartbeat timer tick
        screen._on_heartbeat_tick()
        await pilot.pause()
        header = screen.query_one("#run-header", Static).renderable
        assert isinstance(header, str)
        assert "等待响应" in header


@pytest.mark.asyncio
async def test_command_palette_modal_filter_and_execute() -> None:
    """Ctrl+K opens CommandPaletteModal, typing filters, Enter dispatches."""
    from textual.widgets import Input

    from ubt.tui.app import UBTApp
    from ubt.tui.screens import CommandPaletteModal, ReportModal, RunScreen
    from ubt.tui.state import SessionState

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = RunScreen(SessionState())
        await app.push_screen(screen)
        await pilot.pause()

        # Press Ctrl+K
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert isinstance(app.screen, CommandPaletteModal)

        pal_inp = app.screen.query_one("#palette-input", Input)
        pal_inp.value = "report"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert any(isinstance(s, ReportModal) for s in app.screen_stack)


@pytest.mark.asyncio
async def test_breakpoint_ladder_compact_distinction() -> None:
    """Compact mode (80-110) must keep side rail while hiding telemetry rail."""
    from ubt.tui.app import UBTApp
    from ubt.tui.screens import RunScreen
    from ubt.tui.state import SessionState

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = RunScreen(SessionState())
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.layout_mode == "compact"
        assert screen.query_one("#run-side").display is True
        assert screen.query_one("#run-right").display is False


@pytest.mark.asyncio
async def test_back_to_wizard_action() -> None:
    """Invoking back to wizard or /new safely returns to WizardScreen."""
    from ubt.tui.app import UBTApp
    from ubt.tui.screens import RunScreen, WizardScreen
    from ubt.tui.state import SessionState

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = RunScreen(SessionState())
        await app.push_screen(screen)
        await pilot.pause()

        app.handle_command("/new")
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)


def test_sqlite_bilingual_snippet_query(tmp_path: Path) -> None:
    """The live preview must read the real ledger schema.

    This test used to ``CREATE`` its own ``blocks`` table with a ``text``
    column — a column the ledger has never had — so the production query raised
    ``OperationalError`` on every event, the surrounding ``except: pass``
    swallowed it, and the bilingual preview stayed empty while the suite stayed
    green. Building the fixture through ``SQLiteJobLedger`` makes the test fail
    the moment the query and the schema disagree.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BlockStatus, DocumentIR, FlowID, IRBlock
    from ubt.tui.app import UBTApp

    job_id = "job_test_bilingual"
    db_file = tmp_path / f"{job_id}.sqlite"
    ledger = SQLiteJobLedger(db_file)
    ledger.init_job(
        job_id,
        DocumentIR(
            doc_id="test_doc",
            source_path="/tmp/test_book.epub",
            format_type="epub",
            blocks=[
                IRBlock(
                    id="ch01#b001",
                    flow_id=FlowID.MAIN_STORY,
                    spine_index=1,
                    source_text="Hello world this is source",
                )
            ],
        ),
        target_lang="zh",
    )
    assert ledger.save_checkpoint(
        block_id="ch01#b001",
        status=BlockStatus.MTQE_PASSED,
        target_text="你好世界这是译文",
    )

    app = UBTApp(dry_run_override=True, db_dir=tmp_path)
    app._state.job_id = job_id
    app._query_bilingual_snippet(db_file, job_id, "ch01#b001")
    assert app._state.active_source == "Hello world this is source"
    assert app._state.active_target == "你好世界这是译文"


@pytest.mark.asyncio
async def test_start_run_resets_a_job_id_recorded_for_another_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job_id whose ledger records a different doc_id must be reset.

    The guard called ``ldg.get_job_metadata``, a method ``SQLiteJobLedger``
    does not have. The AttributeError was swallowed by the surrounding
    ``except Exception: pass``, so the check never fired and a stale job_id
    from a different book was silently reused (resume would then read the
    wrong document's blocks). ``doc_id`` is a job_meta *column*, so the
    accessor is ``get_job_snapshot``.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import DocumentIR
    from ubt.tui.app import UBTApp

    job_id = "job_deadbeef0001"
    db_file = tmp_path / f"{job_id}.sqlite"
    ledger = SQLiteJobLedger(db_file)
    try:
        ledger.init_job(
            job_id,
            DocumentIR(
                doc_id="a_completely_different_document",
                source_path="/tmp/other.pdf",
                format_type="pdf",
            ),
            target_lang="zh",
        )
    finally:
        ledger.close()

    source = tmp_path / "book.md"
    source.write_text("# hi\n", encoding="utf-8")

    app = UBTApp(dry_run_override=True, db_dir=tmp_path)
    # The conflict guard runs before the pipeline is launched; stub the launch
    # so this stays a unit test of the guard rather than a dry-run pipeline.
    monkeypatch.setattr(app, "_run_pipeline", lambda: None)
    app._state.input_path = source
    app._state.job_id = job_id

    async with app.run_test(size=(120, 40)):
        app.start_run()
        assert app._state.job_id is None, "a stale job_id from another document was reused"


def test_ctrl_c_and_new_command_present() -> None:
    """Bindings include ctrl+c and /new is recognized."""
    from ubt.tui.app import UBTApp
    from ubt.tui.commands import parse_command

    # The app declares plain (key, action, description) tuples; mypy only sees
    # Textual's wider ClassVar type, whose Binding member is not iterable.
    declared = cast("list[tuple[str, str, str]]", UBTApp.BINDINGS)
    bindings = {key: action for key, action, _description in declared}
    assert "ctrl+c" in bindings and bindings["ctrl+c"] == "quit_app"
    c = parse_command("/new")
    assert c.ok and c.action == "new"
    c2 = parse_command("/wizard")
    assert c2.ok and c2.action == "new"


@pytest.mark.asyncio
async def test_wizard_dual_select_and_shortcuts() -> None:
    """WizardScreen has #dual-select and keyboard shortcuts (b/f/m) switch mode."""
    from textual.widgets import Select

    from ubt.tui.app import UBTApp
    from ubt.tui.screens import WizardScreen

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        state = app.session
        wiz = app.screen
        assert isinstance(wiz, WizardScreen)
        sel = wiz.query_one("#dual-select", Select)
        assert sel.value == "bilingual"
        assert state.dual_label() == "行内双语"

        # Press f for facing spread
        await pilot.press("f")
        await pilot.pause()
        assert sel.value == "facing"
        assert _dual_choice(state) == "facing"
        assert state.effective_dual_mode() == "facing"
        assert state.to_overrides()["facing_spread"] is True

        # Press m for monolingual
        await pilot.press("m")
        await pilot.pause()
        assert sel.value == "monolingual"
        assert _dual_choice(state) == "monolingual"
        assert state.effective_dual_mode() == "monolingual"
        assert state.to_overrides()["facing_spread"] is False

        # Press b for bilingual
        await pilot.press("b")
        await pilot.pause()
        assert sel.value == "bilingual"
        assert _dual_choice(state) == "bilingual"
        assert state.effective_dual_mode() == "inline"


@pytest.mark.asyncio
async def test_run_screen_toggle_dual_shortcut() -> None:
    """RunScreen 'm' key cycles dual mode without restarting."""
    from ubt.tui.app import UBTApp
    from ubt.tui.screens import RunScreen

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        state = app.session
        screen = RunScreen(state)
        await app.push_screen(screen)
        await pilot.pause()
        assert _dual_choice(state) == "bilingual"

        # Press 'm' to cycle to facing
        await pilot.press("m")
        await pilot.pause()
        assert _dual_choice(state) == "facing"

        # Press 'm' to cycle to monolingual
        await pilot.press("m")
        await pilot.pause()
        assert _dual_choice(state) == "monolingual"


def test_translation_output_extractor_never_leaks_prompt_instructions() -> None:
    """The extractor must recover clean source and never leak prompt instructions."""
    from ubt.core.router.extractor import TranslationOutputExtractor

    prompt = (
        "### Source Paragraph to Translate\n"
        "FIG. 1.1: Shrinking gate length makes the subthreshold swing larger.\n\n"
        "Translate only the text under this heading. Any read-only reference "
        "context shown above is not part of the task: do not translate it, "
        "summarise it, repeat it, or carry its citations and equation numbers "
        "into the output.\n\n"
        "### Translation:"
    )

    marker = "### Source Paragraph to Translate\n"
    remainder = prompt.split(marker, 1)[1]
    for end_tag in (
        "\n\nTranslate only the text under this heading",
        "\n\n### Translation:",
        "\n\nProvide the direct",
    ):
        if end_tag in remainder:
            remainder = remainder.split(end_tag, 1)[0]
            break
    src = remainder.strip()
    mock_out = f"<translation>[模拟翻译] {src}</translation>"
    extracted = TranslationOutputExtractor.extract(mock_out)
    assert (
        extracted
        == "[模拟翻译] FIG. 1.1: Shrinking gate length makes the subthreshold swing larger."
    )
    assert "Translate only the text" not in extracted


def test_deepseek_api_key_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEEPSEEK_API_KEY is recognized by UBTConfig with automatic base_url fallback."""
    from ubt.core.config import UBTConfig

    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.delenv("UBT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENCODE_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-unit-test-key")

    cfg = UBTConfig.from_env()
    assert cfg.api_key.get_secret_value() == "sk-deepseek-unit-test-key"
    assert "api.deepseek.com" in cfg.base_url


def test_typst_reconstructor_strips_prompt_leak_and_mock() -> None:
    """Typst reconstructor cleans prompt instructions and suppresses mock drafts."""
    from ubt.adapters.pdf.typst_reconstructor import _polish_target_text, _resolve_content
    from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock

    raw_leaked = (
        "[模拟翻译] FIG. 1.1: Shrinking gate length makes the subV t swing larger.\n\n"
        "Translate only the text under this heading. Any read-only reference context "
        "shown above is not part of the task: do not translate it."
    )
    cleaned = _polish_target_text(raw_leaked)
    assert "[模拟翻译]" not in cleaned
    assert "Translate only the text" not in cleaned
    assert "FIG. 1.1" in cleaned

    # Blocked human block with mock draft falls back to review placeholder rather than mock text
    blk = IRBlock(
        id="test_b01",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="Fig 1.1 sample",
        draft_text="[模拟翻译] Fig 1.1 sample\nTranslate only the text under this heading...",
        target_text='<mark class="ubt-blocked-human">【待人工审校】Fig 1.1 sample</mark>',
        status=BlockStatus.BLOCKED_HUMAN,
        block_type=BlockType.NARRATIVE,
    )
    eff, text, src = _resolve_content(blk)
    assert "[模拟翻译]" not in eff
    assert "Translate only" not in eff
    assert eff == "【待审校: Fig 1.1 sample】"


@pytest.mark.asyncio
async def test_wizard_fresh_mode_toggle(tmp_path: Path) -> None:
    """WizardScreen fresh toggle button and hotkey 'r' toggle session fresh state."""
    from textual.widgets import Button

    from ubt.tui.app import UBTApp
    from ubt.tui.screens import WizardScreen

    dummy_doc = tmp_path / "book.pdf"
    dummy_doc.write_bytes(b"%PDF-1.4 dummy")

    app = UBTApp(initial_file=dummy_doc, db_dir=tmp_path)
    async with app.run_test() as pilot:
        screen = app.screen
        assert isinstance(screen, WizardScreen)
        assert not app.session.fresh

        # Press 'r' shortcut to toggle fresh mode
        await pilot.press("r")
        assert app.session.fresh is True
        btn = screen.query_one("#btn-fresh", Button)
        assert "全新重译" in str(btn.label)

        # Click button to toggle back
        await pilot.click("#btn-fresh")
        assert app.session.fresh is False
        assert "增量续跑" in str(btn.label)


@pytest.mark.asyncio
async def test_cross_document_job_id_cleared_on_wizard_and_run(tmp_path: Path) -> None:
    """Changing files or returning to wizard resets stale job_id to prevent collision."""
    from ubt.tui.app import UBTApp

    doc_a = tmp_path / "doc_a.pdf"
    doc_a.write_bytes(b"%PDF-1.4 doc a content")
    doc_b = tmp_path / "doc_b.pdf"
    doc_b.write_bytes(b"%PDF-1.4 doc b different content")

    app = UBTApp(initial_file=doc_a, db_dir=tmp_path)
    async with app.run_test() as pilot:
        # Simulate an old job_id from doc_a
        app.session.job_id = "job_111111111111_zh"

        # Return to wizard clears job_id
        app.action_back_to_wizard()
        await pilot.pause()
        assert app.session.job_id is None

        # Select doc_b
        app.session.input_path = doc_b
        # If an old job_id from doc_a somehow lingered
        app.session.job_id = "job_111111111111_zh"

        # start_run detects mismatch and clears it
        app.start_run()
        await pilot.pause()
        assert app.session.job_id != "job_111111111111_zh"


def test_stage_aware_pace_text() -> None:
    """During ingest/model loading before blocks complete, pace_text explains parsing instead of false stall."""
    from ubt.tui.state import SessionState

    s = SessionState()
    s.stage = "JOB_STARTED"
    s.total_blocks = 1
    s.completed_blocks = 0
    s.run_started_at = 100.0
    s.last_event_at = 100.0

    # Short elapsed (< 60s)
    text_short = s.pace_text(now=120.0)
    assert "版面与模型解析中" in text_short
    assert "已跑20秒" in text_short

    # Long elapsed (> 60s, which is is_stalled)
    text_long = s.pace_text(now=230.0)
    assert "版面与模型解析中" in text_long
    assert "查看日志" in text_long

    # Once blocks complete, real stall triggers standard stall message
    s.completed_blocks = 5
    s.total_blocks = 10
    s.stage = "DRAFT_BATCH_COMPLETED"
    text_stalled = s.pace_text(now=230.0)
    assert "等待响应 130s" in text_stalled


def test_configure_hf_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """configure_hf_environment sets HF_HUB_OFFLINE=1 if models cached, or mirror if missing."""
    import os

    from ubt.adapters.pdf.docling_parser import configure_hf_environment

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    for p in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(p, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    # 1. Missing models -> defaults to mirror when no proxy
    configure_hf_environment(enrich=True)
    assert os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com"
    assert "HF_HUB_OFFLINE" not in os.environ

    # 2. When models are cached -> sets HF_HUB_OFFLINE=1
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    hub = tmp_path / "hub"
    for repo in (
        "models--docling-project--docling-layout-heron",
        "models--docling-project--docling-models",
        "models--docling-project--CodeFormulaV2",
    ):
        snap = hub / repo / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")

    configure_hf_environment(enrich=True)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


@pytest.mark.asyncio
async def test_completion_freezes_pace_and_updates_widgets() -> None:
    """When a job completes, timers freeze, no stall is triggered, and widgets show completion."""
    from ubt.core.engine.events import EventType, TranslationProgressEvent
    from ubt.tui.app import UBTApp
    from ubt.tui.screens import RunScreen
    from ubt.tui.state import SessionState
    from ubt.tui.widgets import BilingualCard, StageStepper

    s = SessionState()
    s.input_path = Path("/path/to/chapter-1.pdf")
    s.total_blocks = 84
    s.completed_blocks = 84
    s.run_started_at = 100.0
    s.last_event_at = 124.0

    # Simulate EXPORT_COMPLETED event at tick 124.0
    event = TranslationProgressEvent(
        event_type=EventType.EXPORT_COMPLETED,
        job_id="job_afd4744aa3cd_zh",
        total_blocks=84,
        completed_blocks=84,
        current_avg_qe=0.945,
        artifact_path="tmp/output/chapter-1_bilingual.pdf",
    )
    s.apply_event(event, now=124.0)

    assert s.is_completed is True
    assert s.run_completed_at == 124.0
    # Path identity, not str(): on Windows str(Path("tmp/output/x")) renders
    # backslashes, so comparing strings asserted the runner's separator convention.
    assert s.output_path == Path("tmp/output/chapter-1_bilingual.pdf")

    # Even 10 minutes later (now=724.0):
    now = 724.0
    # 1. Elapsed time is frozen at 24s
    assert s.elapsed_secs(now) == 24.0
    # 2. Rate is frozen (84 blocks in 24s = 210.0 blocks/min)
    assert round(s.rate_per_min(now), 1) == 210.0
    # 3. is_stalled is strictly False
    assert s.is_stalled(now) is False
    # 4. pace_text shows completed summary, never '等待响应'
    pace = s.pace_text(now)
    assert "已完成" in pace
    assert "耗时24秒" in pace
    assert "均速210.0 块/分" in pace
    assert "等待响应" not in pace

    # 5. Stepper and BilingualCard display completed artifact in App context
    app = UBTApp(dry_run_override=True)
    async with app.run_test() as pilot:
        run_screen = RunScreen(s)
        app.push_screen(run_screen)
        await pilot.pause()
        run_screen.refresh_all()

        from io import StringIO

        from rich.console import Console

        card = run_screen.query_one("#bilingual", BilingualCard)
        c_io = StringIO()
        Console(file=c_io, color_system=None).print(card.renderable)
        rendered_card = c_io.getvalue()
        assert "全书翻译与导出完成" in rendered_card
        assert "chapter-1_bilingual.pdf" in rendered_card
        assert "等待引擎分配文本任务" not in rendered_card

        stepper = run_screen.query_one("#stage-stepper", StageStepper)
        s_io = StringIO()
        Console(file=s_io, color_system=None).print(stepper.renderable)
        rendered_stepper = s_io.getvalue()
        assert "全部完成" in rendered_stepper


def test_state_apply_cli_request_seeds_and_arms() -> None:
    """Flags parsed before -i resolved must survive into the wizard and, for
    dual/fresh/preset, arm the explicit overrides (L17)."""
    from pathlib import Path as _P

    s = SessionState()
    s.apply_cli_request(
        {
            "target_lang": "ja",
            "source_lang": "fr",
            "preset": "publication",
            "dual_mode": "monolingual",
            "fresh": True,
            "draft_model": "deepseek-chat",
            "glossary": _P("/tmp/g.csv"),
            "pages": "1-10",
            "job_id": "job_seed",
        }
    )
    assert s.target_lang == "ja" and s.source_lang == "fr"
    assert s.preset_explicit and s.dual_explicit and s.fresh_explicit
    assert s.draft_model == "deepseek-chat"
    assert s.glossary == _P("/tmp/g.csv")
    assert s.pages == "1-10"
    assert s.job_id == "job_seed"
    ov = s.to_overrides()
    assert ov["dual_mode"] == "monolingual"
    assert ov["fresh"] is True
    assert ov["draft_model"] == "deepseek-chat"

    # An absent (None) flag must not arm anything.
    s2 = SessionState()
    s2.apply_cli_request({"target_lang": "zh", "dual_mode": None, "fresh": None})
    assert not s2.dual_explicit and not s2.fresh_explicit
    assert "dual_mode" not in s2.to_overrides()


def test_bilingual_snippet_reads_uri_special_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ledger paths with '#', '?' or '%' must not truncate the sqlite URI
    (the f-string form fed them to the parser raw, L19)."""
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BlockStatus, DocumentIR, FlowID, IRBlock
    from ubt.tui.app import UBTApp

    weird_dir = tmp_path / "odd#dir?x%"
    weird_dir.mkdir()
    db = weird_dir / "job_uri.sqlite"
    ledger = SQLiteJobLedger(db)
    doc = DocumentIR(
        doc_id="d_uri",
        source_path="book.md",
        format_type="md",
        metadata={},
        blocks=[
            IRBlock(
                id="ch01#b001",
                flow_id=FlowID.MAIN_STORY,
                spine_index=1,
                source_text="hello",
            )
        ],
    )
    ledger.init_job("uri_job", doc, target_lang="zh")
    ledger.save_checkpoint(
        block_id="ch01#b001", status=BlockStatus.DRAFTED, target_text="こんにちは"
    )
    ledger.close()

    app = UBTApp()
    seen: list[tuple[str, str]] = []

    def _capture(source: str | None, target: str | None, block_id: str | None) -> None:
        seen.append((source or "", target or ""))

    monkeypatch.setattr(app._state, "update_bilingual_snippet", _capture)
    app._query_bilingual_snippet(db, "uri_job", "ch01#b001")
    assert seen == [("hello", "こんにちは")]


@pytest.mark.asyncio
async def test_status_command_rejects_a_traversal_job_id() -> None:
    """``/status`` must apply the same job-id rule as /resume, the CLI and the API.

    Without it ``ledger_dir / "../../tmp/other.sqlite"`` resolved outside the
    ledger dir and ran the schema migrations on an arbitrary SQLite file.
    """
    from ubt.tui.app import UBTApp

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._show_status("../../tmp/other")
        await pilot.pause()
        messages = [msg for _ts, stage, msg in app.session.event_tail if stage == "STATUS"]
        assert any("仅允许" in m for m in messages), messages
        assert not any("账本不存在" in m for m in messages), messages


def test_cli_profile_seed_reaches_the_run_kwargs() -> None:
    """``ubt translate -i --profile paper`` must run as the paper profile.

    CLI, API and MCP all map ``profile`` onto ``run(profile_name=...)``; the
    wizard dropped it, so the same request interactively lost the academic
    prompt path, the profile's seed glossary and the rolling-summary rule.
    """
    s = SessionState()
    s.apply_cli_request({"profile": "paper"})
    assert s.to_run_kwargs()["profile_name"] == "paper"

    # Untouched: the engine's own default, not a value invented here.
    assert SessionState().to_run_kwargs()["profile_name"] == "general"


class _ScriptedOrchestrator:
    """An orchestrator that yields a fixed event script, then fails or not."""

    def __init__(self, events: list[Any], *, boom: bool) -> None:
        self._events = events
        self._boom = boom
        self.kwargs: dict[str, Any] = {}

    async def run(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.kwargs = kwargs
        for event in self._events:
            yield event
        if self._boom:
            raise RuntimeError("budget exceeded")


def _ev(kind: Any, **kw: Any) -> Any:
    from ubt.core.engine.events import TranslationProgressEvent

    base: dict[str, Any] = {
        "event_type": kind,
        "job_id": "job_tui",
        "total_blocks": 10,
        "completed_blocks": 10,
    }
    base.update(kw)
    return TranslationProgressEvent(**base)


@pytest.mark.parametrize(("boom", "expect_done"), [(True, False), (False, True)])
async def test_run_only_reports_completion_after_the_export_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    boom: bool,
    expect_done: bool,
) -> None:
    """A raised pipeline or a /cancel must not render as "全书翻译与导出完成".

    The run wrapper used to call ``mark_completed`` from its ``finally`` block,
    so every exit path -- including the failure one -- flipped the status card
    to 已完成 and printed an artifact path the wizard had merely *declared*.
    Completion is now the EXPORT_COMPLETED event's verdict alone.
    """
    from ubt.core.engine import dry_run
    from ubt.core.engine.events import EventType
    from ubt.tui.app import UBTApp

    events = (
        [_ev(EventType.PREPROCESSING_DONE)]
        if boom
        else [_ev(EventType.EXPORT_COMPLETED, artifact_path=str(tmp_path / "out.md"))]
    )
    fake = _ScriptedOrchestrator(events, boom=boom)
    monkeypatch.setattr(dry_run, "create_dry_run_orchestrator", lambda config: fake)

    app = UBTApp(dry_run_override=True)
    async with app.run_test(size=(120, 40)) as pilot:
        app._state.input_path = tmp_path / "book.md"
        app._state.output_path = tmp_path / "declared-but-never-written.md"
        app._run_pipeline()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.session.is_completed is expect_done, (
            f"boom={boom} reported completion {app.session.is_completed}"
        )
        if not expect_done:
            # The declared path is not evidence of a deliverable.
            assert "未产出" in "".join(msg for _ts, _kind, msg in app.session.event_tail)


def test_tui_start_button_preserves_job_id() -> None:
    from unittest.mock import MagicMock, patch

    from ubt.tui.screens import WizardScreen

    screen = WizardScreen.__new__(WizardScreen)
    screen._state = MagicMock()
    screen._state.input_path = Path("/tmp/book.pdf")
    screen._state.job_id = "job_custom123"

    with patch("ubt.tui.screens._ubt_app") as mock_app:
        mock_app.return_value.start_run = MagicMock()
        screen._on_start_btn(MagicMock())
        # job_id must NOT be wiped to None!
        assert screen._state.job_id == "job_custom123"


def test_tui_action_show_report_passes_body() -> None:
    from unittest.mock import MagicMock, patch

    from ubt.tui.screens import RunScreen

    screen = RunScreen.__new__(RunScreen)
    mock_app_instance = MagicMock()
    mock_app_instance._report_body.return_value = "Detailed Quality Report"

    with patch("ubt.tui.screens._ubt_app", return_value=mock_app_instance):
        screen.action_show_report()
        mock_app_instance.push_screen.assert_called_once()
        modal = mock_app_instance.push_screen.call_args[0][0]
        assert modal._body == "Detailed Quality Report"


@pytest.mark.fast
def test_tui_custom_job_id_preserved_when_no_conflict(tmp_path: Path) -> None:
    from unittest.mock import patch

    from ubt.tui.app import UBTApp

    dummy_file = tmp_path / "book.md"
    dummy_file.write_text("# Test", encoding="utf-8")

    app = UBTApp()
    s = app._state
    s.input_path = dummy_file
    s.job_id = "my_custom_translation_job"

    # Mock _run_pipeline and _show_run_screen so start_run doesn't launch background worker
    with patch.object(app, "_show_run_screen"), patch.object(app, "_run_pipeline"):
        app.start_run()

    # Custom job_id should NOT be reset to None!
    assert s.job_id == "my_custom_translation_job"


@pytest.mark.fast
def test_telemetry_snapshot_properties() -> None:
    """SessionState.to_snapshot() produces a frozen TelemetrySnapshot with accurate properties."""
    from dataclasses import FrozenInstanceError

    from ubt.tui.state import SessionState, TelemetrySnapshot

    s = SessionState(
        stage="DRAFT",
        total_blocks=100,
        completed_blocks=50,
        avg_qe=0.885,
        bottom15_qe=0.742,
        cost_usd=0.1234,
        run_started_at=100.0,
    )
    snap = s.to_snapshot()
    assert isinstance(snap, TelemetrySnapshot)
    assert snap.progress_pct() == 50.0
    assert snap.cost_label() == "$0.1234"
    assert snap.elapsed_secs(160.0) == 60.0
    assert snap.rate_per_min(160.0) == 50.0
    assert "50.0 块/分" in snap.pace_text(160.0)

    # Verify immutability
    with pytest.raises(FrozenInstanceError):
        snap.completed_blocks = 99  # type: ignore[misc]


@pytest.mark.fast
def test_probe_card_renders_assessment_quote(sample_math_md: Path) -> None:
    """render_probe_card includes AssessQuoteDeck when assessment report is present."""
    report = DocumentAdvisor.analyze(sample_math_md)
    assert report.assessment is not None
    text = _render_text(render_probe_card(report))
    assert "译前报价与成本预估" in text
    assert "计费分块:" in text
    assert "预期总成本:" in text


@pytest.mark.fast
def test_chapter_tree_heatmap_rendering() -> None:
    """ChapterTreeHeatmap renders chapters and block heatmaps cleanly."""
    from ubt.tui.widgets import ChapterHeatmapData, ChapterTreeHeatmap

    heatmap = ChapterTreeHeatmap()
    assert "等待章节发现" in heatmap.format_heatmap_text()

    data = [
        ChapterHeatmapData(
            chapter_index=1,
            title="Introduction",
            total_blocks=4,
            completed_blocks=4,
            repaired_blocks=1,
            failed_blocks=0,
            needs_human_blocks=0,
            avg_qe=0.91,
            block_statuses=(
                ("completed", 0.95),
                ("completed", 0.88),
                ("repaired", 0.90),
                ("completed", 0.72),
            ),
        ),
        ChapterHeatmapData(
            chapter_index=2,
            title="Methods",
            total_blocks=3,
            completed_blocks=1,
            repaired_blocks=0,
            failed_blocks=1,
            needs_human_blocks=1,
            avg_qe=0.65,
            block_statuses=(
                ("completed", 0.86),
                ("needs_human", 0.60),
                ("failed", None),
            ),
        ),
    ]
    heatmap.update_chapters(data)
    rendered = heatmap.format_heatmap_text()
    assert "第 1 章" in rendered
    assert "Introduction" in rendered
    assert "第 2 章" in rendered
    assert "Methods" in rendered
    assert "图例:" in rendered
    assert "■" in rendered


@pytest.mark.fast
def test_pe_slash_command_and_bindings() -> None:
    """Slash command /pe parses correctly and RunScreen has 'p' binding."""
    from ubt.tui.commands import parse_command
    from ubt.tui.screens import RunScreen

    cmd = parse_command("/pe")
    assert cmd.ok and cmd.action == "pe" and cmd.args.get("block_id") is None

    cmd_block = parse_command("/pe ch01_b005")
    assert (
        cmd_block.ok and cmd_block.action == "pe" and cmd_block.args.get("block_id") == "ch01_b005"
    )

    declared = cast("list[tuple[str, str, str]]", RunScreen.BINDINGS)
    bindings = {key: action for key, action, _description in declared}
    assert "p" in bindings and bindings["p"] == "post_edit"


@pytest.mark.fast
def test_pe_modal_save_atomic_writeback(tmp_path: Path) -> None:
    """PEModal saves revisions to SQLite ledger with BlockStatus.REPAIRED and syncs to TM."""
    from unittest.mock import MagicMock, patch

    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BlockStatus, FlowID, IRBlock
    from ubt.core.memory.tm import TranslationMemory
    from ubt.tui.screens import PEModal

    db_path = tmp_path / "test_job.sqlite"
    from ubt.core.ir.models import DocumentIR

    block = IRBlock(
        id="b001",
        flow_id=FlowID.MAIN_STORY,
        spine_index=0,
        source_text="Hello world.",
        target_text="Draft hello.",
        status=BlockStatus.NEEDS_HUMAN,
    )
    doc_ir = DocumentIR(
        doc_id="test_doc",
        source_path=str(tmp_path / "test.md"),
        format_type="md",
        blocks=[block],
        metadata={"source_lang": "en"},
    )
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job("test_job", doc_ir, target_lang="zh")

    modal = PEModal(
        job_id="test_job",
        block_id="b001",
        flow_id="ch01",
        source_text="Hello world.",
        target_text="Draft hello.",
        status="needs_human",
        defect_note="MQM critical blocked",
        db_path=db_path,
        source_lang="en",
        target_lang="zh",
    )

    mock_text_area = MagicMock()
    mock_text_area.text = "你好，世界。"
    dismiss = MagicMock()

    with (
        patch.object(modal, "query_one", MagicMock(return_value=mock_text_area)),
        patch.object(modal, "dismiss", dismiss),
    ):
        modal.action_save()

    # Modal should dismiss with True (success)
    dismiss.assert_called_once_with(True)

    # Verify SQLite WAL ledger was updated atomically!
    with SQLiteJobLedger(db_path) as ledger:
        b = ledger.get_block("b001")
        assert b is not None
        assert b.status == BlockStatus.REPAIRED
        assert b.target_text == "你好，世界。"
        assert "human_pe_imported" in b.error_flags

    # Verify TM was populated with PROVENANCE_HUMAN_PE
    tm_path = tmp_path / "tm.sqlite"
    assert tm_path.exists()
    tm = TranslationMemory(tm_path)
    res = tm.lookup_exact("en", "zh", "Hello world.")
    assert res is not None
    assert res.target_text == "你好，世界。"
    assert res.provenance == "human_pe"


@pytest.mark.fast
def test_full_quality_report_formatting() -> None:
    """_format_full_quality_report formats all 15 dimensions into human-readable text."""
    from datetime import UTC, datetime

    from ubt.core.engine.reporter import (
        QualityReport,
        ReportEntityConsistency,
        ReportPlaceholderMetrics,
        ReportRenderCoverage,
        ReportRepairBreakdown,
        ReportScoreMetrics,
        ReportSummary,
        ReportTerminologyMetrics,
    )
    from ubt.tui.app import UBTApp

    app = UBTApp()
    qr = QualityReport(
        job_id="job_full_test",
        doc_id="doc_full_test",
        book_title="Test Publication",
        source_path="/path/test.pdf",
        output_path="/path/test_out.pdf",
        target_lang="zh",
        generated_at=datetime.now(UTC),
        summary=ReportSummary(
            total_blocks=10,
            completed_blocks=10,
            repaired_blocks=2,
            failed_blocks=0,
            needs_human_blocks=0,
            blocked_human_blocks=0,
            pass_rate=1.0,
            estimated_cost_usd=0.08,
        ),
        score_metrics=ReportScoreMetrics(
            avg_qe=0.92,
            min_qe=0.81,
            max_qe=0.98,
            p10_qe=0.84,
            p50_qe=0.93,
            p90_qe=0.97,
            bottom_15_avg_qe=0.83,
            scored_blocks=10,
        ),
        repair_breakdown=ReportRepairBreakdown(
            direct_pass_count=8,
            round_1_repaired_count=2,
            round_2_repaired_count=0,
            exhausted_count=0,
        ),
        terminology=ReportTerminologyMetrics(
            terms_expected=5,
            terms_rendered=5,
            term_precision=1.0,
            fuzzy_term_precision=1.0,
            term_recall=1.0,
        ),
        entity_consistency=ReportEntityConsistency(
            terms_audited=4,
            terms_with_drift=0,
            top_drifted=[],
        ),
        placeholder=ReportPlaceholderMetrics(
            masked_spans=2,
            corrupt_spans=0,
            retention_rate=1.0,
            masked_blocks=2,
            corrupt_blocks=0,
        ),
        render_coverage=ReportRenderCoverage(
            rendered_blocks=10,
            skipped_blocks=0,
            render_coverage=1.0,
        ),
        defect_flags={"minor_typo": 1},
    )

    formatted = app._format_full_quality_report(qr)
    assert "job_full_test" in formatted
    assert "初通率 (Pass Rate)：100.0%" in formatted
    assert "平均分 (Avg)：0.9200" in formatted
    assert "术语精确率：100.0%" in formatted
    assert "占位符保护率：100.0%" in formatted
    assert "minor_typo" in formatted

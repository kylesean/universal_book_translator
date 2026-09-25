"""Screens for TUI v2: wizard, run, and modals.

Interaction rules (tui-design skill, interaction-patterns):
- ``?`` help lives on the App only (screens must not double-bind it).
- Ctrl+C is never rebound: it quits cleanly, the ledger keeps every state.
- Destructive confirms default to No (focus starts on Cancel).
- Validation waits: path input only acts on Enter, never per keystroke.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
from textual import events, on, work
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
    Rule,
    Select,
    Static,
    TextArea,
)

from ubt.tui.advisor import AdvisoryReport, DocumentAdvisor
from ubt.tui.commands import PALETTE_COMMANDS, help_text
from ubt.tui.events import stage_label
from ubt.tui.probe import render_probe_card
from ubt.tui.state import SessionState
from ubt.tui.theme import ACCENT, ERROR, MUTED_RICH, WARNING
from ubt.tui.widgets import (
    BilingualCard,
    ChapterHeatmapData,
    ChapterTreeHeatmap,
    EventLogView,
    StageStepper,
    TelemetryPanel,
    progress_bar,
)

if TYPE_CHECKING:
    from ubt.tui.app import UBTApp

logger = logging.getLogger(__name__)

RUN_HINTS = "? 帮助 · / 命令 · Ctrl+K 命令面板 · Tab 切换 · p 审校 · r 重试 · o 打开 · d 报告 · Esc 向导 · Ctrl+C 退出"

KEYS_HELP = """按键：
  Tab / Shift+Tab   在面板间切换
  1 / 2 / 3         切换质量档（运行前）
  r                 重试失败块      p  人工审校(PE)  o  打开产物      d  质量报告
  /                 聚焦命令输入    Ctrl+K  打开命令面板
  Esc               返回主向导      ?  帮助
  Ctrl+C            退出（账本保留，可续跑）"""

# Responsive floor (visual-patterns: breakpoint ladder + minimum state).
MIN_WIDTH = 60
MIN_HEIGHT = 16
COMPACT_BELOW = 110  # under: telemetry pane hides, drill in with d
MINIMAL_BELOW = 80  # under: single center pane


def _ubt_app(screen: Screen[None]) -> UBTApp:
    from ubt.tui.app import UBTApp as _App

    app = screen.app
    assert isinstance(app, _App)
    return app


class WizardScreen(Screen[None]):
    """Home wizard: pick file, probe, preset, confirm. All keyboard reachable."""

    BINDINGS = [
        ("ctrl+p", "focus_path", "聚焦路径输入"),
        ("1", "preset_pub", "出版级"),
        ("2", "preset_std", "标准"),
        ("3", "preset_prev", "预览"),
        ("b", "dual_bilingual", "行内双语"),
        ("f", "dual_facing", "左右对照"),
        ("m", "dual_monolingual", "纯单语"),
        ("d", "toggle_dryrun", "演练模式"),
        ("r", "toggle_fresh", "重译模式"),
        ("p", "probe_current", "预检文档"),
    ]

    def __init__(self, state: SessionState, initial_file: Path | None = None) -> None:
        super().__init__()
        self._state = state
        self._initial_file = initial_file
        self._report: AdvisoryReport | None = None
        self._candidates: list[Path] = []

    def compose(self) -> Any:
        with Vertical(id="wiz-root"):
            yield Static(
                "[bold cyan]UBT 2.0[/] · 通用长文档工业级翻译向导  [dim]键盘优先 · 响应式监视[/]",
                id="wiz-title",
            )
            with Horizontal(id="wiz-body"):
                with Vertical(id="wiz-left"):
                    with Vertical(id="card-files", classes="wiz-card"):
                        yield Label("1 · 选择待译文档（上下键移动，Enter 预检）")
                        yield ListView(id="file-list")
                        yield Input(
                            placeholder="粘贴/拖入文件路径，或输入关键字模糊过滤…", id="path-input"
                        )
                    with Vertical(id="card-config", classes="wiz-card"):
                        yield Label("2 · 质量档位（快捷键 1/2/3）")
                        yield Select(
                            [
                                ("出版级 · 最佳质量（重排/对照全覆盖，3-pass）", "publication"),
                                ("标准 · 质量与成本平衡（生产推荐，2-pass）", "standard"),
                                ("预览 · 抽检/演练最快（1-pass 无质检）", "preview"),
                            ],
                            value=self._state.preset.value,
                            id="preset-select",
                            allow_blank=False,
                        )
                        yield Label("3 · 译文形态（快捷键 b 行内双语 / f 左右对照 / m 纯单语）")
                        yield Select(
                            [
                                ("行内双语 (Inline) · 原文译文逐段对照（研读推荐）", "bilingual"),
                                ("左右跨页 (Facing) · 左页原文右页译文（学术/出版推荐）", "facing"),
                                ("纯目标语 (Monolingual) · 纯译文发行（流畅通读）", "monolingual"),
                            ],
                            value=self._state.dual_choice,
                            id="dual-select",
                            allow_blank=False,
                        )
                    with Vertical(id="card-env", classes="wiz-card"):
                        yield Static("", id="api-status")
                with (
                    Vertical(id="wiz-right"),
                    Vertical(id="card-probe", classes="wiz-card"),
                ):
                    yield Label("4 · 文档预检事实与结构分析（只读探测）")
                    with VerticalScroll(id="probe-scroll"):
                        yield Static("尚未预检", id="probe-card")
            with Vertical(id="wiz-bottom"):
                with Horizontal(id="wiz-actions"):
                    yield Button("开始翻译", id="btn-start")
                    yield Button("预检", id="btn-probe")
                    yield Button("模式：增量续跑", id="btn-fresh")
                    yield Button("演练模式：关", id="btn-dryrun")
                    yield Button("退出", id="btn-quit")
                yield Static("", id="wiz-status")

    def on_mount(self) -> None:
        from ubt.core.config import MOCK_API_KEY, UBTConfig
        from ubt.tui.probe import scan_local_books

        cfg = UBTConfig.from_env()
        has_key = cfg.api_key.get_secret_value() != MOCK_API_KEY
        if has_key:
            self.query_one("#api-status", Static).update(
                f"[green]● 密钥已就绪[/] · 默认模型: [bold cyan]{cfg.draft_model}[/] · 真实翻译模式"
            )
        else:
            self.query_one("#api-status", Static).update(
                "[bold yellow]▲ 未检测到有效 API 密钥[/]"
                "（自动演练模式，配置 UBT_LLM_API_KEY 或 OPENAI_API_KEY 开启真实翻译）"
            )

        if self._initial_file is not None and self._initial_file.exists():
            self._state.input_path = self._initial_file.resolve()
            self._probe_current()
            return
        try:
            cands = scan_local_books()
        except Exception:
            cands = []
        self._candidates = cands[:20]
        lv = self.query_one("#file-list", ListView)
        lv.clear()
        for p in self._candidates:
            try:
                rel = str(p.relative_to(Path.cwd()))
            except ValueError:
                rel = str(p)
            size = round(p.stat().st_size / (1024 * 1024), 2) if p.exists() else 0.0
            lv.append(ListItem(Label(f"{p.name}  {size}MB  {rel}")))
        if not self._candidates:
            self.query_one("#wiz-status", Static).update("未发现本地候选，直接粘贴路径即可。")

    @on(ListView.Selected, "#file-list")
    def _on_file_selected(self, event: ListView.Selected) -> None:
        idx = self.query_one("#file-list", ListView).index
        if idx is not None and 0 <= idx < len(self._candidates):
            self._probe_path(self._candidates[idx])

    @on(Input.Submitted, "#path-input")
    def _on_path_submitted(self, event: Input.Submitted) -> None:
        raw = event.value.strip().strip("'\"")
        if not raw:
            return
        p = Path(raw).expanduser()
        if p.exists() and p.is_file():
            self._probe_path(p)
            event.input.value = ""
            return
        try:
            from rapidfuzz import fuzz

            scored = sorted(
                ((fuzz.WRatio(raw, c.name), c) for c in self._candidates),
                key=lambda t: t[0],
                reverse=True,
            )
            top = [c for s, c in scored if s > 40][:20]
            lv = self.query_one("#file-list", ListView)
            lv.clear()
            self._candidates = top
            for c in top:
                lv.append(ListItem(Label(c.name)))
            self.query_one("#wiz-status", Static).update(f"模糊搜 {raw!r}：{len(top)} 个候选")
        except Exception as exc:
            self.query_one("#wiz-status", Static).update(f"路径不存在：{raw}（{exc}）")

    @on(Select.Changed, "#preset-select")
    def _on_preset_changed(self, event: Select.Changed) -> None:
        from ubt.tui.presets import Preset as _Preset

        try:  # noqa: SIM105
            chosen = _Preset(str(event.value))
        except Exception:
            return
        if chosen is self._state.preset and not self._state.preset_explicit:
            # The probe's recommendation pre-fill echoing back. A
            # recommendation is shown, not adopted: only an affirmative pick
            # arms the bundle's engine knobs (see SessionState.preset_explicit).
            return
        self._state.choose_preset(chosen)

    @on(Select.Changed, "#dual-select")
    def _on_dual_changed(self, event: Select.Changed) -> None:
        v = str(event.value)
        if v in ("bilingual", "monolingual", "facing"):
            self._state.choose_dual(v)  # type: ignore[arg-type]
            self.query_one("#wiz-status", Static).update(
                f"输出形态已设为：{self._state.dual_label()}"
            )

    @on(Button.Pressed, "#btn-probe")
    def _on_probe_btn(self, event: Button.Pressed) -> None:
        self._probe_current()

    @on(Button.Pressed, "#btn-dryrun")
    def _on_dryrun_btn(self, event: Button.Pressed) -> None:
        self._state.dry_run = not self._state.dry_run
        event.button.label = "演练模式：开" if self._state.dry_run else "演练模式：关"
        self.query_one("#wiz-status", Static).update(
            "演练模式开（零消耗）" if self._state.dry_run else "演练模式关（真实 API）"
        )

    @on(Button.Pressed, "#btn-fresh")
    def _on_fresh_btn(self, event: Button.Pressed) -> None:
        self.action_toggle_fresh()

    @on(Button.Pressed, "#btn-start")
    def _on_start_btn(self, event: Button.Pressed) -> None:
        if self._state.input_path is None:
            self.query_one("#wiz-status", Static).update("先选一个文件再开始。")
            return
        _ubt_app(self).start_run()

    @on(Button.Pressed, "#btn-quit")
    def _on_quit_btn(self, event: Button.Pressed) -> None:
        self.app.exit()

    def _probe_current(self) -> None:
        if self._state.input_path is not None:
            self._probe_path(self._state.input_path)

    def _probe_path(self, path: Path) -> None:
        self.query_one("#wiz-status", Static).update("正在预检文档与环境…")
        # Capture the app on the UI thread: worker threads cannot resolve it.
        self._run_probe(_ubt_app(self), path)

    @work(thread=True, exclusive=True)
    def _run_probe(self, app: UBTApp, path: Path) -> None:
        """Probe off the UI thread; results re-enter via call_from_thread."""
        try:
            report = DocumentAdvisor.analyze(path)
        except Exception as exc:
            app.call_from_thread(self._finish_probe_fail, str(exc))
        else:
            app.call_from_thread(self._finish_probe_ok, str(path), report)

    def _finish_probe_fail(self, error: str) -> None:
        if not self.is_mounted:
            return
        self.query_one("#wiz-status", Static).update(f"预检失败：{error}")

    def _finish_probe_ok(self, path_str: str, report: AdvisoryReport) -> None:
        if not self.is_mounted:
            return
        path = Path(path_str)
        self._report = report
        self._state.input_path = path.resolve()
        self._state.job_id = None
        self._state.recommended_glossary = report.recommended_glossary
        self._state.detected_domain = report.detected_domain
        if report.detected_domain != "general":
            self._state.domain = report.detected_domain
        self._state.preset = report.recommended_preset
        try:  # noqa: SIM105
            self.query_one("#preset-select", Select).value = report.recommended_preset.value
        except Exception:
            pass
        self.query_one("#probe-card", Static).update(render_probe_card(report))
        history_note = ""
        try:
            # One resolution point with the run page sidebar: a raw _db_dir
            # fallback skips the UBT_DB_DIR / ubt.toml chain, so the wizard's
            # history scan and the sidebar then disagreed about where ledgers
            # live.
            db_dir = _ubt_app(self)._ledger_dir()
            if db_dir.exists():
                from ubt.core.engine.ledger import SQLiteJobLedger

                for jf in db_dir.glob("job_*.sqlite"):
                    try:
                        with SQLiteJobLedger(jf) as led:
                            snap = led.get_job_snapshot(jf.stem)
                            if (
                                snap
                                and snap.get("total", 0) > 0
                                and path.stem in snap.get("source_path", "")
                            ):
                                comp = snap.get("completed", 0) + snap.get("repaired", 0)
                                history_note = f" · 发现历史进度: {comp}/{snap['total']} 块 (按 r 可切换全新重译)"
                                break
                    except Exception:
                        pass
        except Exception:
            pass

        self.query_one("#wiz-status", Static).update(
            f"已预检：{report.file_name}，建议档 {report.recommended_preset.value}{history_note}"
        )

    def action_toggle_fresh(self) -> None:
        self._state.choose_fresh(not self._state.fresh)
        btn = self.query_one("#btn-fresh", Button)
        btn.label = "模式：全新重译 (--fresh)" if self._state.fresh else "模式：增量续跑 (默认)"
        self.query_one("#wiz-status", Static).update(
            "全新重译模式：将清理历史旧账本，从头全新翻译"
            if self._state.fresh
            else "增量续跑模式：命中历史缓存将秒级跳过"
        )

    def action_focus_path(self) -> None:
        self.query_one("#path-input", Input).focus()

    def action_preset_pub(self) -> None:
        self._set_preset("publication")

    def action_preset_std(self) -> None:
        self._set_preset("standard")

    def action_preset_prev(self) -> None:
        self._set_preset("preview")

    def action_dual_bilingual(self) -> None:
        self._set_dual("bilingual")

    def action_dual_facing(self) -> None:
        self._set_dual("facing")

    def action_dual_monolingual(self) -> None:
        self._set_dual("monolingual")

    def _set_dual(self, v: str) -> None:
        try:  # noqa: SIM105
            self.query_one("#dual-select", Select).value = v
            self._state.choose_dual(v)  # type: ignore[arg-type]
            self.query_one("#wiz-status", Static).update(
                f"输出形态已设为：{self._state.dual_label()}"
            )
        except Exception:
            pass

    def action_toggle_dryrun(self) -> None:
        btn = self.query_one("#btn-dryrun", Button)
        self._state.dry_run = not self._state.dry_run
        btn.label = "演练模式：开" if self._state.dry_run else "演练模式：关"
        self.query_one("#wiz-status", Static).update(
            "演练模式开（零消耗）" if self._state.dry_run else "演练模式关（真实 API）"
        )

    def action_probe_current(self) -> None:
        self._probe_current()

    def _set_preset(self, v: str) -> None:
        from ubt.tui.presets import Preset as _Preset

        try:  # noqa: SIM105
            self.query_one("#preset-select", Select).value = v
            self._state.choose_preset(_Preset(v))
        except Exception:
            pass


class RunScreen(Screen[None]):
    """Three-pane run monitor: sidebar / stream / telemetry + footer command."""

    BINDINGS = [
        ("slash", "focus_input", "命令"),
        ("ctrl+k", "command_palette", "命令面板"),
        ("r", "retry_failed", "重试失败块"),
        ("p", "post_edit", "人工审校(PE)"),
        ("o", "open_artifact", "打开产物"),
        ("d", "show_report", "质量报告"),
        ("m", "toggle_dual_mode", "切换双语/单语"),
        ("escape", "back_to_wizard", "返回向导"),
    ]

    def __init__(self, state: SessionState) -> None:
        super().__init__()
        self._state = state
        self.layout_mode = "full"

    def compose(self) -> Any:
        with Vertical(id="run-root"):
            yield Static("", id="too-small")
            yield Static("", id="run-header")
            yield Rule(id="header-rule")
            with Horizontal(id="run-body"):
                with Vertical(id="run-side"):
                    yield Label("任务 (Enter 查看详情/回放)")
                    yield ListView(id="job-list")
                    yield Label("全书章节与分块全景", id="chapter-label")
                    with VerticalScroll(id="chapter-tree-scroll"):
                        yield ChapterTreeHeatmap(id="chapter-tree")
                    yield ListView(id="run-files")
                with Vertical(id="run-center"):
                    yield StageStepper(id="stage-stepper")
                    with VerticalScroll(id="preview-scroll"):
                        yield BilingualCard(id="bilingual")
                    yield Static("", id="failure-list")
                with Vertical(id="run-right"):
                    yield Label("遥测")
                    yield TelemetryPanel(id="telemetry")
                    yield Static("", id="engine-notes")
            with Vertical(id="run-footer"):
                yield EventLogView(id="event-log")
                yield Input(placeholder="输入 / 查看命令，或按 Ctrl+K 打开命令面板", id="cmd-input")
                yield Static(RUN_HINTS, id="run-hints")

    def on_mount(self) -> None:
        self.refresh_all()
        self.refresh_sidebar()
        self.set_interval(1.0, self._on_heartbeat_tick)
        self._apply_breakpoints()

    def _on_heartbeat_tick(self) -> None:
        """Periodic 1Hz tick: drives duration, pace, and honest >60s stall notes."""
        if self._state.run_started_at is not None:
            self.refresh_all()
            self._refresh_heatmap()

    def on_resize(self, event: events.Resize) -> None:
        self._apply_breakpoints()

    def refresh_sidebar(self) -> None:
        """Populate files and dynamic history job list from ledger directory."""
        self._refresh_heatmap()
        try:
            fl = self.query_one("#run-files", ListView)
            fl.clear()
            if self._state.input_path is not None:
                fl.append(ListItem(Label(str(self._state.input_path.name))))

            jl = self.query_one("#job-list", ListView)
            jl.clear()
            cur_jid = self._state.job_id
            if cur_jid:
                jl.append(ListItem(Label(f"[bold cyan]{cur_jid}[/] (当前)"), name=cur_jid))
            else:
                jl.append(ListItem(Label("[grey50]暂无运行任务[/]"), name=""))

            d = _ubt_app(self)._ledger_dir()
            if d.exists():
                files = sorted(
                    d.glob("job_*.sqlite"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )[:15]
                for p in files:
                    jid = p.stem
                    if jid != cur_jid:
                        mtime_str = datetime.fromtimestamp(p.stat().st_mtime).strftime(
                            "%m-%d %H:%M"
                        )
                        jl.append(ListItem(Label(f"{jid} · [grey50]{mtime_str}[/]"), name=jid))
        except Exception:
            pass

    def _refresh_heatmap(self) -> None:
        try:
            tree = self.query_one("#chapter-tree", ChapterTreeHeatmap)
        except Exception:
            return

        cur_jid = self._state.job_id
        if not cur_jid:
            tree.update_chapters([])
            return

        db_path = _ubt_app(self)._ledger_dir() / f"{cur_jid}.sqlite"
        if not db_path.exists():
            return

        try:
            import sqlite3
            from collections import defaultdict

            # Percent-encode the path (spaces, '#', '?' …): a raw
            # ``file:{path}?mode=ro`` URI silently opens the wrong file for a
            # db_dir containing URI metacharacters.
            with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT block_id, flow_id, spine_index, status, mtqe_score FROM blocks "
                    "WHERE job_id = ? ORDER BY spine_index ASC",
                    (cur_jid,),
                ).fetchall()
                if not rows:
                    return

                chapters_dict: dict[str, list[sqlite3.Row]] = defaultdict(list)
                for r in rows:
                    flow = str(r["flow_id"] or "ch01")
                    chapters_dict[flow].append(r)

                data: list[ChapterHeatmapData] = []
                for idx, (flow, blocks) in enumerate(chapters_dict.items(), 1):
                    tot = len(blocks)
                    comp = sum(
                        1
                        for b in blocks
                        if (b["status"] or "").lower()
                        in ("completed", "translated", "mtqe_passed", "repaired")
                    )
                    rep = sum(1 for b in blocks if (b["status"] or "").lower() == "repaired")
                    fail = sum(1 for b in blocks if (b["status"] or "").lower() == "failed")
                    pe = sum(
                        1
                        for b in blocks
                        if (b["status"] or "").lower() in ("needs_human", "blocked_human")
                    )
                    scored = [float(b["mtqe_score"]) for b in blocks if b["mtqe_score"] is not None]
                    avg_qe = sum(scored) / len(scored) if scored else None
                    statuses = tuple(
                        (
                            str(b["status"] or "pending"),
                            (float(b["mtqe_score"]) if b["mtqe_score"] is not None else None),
                        )
                        for b in blocks
                    )
                    data.append(
                        ChapterHeatmapData(
                            chapter_index=idx,
                            title=flow,
                            total_blocks=tot,
                            completed_blocks=comp,
                            repaired_blocks=rep,
                            failed_blocks=fail,
                            needs_human_blocks=pe,
                            avg_qe=avg_qe,
                            block_statuses=statuses,
                        )
                    )
                tree.update_chapters(data)
        except Exception as exc:
            logger.debug("Heatmap query failed: %s", exc)

    @on(ListView.Selected, "#job-list")
    def _on_job_selected(self, event: ListView.Selected) -> None:
        jid = getattr(event.item, "name", None)
        if not jid or not jid.startswith("job_"):
            return
        db = _ubt_app(self)._ledger_dir() / f"{jid}.sqlite"
        if not db.exists():
            return
        self.app.push_screen(JobDetailModal(jid, db))

    @on(Input.Submitted, "#cmd-input")
    def _on_cmd_submitted(self, event: Input.Submitted) -> None:
        val = event.value.strip()
        event.input.value = ""
        if val:
            _ubt_app(self).handle_command(val)

    def _apply_breakpoints(self) -> None:
        """Breakpoint ladder: full -> compact -> minimal -> too-small."""
        try:
            w, h = self.size.width, self.size.height
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                mode = "too-small"
            elif w < MINIMAL_BELOW:
                mode = "minimal"
            elif w < COMPACT_BELOW:
                mode = "compact"
            else:
                mode = "full"
            self.layout_mode = mode
            self.query_one("#run-side").display = mode in ("full", "compact")
            self.query_one("#run-right").display = mode == "full"
            notice = self.query_one("#too-small", Static)
            if mode == "too-small":
                notice.update(f"终端太小（当前 {w}x{h}），至少需要 {MIN_WIDTH}x{MIN_HEIGHT}")
                notice.display = True
                self.query_one("#run-header").display = False
                self.query_one("#header-rule").display = False
                self.query_one("#run-body").display = False
                self.query_one("#run-footer").display = False
            else:
                notice.display = False
                self.query_one("#run-header").display = True
                self.query_one("#header-rule").display = True
                self.query_one("#run-body").display = True
                self.query_one("#run-footer").display = True
        except Exception:  # noqa: SIM105
            pass

    def refresh_all(self) -> None:
        s = self._state
        try:  # noqa: SIM105
            now = time.monotonic()
            name = s.input_path.name if s.input_path else "—"
            prog = (
                f"{progress_bar(s.progress_frac(), 16)} "
                f"{s.completed_blocks}/{s.total_blocks} ({s.progress_pct()}%)"
            )
            mode_badge = f"[{ACCENT}]{s.dual_label()}[/]"
            dry_badge = f" [{WARNING} bold on grey23]演练模式[/]" if s.dry_run else ""
            if s.is_completed:
                stage_str = f"[bold green][完成] {stage_label(s.stage)}[/]{dry_badge}"
            else:
                stage_str = stage_label(s.stage) + dry_badge
            parts = [name, stage_str, mode_badge, prog, s.pace_text(now)]
            # Exception-only marker: failures surface here solely when nonzero.
            if s.failed_blocks:
                parts.append(f"[{ERROR}]失败 {s.failed_blocks}[/]")
            if s.job_id:
                parts.append(s.job_id)
            self.query_one("#run-header", Static).update(" · ".join(parts))
            self.query_one("#stage-stepper", StageStepper).update_stage(s.stage)
            self.query_one("#bilingual", BilingualCard).update_state(s)
            self.query_one("#telemetry", TelemetryPanel).update_state(s)
            fails = ""
            if s.failed_blocks:
                fails = f"失败块 {s.failed_blocks}（按 r 重试，再跑同 job 自动续）"
            self.query_one("#failure-list", Static).update(fails)
        except Exception:
            pass

    def push_log(self, ts: str, stage: str, msg: str) -> None:
        try:  # noqa: SIM105
            # One signal per line: event lines already start with their label.
            line = msg if (msg and stage and msg.startswith(stage)) else f"{stage} {msg}".strip()
            self.query_one("#event-log", EventLogView).push_event(ts, line)
        except Exception:
            pass

    def action_focus_input(self) -> None:
        self.query_one("#cmd-input", Input).focus()

    def action_command_palette(self) -> None:
        _ubt_app(self).action_command_palette()

    def action_back_to_wizard(self) -> None:
        _ubt_app(self).action_back_to_wizard()

    def action_retry_failed(self) -> None:
        _ubt_app(self).notify_retry()

    def action_post_edit(self) -> None:
        _ubt_app(self).action_post_edit()

    def action_open_artifact(self) -> None:
        _ubt_app(self).open_artifact()

    def action_show_report(self) -> None:
        _ubt_app(self).push_screen(ReportModal(_ubt_app(self)._report_body()))

    def action_toggle_dual_mode(self) -> None:
        """Cycle dual mode: bilingual -> facing -> monolingual -> bilingual."""
        cycle = {"bilingual": "facing", "facing": "monolingual", "monolingual": "bilingual"}
        nxt = cycle.get(self._state.dual_choice, "bilingual")
        _ubt_app(self).handle_command(f"/dual {nxt}")


class CommandPaletteModal(ModalScreen[None]):
    """Fuzzy command palette. Centered modal, keyboard-driven."""

    def compose(self) -> Any:
        yield Vertical(
            Static("[bold]命令面板[/]  [grey50](Esc 关闭 · ↑/↓ 移动 · Enter 执行)[/]"),
            Input(placeholder="输入命令或关键词模糊匹配…", id="palette-input"),
            ListView(id="palette-list"),
            id="palette-box",
        )

    def on_mount(self) -> None:
        self._filter_commands("")
        self.query_one("#palette-input", Input).focus()

    def _filter_commands(self, query: str) -> None:
        from rapidfuzz import fuzz

        lv = self.query_one("#palette-list", ListView)
        lv.clear()
        q = query.strip().lower()
        scored: list[tuple[float, tuple[str, str, str]]] = []
        for cmd, desc, cat in PALETTE_COMMANDS:
            if not q:
                scored.append((100.0, (cmd, desc, cat)))
            else:
                score_cmd = fuzz.partial_ratio(q, cmd.lower())
                score_desc = fuzz.partial_ratio(q, desc.lower())
                score = max(score_cmd * 1.2, score_desc)
                if score > 35 or q in cmd.lower() or q in desc.lower():
                    scored.append((score, (cmd, desc, cat)))
        scored.sort(key=lambda t: t[0], reverse=True)
        for _, (cmd, desc, cat) in scored:
            lv.append(
                ListItem(
                    Label(f"[bold cyan]{cmd:<12}[/] [grey50][{cat}][/] {desc}"),
                    name=cmd,
                )
            )
        if len(lv.children) > 0:
            lv.index = 0

    @on(Input.Changed, "#palette-input")
    def _on_input_changed(self, event: Input.Changed) -> None:
        self._filter_commands(event.value)

    @on(Input.Submitted, "#palette-input")
    def _on_input_submitted(self, event: Input.Submitted) -> None:
        lv = self.query_one("#palette-list", ListView)
        if lv.highlighted_child is not None and getattr(lv.highlighted_child, "name", None):
            cmd = str(lv.highlighted_child.name)
            self.dismiss(None)
            _ubt_app(self).handle_command(cmd)
        elif event.value.strip():
            cmd = event.value.strip()
            self.dismiss(None)
            _ubt_app(self).handle_command(cmd)

    @on(ListView.Selected, "#palette-list")
    def _on_list_selected(self, event: ListView.Selected) -> None:
        cmd = getattr(event.item, "name", "")
        if cmd:
            self.dismiss(None)
            _ubt_app(self).handle_command(str(cmd))

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(None)
        elif event.key == "down":
            lv = self.query_one("#palette-list", ListView)
            if lv.index is not None and lv.index < len(lv.children) - 1:
                lv.index += 1
            elif lv.index is None and len(lv.children) > 0:
                lv.index = 0
            event.stop()
        elif event.key == "up":
            lv = self.query_one("#palette-list", ListView)
            if lv.index is not None and lv.index > 0:
                lv.index -= 1
            event.stop()


class JobDetailModal(ModalScreen[None]):
    """Detailed ledger view for a selected job record."""

    def __init__(self, job_id: str, db_path: Path) -> None:
        super().__init__()
        self._job_id = job_id
        self._db_path = db_path
        self._source_path: str | None = None

    def compose(self) -> Any:
        stats_text = "正在查询账本数据…"
        if self._db_path.exists():
            try:
                from ubt.core.engine.ledger import SQLiteJobLedger

                ledger = SQLiteJobLedger(self._db_path)
                snap = ledger.get_job_snapshot(self._job_id)
                if snap:
                    src = (
                        Path(snap.get("source_path", "")).name
                        if snap.get("source_path")
                        else "未知"
                    )
                    self._source_path = snap.get("source_path")
                    status_text = {
                        "completed": "[green]已完成[/]",
                        "failed": "[red]失败[/]",
                        "in_progress": "[yellow]进行中[/]",
                        "initialized": "[cyan]已初始化[/]",
                    }.get(snap.get("status", ""), snap.get("status", "未知"))
                    lines = [
                        f"任务编号: {self._job_id}",
                        f"原文档名: {src}",
                        f"任务状态: {status_text}",
                        f"目标语言: {snap.get('target_lang', 'zh')}",
                        f"总块数:   {snap.get('total', 0)} 块",
                        f"已完成:   {snap.get('completed', 0)} 块 ({round((snap.get('completed', 0) / max(snap.get('total', 1), 1)) * 100, 1)}%)",
                        f"修复次数: {snap.get('repaired', 0)} 块",
                        f"失败块数: {snap.get('failed', 0)} 块",
                        f"平均 QE:  {float(snap.get('avg_qe_score', 0.0)):.3f}",
                        f"末位15%:  {float(snap.get('bottom_15_avg_qe', 0.0)):.3f}",
                        f"开始时间: {snap.get('created_at', '—')}",
                    ]
                    stats_text = "\n".join(lines)
                else:
                    stats_text = f"账本未记录作业 {self._job_id} 的元数据"
            except Exception as exc:
                stats_text = f"读取账本失败: {exc}"
        else:
            stats_text = f"账本文件不存在: {self._db_path}"

        with Vertical(id="job-detail-box"):
            yield Static(f"[bold cyan]历史作业详情 · {self._job_id}[/]\n")
            with VerticalScroll(id="job-detail-scroll"):
                yield Static(stats_text, id="job-detail-text")
            with Horizontal(id="job-detail-actions"):
                yield Button("续跑此任务", id="btn-job-resume")
                yield Button("查看报告", id="btn-job-report")
                yield Button("关闭", id="btn-job-close")

    def on_mount(self) -> None:
        self.query_one("#btn-job-close", Button).focus()

    @on(Button.Pressed, "#btn-job-close")
    def _close(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#btn-job-resume")
    def _resume(self, event: Button.Pressed) -> None:
        self.dismiss(None)
        if self._source_path:
            p = Path(self._source_path)
            if p.exists():
                _ubt_app(self)._state.input_path = p.resolve()
        _ubt_app(self).handle_command(f"/resume {self._job_id}")

    @on(Button.Pressed, "#btn-job-report")
    def _report(self, event: Button.Pressed) -> None:
        self.dismiss(None)
        _ubt_app(self).handle_command("/report")

    def on_key(self, event: events.Key) -> None:
        if event.key in ("escape", "q"):
            event.stop()
            self.dismiss(None)


class HelpModal(ModalScreen[None]):
    """Keyboard + slash help. Single overlay border; content unboxed."""

    def compose(self) -> Any:

        yield Vertical(
            Static("[bold]帮助[/]  （? / Esc 关闭）"),
            Static(help_text()),
            Static(KEYS_HELP),
            Button("关闭", id="help-close"),
            id="help-box",
        )

    @on(Button.Pressed, "#help-close")
    def _close(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def on_key(self, event: events.Key) -> None:
        if event.key in ("escape", "question_mark", "q"):
            event.stop()
            self.dismiss(None)


class ConfirmModal(ModalScreen[bool]):
    """Dangerous-action confirm. Friction matches consequence; defaults to No."""

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> Any:
        yield Vertical(
            Static(f"[bold]{self._title}[/]\n{self._body}"),
            Horizontal(
                Button("确认", id="cf-yes"),
                Button("取消", id="cf-no"),
                id="cf-row",
            ),
            id="cf-box",
        )

    def on_mount(self) -> None:
        self.query_one("#cf-no", Button).focus()

    @on(Button.Pressed, "#cf-yes")
    def _yes(self, event: Button.Pressed) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#cf-no")
    def _no(self, event: Button.Pressed) -> None:
        self.dismiss(False)


class DoctorModal(ModalScreen[None]):
    """Environment diagnostics (read-only)."""

    def __init__(self, report_text: str) -> None:
        super().__init__()
        self._text = report_text

    def compose(self) -> Any:
        with Vertical(id="doc-box"):
            yield Static("[bold cyan]环境诊断报告[/]\n")
            with VerticalScroll(id="doc-scroll"):
                yield Static(self._text or "暂无诊断数据", id="doc-text")
            yield Button("关闭", id="doc-close")

    @on(Button.Pressed, "#doc-close")
    def _close(self, event: Button.Pressed) -> None:
        self.dismiss(None)


class ReportModal(ModalScreen[None]):
    """Quality report viewer."""

    def __init__(self, body: str = "") -> None:
        super().__init__()
        self._body = body

    def compose(self) -> Any:
        with Vertical(id="rep-box"):
            yield Static("[bold cyan]全书翻译与出版质量报告[/]\n")
            with VerticalScroll(id="rep-scroll"):
                yield Static(self._body or "暂无报告", id="rep-text")
            yield Button("关闭", id="rep-close", variant="default")

    @on(Button.Pressed, "#rep-close")
    def _close(self, event: Button.Pressed) -> None:
        self.dismiss(None)


class PEModal(ModalScreen[bool]):
    """Interactive in-terminal Post-Editing (PE) review and correction modal."""

    BINDINGS = [
        ("ctrl+s", "save", "保存并写回"),
        ("escape", "cancel", "取消"),
    ]

    def __init__(
        self,
        job_id: str,
        block_id: str,
        flow_id: str,
        source_text: str,
        target_text: str,
        status: str,
        defect_note: str = "",
        db_path: Path | None = None,
        source_lang: str = "en",
        target_lang: str = "zh",
    ) -> None:
        super().__init__()
        self._job_id = job_id
        self._block_id = block_id
        self._flow_id = flow_id
        self._source_text = source_text
        self._target_text = target_text
        self._status = status
        self._defect_note = defect_note
        self._db_path = db_path
        self._source_lang = source_lang
        self._target_lang = target_lang

    def compose(self) -> Any:
        with Vertical(id="pe-box"):
            yield Static(f"[bold cyan]人工后编辑 (Post-Editing) · {escape(self._block_id)}[/]")
            meta_str = f"所属章节: {escape(self._flow_id)} · 原始状态: [bold magenta]{escape(self._status)}[/]"
            if self._defect_note:
                meta_str += f" · [yellow]{escape(self._defect_note)}[/]"
            yield Static(f"[{MUTED_RICH}]{meta_str}[/]\n")

            with VerticalScroll(id="pe-scroll"):
                yield Panel(
                    escape(self._source_text),
                    title="[bold]原文 (Source)[/]",
                    title_align="left",
                    border_style=MUTED_RICH,
                    expand=True,
                )
                yield Static(
                    Text.from_markup("\n[bold]译文修订 (可在下方直接编辑并按 Ctrl+S 写回):[/]")
                )
                yield TextArea(self._target_text, id="pe-target-input")

            with Horizontal(id="pe-actions"):
                yield Button("保存并写回 (Ctrl+S)", id="btn-pe-save", variant="default")
                yield Button("取消 (Esc)", id="btn-pe-cancel", variant="default")

    def on_mount(self) -> None:
        self.query_one("#pe-target-input", TextArea).focus()

    @on(Button.Pressed, "#btn-pe-cancel")
    def _on_cancel_btn(self, event: Button.Pressed) -> None:
        self.action_cancel()

    @on(Button.Pressed, "#btn-pe-save")
    def _on_save_btn(self, event: Button.Pressed) -> None:
        self.action_save()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_save(self) -> None:
        new_target = self.query_one("#pe-target-input", TextArea).text.strip()
        if not new_target:
            self.notify("译文内容不能为空", severity="error")
            return

        if self._db_path and self._db_path.exists():
            try:
                from ubt.core.engine.ledger import SQLiteJobLedger
                from ubt.core.ir.models import BlockStatus

                with SQLiteJobLedger(self._db_path) as ledger:
                    ledger.save_checkpoints_batch(
                        [
                            {
                                "block_id": self._block_id,
                                "status": BlockStatus.REPAIRED,
                                "target_text": new_target,
                                "error_flags": ["human_pe_imported"],
                            }
                        ],
                        clear_verdict_for=[self._block_id],
                    )

                try:
                    from ubt.core.memory.tm import (
                        PROVENANCE_HUMAN_PE,
                        TMPendingEntry,
                        TranslationMemory,
                    )

                    tm_path = self._db_path.parent / "tm.sqlite"
                    tm = TranslationMemory(tm_path)
                    tm.writeback(
                        [
                            TMPendingEntry(
                                src_lang=self._source_lang,
                                tgt_lang=self._target_lang,
                                source_text=self._source_text,
                                target_text=new_target,
                                provenance=PROVENANCE_HUMAN_PE,
                            )
                        ]
                    )
                except Exception as tm_exc:
                    logger.debug("TM writeback skipped/failed: %s", tm_exc)

            except Exception as exc:
                self.notify(f"账本写回失败: {exc}", severity="error")
                return

        self.dismiss(True)

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.action_cancel()

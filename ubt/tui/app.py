"""Fullscreen Textual application for UBT TUI v2."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App
from textual.widgets import Static

from ubt.core.config import MOCK_API_KEY, UBTConfig
from ubt.core.job_options import apply_config_overrides
from ubt.tui.commands import parse_command
from ubt.tui.events import describe_event, stage_label
from ubt.tui.logsetup import route_logs_to_file
from ubt.tui.screens import (
    ConfirmModal,
    DoctorModal,
    HelpModal,
    PEModal,
    ReportModal,
    RunScreen,
    WizardScreen,
)
from ubt.tui.state import SessionState

logger = logging.getLogger(__name__)

# UI refresh policy: pipeline events arrive in bursts (concurrency N blocks
# completing at once) while a full dashboard rebuild is expensive. Repaint at
# most ~7Hz; stage transitions that change what the user watches always win.
UI_REFRESH_INTERVAL = 0.15
FORCE_REFRESH_CODES = frozenset(
    {
        "JOB_STARTED",
        "EXPORT_COMPLETED",
        "PIPELINE_FAILED",
        "MODE_ADVISED",
    }
)


def should_refresh(code: str, now: float, last: float) -> bool:
    """Pure refresh decision for one pipeline event."""
    return code in FORCE_REFRESH_CODES or (now - last) >= UI_REFRESH_INTERVAL


class UBTApp(App[None]):
    """Code-agent grade fullscreen TUI: wizard -> run monitor."""

    CSS_PATH = ["styles/app.tcss"]

    BINDINGS = [
        ("question_mark", "show_help", "帮助"),
        ("ctrl+q", "quit_app", "退出"),
        ("ctrl+c", "quit_app", "退出"),
    ]

    def __init__(
        self,
        initial_file: Path | None = None,
        dry_run_override: bool = False,
        db_dir: Path | None = None,
        log_path: Path | None = None,
        request: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._state = SessionState(dry_run=dry_run_override)
        if request:
            self._state.apply_cli_request(request)
        self._log_path = log_path
        self._initial_file = initial_file
        self._db_dir = db_dir
        self._config = UBTConfig.from_env()
        self._pipeline_running = False
        self._cancel_requested = False
        self._artifact: Path | None = None
        self._engine_notes: list[str] = []
        self._last_ui_refresh = 0.0
        self._last_snippet_query = 0.0
        self._trailing_timer: Any = None

    @property
    def session(self) -> SessionState:
        return self._state

    def on_mount(self) -> None:
        self.push_screen(WizardScreen(self._state, self._initial_file))

    def action_show_help(self) -> None:
        from textual.screen import ModalScreen

        if any(isinstance(s, ModalScreen) for s in self.screen_stack):
            return
        self.push_screen(HelpModal())

    def action_quit_app(self) -> None:
        if self._pipeline_running:

            def _after(result: bool | None) -> None:
                if result:
                    self._cancel_requested = True
                    self.exit()

            self.push_screen(
                ConfirmModal(
                    "退出确认", "管线运行中，退出将中止本次运行（账本保留，可续跑）。确认退出？"
                ),
                _after,
            )
        else:
            self.exit()

    def start_run(self) -> None:
        s = self._state
        if s.input_path is None:
            self.notify("先选一个文件再开始", severity="warning")
            return
        if s.job_id is not None:
            try:
                s.validate_job_id(s.job_id)
                # Verify that job_id actually corresponds to current input_path
                from ubt.core.ir.serializer import compute_file_sha256

                if s.input_path.is_file():
                    import re

                    file_hash = compute_file_sha256(s.input_path)[:12]
                    db = self._ledger_dir() / f"{s.job_id}.sqlite"
                    conflict = False
                    if db.exists():
                        try:
                            from ubt.core.engine.ledger import SQLiteJobLedger

                            with SQLiteJobLedger(db, read_only=True) as ldg:
                                meta = ldg.get_job_snapshot(s.job_id)
                                if meta and meta.get("doc_id"):
                                    recorded_doc = str(meta["doc_id"])[:12]
                                    if recorded_doc != file_hash:
                                        conflict = True
                        except Exception:
                            pass
                    elif re.match(r"^job_[0-9a-f]{12}", s.job_id) and not s.job_id.startswith(
                        f"job_{file_hash}"
                    ):
                        conflict = True

                    if conflict:
                        self._emit_log(
                            "STATE", f"检测到不同文档，重置历史任务 ID：{s.job_id} -> 自动派生"
                        )
                        s.job_id = None
            except ValueError as exc:
                self.notify(str(exc), severity="error")
                return
            except Exception:
                s.job_id = None
        self._show_run_screen()
        s.run_completed_at = None
        s.run_started_at = None
        # Clear the previous run's artifact: the consume-finally marks the run
        # completed whenever ``_artifact`` is set, so a failed rerun would
        # otherwise report "completed" with the earlier run's file.
        self._artifact = None
        self._pipeline_running = True
        self._cancel_requested = False
        self._last_ui_refresh = 0.0
        self._run_pipeline()

    def _show_run_screen(self) -> None:
        """Exactly one RunScreen exists: push from the wizard, reuse afterwards.

        Reusing keeps the visible event log across resume/retry; pushing a new
        screen every run leaked screens and left stale ones receiving updates.
        """
        if not isinstance(self.screen, RunScreen):
            self.push_screen(RunScreen(self._state))

    @work(thread=True, exclusive=True, exit_on_error=False)
    def _run_pipeline(self) -> None:
        """Run the pipeline on a worker thread with its own event loop.

        docling ingest, model loads and Typst each block for seconds: on the
        UI loop that reads as a frozen app while the job keeps progressing.
        Everything UI-bound crosses back via call_from_thread only.
        """
        asyncio.run(self._consume_pipeline())

    async def _consume_pipeline(self) -> None:
        post = self.call_from_thread
        s = self._state
        assert s.input_path is not None
        db_dir = self._db_dir or self._config.db_dir
        overrides = s.to_overrides(db_dir=db_dir)
        try:
            config = apply_config_overrides(UBTConfig.from_env(), overrides)
        except Exception as exc:
            post(self._emit_log, "CONFIG", f"参数校验失败：{exc}")
            self._pipeline_running = False
            return
        if not s.dry_run and config.api_key.get_secret_value() == MOCK_API_KEY:
            post(self._emit_log, "AUTH", "未检测到有效 API 密钥，已自动切演练模式（零消耗）")
            s.dry_run = True
        try:  # noqa: SIM105
            from ubt.tui.advisor import DocumentAdvisor as _DA

            rep = _DA.analyze(s.input_path)
            for w in rep.check_conflict(
                str(overrides.get("render_engine", "")), str(overrides.get("dual_mode", ""))
            ):
                self._engine_notes.append(w)
                post(self._emit_log, "LAYOUT", w)
        except Exception:
            pass
        post(self._refresh_notes)
        try:
            if s.dry_run:
                from ubt.core.engine.dry_run import create_dry_run_orchestrator

                orchestrator = create_dry_run_orchestrator(config)
            else:
                from ubt.core.engine.pipeline import PipelineOrchestrator

                orchestrator = PipelineOrchestrator(config=config)
        except Exception as exc:
            post(self._emit_log, "INIT", f"引擎初始化失败：{exc}")
            self._pipeline_running = False
            return
        rk = s.to_run_kwargs()
        post(
            self._emit_log,
            "JOB",
            f"开始：{s.input_path.name} preset={s.preset.value} dry_run={s.dry_run}",
        )
        if s.preset.value == "publication":
            post(
                self._emit_log,
                "INGEST",
                "出版级预设：正在运行 Docling VLM 视觉解析版面与公式 (GPU)...",
            )
        else:
            post(
                self._emit_log,
                "INGEST",
                f"正在解析源文档结构：{s.input_path.name}",
            )
        if s.job_id and s.job_id_dry_run != s.dry_run:
            # The captured id came from the other provider mode. A mock run's
            # ledger holds simulated drafts; resuming it from a real run would
            # export them without a single API call.
            post(
                self._emit_log,
                "STATE",
                f"翻译模式已切换，不再复用账本 {s.job_id}（改用按模式派生的新 ID）",
            )
            s.job_id = None
        gen = orchestrator.run(
            input_path=s.input_path,
            output_path=s.output_path,
            target_lang=s.target_lang,
            profile_name=str(rk.get("profile_name", "general")),
            source_lang=s.source_lang,
            job_id=s.job_id,
        )
        try:
            async for event in gen:
                if self._cancel_requested:
                    post(self._emit_log, "CANCEL", "已请求中止，保留账本，可 /resume 续跑")
                    break
                self._on_pipeline_event(event)
        except Exception as exc:
            post(self._emit_log, "FAILED", f"管线失败：{exc}（同 job_id 重跑自动续）")
        finally:
            # The declared type is AsyncIterator; the runtime object is an
            # async generator, so close it when the protocol allows.
            aclose = getattr(gen, "aclose", None)
            if aclose is not None:
                await aclose()
            self._pipeline_running = False
            # Completion is the export event's verdict, not this block's.
            # ``apply_event`` already marked the session done when
            # EXPORT_COMPLETED arrived with an artifact; marking done here too
            # made a raised pipeline or a /cancel break render as
            # "全书翻译与导出完成" over a path that was never written.
            if self._artifact is not None:
                s.mark_completed(time.monotonic(), artifact_path=self._artifact)
            post(self._refresh_run)
            post(
                self._emit_log,
                "DONE",
                f"结束：产物 {self._artifact or '未产出'} · /report 看质量 · /resume 续跑",
            )

    def _on_pipeline_event(self, event: Any) -> None:
        """Fold one pipeline event (worker thread): cheap state, throttled paint."""
        s = self._state
        jid = str(getattr(event, "job_id", "") or "")
        if jid and not s.job_id:
            s.job_id = jid
            s.job_id_dry_run = s.dry_run
        s.apply_event(event)
        et = getattr(event, "event_type", None)
        code = str(getattr(et, "value", et) or "").upper()
        if code == "EXPORT_COMPLETED":
            ap = getattr(event, "artifact_path", None) or getattr(event, "message", "")
            try:
                self._artifact = Path(str(ap)) if ap else None
                s.output_path = self._artifact
            except Exception:
                self._artifact = None
            s.mark_completed(time.monotonic(), artifact_path=self._artifact)
        # Query real bilingual snippet from SQLite WAL if available
        bid = getattr(event, "active_block_id", None)
        now = time.monotonic()
        if s.job_id and (now - self._last_snippet_query >= 0.2 or code in FORCE_REFRESH_CODES):
            self._last_snippet_query = now
            try:
                db = self._ledger_dir() / f"{s.job_id}.sqlite"
                if db.exists():
                    self._query_bilingual_snippet(db, s.job_id, str(bid) if bid else None)
            except Exception:
                pass
        refresh = should_refresh(code, now, self._last_ui_refresh)
        if refresh:
            self._last_ui_refresh = now
            if self._trailing_timer is not None:
                with contextlib.suppress(Exception):
                    self._trailing_timer.cancel()
                self._trailing_timer = None
        else:
            if self._trailing_timer is None:
                with contextlib.suppress(Exception):
                    self._trailing_timer = self.set_timer(
                        UI_REFRESH_INTERVAL, self._on_trailing_refresh
                    )
        try:
            self.call_from_thread(self._render_event, event, refresh)
        except RuntimeError:
            # Same-thread caller (tests, sanity scripts): paint directly.
            self._render_event(event, refresh)

    def _on_trailing_refresh(self) -> None:
        """Trailing edge timer callback to ensure final burst events repaint."""
        self._trailing_timer = None
        self._last_ui_refresh = time.monotonic()
        self._refresh_run()

    def _query_bilingual_snippet(self, db_path: Path, job_id: str, block_id: str | None) -> None:
        """Fetch true source and target text from SQLite WAL without locking."""
        import sqlite3

        try:
            # as_uri percent-encodes (#, ?, spaces, non-ASCII): an f-string
            # would let those characters truncate or mangle the URI.
            with sqlite3.connect(
                f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=0.5
            ) as conn:
                conn.row_factory = sqlite3.Row
                row = None
                if block_id:
                    row = conn.execute(
                        "SELECT source_text, target_text, block_id FROM blocks WHERE block_id = ? AND target_text IS NOT NULL AND length(target_text) > 0",
                        (block_id,),
                    ).fetchone()
                if not row:
                    row = conn.execute(
                        "SELECT source_text, target_text, block_id FROM blocks WHERE job_id = ? AND target_text IS NOT NULL AND length(target_text) > 0 ORDER BY rowid DESC LIMIT 1",
                        (job_id,),
                    ).fetchone()
                if row:
                    src = str(row["source_text"] or "")
                    tgt = str(row["target_text"] or "")
                    b_id = str(row["block_id"] or block_id or "")
                    self._state.update_bilingual_snippet(source=src, target=tgt, block_id=b_id)
        except Exception as exc:
            # The preview is opportunistic, but silence made a broken
            # ledger path indistinguishable from "no translated row yet".
            logging.getLogger("ubt.tui").debug(
                "bilingual snippet unavailable for %s: %s", db_path, exc
            )

    def _render_event(self, event: Any, refresh: bool) -> None:
        """Paint one folded event (UI thread only)."""
        et = getattr(event, "event_type", None)
        code = str(getattr(et, "value", et) or "")
        self._emit_log(stage_label(code), describe_event(event), refresh=refresh)

    def _emit_log(self, stage: str, msg: str, ts: str | None = None, refresh: bool = False) -> None:
        ts = ts or datetime.now().strftime("%H:%M:%S")
        logging.getLogger("ubt.tui.screen").info("[%s] %s", stage, msg)
        self._state.event_tail.append((ts, stage, msg))
        if len(self._state.event_tail) > 300:
            self._state.event_tail.pop(0)
        try:  # noqa: SIM105
            for screen in reversed(self.screen_stack):
                if isinstance(screen, RunScreen):
                    screen.push_log(ts, stage, msg)
                    if refresh:
                        screen.refresh_all()
                    break
        except Exception:
            pass

    def _refresh_notes(self) -> None:
        try:  # noqa: SIM105
            for screen in reversed(self.screen_stack):
                if isinstance(screen, RunScreen):
                    try:  # noqa: SIM105
                        screen.query_one("#engine-notes", Static).update(
                            " · ".join(self._engine_notes) or ""
                        )
                    except Exception:
                        pass
                    break
        except Exception:
            pass

    def handle_command(self, text: str) -> None:
        cmd = parse_command(text)
        if not cmd.ok:
            self._emit_log("CMD", cmd.error or "命令错误")
            self.notify(cmd.error or "命令错误", severity="error")
            return
        a = cmd.action
        if a == "noop":
            return
        if a == "help":
            self.push_screen(HelpModal())
        elif a == "quit":
            self.action_quit_app()
        elif a == "cancel":
            if self._pipeline_running:
                self._cancel_requested = True
                self._emit_log("CMD", "已请求中止（保留账本）")
            else:
                self.notify("当前没有运行中的任务", severity="warning")
        elif a == "fresh":

            def _after(result: bool | None) -> None:
                if result:
                    self._state.choose_fresh(True)
                    self._emit_log("CMD", "已标记 fresh：下次运行从零重 ingest")

            self.push_screen(
                ConfirmModal("重来确认", "fresh 将丢弃该 job 已有账本从零开始。确认？"), _after
            )
        elif a == "preset":
            from ubt.tui.presets import Preset as _Preset

            if self._pipeline_running:
                self.notify("运行中不可切档（先 /cancel）", severity="warning")
                return
            try:
                self._state.choose_preset(_Preset(str(cmd.args["preset"])))
                self._emit_log("CMD", f"档位 → {self._state.preset.value}")
                self._refresh_run()
            except Exception as exc:
                self.notify(str(exc), severity="error")
        elif a == "pages":
            self._state.pages = cmd.args.get("pages")
            self._emit_log("CMD", f"页码 → {self._state.pages or '全部'}")
        elif a == "dual":
            v = str(cmd.args["dual"])
            if v == "monolingual":
                self._state.choose_dual("monolingual")
            elif v == "facing":
                self._state.choose_dual("facing")
            else:
                self._state.choose_dual("bilingual")
            self._emit_log("CMD", f"输出形态 → {self._state.dual_label()} ({v})")
            self._refresh_run()
        elif a == "glossary":
            mode = str(cmd.args.get("mode", "auto"))
            if mode == "auto":
                self._state.glossary = self._state.recommended_glossary
                self._state.glossary_auto = True
            elif mode == "none":
                self._state.glossary = None
                self._state.glossary_auto = True
            else:
                p = Path(str(cmd.args.get("path", ""))).expanduser()
                if not p.exists():
                    self.notify(f"术语表不存在：{p}", severity="error")
                    return
                self._state.glossary = p.resolve()
                self._state.glossary_auto = False
            self._emit_log("CMD", f"术语表 → {mode}")
        elif a == "model":
            m = cmd.args.get("model")
            self._state.draft_model = m
            self._state.repair_model = m
            self._emit_log("CMD", f"模型 → {m or '引擎默认'}")
        elif a == "select_file":
            p = Path(str(cmd.args["path"])).expanduser()
            if not p.exists():
                self.notify(f"文件不存在：{p}", severity="error")
                return
            if self._pipeline_running:
                self.notify("运行中不可切文件（先 /cancel）", severity="warning")
                return
            self._state.input_path = p.resolve()
            self._emit_log("CMD", f"已选中 {p.name}")
            self._refresh_run()
        elif a == "fuzzy":
            self._emit_log("CMD", f"模糊搜：{cmd.args.get('query', '')}（回向导页用列表更快）")
        elif a == "translate":
            path = cmd.args.get("path")
            if path:
                p = Path(str(path)).expanduser()
                if not p.exists():
                    self.notify(f"文件不存在：{p}", severity="error")
                    return
                if self._pipeline_running:
                    self.notify("运行中（先 /cancel）", severity="warning")
                    return
                self._state.input_path = p.resolve()
            if cmd.args.get("preset"):
                from ubt.tui.presets import Preset as _Preset

                self._state.choose_preset(_Preset(str(cmd.args["preset"])))
            if cmd.args.get("pages") is not None:
                self._state.pages = cmd.args.get("pages")
            if cmd.args.get("dry_run"):
                self._state.dry_run = True
            if self._pipeline_running:
                self.notify("已在运行中", severity="warning")
                return
            self.start_run()
        elif a == "resume":
            try:
                jid = self._state.validate_job_id(str(cmd.args["job_id"]))
            except ValueError as exc:
                self.notify(str(exc), severity="error")
                return
            self._state.job_id = jid
            self._state.choose_fresh(False)
            # Auto-restore input_path from ledger if available
            db = self._ledger_dir() / f"{jid}.sqlite"
            if db.exists():
                try:
                    from ubt.core.engine.ledger import SQLiteJobLedger

                    with SQLiteJobLedger(db) as led:
                        snap = led.get_job_snapshot(jid)
                        if snap and snap.get("source_path"):
                            sp = Path(snap["source_path"])
                            if sp.exists():
                                self._state.input_path = sp
                except Exception:
                    pass
            self._emit_log("CMD", f"续跑 {self._state.job_id}")
            if not self._pipeline_running and self._state.input_path is not None:
                self.start_run()
        elif a == "jobs":
            self._show_jobs()
        elif a == "status":
            self._show_status(cmd.args.get("job_id"))
        elif a == "report":
            self.push_screen(ReportModal(self._report_body()))
        elif a == "pe":
            self.action_post_edit(cmd.args.get("block_id"))
        elif a == "doctor":
            self.push_screen(DoctorModal(self._doctor_body()))
        elif a == "open":
            self.open_artifact(cmd.args.get("target"))
        elif a == "new":
            self.action_back_to_wizard()
        else:
            self.notify(f"未知动作 {a}，/help 查看", severity="error")

    def action_post_edit(self, target_block_id: str | None = None) -> None:
        """Open in-terminal Post-Editing review modal for a specific block or first needs_human block."""
        jid = self._state.job_id
        if not jid:
            self.notify("当前没有正在运行或选中的翻译任务", severity="warning")
            return

        db_path = self._ledger_dir() / f"{jid}.sqlite"
        if not db_path.exists():
            self.notify(f"任务账本不存在: {jid}", severity="warning")
            return

        from ubt.core.engine.ledger import SQLiteJobLedger
        from ubt.core.ir.models import BlockStatus

        target_block = None
        try:
            with SQLiteJobLedger(db_path) as ledger:
                if target_block_id:
                    target_block = ledger.get_block(target_block_id)
                if target_block is None:
                    nh_blocks = ledger.fetch_blocks_by_status(jid, BlockStatus.NEEDS_HUMAN)
                    if not nh_blocks:
                        nh_blocks = ledger.fetch_blocks_by_status(jid, BlockStatus.BLOCKED_HUMAN)
                    if not nh_blocks:
                        nh_blocks = ledger.fetch_blocks_by_status(jid, BlockStatus.FAILED)
                    if nh_blocks:
                        target_block = nh_blocks[0]
                    elif self._state.active_block_id:
                        target_block = ledger.get_block(self._state.active_block_id)
        except Exception as exc:
            self.notify(f"读取账本失败: {exc}", severity="error")
            return

        if target_block is None:
            self.notify("当前没有待人工审校或可编辑的文本块", severity="information")
            return

        # Only a block awaiting human review increments the repaired counter on edit,
        # ensuring consistency with ledger block statuses.
        was_human_review = target_block.status in (
            BlockStatus.NEEDS_HUMAN,
            BlockStatus.BLOCKED_HUMAN,
        )
        counted = False

        def _on_pe_dismissed(saved: bool | None) -> None:
            nonlocal counted
            if saved:
                self.notify(
                    f"块 {target_block.id} 审校已保存并写回账本与 TM", severity="information"
                )
                self._emit_log("PE", f"人工修订块 {target_block.id}")
                if was_human_review and not counted:
                    counted = True
                    if self._state.needs_human_blocks > 0:
                        self._state.needs_human_blocks -= 1
                    self._state.repaired_blocks += 1
                if self._state.active_block_id == target_block.id:
                    self._state.active_target = target_block.target_text or ""
                if isinstance(self.screen, RunScreen):
                    self.screen.refresh_all()
                    self.screen.refresh_sidebar()

        self.push_screen(
            PEModal(
                job_id=jid,
                block_id=target_block.id,
                flow_id=str(target_block.flow_id or "ch01"),
                source_text=target_block.source_text or "",
                target_text=target_block.target_text or "",
                status=str(target_block.status),
                defect_note="；".join(target_block.error_flags) if target_block.error_flags else "",
                db_path=db_path,
                source_lang=self._state.source_lang,
                target_lang=self._state.target_lang,
            ),
            _on_pe_dismissed,
        )

    def action_command_palette(self) -> None:
        """Open the fuzzy command palette modal."""
        from textual.screen import ModalScreen

        from ubt.tui.screens import CommandPaletteModal

        if any(isinstance(s, ModalScreen) for s in self.screen_stack):
            return
        self.push_screen(CommandPaletteModal())

    def action_back_to_wizard(self) -> None:
        """Safely return to wizard screen to select or probe another document."""
        if self._pipeline_running:

            def _after(result: bool | None) -> None:
                if result:
                    self._cancel_requested = True
                    self._go_to_wizard()

            self.push_screen(
                ConfirmModal(
                    "返回向导确认", "管线运行中，返回向导将中止本次运行（账本保留）。确认？"
                ),
                _after,
            )
        else:
            self._go_to_wizard()

    def _go_to_wizard(self) -> None:
        from ubt.tui.screens import RunScreen, WizardScreen

        self._state.job_id = None
        self._state.reset_run()
        self._artifact = None
        while len(self.screen_stack) > 1 and not isinstance(
            self.screen_stack[-1], (WizardScreen, RunScreen)
        ):
            self.pop_screen()
        if len(self.screen_stack) > 1 and isinstance(self.screen_stack[0], WizardScreen):
            while len(self.screen_stack) > 1:
                self.pop_screen()
        else:
            self.switch_screen(WizardScreen(self._state))

    def _refresh_run(self) -> None:
        try:  # noqa: SIM105
            for screen in reversed(self.screen_stack):
                if isinstance(screen, RunScreen):
                    screen.refresh_all()
                    break
        except Exception:
            pass

    def notify_retry(self) -> None:
        s = self._state
        if not s.failed_blocks:
            self.notify("没有失败块可重试", severity="warning")
            return
        if self._pipeline_running:
            self.notify("运行中，失败块会在 repair 阶段自动重试", severity="warning")
            return
        s.run_started_at = None
        s.run_completed_at = None
        self._emit_log("CMD", "重试失败块：同 job_id 续跑（已完成块不重复计费）")
        self._pipeline_running = True
        self._cancel_requested = False
        self._last_ui_refresh = 0.0
        self._run_pipeline()

    def open_artifact(self, target_arg: str | None = None) -> None:
        """Open the last artifact, or an explicit path from ``/open <path>``.

        ``/open`` accepts an optional path argument; an explicit existing path
        is opened directly, otherwise the run's latest artifact is opened.
        """
        if target_arg is not None and target_arg != "artifact":
            explicit = Path(target_arg).expanduser()
            if not explicit.exists():
                self.notify(f"路径不存在：{explicit}", severity="warning")
                return
            self._launch_artifact(explicit)
            return
        target = self._artifact
        if target is None or not target.exists():
            self.notify("产物尚未生成（跑完后自动可开）", severity="warning")
            return
        self._launch_artifact(target)

    def _launch_artifact(self, target: Path) -> None:
        try:
            launch_detached(str(target.resolve()))
            self._emit_log("OPEN", f"已调用系统查看器：{target.resolve()}")
        except Exception as exc:
            self.notify(f"打开失败：{exc}", severity="error")

    def _ledger_dir(self) -> Path:
        try:
            return self._db_dir or self._config.db_dir
        except Exception:
            return Path(".ubt/ledgers")

    def _show_jobs(self) -> None:
        d = self._ledger_dir()
        try:
            files = sorted(d.glob("*.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
        except Exception:
            files = []
        if not files:
            self._emit_log("JOBS", f"暂无历史（{d} 为空）")
            return
        lines = [
            f"{p.stem} · {datetime.fromtimestamp(p.stat().st_mtime).strftime('%m-%d %H:%M')}"
            for p in files
        ]
        self._emit_log("JOBS", "历史jobs: " + " | ".join(lines[:8]))

    def _show_status(self, job_id: str | None) -> None:
        raw = job_id or self._state.job_id
        if not raw:
            self._emit_log("STATUS", "用法：/status <job_id>（或先跑一个任务）")
            return
        try:
            # Same rule as /resume, CLI and API: without it a traversal like
            # ``../../tmp/other`` resolved the ledger path outside the ledger
            # dir and ran the schema migrations on an arbitrary SQLite file.
            jid = self._state.validate_job_id(raw)
        except ValueError as exc:
            self._emit_log("STATUS", str(exc))
            return
        db = self._ledger_dir() / f"{jid}.sqlite"
        if not db.exists():
            self._emit_log("STATUS", f"{jid}：账本不存在（{db}）")
            return
        try:
            from ubt.core.engine.ledger import SQLiteJobLedger

            ledger = SQLiteJobLedger(db)
            stats = ledger.get_job_stats(jid)
            self._emit_log("STATUS", f"{jid}：{stats}")
        except Exception as exc:
            self._emit_log("STATUS", f"{jid} 查询失败：{exc}")

    def _report_body(self) -> str:
        # Check if output artifact has quality_report.json sidecar
        if self._artifact:
            from ubt.core.job_options import sidecar_path

            qr_path = sidecar_path(Path(self._artifact), "quality_report.json")
            if qr_path.exists():
                try:
                    from ubt.core.engine.reporter import QualityReport

                    qr = QualityReport.model_validate_json(qr_path.read_text(encoding="utf-8"))
                    return self._format_full_quality_report(qr)
                except Exception as exc:
                    logger.debug("Failed to parse quality_report.json: %s", exc)

        s = self._state
        lines = [
            f"文档：{s.input_path or '—'}",
            f"档位：{s.preset.value} · 阶段：{stage_label(s.stage)}",
            f"完成：{s.completed_blocks}/{s.total_blocks}（{s.progress_pct()}%）",
            f"MTQE：{s.avg_qe:.3f} · 末位15%：{s.bottom15_qe:.3f}",
            f"修复：{s.repaired_blocks} · 失败：{s.failed_blocks}",
            f"花费：{f'{s.cost_label()} USD' if s.cost_usd is not None else '未知'} · 缓存：{round(s.cache_hit * 100, 1)}%",
            f"产物：{self._artifact or '尚未生成'}",
        ]
        if s.needs_human_blocks:
            lines.append(f"待审：{s.needs_human_blocks} 块待人工后编辑（按 p 审校）")
        if self._engine_notes:
            lines.append("引擎决策：" + "；".join(self._engine_notes))
        return "\n".join(lines)

    def _format_full_quality_report(self, qr: Any) -> str:
        """Format complete 15-dimension publication-grade QualityReport."""
        lines = [
            f"任务编号：{qr.job_id}",
            f"原文档：{qr.source_path}",
            f"译文产物：{qr.output_path}",
            f"目标语言：{qr.target_lang}",
            "",
            "【1. 产出概览与履约指标】",
            f"总分块数：{qr.summary.total_blocks} 块 · 成功完成：{qr.summary.completed_blocks} 块",
            f"初通率 (Pass Rate)：{qr.summary.pass_rate * 100:.1f}%",
            f"靶向修复：{qr.summary.repaired_blocks} 块 · 失败：{qr.summary.failed_blocks} 块",
            f"待人工后编辑 (PE)：{qr.summary.needs_human_blocks} 块 (阻断: {qr.summary.blocked_human_blocks})",
            f"消耗成本：{f'${qr.summary.estimated_cost_usd:.4f}' if qr.summary.estimated_cost_usd is not None else '未知'}",
            "",
            "【2. MTQE 质量多维度分布】",
            f"平均分 (Avg)：{qr.score_metrics.avg_qe:.4f} · 末位 15% (Bottom-15)：{qr.score_metrics.bottom_15_avg_qe:.4f}",
            f"中位数 (P50)：{qr.score_metrics.p50_qe:.4f} · P10/P90：{qr.score_metrics.p10_qe:.4f} / {qr.score_metrics.p90_qe:.4f}",
            f"评分区间：[{qr.score_metrics.min_qe:.4f} ~ {qr.score_metrics.max_qe:.4f}]",
            "",
            "【3. 术语与实体一致性】",
            f"术语精确率：{qr.terminology.term_precision * 100:.1f}% (模糊匹配: {qr.terminology.fuzzy_term_precision * 100:.1f}%)",
            f"术语召回率：{qr.terminology.term_recall * 100:.1f}%",
            f"实体漂移审计：{qr.entity_consistency.terms_audited} 个术语 · 发生漂移: {qr.entity_consistency.terms_with_drift} 个",
            "",
            "【4. 公式与版式保真度】",
            f"占位符保护率：{qr.placeholder.retention_rate * 100:.1f}% (损坏 span: {qr.placeholder.corrupt_spans})",
            f"公式图例替代：{len(qr.formula_witness_fallbacks)}/{qr.formula_blocks} 块",
            f"排版覆盖率：{qr.render_coverage.render_coverage * 100:.1f}% (跳过: {qr.render_coverage.skipped_blocks})",
        ]
        if qr.syntax_fallbacks:
            lines.append(f"语法自愈回退行：{len(qr.syntax_fallbacks)} 行")
        if qr.defect_flags:
            lines.append("")
            lines.append("【5. 缺陷标志统计】")
            for flag, count in sorted(qr.defect_flags.items(), key=lambda t: t[1], reverse=True)[
                :5
            ]:
                lines.append(f" · {flag[:80]}: {count} 次")
        return "\n".join(lines)

    def _doctor_body(self) -> str:
        import shutil

        from ubt.core.config import UBTConfig as _C

        try:
            cfg = _C.from_env()
            key_ok = bool(
                cfg.api_key.get_secret_value() and cfg.api_key.get_secret_value() != MOCK_API_KEY
            )
        except Exception:
            key_ok = False
        try:
            from ubt.core.env import has_accelerator

            gpu = bool(has_accelerator())
        except Exception:
            gpu = False
        lines = [
            f"密钥：{'已配置' if key_ok else '未配置（可用 --dry-run 演练）'}",
            f"GPU：{'就绪' if gpu else 'CPU 模式'}",
            f"Typst：{'可用' if shutil.which('typst') else '未安装'}",
            f"账本：{self._ledger_dir()}",
            f"终端：{os.environ.get('TERM', '?')} · NO_COLOR={os.environ.get('NO_COLOR', '')}",
            f"日志文件：{self._log_path or '未重定向'}",
        ]
        return "\n".join(lines)


def launch_detached(path: str) -> None:
    """Open a file with the OS viewer without blocking or sharing stdio.

    Fire-and-forget: ``subprocess.run(["xdg-open", ...])`` on the UI thread
    stalls the whole TUI until the launcher exits (DBus activation can take
    seconds), and an inheriting child sprays diagnostics such as Evince
    ``Gdk-WARNING ... Broken pipe`` onto the user's terminal -- or worse,
    onto the active alternate screen. Detached + DEVNULL fds avoid both.
    """
    if sys.platform == "win32":
        # Looked up rather than named: `os.startfile` only exists on Windows, so
        # a `type: ignore[attr-defined]` is required on Linux and *unused* (an
        # error under --strict) on the Windows CI matrix.
        os.startfile(path)
        return
    cmd = ["open", path] if sys.platform == "darwin" else ["xdg-open", path]
    # The launcher needs DISPLAY/XDG but never the LLM credentials; keep the
    # documented "every spawn gets a scrubbed env" invariant.
    from ubt.core.env import subprocess_env

    subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=subprocess_env(),
    )


def launch_tui(
    file_path: Path | None = None,
    dry_run: bool = False,
    db_dir: Path | None = None,
    request: dict[str, Any] | None = None,
) -> int | None:
    """Launch fullscreen v2 app. Returns app exit value.

    ``request`` carries the flags parsed before ``-i`` resolved (CLI/translate
    path): seeding the wizard is what keeps ``ubt translate -i -l ja
    --preset publication`` from silently discarding them.
    """
    log_path = route_logs_to_file()
    app = UBTApp(
        initial_file=file_path,
        dry_run_override=dry_run,
        db_dir=db_dir,
        log_path=log_path,
        request=request,
    )
    return app.run()

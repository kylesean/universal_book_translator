"""Session state for TUI v2. Single source of overrides, same as CLI/REST/MCP."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ubt.core.job_options import (
    JOB_ID_MAX_LEN,
    job_id_is_valid,
)
from ubt.core.job_options import (
    JOB_ID_RE as JOB_ID_RE,
)
from ubt.tui.presets import Preset

# No event for this long: say so instead of looking dead. Long silent phases
# (model download, docling ingest) are normal; the text must read as waiting,
# never as stuck, and must point at the log file for proof.
STALL_AFTER_SECS = 60.0
PresetName = Literal["publication", "standard", "preview"]
DualChoice = Literal["bilingual", "monolingual", "facing"]


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """Immutable point-in-time telemetry snapshot for thread-safe UI rendering."""

    stage: str = "IDLE"
    total_blocks: int = 1
    completed_blocks: int = 0
    drafted_blocks: int = 0
    repaired_blocks: int = 0
    failed_blocks: int = 0
    needs_human_blocks: int = 0
    blocked_human_blocks: int = 0
    avg_qe: float = 0.0
    bottom15_qe: float = 0.0
    cost_usd: float | None = None
    cache_hit: float = 0.0
    active_block_id: str | None = None
    active_source: str = ""
    active_target: str = ""
    progress_pct_val: float = 0.0
    cost_label_str: str = "未知"
    dual_label_str: str = "行内双语"
    is_completed_flag: bool = False
    run_started_at: float | None = None
    last_event_at: float | None = None
    run_completed_at: float | None = None

    def progress_pct(self) -> float:
        return self.progress_pct_val

    def cost_label(self) -> str:
        return self.cost_label_str

    def dual_label(self) -> str:
        return self.dual_label_str

    @property
    def is_completed(self) -> bool:
        return self.is_completed_flag

    def elapsed_secs(self, now: float) -> float:
        if self.run_started_at is None:
            return 0.0
        anchor = self.run_completed_at if self.run_completed_at is not None else now
        return max(0.0, anchor - self.run_started_at)

    def rate_per_min(self, now: float) -> float:
        elapsed = self.elapsed_secs(now)
        if elapsed < 1.0 or self.completed_blocks <= 0:
            return 0.0
        return (self.completed_blocks / elapsed) * 60.0

    @staticmethod
    def fmt_duration(secs: float) -> str:
        s = int(secs)
        if s < 60:
            return f"{s}秒"
        m, sec = divmod(s, 60)
        if m < 60:
            return f"{m}分{sec:02d}秒"
        h, m = divmod(m, 60)
        return f"{h}时{m:02d}分"

    def pace_text(self, now: float) -> str:
        if self.is_completed_flag:
            rate = self.rate_per_min(now)
            rate_str = f" · 均速{rate:.1f} 块/分" if rate > 0 else ""
            return f"已完成 · 耗时{self.fmt_duration(self.elapsed_secs(now))}{rate_str}"
        rate = self.rate_per_min(now)
        if rate <= 0:
            if self.completed_blocks == 0 and (
                "INGEST" in self.stage or "STARTED" in self.stage or "PREPROCESSING" in self.stage
            ):
                return f"版面与模型解析中 · 已跑{self.fmt_duration(self.elapsed_secs(now))}"
            return "启动中"
        return f"{rate:.1f} 块/分 · 已跑{self.fmt_duration(self.elapsed_secs(now))}"


@dataclass
class SessionState:
    """User-facing wizard state. Technical names never reach the UI layer."""

    input_path: Path | None = None
    source_lang: str = "en"
    target_lang: str = "zh"
    dual_choice: DualChoice = "bilingual"
    preset: Preset = Preset.STANDARD
    # Engine knobs are injected only for a preset the user actually picked.
    # Applying the default unconditionally would rewrite UBT_RENDER_ENGINE /
    # UBT_MATH_BACKEND / UBT_PROMPT_STRATEGY (and the same keys from ubt.toml)
    # on every TUI run — the same class of bug guarded against for
    # translate_chrome / cover_mode / formula_mode below.
    preset_explicit: bool = False
    # Same rule as preset_explicit for the wizard defaults that map onto
    # config keys: an untouched output-shape or fresh toggle must not
    # rewrite UBT_DUAL_MODE / UBT_FACING_SPREAD / UBT_FRESH.
    dual_explicit: bool = False
    fresh_explicit: bool = False
    draft_model: str | None = None
    repair_model: str | None = None
    glossary: Path | None = None
    glossary_auto: bool = True
    domain: str | None = None
    pages: str | None = None
    #: The document profile the run is judged against. The other three entry
    #: surfaces take it from ``--profile``; the wizard starts at the same
    #: default ``PipelineOrchestrator.run`` does.
    profile: str = "general"
    dry_run: bool = False
    formula_strategy: str = "engine"  # engine | source-image
    job_id: str | None = None
    # Which provider mode produced ``job_id``. A ledger captured from a
    # simulated run holds mock drafts, so it must not be resumed by a real run.
    job_id_dry_run: bool = False
    fresh: bool = False
    output_path: Path | None = None
    # Filled after probe
    recommended_glossary: Path | None = None
    detected_domain: str = "general"

    # -- runtime telemetry (updated from TranslationProgressEvent) --
    stage: str = "IDLE"
    total_blocks: int = 1
    completed_blocks: int = 0
    drafted_blocks: int = 0
    repaired_blocks: int = 0
    failed_blocks: int = 0
    needs_human_blocks: int = 0
    blocked_human_blocks: int = 0
    avg_qe: float = 0.0
    bottom15_qe: float = 0.0
    cost_usd: float | None = None
    cache_hit: float = 0.0
    active_block_id: str | None = None
    active_source: str = ""
    active_target: str = ""
    qe_history: list[float] = field(default_factory=list)
    event_tail: list[tuple[str, str, str]] = field(default_factory=list)
    # monotonic clocks for rate / ETA / stall detection (set on first event)
    run_started_at: float | None = None
    last_event_at: float | None = None
    run_completed_at: float | None = None

    def validate_job_id(self, raw: str) -> str:
        """Validate explicit job id, same rule as CLI/API."""
        if not job_id_is_valid(raw):
            raise ValueError(f"job-id 仅允许字母/数字/-/_（最长 {JOB_ID_MAX_LEN} 字符）：{raw!r}")
        return raw

    def effective_dual_mode(self) -> str:
        """Map user-facing dual_choice to engine dual_mode."""
        if self.dual_choice == "monolingual":
            return "monolingual"
        if self.dual_choice == "facing":
            return "facing"
        return "inline"

    def dual_label(self) -> str:
        if self.dual_choice == "monolingual":
            return "纯单语"
        if self.dual_choice == "facing":
            return "左右对照"
        return "行内双语"

    def choose_preset(self, preset: Preset) -> None:
        """Record an affirmative preset pick, which is what arms the engine bundle."""
        self.preset = preset
        self.preset_explicit = True

    def choose_dual(self, value: DualChoice) -> None:
        """Record a user-driven output-shape change (arms the override)."""
        self.dual_choice = value
        self.dual_explicit = True

    def choose_fresh(self, value: bool) -> None:
        """Record a user-driven fresh/incremental choice (arms the override)."""
        self.fresh = value
        self.fresh_explicit = True

    def apply_cli_request(self, request: dict[str, Any]) -> None:
        """Seed the wizard from flags parsed before ``-i`` was resolved.

        Called only on the interactive path, where the alternative used to
        be dropping every flag but the input file. Only keys the wizard can
        express are mapped; ``dual_mode``/``fresh``/``preset`` arm their
        explicit flags (their typer default is None, so a present value is
        an affirmative choice that must survive into to_overrides).
        """
        if request.get("source_lang"):
            self.source_lang = str(request["source_lang"])
        if request.get("target_lang"):
            self.target_lang = str(request["target_lang"])
        if request.get("draft_model"):
            self.draft_model = str(request["draft_model"])
        if request.get("repair_model"):
            self.repair_model = str(request["repair_model"])
        if request.get("glossary") is not None:
            self.glossary = Path(request["glossary"])
        if request.get("domain"):
            self.domain = str(request["domain"])
        if request.get("profile"):
            self.profile = str(request["profile"])
        if request.get("pages"):
            self.pages = str(request["pages"])
        if request.get("output_path") is not None:
            self.output_path = Path(request["output_path"])
        if request.get("preset") is not None:
            self.choose_preset(Preset(str(request["preset"])))
        dual = request.get("dual_mode")
        if dual is not None:
            mode = str(dual)
            self.choose_dual(
                "monolingual"
                if mode == "monolingual"
                else "facing"
                if mode == "facing"
                else "bilingual"
            )
        elif request.get("facing_spread") is True:
            self.choose_dual("facing")
        if request.get("fresh") is not None:
            self.choose_fresh(bool(request["fresh"]))
        if request.get("job_id"):
            self.job_id = self.validate_job_id(str(request["job_id"]))

    def to_overrides(self, db_dir: Path | None = None) -> dict[str, Any]:
        """Build engine overrides dict for ``apply_config_overrides``."""
        from ubt.core.presets import resolve_engine_params

        # An untouched preset contributes nothing: the config layer (env >
        # ubt.toml > defaults) keeps control until the user picks a bundle.
        eng = resolve_engine_params(self.preset, {}) if self.preset_explicit else {}
        # Only wizard-backed choices are injected. translate_chrome /
        # cover_mode / formula_mode are not exposed anywhere in the TUI and
        # must never be hard-injected as constants: that would silently
        # override UBT_TRANSLATE_CHROME / UBT_COVER_MODE / UBT_FORMULA_MODE
        # (same class of bug guarded against in the CLI typer literals).
        # dual/fresh follow the same rule: injected only when the user
        # actually touched them, and after the bundle so an explicit pick
        # wins over preset defaults.
        overrides: dict[str, Any] = {}
        overrides.update(eng)
        if self.dual_explicit:
            eff = self.effective_dual_mode()
            overrides["dual_mode"] = eff
            overrides["facing_spread"] = eff == "facing"
        if self.fresh_explicit:
            overrides["fresh"] = self.fresh
        if self.draft_model:
            overrides["draft_model"] = self.draft_model
            overrides["repair_model"] = self.repair_model or self.draft_model
        elif self.repair_model:
            overrides["repair_model"] = self.repair_model
        if self.glossary is not None:
            overrides["glossary_path"] = self.glossary
        if self.domain:
            overrides["domain"] = self.domain
        if self.pages:
            overrides["pages"] = self.pages
        if db_dir is not None:
            overrides["db_dir"] = db_dir
        return overrides

    def to_run_kwargs(self) -> dict[str, Any]:
        """Run-only kwargs for ``PipelineOrchestrator.run``."""
        return {
            "target_lang": self.target_lang,
            "source_lang": self.source_lang,
            "profile_name": self.profile,
            "job_id": self.job_id,
        }

    @property
    def is_completed(self) -> bool:
        return self.run_completed_at is not None or self.stage in (
            "EXPORT_COMPLETED",
            "DONE",
            "COMPLETED",
        )

    def mark_completed(self, tick: float | None = None, artifact_path: Any = None) -> None:
        """Mark pipeline as completed, freezing timers, clocks, and telemetry."""
        now = time.monotonic() if tick is None else tick
        if self.run_completed_at is None:
            self.run_completed_at = now
        if artifact_path:
            with contextlib.suppress(Exception):
                self.output_path = Path(str(artifact_path))

    def apply_event(self, event: Any, now: float | None = None) -> None:
        """Fold a TranslationProgressEvent into telemetry (pure, testable)."""
        tick = time.monotonic() if now is None else now
        if self.run_started_at is None:
            self.run_started_at = tick
        self.last_event_at = tick
        total = int(getattr(event, "total_blocks", 0) or 0)
        if total > 0:
            self.total_blocks = total
        self.completed_blocks = int(getattr(event, "completed_blocks", 0) or 0)
        self.drafted_blocks = int(getattr(event, "drafted_blocks", 0) or 0)
        self.repaired_blocks = int(getattr(event, "repaired_blocks", 0) or 0)
        self.failed_blocks = int(getattr(event, "failed_blocks", 0) or 0)
        self.needs_human_blocks = int(getattr(event, "needs_human_blocks", 0) or 0)
        self.blocked_human_blocks = int(getattr(event, "blocked_human_blocks", 0) or 0)
        self.avg_qe = float(getattr(event, "current_avg_qe", 0.0) or 0.0)
        self.bottom15_qe = float(getattr(event, "bottom_15_avg_qe", 0.0) or 0.0)
        cost = getattr(event, "estimated_cost_usd", None)
        self.cost_usd = float(cost) if cost is not None else None
        self.cache_hit = float(getattr(event, "cache_hit_rate", 0.0) or 0.0)
        et = getattr(event, "event_type", None)
        self.stage = str(getattr(et, "value", et) or "").upper() or self.stage
        if self.stage in ("EXPORT_COMPLETED", "DONE", "COMPLETED"):
            art = getattr(event, "artifact_path", None) or getattr(event, "message", None)
            self.mark_completed(tick, artifact_path=art)
        bid = getattr(event, "active_block_id", None)
        if bid:
            self.active_block_id = str(bid)
        msg = str(getattr(event, "message", "") or "")
        # Keep snippets short for the bilingual card, ignoring internal debug messages
        _debug_prefixes = ("drafted blocks", "targeted repair", "processing", "saving", "extracted")
        is_debug_log = any(msg.lower().startswith(p) for p in _debug_prefixes)
        if self.active_block_id and msg and len(msg) < 500 and not is_debug_log:
            if "draft" in self.stage.lower() or "repair" in self.stage.lower():
                self.active_target = msg[:300]
            else:
                self.active_source = msg[:300]
        if self.avg_qe > 0:
            self.qe_history.append(round(self.avg_qe, 4))
            if len(self.qe_history) > 60:
                self.qe_history.pop(0)

    def update_bilingual_snippet(
        self,
        source: str | None = None,
        target: str | None = None,
        block_id: str | None = None,
    ) -> None:
        """Update active block source and target snippets with real document text."""
        if block_id:
            self.active_block_id = str(block_id)
        if source:
            self.active_source = source.strip()[:400]
        if target:
            self.active_target = target.strip()[:400]

    def processed_blocks(self) -> int:
        """Terminal-state blocks (completed + failed + needs/blocked human).

        Progress and ETA must count these: a run whose blocks all went to the
        human queue finished — it is not "0%" (same
        rule as core.engine.progress.processed_blocks).
        """
        return (
            self.completed_blocks
            + self.failed_blocks
            + self.needs_human_blocks
            + self.blocked_human_blocks
        )

    def reset_run(self) -> None:
        """Clear every per-run telemetry field so the wizard starts a book clean.

        Resets all fields completely to avoid leaking avg_qe/cost/snippets/qe_history
        from the previous book into the next one.
        """
        self.stage = "IDLE"
        self.total_blocks = 1
        self.completed_blocks = 0
        self.drafted_blocks = 0
        self.repaired_blocks = 0
        self.failed_blocks = 0
        self.needs_human_blocks = 0
        self.blocked_human_blocks = 0
        self.avg_qe = 0.0
        self.bottom15_qe = 0.0
        self.cost_usd = None
        self.cache_hit = 0.0
        self.active_block_id = None
        self.active_source = ""
        self.active_target = ""
        self.qe_history.clear()
        self.event_tail.clear()
        self.run_started_at = None
        self.last_event_at = None
        self.run_completed_at = None

    def progress_pct(self) -> float:
        """0-100 progress percent (terminal-state分子)."""
        if self.total_blocks <= 0:
            return 0.0
        return round(min(1.0, self.processed_blocks() / self.total_blocks) * 100, 1)

    def progress_frac(self) -> float:
        """0-1 fraction for bars (terminal-state分子)."""
        if self.total_blocks <= 0:
            return 0.0
        return max(0.0, min(1.0, self.processed_blocks() / self.total_blocks))

    def elapsed_secs(self, now: float) -> float:
        """Seconds since the first folded event (frozen once completed)."""
        if self.run_started_at is None:
            return 0.0
        anchor = self.run_completed_at if self.run_completed_at is not None else now
        return max(0.0, anchor - self.run_started_at)

    def rate_per_min(self, now: float) -> float:
        """Processed blocks per minute since start (0 when unmeasurable)."""
        elapsed = self.elapsed_secs(now)
        done = self.processed_blocks()
        if elapsed < 1.0 or done <= 0:
            return 0.0
        return done / elapsed * 60.0

    def stalled_for(self, now: float) -> float:
        """Seconds since the last folded event (0 when completed or nothing ran yet)."""
        if self.is_completed:
            return 0.0
        anchor = self.last_event_at if self.last_event_at is not None else self.run_started_at
        if anchor is None:
            return 0.0
        return max(0.0, now - anchor)

    def is_stalled(self, now: float, after: float = STALL_AFTER_SECS) -> bool:
        """True when silent longer than the stall threshold (never True once completed)."""
        if self.is_completed:
            return False
        return self.run_started_at is not None and self.stalled_for(now) > after

    def eta_text(self, now: float) -> str:
        """ETA for long ops; '—' when there is nothing to project from."""
        rate = self.rate_per_min(now)
        remaining = max(0, self.total_blocks - self.processed_blocks())
        if rate <= 0 or remaining <= 0 or self.total_blocks <= 0:
            return "—"
        secs = remaining / rate * 60.0
        if secs >= 120:
            return f"剩余约{int(secs // 60)}分"
        if secs >= 60:
            return "剩余约1分"
        return f"剩余约{int(secs)}秒"

    @staticmethod
    def fmt_duration(secs: float) -> str:
        """Short Chinese duration: 45秒 / 3分 / 1时05分."""
        total = max(0, int(secs))
        if total < 60:
            return f"{total}秒"
        if total < 3600:
            return f"{total // 60}分"
        return f"{total // 3600}时{total % 3600 // 60:02d}分"

    def pace_text(self, now: float) -> str:
        """One-line liveness: rate + elapsed + ETA, or an honest completion / stall note."""
        if self.is_completed:
            elapsed = self.fmt_duration(self.elapsed_secs(now))
            rate = self.rate_per_min(now)
            if rate > 0:
                return f"已完成 · 耗时{elapsed} · 均速{rate:.1f} 块/分"
            return f"已完成 · 耗时{elapsed}"
        if self.is_stalled(now):
            if self.completed_blocks == 0 and (
                "INGEST" in self.stage
                or "STARTED" in self.stage
                or "IDLE" in self.stage
                or "PREPROCESSING" in self.stage
            ):
                return (
                    f"版面与模型解析中 (已跑{self.fmt_duration(self.elapsed_secs(now))} · 查看日志)"
                )
            return f"等待响应 {int(self.stalled_for(now))}s"
        rate = self.rate_per_min(now)
        if rate <= 0:
            if self.completed_blocks == 0 and (
                "INGEST" in self.stage or "STARTED" in self.stage or "PREPROCESSING" in self.stage
            ):
                return f"版面与模型解析中 · 已跑{self.fmt_duration(self.elapsed_secs(now))}"
            return "启动中"
        return f"{rate:.1f} 块/分 · 已跑{self.fmt_duration(self.elapsed_secs(now))} · {self.eta_text(now)}"

    def cost_label(self) -> str:
        """Never fabricate 0.00 for unknown cost (None = 未知)."""
        if self.cost_usd is None:
            return "未知"
        return f"${self.cost_usd:.4f}"

    @staticmethod
    def qe_color(qe: float) -> str:
        """QE color token: green >= .85, yellow >= .70, else red."""
        if qe >= 0.85:
            return "green"
        if qe >= 0.70:
            return "yellow"
        return "red"

    def to_snapshot(self) -> TelemetrySnapshot:
        """Create an immutable snapshot of current telemetry state."""
        return TelemetrySnapshot(
            stage=self.stage,
            total_blocks=self.total_blocks,
            completed_blocks=self.completed_blocks,
            drafted_blocks=self.drafted_blocks,
            repaired_blocks=self.repaired_blocks,
            failed_blocks=self.failed_blocks,
            needs_human_blocks=self.needs_human_blocks,
            blocked_human_blocks=self.blocked_human_blocks,
            avg_qe=self.avg_qe,
            bottom15_qe=self.bottom15_qe,
            cost_usd=self.cost_usd,
            cache_hit=self.cache_hit,
            active_block_id=self.active_block_id,
            active_source=self.active_source,
            active_target=self.active_target,
            progress_pct_val=self.progress_pct(),
            cost_label_str=self.cost_label(),
            dual_label_str=self.dual_label(),
            is_completed_flag=self.is_completed,
            run_started_at=self.run_started_at,
            last_event_at=self.last_event_at,
            run_completed_at=self.run_completed_at,
        )

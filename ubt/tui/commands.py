"""Slash-command parser for TUI v2. Pure functions, fully unit tested."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParsedCommand:
    """Result of parsing one input line."""

    action: str
    args: dict[str, Any]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


_VALID_PRESETS = ("publication", "standard", "preview")
_VALID_DUALS = ("inline", "facing", "alternating", "monolingual", "bilingual")

_HELP_TEXT = """命令一览：
/translate [path] [--preset p|s|preview] [--pages 1-3] [--dry-run]  开跑
/preset <publication|standard|preview>   切换质量档
/pages 1-5 | /pages none                页码范围（仅 PDF 抽样）
/dual inline|facing|monolingual         输出形态
/glossary auto|none|<path>              术语表
/model <name> | /model none             模型（空=引擎默认）
/resume <job_id>  /cancel  /fresh       续跑 / 中止(保留账本) / 重来(二次确认)
/new  /wizard                           返回向导屏选择新文档
/jobs  /status <job_id>  /report        历史与报告
/doctor  /open  /help  /quit            诊断 / 打开产物 / 帮助 / 退出
直接粘贴文件路径（含引号/拖入）也可选中文件；1/2/3 快捷切档（运行前）。
"""

PALETTE_COMMANDS: list[tuple[str, str, str]] = [
    (
        "/translate",
        "开始运行翻译管线（可选 [path] [--preset p] [--pages 1-3] [--dry-run]）",
        "运行",
    ),
    ("/preset", "切换翻译质量档位 (publication / standard / preview)", "参数"),
    ("/dual inline", "设置译文形态为行内双语对照 (Bilingual Inline - 研读推荐)", "形态"),
    ("/dual facing", "设置译文形态为左右跨页对照 (Facing Spread - 学术/出版推荐)", "形态"),
    ("/dual monolingual", "设置译文形态为纯目标语言单语 (Monolingual - 纯译文发行)", "形态"),
    ("/model", "覆盖初译与修复模型名称", "参数"),
    ("/report", "查看当前任务的 MTQE 质量与进度报告", "监控"),
    ("/pe", "打开人工后编辑与审校 (Post-Editing) 界面 (/pe [block_id])", "审校"),
    ("/doctor", "检查 API 密钥、GPU 加速与 Typst 运行环境诊断", "工具"),
    ("/open", "调用系统原生查看器打开生成的双语产物", "工具"),
    ("/cancel", "中止当前运行的任务（保留 SQLite 账本）", "控制"),
    ("/resume", "断点续跑指定或当前任务 (/resume [job_id])", "控制"),
    ("/fresh", "丢弃账本并从头重新翻译", "控制"),
    ("/new", "返回向导屏选择新文档开启新任务", "导航"),
    ("/jobs", "查看近期本地历史翻译账本任务", "历史"),
    ("/status", "查询账本块统计数据 (/status [job_id])", "历史"),
    ("/help", "打开快捷键与斜杠命令帮助浮层", "帮助"),
    ("/quit", "退出 UBT 终端应用", "系统"),
]


def help_text() -> str:
    """Return help copy."""
    return _HELP_TEXT


def parse_command(text: str) -> ParsedCommand:
    """Parse one input line into an action.

    Never raises; failures come back as ``error`` so the UI stays alive.
    """
    DOC_SUFFIXES = {".pdf", ".epub", ".docx", ".md", ".txt", ".html"}
    raw = (text or "").strip().strip("'\"")
    if not raw:
        return ParsedCommand(action="noop", args={}, error=None)
    # Absolute / relative file paste wins over slash-commands (e.g.
    # /tmp/book.pdf or ./docs/ch.pdf must not parse as /tmp command), but a
    # slash-command that carries a document argument (/translate book.epub)
    # must not be mistaken for a path: a leading "/" plus a space is a command,
    # and only an existing file overrides that.
    _maybe_path = Path(raw.strip("'\"")).expanduser()
    if _maybe_path.suffix.lower() in DOC_SUFFIXES and (
        _maybe_path.exists() or not raw.startswith("/") or not any(c.isspace() for c in raw)
    ):
        return ParsedCommand(action="select_file", args={"path": str(_maybe_path)})
    if not raw.startswith("/"):
        # Plain path paste or fuzzy query
        p = Path(raw.strip("'\"")).expanduser()
        if p.suffix.lower() in {".pdf", ".epub", ".docx", ".md", ".txt", ".html"}:
            return ParsedCommand(action="select_file", args={"path": str(p)})
        return ParsedCommand(action="fuzzy", args={"query": raw})
    try:
        parts = shlex.split(raw)
    except ValueError as exc:
        return ParsedCommand(action="unknown", args={"raw": raw}, error=f"解析失败：{exc}")
    if not parts:
        return ParsedCommand(action="noop", args={})
    cmd = parts[0].lower().lstrip("/")
    rest = parts[1:]

    if cmd in ("help", "h", "?"):
        return ParsedCommand(action="help", args={})
    if cmd in ("quit", "q", "exit"):
        return ParsedCommand(action="quit", args={})
    if cmd in ("cancel", "stop"):
        return ParsedCommand(action="cancel", args={})
    if cmd in ("fresh",):
        return ParsedCommand(action="fresh", args={})
    if cmd in ("new", "wizard", "w"):
        return ParsedCommand(action="new", args={})
    if cmd in ("jobs",):
        return ParsedCommand(action="jobs", args={})
    if cmd in ("report",):
        return ParsedCommand(action="report", args={})
    if cmd in ("pe", "postedit"):
        return ParsedCommand(action="pe", args={"block_id": rest[0] if rest else None})
    if cmd in ("doctor",):
        return ParsedCommand(action="doctor", args={})
    if cmd in ("open",):
        target = rest[0] if rest else "artifact"
        return ParsedCommand(action="open", args={"target": target})
    if cmd in ("preset", "p"):
        if not rest:
            return ParsedCommand(
                action="unknown",
                args={"raw": raw},
                error="用法：/preset publication|standard|preview",
            )
        name = rest[0].lower()
        aliases = {
            "pub": "publication",
            "publication": "publication",
            "std": "standard",
            "s": "standard",
            "standard": "standard",
            "prev": "preview",
            "preview": "preview",
        }
        if name not in aliases:
            return ParsedCommand(
                action="unknown",
                args={"raw": raw},
                error=f"未知档位 {rest[0]!r}，可选 publication/standard/preview",
            )
        return ParsedCommand(action="preset", args={"preset": aliases[name]})
    if cmd in ("pages",):
        if not rest or rest[0].lower() in ("none", "all", "clear"):
            return ParsedCommand(action="pages", args={"pages": None})
        return ParsedCommand(action="pages", args={"pages": rest[0]})
    if cmd in ("dual",):
        if not rest or rest[0].lower() not in _VALID_DUALS:
            return ParsedCommand(
                action="unknown", args={"raw": raw}, error="用法：/dual inline|facing|monolingual"
            )
        val = rest[0].lower()
        if val == "bilingual":
            val = "inline"
        return ParsedCommand(action="dual", args={"dual": val})
    if cmd in ("glossary", "g"):
        if not rest or rest[0].lower() == "auto":
            return ParsedCommand(action="glossary", args={"mode": "auto"})
        if rest[0].lower() in ("none", "off"):
            return ParsedCommand(action="glossary", args={"mode": "none"})
        return ParsedCommand(action="glossary", args={"mode": "custom", "path": rest[0]})
    if cmd in ("model", "m"):
        if not rest or rest[0].lower() in ("none", "default", "clear"):
            return ParsedCommand(action="model", args={"model": None})
        return ParsedCommand(action="model", args={"model": rest[0]})
    if cmd in ("resume", "r"):
        if not rest:
            return ParsedCommand(
                action="unknown", args={"raw": raw}, error="用法：/resume <job_id>"
            )
        return ParsedCommand(action="resume", args={"job_id": rest[0]})
    if cmd in ("status", "st"):
        return ParsedCommand(action="status", args={"job_id": rest[0] if rest else None})
    if cmd in ("translate", "t", "run"):
        path: str | None = None
        preset: str | None = None
        pages: str | None = None
        dry_run = False
        positional: list[str] = []
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok == "--dry-run":
                dry_run = True
            elif tok == "--preset" and i + 1 < len(rest):
                preset = rest[i + 1].lower()
                i += 1
            elif tok.startswith("--preset="):
                preset = tok.split("=", 1)[1].lower()
            elif tok == "--pages" and i + 1 < len(rest):
                pages = rest[i + 1]
                i += 1
            elif tok.startswith("--pages="):
                pages = tok.split("=", 1)[1]
            elif tok.startswith("--"):
                return ParsedCommand(action="unknown", args={"raw": raw}, error=f"未知选项 {tok!r}")
            else:
                positional.append(tok)
            i += 1
        if positional:
            path = positional[0]
        if preset is not None and preset not in _VALID_PRESETS:
            aliases = {"pub": "publication", "s": "standard", "std": "standard"}
            preset = aliases.get(preset, preset)
            if preset not in _VALID_PRESETS:
                return ParsedCommand(
                    action="unknown", args={"raw": raw}, error=f"未知档位 {preset!r}"
                )
        return ParsedCommand(
            action="translate",
            args={"path": path, "preset": preset, "pages": pages, "dry_run": dry_run},
        )
    return ParsedCommand(
        action="unknown", args={"raw": raw}, error=f"未知命令 /{cmd}，输入 /help 查看"
    )

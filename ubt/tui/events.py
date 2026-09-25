"""Chinese presentation of engine progress events.

Screen-consistency rule: the fullscreen UI speaks Chinese only. Raw English
engine messages stay in the session log file (see ``ubt.tui.logsetup``);
these helpers render the short Chinese skeleton for the header, the stepper
context, and the event ticker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

STAGE_LABELS: dict[str, str] = {
    "IDLE": "就绪",
    "JOB_STARTED": "任务开始",
    "PREPROCESSING_DONE": "预处理完成",
    "BIBLE_EXTRACTED": "术语圣经就绪",
    "DRAFT_BATCH_COMPLETED": "初译",
    "MTQE_EVALUATED": "质检",
    "REPAIR_BATCH_COMPLETED": "修复",
    "TRIAGE_COMPLETED": "分诊",
    "CTEXT_COMPLETED": "译文定稿",
    "MODE_ADVISED": "引擎建议",
    "EXPORT_COMPLETED": "导出完成",
    "PIPELINE_FAILED": "管线失败",
}


def stage_label(code: str | None) -> str:
    """Map a raw engine stage code to its Chinese label (passthrough if unknown)."""
    if not code:
        return STAGE_LABELS["IDLE"]
    key = str(code).upper()
    return STAGE_LABELS.get(key, key)


def _event_code(event: Any) -> str:
    et = getattr(event, "event_type", None)
    return str(getattr(et, "value", et) or "").upper()


def describe_event(event: Any) -> str:
    """One Chinese line per engine event for the on-screen ticker."""
    code = _event_code(event)
    total = int(getattr(event, "total_blocks", 0) or 0)
    done = int(getattr(event, "completed_blocks", 0) or 0)
    if code == "JOB_STARTED":
        return f"任务开始 {getattr(event, 'job_id', '')}"
    if code == "PREPROCESSING_DONE":
        return "预处理完成"
    if code == "BIBLE_EXTRACTED":
        return "术语圣经就绪"
    if code == "DRAFT_BATCH_COMPLETED":
        return f"初译 {done}/{total}" if total else "初译推进"
    if code == "MTQE_EVALUATED":
        avg = float(getattr(event, "current_avg_qe", 0.0) or 0.0)
        return f"质检 平均分 {avg:.2f}"
    if code == "REPAIR_BATCH_COMPLETED":
        fixed = int(getattr(event, "repaired_blocks", 0) or 0)
        return f"修复完成 {fixed} 块"
    if code == "TRIAGE_COMPLETED":
        return "分诊完成"
    if code == "CTEXT_COMPLETED":
        return "译文定稿"
    if code == "MODE_ADVISED":
        return "引擎建议已应用（详情见日志文件）"
    if code == "EXPORT_COMPLETED":
        artifact = getattr(event, "artifact_path", None) or ""
        name = Path(str(artifact)).name if artifact else ""
        return f"导出完成 {name}".rstrip()
    if code == "PIPELINE_FAILED":
        return "管线失败（同 job 可续跑，详见日志文件）"
    message = str(getattr(event, "message", "") or "")[:80]
    return message or code

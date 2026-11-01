from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MISSION_CONTROL = _REPO_ROOT / "web" / "src" / "views" / "mission-control" / "MissionControl.tsx"


def _source(path: Path) -> str:
    if not path.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    return path.read_text(encoding="utf-8")


def test_the_console_log_survives_leaving_the_page() -> None:
    source = _source(_MISSION_CONTROL)

    assert "LOG_HISTORY" in source
    assert "historyFor" in source
    assert "rememberLogs" in source
    assert "useState<LogEntry[]>(() => historyFor(" in source
    assert "setLogs(historyFor(currentJobId))" in source
    assert "appendLog(" in source
    stray = [
        line.strip()
        for line in source.splitlines()
        if "setLogs(" in line and "appendLog" not in line
    ]
    assert len(stray) == 2, f"a log write bypasses appendLog: {stray}"
    assert any("setLogs(historyFor(" in line for line in stray), (
        "the job-switch reset must reload the new job's own history"
    )

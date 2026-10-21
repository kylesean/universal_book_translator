"""Guard: the Mission Control log must survive leaving and returning.

The console has no frontend test runner, so this class of defect ships
silently. Measured in a real browser against the reference job: two lines after
navigating away and back, on a run whose stream had produced dozens.
"""

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
    """A finished job replays two frames, so an unmounted panel came back with two.

    The log lived in component state and Mission Control unmounts whenever the
    operator opens the review workbench. Measured on the reference job: two
    lines after navigating away and back, both the same final snapshot, on a
    run whose stream had produced dozens of distinct messages.
    """
    source = _source(_MISSION_CONTROL)

    # The history is kept outside the component and keyed by job...
    assert "LOG_HISTORY" in source
    assert "historyFor" in source
    assert "rememberLogs" in source
    # ...seeded on mount, and re-seeded when the route switches jobs in place.
    assert "useState<LogEntry[]>(() => historyFor(" in source
    assert "setLogs(historyFor(currentJobId))" in source
    # Every append goes through one sink that writes both stores. Exactly two
    # setLogs calls may remain: the sink's own body, and the job-switch reset.
    # A third would be a stray write restoring the leak silently.
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

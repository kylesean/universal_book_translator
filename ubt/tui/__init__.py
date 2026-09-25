"""UBT Terminal User Interface (fullscreen Textual app)."""

from ubt.tui.advisor import AdvisoryReport, DocCategory, DocumentAdvisor, MathDensity
from ubt.tui.app import UBTApp, launch_tui
from ubt.tui.commands import ParsedCommand, help_text, parse_command
from ubt.tui.events import describe_event, stage_label
from ubt.tui.logsetup import route_logs_to_file
from ubt.tui.presets import PRESETS, Preset, PresetPolicy
from ubt.tui.state import SessionState

__all__ = [
    "AdvisoryReport",
    "DocCategory",
    "DocumentAdvisor",
    "MathDensity",
    "PRESETS",
    "ParsedCommand",
    "Preset",
    "PresetPolicy",
    "SessionState",
    "UBTApp",
    "describe_event",
    "help_text",
    "launch_tui",
    "parse_command",
    "route_logs_to_file",
    "stage_label",
]

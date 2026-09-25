"""Semantic visual tokens for the fullscreen TUI.

Single vocabulary for color and markers, shared by Rich markup (this module)
and Textual CSS (``UBTApp.CSS`` mirrors these names in its ``$`` variables).

Contract (tui-design skill, visual-patterns):
- ANSI theme variables only, never hex: the user's terminal theme is sacred.
- Color never carries meaning alone: every status also has words or numbers.
- No emoji, no Nerd Font glyphs anywhere: plain ASCII/Unicode box + block
  characters only (there is no font detection, only opt-in; we do not opt in).
"""

from __future__ import annotations

ACCENT = "cyan"
SUCCESS = "green"
WARNING = "yellow"
ERROR = "red"
# NOTE: Rich markup and Textual CSS parse different grey spellings
# (verified against the installed versions: Rich takes ``grey50`` and rejects
# ``grey``/``gray`` in its get_style path; Textual takes ``grey`` and rejects
# ``grey50``/``bright_black``). One token, two spellings, same muted meaning.
MUTED = "grey"  # Textual CSS only
MUTED_RICH = "grey50"  # Rich markup only

# Stepper joints: plain ASCII separator, dimmed at render time.
STEP_JOINT = " > "

# QE thresholds mirror SessionState.qe_color.
QE_GOOD = 0.85
QE_WARN = 0.70

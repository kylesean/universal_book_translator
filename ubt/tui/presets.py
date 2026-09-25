"""TUI preset surface.

The single source of truth moved to:mod:`ubt.core.presets` so the
CLI ``--preset`` layer and the wizard resolve the same bundles. This module
stays as the TUI import surface.
"""

from __future__ import annotations

from ubt.core.presets import PRESETS, Preset, PresetPolicy, preset_options

__all__ = ["PRESETS", "Preset", "PresetPolicy", "preset_options"]

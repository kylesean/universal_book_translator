"""Hatch build hook: include the fetched font set when the checkout has it.

`fonts/` is fetched by ``scripts/fetch_fonts.py`` and is deliberately not
committed. A static ``force-include`` made every wheel build fail on a clean
checkout (Hatch raises "Forced include not found"), including the Docker build
whose context has no fonts. Add the mapping only when the directory exists;
without it the wheel falls back to system fonts exactly as before.
"""

from __future__ import annotations

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class FontBuildHook(BuildHookInterface):
    """Map ``fonts/`` into ``ubt/fonts`` only when it was fetched."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        del version
        fonts = Path(self.root) / "fonts"
        if fonts.is_dir() and any(fonts.iterdir()):
            force_include = build_data.setdefault("force_include", {})
            assert isinstance(force_include, dict)
            force_include[str(fonts)] = "ubt/fonts"

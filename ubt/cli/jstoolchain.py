"""Locating the JavaScript toolchain the console's build/dev steps need.

**Why this is not just `subprocess.run(["pnpm", ...])`.** The CLI hardcoded
``pnpm`` in two places — ``ubt console --dev`` and ``scripts/build_console_assets.py``
— with no probe and no fallback. On a machine without pnpm that is a raw
``FileNotFoundError`` traceback, and the build path is worse: ``console`` invokes
the script automatically when the bundled assets are missing, so a first-time
``ubt console`` (no ``--dev``) failed too, reporting only "Failed to build static
assets".

The preference order is deliberate:

1. **The project's own ``node_modules/.bin``.** When dependencies are installed,
   ``vite`` and ``tsc`` are already there as executables. Running them directly is
   the fastest path and needs no package manager at all — which is exactly the
   situation this repository ships in (``web/node_modules`` is populated).
2. **A package manager**, in the order ``pnpm`` → ``npm`` → ``yarn``. Needed only
   for the very first install, or when ``node_modules`` is absent.
3. **A readable error.** Every failure names the missing tool and the command that
   installs it, instead of surfacing ``[Errno 2]``.

Nothing here installs dependencies or rewrites lockfiles: choosing a package
manager is the operator's call, and this module only reports what is available.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

#: Package managers, in preference order. ``pnpm`` first because the committed
#: lockfile is ``pnpm-lock.yaml`` and CI uses pnpm; the others exist so a machine
#: with only npm is not blocked from a build whose scripts are plain npm scripts.
_PACKAGE_MANAGERS: tuple[str, ...] = ("pnpm", "npm", "yarn")

#: Local binaries the console needs, resolved from ``web/node_modules/.bin``.
_LOCAL_TOOLS: dict[str, str] = {
    "vite": "vite",
    "tsc": "tsc",
    "oxlint": "oxlint",
}


@dataclass(frozen=True)
class JsToolchain:
    """What is available for driving the frontend, and how to invoke it."""

    web_dir: Path
    #: Absolute path to a local ``node_modules/.bin/<tool>``, or ``None``.
    local_bin: Path | None
    #: A package manager on ``PATH``, or ``None``.
    package_manager: str | None

    @property
    def can_build(self) -> bool:
        """Whether ``build`` can run: local ``vite`` plus ``tsc``, or a manager."""
        return (self._has_local("vite") and self._has_local("tsc")) or bool(self.package_manager)

    @property
    def can_dev(self) -> bool:
        """Whether ``dev`` can run. Only ``vite`` is required."""
        return self._has_local("vite") or bool(self.package_manager)

    def _has_local(self, tool: str) -> bool:
        if self.local_bin is None:
            return False
        candidate = self.local_bin / _LOCAL_TOOLS[tool]
        return candidate.is_file()

    def local(self, tool: str) -> Path | None:
        """Absolute path to a local tool, or ``None`` when it is not installed."""
        if self.local_bin is None:
            return None
        candidate = self.local_bin / _LOCAL_TOOLS[tool]
        return candidate if candidate.is_file() else None

    def build_steps(self) -> list[list[str]]:
        """Commands that type-check and bundle the console, in order.

        A *list* of argument vectors rather than one command: ``tsc -b`` and
        ``vite build`` are two processes, and joining them with ``&&`` would pass
        a literal ``&&`` argument to ``tsc`` unless the caller opted into a shell.
        The caller runs each step and stops at the first non-zero exit, which is
        what the ``&&`` in ``package.json``'s build script means.

        Prefers the local binaries, reproducing ``pnpm build`` without a package
        manager; falls back to the manager's own ``build`` script, which is the
        same two steps.
        """
        tsc = self.local("tsc")
        vite = self.local("vite")
        if tsc is not None and vite is not None:
            return [[str(tsc), "-b"], [str(vite), "build"]]
        if self.package_manager:
            return [[self.package_manager, "run", "build"]]
        raise JsToolchainError(self._missing_message("build"))

    def dev_argv(self) -> list[str]:
        """Command that starts the Vite dev server with HMR."""
        vite = self.local("vite")
        if vite is not None:
            return [str(vite)]
        if self.package_manager:
            return [self.package_manager, "run", "dev"]
        raise JsToolchainError(self._missing_message("dev"))

    def _missing_message(self, action: str) -> str:
        verb = "build" if action == "build" else "start the dev server for"
        return (
            f"Cannot {verb} the web console: no JavaScript toolchain found.\n"
            f"  looked for: {self.web_dir / 'node_modules' / '.bin'}"
            f"/{{{'|'.join(_LOCAL_TOOLS)}}}\n"
            f"  looked for a package manager on PATH: {', '.join(_PACKAGE_MANAGERS)}\n"
            "\n"
            "Fix it with either:\n"
            f"  cd {self.web_dir} && npm install     # or pnpm install\n"
            "or run the console without --dev, which serves the prebuilt bundle."
        )


class JsToolchainError(RuntimeError):
    """No usable JavaScript toolchain, with a message that says what to do."""


def detect_js_toolchain(web_dir: Path) -> JsToolchain:
    """Probe ``web_dir`` for a local toolchain, then ``PATH`` for a manager."""
    local_bin = web_dir / "node_modules" / ".bin"
    manager = next((m for m in _PACKAGE_MANAGERS if shutil.which(m)), None)
    return JsToolchain(
        web_dir=web_dir,
        local_bin=local_bin if local_bin.is_dir() else None,
        package_manager=manager,
    )

#!/usr/bin/env python3
"""Build the web console bundle and sync it into ``ubt/api/static``.

Run from the repository root: ``python scripts/build_console_assets.py``.

**No package manager is required when dependencies are installed.** The script
prefers ``web/node_modules/.bin/{tsc,vite}`` and only falls back to a package
manager (``pnpm`` → ``npm`` → ``yarn``) when those are absent. It used to run a
bare ``pnpm build``, so a machine with npm but no pnpm failed here — and because
``ubt console`` calls this script automatically when the bundle is missing, a
first-time ``ubt console`` failed too, reporting only "Failed to build static
assets" with the real cause buried.
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
DIST_DIR = WEB_DIR / "dist"
STATIC_DIR = REPO_ROOT / "ubt" / "api" / "static"

# Make ``ubt`` importable when this runs as a plain script from the repo root.
sys.path.insert(0, str(REPO_ROOT))

from ubt.cli.jstoolchain import JsToolchainError, detect_js_toolchain  # noqa: E402


def _run_steps(steps: list[list[str]]) -> None:
    """Run each command in order, stopping at the first failure.

    Separate processes rather than one ``&&``-joined string: ``&&`` is a shell
    operator, so a non-shell ``subprocess.run`` would pass it to ``tsc`` as an
    argument. Stopping on the first non-zero exit is the same fail-fast behaviour
    the ``&&`` in ``package.json``'s build script provides.
    """
    for argv in steps:
        print(f"      $ {' '.join(argv)}")
        result = subprocess.run(argv, cwd=WEB_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            if result.stdout:
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            sys.exit(result.returncode)


def main() -> None:
    print("[1/3] Building Web Console SPA...")
    toolchain = detect_js_toolchain(WEB_DIR)
    if toolchain.local("vite"):
        print("      using the local node_modules toolchain (no package manager)")
    else:
        print(f"      no local toolchain; using {toolchain.package_manager or 'none available'}")
    try:
        _run_steps(toolchain.build_steps())
    except JsToolchainError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print("      Build succeeded.")

    print(f"[2/3] Preparing static target: {STATIC_DIR}...")
    if STATIC_DIR.exists():
        shutil.rmtree(STATIC_DIR)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[3/3] Copying {DIST_DIR} -> {STATIC_DIR}...")
    shutil.copytree(DIST_DIR, STATIC_DIR, dirs_exist_ok=True)
    file_count = len(list(STATIC_DIR.rglob("*")))
    print(f"      Synced {file_count} files to {STATIC_DIR}.")
    print("Done! Web Console is ready to be served by FastAPI.")


if __name__ == "__main__":
    main()

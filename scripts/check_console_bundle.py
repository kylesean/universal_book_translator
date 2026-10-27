#!/usr/bin/env python3
"""Fail when the committed console bundle does not match the frontend source.

CI rebuilds the console and fails if ``ubt/api/static`` differs afterwards.
This is the local mirror of that check for the pre-push hook (see
``.pre-commit-config.yaml``): a push that touched ``web/`` or the sync script
could otherwise pass every local gate and still go red on CI. The rebuild
itself costs about a second, so it runs unconditionally rather than trying to
guess from mtimes whether it would be needed.

Exit codes: 0 the bundle matches; 1 it is stale (diff shown); 2 the build or
toolchain failed.
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = "ubt/api/static"


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True)


def _pnpm() -> str | None:
    return shutil.which("pnpm")


def main() -> int:
    pnpm = _pnpm()
    if pnpm:
        # Refresh node_modules from the lockfile when it changed; pnpm makes
        # this a sub-second no-op otherwise. A failure here (offline, unwritable
        # store) is an environment problem, not a stale bundle: warn and build
        # with what is installed. CI still runs the strict frozen-lockfile
        # install, so a lockfile-divergent bundle cannot reach main.
        install = _run([pnpm, "--dir", "web", "install", "--frozen-lockfile"])
        if install.returncode != 0:
            print(
                "warning: pnpm install --frozen-lockfile failed; building with the "
                "existing web/node_modules (CI installs strictly)",
                file=sys.stderr,
            )
    else:
        print(
            "note: pnpm not found; building with the existing web/node_modules "
            "(CI also installs from the lockfile)",
            file=sys.stderr,
        )

    build = _run([sys.executable, "scripts/build_console_assets.py"])
    if build.returncode != 0:
        sys.stderr.write(build.stdout + build.stderr)
        print("::error::console bundle build failed", file=sys.stderr)
        return 2

    # --porcelain also reports untracked/missing files, which `git diff`
    # ignores; a bundle that grew a file no source build produces is stale too.
    status = _run(["git", "status", "--porcelain", "--", STATIC_DIR])
    if status.returncode != 0:
        sys.stderr.write(status.stdout + status.stderr)
        return 2
    if status.stdout.strip():
        diff = _run(["git", "diff", "--stat", "--", STATIC_DIR])
        print(
            f"::error::{STATIC_DIR} is stale -- run 'pnpm --dir web build && "
            f"{sys.executable} scripts/build_console_assets.py' and commit the result.",
            file=sys.stderr,
        )
        print(status.stdout, file=sys.stderr)
        print(diff.stdout, file=sys.stderr)
        return 1
    print(f"console bundle matches source: {STATIC_DIR} is up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

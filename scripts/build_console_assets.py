#!/usr/bin/env python3
"""Build web frontend assets and sync to ubt/api/static directory."""

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
DIST_DIR = WEB_DIR / "dist"
STATIC_DIR = REPO_ROOT / "ubt" / "api" / "static"


def main() -> None:
    print("[1/3] Building Web Console SPA with Vite...")
    res = subprocess.run(["pnpm", "build"], cwd=WEB_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error during pnpm build:\n{res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)
    print("      Vite build succeeded.")

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

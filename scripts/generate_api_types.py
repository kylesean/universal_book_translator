#!/usr/bin/env python3
"""Export FastAPI OpenAPI schema to JSON and invoke openapi-typescript."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
SCHEMA_PATH = WEB_DIR / "openapi.json"
OUT_PATH = WEB_DIR / "src" / "api" / "generated-types.ts"


def main() -> None:
    print("[1/2] Generating OpenAPI schema from ubt.api.app...")
    from ubt.api.app import create_app

    app = create_app()
    schema = app.openapi()
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"      Saved to {SCHEMA_PATH} (endpoints: {len(schema.get('paths', {}))})")

    print("[2/2] Running openapi-typescript...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["npx", "openapi-typescript", str(SCHEMA_PATH), "-o", str(OUT_PATH)],
        cwd=WEB_DIR,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"Error: {res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)
    print(f"      Success! TypeScript types generated at {OUT_PATH}")


if __name__ == "__main__":
    main()

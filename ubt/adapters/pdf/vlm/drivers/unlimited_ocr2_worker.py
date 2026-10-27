"""Resident Unlimited-OCR-2 inference worker: llama.cpp behind a process wall.

Launched by ``UnlimitedOcr2Driver`` as ``python -m
ubt.adapters.pdf.vlm.drivers.unlimited_ocr2_worker``. The child starts a
patched ``llama-mtmd-cli`` binary once and then answers JSON-lines requests
``{"id": N, "image_b64": ...}`` with ``{"id": N, "text": ...}`` (or
``{"id": N, "error": ...}`` for a single failed page — one bad request must
not kill the session). stdin EOF ends the loop: when the driver dies the pipes
close, so this process is never orphaned.

Why a subprocess at all: the model runs native code that can segfault or OOM,
and the 2.92 GiB weights (Q8_0) must be releasable by killing one process
instead of taking the whole translation run down with them.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _worker_dir() -> Path:
    """The directory that carries model.gguf and mmproj.gguf."""
    raw = os.environ.get("UBT_UNLIMITED_OCR2_MODEL", "").strip()
    if not raw:
        raise RuntimeError(
            "UBT_UNLIMITED_OCR2_MODEL is unset: point it at the directory "
            "containing model.gguf and mmproj.gguf from the patched llama.cpp build"
        )
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"UBT_UNLIMITED_OCR2_MODEL is not a directory: {path}")
    model = path / "model.gguf"
    mmproj = path / "mmproj.gguf"
    if not model.is_file():
        raise RuntimeError(f"model.gguf not found in {path}")
    if not mmproj.is_file():
        raise RuntimeError(f"mmproj.gguf not found in {path}")
    return path


def _cli_binary() -> str:
    """The patched llama-mtmd-cli to run."""
    raw = os.environ.get("UBT_UNLIMITED_OCR2_CLI", "").strip()
    if raw:
        return raw
    raise RuntimeError(
        "UBT_UNLIMITED_OCR2_CLI is unset: point it at the patched "
        "llama-mtmd-cli binary (mainline llama.cpp cannot load the architecture)"
    )


def _infer_text(image_b64: str) -> str:
    """One page inference through a fresh llama-mtmd-cli process."""
    path = _worker_dir()
    cli = _cli_binary()
    prompt = os.environ.get("UBT_UNLIMITED_OCR2_PROMPT", "<PAGE><__media__>Multi page parsing.")
    ctx = os.environ.get("UBT_UNLIMITED_OCR2_CTX", "16384")
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "page.png"
        png.write_bytes(base64.b64decode(image_b64))
        result = subprocess.run(
            [
                cli,
                "-m",
                str(path / "model.gguf"),
                "--mmproj",
                str(path / "mmproj.gguf"),
                "--jinja",
                "--temp",
                "0",
                "-c",
                ctx,
                "-n",
                "4000",
                "--image",
                str(png),
                "-p",
                prompt,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=540,
            check=False,
        )
        if result.returncode != 0:
            stderr_tail = result.stderr.strip().splitlines()[-3:]
            raise RuntimeError(
                f"llama-mtmd-cli exited {result.returncode}: {' | '.join(stderr_tail)}"
            )
        return result.stdout


def main() -> int:
    """JSON-lines loop: one request per line, one reply per line."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id: Any = None
        try:
            request = json.loads(line)
            req_id = request.get("id")
            if request.get("cmd") == "quit":
                return 0
            text = _infer_text(str(request.get("image_b64", "")))
            reply: dict[str, Any] = {"id": req_id, "text": text}
        except Exception as exc:
            reply = {"id": req_id, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

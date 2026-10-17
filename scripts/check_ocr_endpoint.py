#!/usr/bin/env python3
"""Probe an OpenAI-compatible vision OCR endpoint with one real page.

A wrong endpoint used to surface only after a full translation: the run
finished, every scanned page stayed source, and nothing said why. This posts one
rendered page and checks four things in seconds:

1. the endpoint answers;
2. it returns text at all;
3. the text is not the prompt echoing back (a model that refuses vision commonly
   echoes its instruction, and an echo would be scored as a successful OCR);
4. it finished inside a sane time budget.

Exit code 1 on any failure, so it composes with a shell gate.

Usage::

    uv run python scripts/check_ocr_endpoint.py --pdf paper.pdf --page 4
    uv run python scripts/check_ocr_endpoint.py --pdf paper.pdf \
        --endpoint http://127.0.0.1:8185/v1/chat/completions \
        --model PaddlePaddle/PaddleOCR-VL-1.6
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

#: The same instruction the CloudOcrDriver's vision route sends. Kept in one
#: place so the probe exercises the request shape production actually uses.
VISION_PROMPT = "OCR:"

#: Fewest characters that can be a real transcription once the prompt is
#: stripped. A one-glyph answer is an echo with noise, not a page of text.
_MIN_USABLE_CHARS = 8

#: Matches the OCR prompt however many times a refusing model repeats it.
_ECHO_PROMPT_RE = re.compile(re.escape(VISION_PROMPT), re.IGNORECASE)

#: A single page should not take longer than this. A 40-page document at this
#: budget is ~20 minutes, which is the point at which the operator needs to know.
DEFAULT_TIMEOUT_S = 300


def _render_page(pdf: Path, page_no: int, dpi: int = 150) -> bytes:
    with tempfile.TemporaryDirectory(prefix="ubt-probe-") as tmp:
        stem = Path(tmp) / "page"
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                str(dpi),
                "-f",
                str(page_no),
                "-l",
                str(page_no),
                str(pdf),
                str(stem),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        produced = sorted(Path(tmp).glob("page*.png"))
        if not produced:
            raise RuntimeError(f"pdftoppm produced no image for page {page_no}")
        return produced[0].read_bytes()


def _post(endpoint: str, model: str, api_key: str, png: bytes, timeout: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")
                        },
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    try:
        return str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected response shape: {str(body)[:300]}") from exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pdf", type=Path, required=True, help="a PDF to take one page from")
    ap.add_argument("--page", type=int, default=1, help="1-based page number (default 1)")
    ap.add_argument(
        "--endpoint",
        default=os.environ.get("UBT_OCR_ENDPOINT", "http://127.0.0.1:8185/v1/chat/completions"),
        help="OpenAI-compatible /chat/completions endpoint",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("UBT_OCR_MODEL", "PaddlePaddle/PaddleOCR-VL-1.6"),
        help="model name the service was started with",
    )
    ap.add_argument(
        "--api-key",
        default=os.environ.get("UBT_OCR_API_KEY", "local-only"),
        help="bearer token (vLLM ignores it when no key is configured)",
    )
    ap.add_argument(
        "--timeout", type=float, default=float(os.environ.get("UBT_OCR_TIMEOUT", DEFAULT_TIMEOUT_S))
    )
    args = ap.parse_args(argv)

    if not args.pdf.exists():
        print(f"FAIL  no such file: {args.pdf}", file=sys.stderr)
        return 2

    print(f"endpoint : {args.endpoint}")
    print(f"model    : {args.model}")
    print(f"page     : {args.page} of {args.pdf.name}")

    try:
        png = _render_page(args.pdf, args.page)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f"FAIL  could not render the page: {exc}", file=sys.stderr)
        return 2
    print(f"raster   : {len(png) // 1024} KiB")

    started = time.monotonic()
    try:
        text = _post(args.endpoint, args.model, args.api_key, png, args.timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        print(f"FAIL  HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"FAIL  cannot reach the endpoint: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started

    print(f"latency  : {elapsed:.1f}s")
    print(f"chars    : {len(text)}")

    if not text.strip():
        print("FAIL  the endpoint returned no text", file=sys.stderr)
        return 1

    # An echo of the instruction is not a transcription. A model that refuses the
    # image commonly returns the prompt back, once or repeated. Judged by what
    # remains after stripping the prompt, not by total length: a three-times
    # repeated "OCR: OCR: OCR:" cleared the old length bound and was scored as
    # usable text (caught by this file's own test).
    residue = _ECHO_PROMPT_RE.sub("", text).strip()
    if len(residue) < _MIN_USABLE_CHARS:
        print(
            f"FAIL  the response is the prompt echoing back, not a transcription: {text[:120]!r}",
            file=sys.stderr,
        )
        return 1

    print("\n--- first 400 chars ---")
    print(text[:400])
    print("\nOK    the endpoint returned usable text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

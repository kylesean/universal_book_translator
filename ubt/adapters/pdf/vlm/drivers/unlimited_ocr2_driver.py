"""Baidu Unlimited-OCR-2 driver: page-image → ordered text, no geometry.

The model runs behind a patched llama.cpp (mainline llama.cpp rejects the
architecture), so this driver speaks JSON-lines to a resident subprocess that
owns the llama-mtmd-cli lifecycle. A crash inside native code kills only the
worker, never the translation run; the parent restarts it once per page before
surfacing the failure.

Like every LLM-VLM driver, ``measured_boxes=False``: the model emits text in
reading order and no trustworthy coordinates. The anchoring layer refuses it
in recognition mode (a true scan cannot be transcribed by a text-only
witness) and uses it in proofread mode (pages that already have a text layer,
where the driver's job is cross-checking the stronger pdfium witness).

VRAM budget: the Q8_0 GGUF is 2.92 GiB and the mmproj 0.77 GiB; ~6 GiB free
is the practical floor once compute buffers land. A Q4_K_M build fits a
smaller card. The worker refuses to start below the floor with an actionable
message instead of OOMing mid-page.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import os
import selectors
import subprocess
import sys
import threading
import time
from typing import Any

from ubt.adapters.pdf.vlm.types import PageTranscript, VlmLine

logger = logging.getLogger(__name__)


class UnlimitedOcr2Driver:
    """Unlimited-OCR-2 behind the VlmDriver contract (proofread-only).

    The llama.cpp runtime lives in ``unlimited_ocr2_worker`` (a resident
    subprocess speaking JSON-lines over stdio): a native crash or OOM inside
    inference kills only the worker, the weights are released by killing one
    process, and a dead worker is restarted once per page before the failure
    surfaces. ``worker_cmd`` is injectable for tests.
    """

    name = "unlimited-ocr2"
    measured_boxes = False
    #: A page inference after a warm model load; the first request also pays
    #: the initial import, so the bound is generous.
    REQUEST_TIMEOUT_S = 600.0

    def __init__(self, worker_cmd: list[str] | None = None) -> None:
        self._worker_cmd = worker_cmd or [
            sys.executable,
            "-m",
            "ubt.adapters.pdf.vlm.drivers.unlimited_ocr2_worker",
        ]
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_buffer = bytearray()
        self._req_seq = 0

    def close(self) -> None:
        """Terminate the worker subprocess immediately and release GPU VRAM."""
        self._close_worker(kill_now=True)

    def __del__(self) -> None:
        self.close()

    def _ensure_loaded(self) -> subprocess.Popen[str]:
        """Start (or return the live) worker process; gate the binary first."""
        model_path = os.environ.get("UBT_UNLIMITED_OCR2_MODEL", "").strip()
        if not model_path:
            raise RuntimeError(
                "unlimited-ocr2 needs the path to a patched-llama.cpp GGUF model "
                "directory: set UBT_UNLIMITED_OCR2_MODEL to the directory "
                "containing model.gguf and mmproj.gguf (see docs/guides/"
                "OCR_DRIVERS.md), or choose a different driver (rapidocr, sidecar, vlm)."
            )
        from ubt.core.env import subprocess_env

        proc = self._proc
        if proc is not None and proc.poll() is None:
            return proc
        self._close_worker()
        try:
            proc = subprocess.Popen(
                self._worker_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=subprocess_env(),
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start unlimited-ocr2 worker: {exc}") from exc
        self._proc = proc
        return proc

    def _close_worker(self, kill_now: bool = False) -> None:
        """Tear the worker down; ``kill_now`` for sessions that cannot talk."""
        proc = self._proc
        self._proc = None
        self._stdout_buffer.clear()
        if proc is None:
            return
        try:
            if kill_now:
                with contextlib.suppress(OSError):
                    proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                    proc.wait(timeout=5)
                return
            with contextlib.suppress(OSError):
                if proc.stdin is not None and proc.poll() is None:
                    proc.stdin.write('{"cmd": "quit"}\n')
                    proc.stdin.flush()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                proc.wait(timeout=10)
            if proc.poll() is None:
                with contextlib.suppress(OSError):
                    proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                    proc.wait(timeout=5)
        finally:
            for pipe in (proc.stdin, proc.stdout):
                if pipe is not None:
                    with contextlib.suppress(OSError, ValueError):
                        pipe.close()

    def _write_line(self, proc: subprocess.Popen[str], line: str, deadline: float) -> None:
        """Write one request line under the shared deadline."""
        error: list[BaseException] = []

        def _writer() -> None:
            try:
                proc.stdin.write(line)  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
            except BaseException as exc:
                error.append(exc)

        thread = threading.Thread(target=_writer, daemon=True)
        thread.start()
        thread.join(max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            raise TimeoutError("unlimited-ocr2 worker stopped draining stdin")
        if error:
            raise error[0]

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One JSON-lines round trip; restart the worker once per request."""
        for attempt in (1, 2):
            proc = self._ensure_loaded()
            self._req_seq += 1
            req_id = self._req_seq
            if proc.stdin is None or proc.stdout is None:
                raise RuntimeError("unlimited-ocr2 worker pipes unavailable")
            deadline = time.monotonic() + self.REQUEST_TIMEOUT_S
            try:
                self._write_line(
                    proc,
                    json.dumps({"id": req_id, **payload}, ensure_ascii=False) + "\n",
                    deadline,
                )
            except TimeoutError:
                if attempt == 2:
                    self._close_worker(kill_now=True)
                    raise RuntimeError("unlimited-ocr2 worker pipe died or wedged") from None
                self._close_worker(kill_now=True)
                continue
            except OSError:
                if attempt == 2:
                    self._close_worker(kill_now=True)
                    raise RuntimeError("unlimited-ocr2 worker pipe died or wedged") from None
                self._close_worker()
                continue
            line = ""
            selector = selectors.DefaultSelector()
            try:
                fd = proc.stdout.fileno()
                os.set_blocking(fd, False)
                selector.register(fd, selectors.EVENT_READ)
                while b"\n" not in self._stdout_buffer:
                    remaining = max(0.0, deadline - time.monotonic())
                    if remaining <= 0 or not selector.select(remaining):
                        logger.warning(
                            "unlimited-ocr2 worker timed out after %.0fs",
                            self.REQUEST_TIMEOUT_S,
                        )
                        break
                    try:
                        chunk = os.read(fd, 64 * 1024)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        break
                    self._stdout_buffer.extend(chunk)
                if b"\n" in self._stdout_buffer:
                    raw_line, _, remainder = self._stdout_buffer.partition(b"\n")
                    self._stdout_buffer = bytearray(remainder)
                    line = raw_line.decode("utf-8", errors="replace")
            except (OSError, ValueError):
                line = ""
            finally:
                selector.close()
            if not line:
                if attempt == 2:
                    raise RuntimeError("unlimited-ocr2 worker crashed or timed out serving a page")
                self._close_worker()
                continue
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                if attempt == 2:
                    raise RuntimeError("unlimited-ocr2 worker sent an unparseable reply") from None
                self._close_worker()
                continue
            if not isinstance(reply, dict) or reply.get("id") != req_id:
                if attempt == 2:
                    raise RuntimeError("unlimited-ocr2 worker reply desynced")
                self._close_worker()
                continue
            if "error" in reply:
                raise RuntimeError(f"unlimited-ocr2 page inference failed: {reply['error']}")
            return reply
        raise AssertionError("unreachable")

    def recognize(
        self,
        image: Any,
        page_size_pt: tuple[float, float],
        scale: float,
        rotation: int = 0,
        origin_pt: tuple[float, float] = (0.0, 0.0),
    ) -> PageTranscript:
        """Text-only witness: the model reads the page, anchoring owns geometry."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        reply = self._request({"image_b64": base64.b64encode(buf.getvalue()).decode("ascii")})
        text = str(reply.get("text", ""))
        lines = tuple(VlmLine(text=ln.strip()) for ln in text.splitlines() if ln.strip())
        return PageTranscript(lines=lines, engine=self.name, measured_boxes=False)

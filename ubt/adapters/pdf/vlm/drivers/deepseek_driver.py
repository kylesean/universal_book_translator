"""DeepSeek-OCR driver: VLM second witness (proofread grade). STATUS: experimental.

Registered as ``deepseek-ocr`` in registry.py (explicit selection only — the
auto chain prefers rapidocr/sidecar/cloud). The generate() 4d-mask crash
described below is worked around in-process by the two compat shims
(``_install_llama_flash_compat_shim`` / ``_install_cache_compat_shim``);
the historical failure mode is kept here because it defines the trust
boundary: any mask misalignment risks SILENTLY WRONG witness output —
worse than absence. If DeepSeek output ever disagrees with rapidocr,
suspect this path first.

DeepSeek-OCR emits MARKDOWN/text with no trustworthy geometry, so
``measured_boxes=False``: recognition mode (textless pages) refuses it and
rapidocr remains the only geometry source. Its value is as a SECOND
witness on text-having pages — cross-engine disagreement feeds the MQM
sampler — plus a future table-markdown upgrade path.

GPU orchestration (operator duty, enforced by a fail-closed guard):
the 6.7 GB weights need ~8 GB free VRAM, but ollama residents hold ~5 GB.
Stop them first (``ollama stop <models>``), run, restart. The driver
refuses to load with an actionable error instead of OOMing mid-page.
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

MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
FREE_OCR_PROMPT = "<image>\nFree OCR."
# 6.7 GB weights + ~1 GB runtime. 9 GB comfortable, 8 GB floor (OOM below
# learns the hard way — the error tells the operator exactly what to free).
MIN_FREE_BYTES = 8 * 1024**3


def _install_llama_flash_compat_shim() -> None:
    """Alias the removed ``LlamaFlashAttention2`` to eager attention.

    DeepSeek-OCR's modeling file targets transformers~=4.46, whose llama
    module still exported ``LlamaFlashAttention2``; current transformers
    removed it (flash moved to backend dispatch). The alias keeps the class
    hierarchy mechanically intact (subclassing still works); attention runs
    the eager path instead of flash kernels — numerically the same ops,
    slower, more memory. Inference-only witness use: outputs are audited
    against pdfium/rapidocr downstream, never trusted blindly. Revisit when
    DeepSeek refreshes their modeling file. No-op on old transformers.
    """
    try:
        from transformers.models.llama import modeling_llama as llama_mod
    except ImportError:
        return
    if not hasattr(llama_mod, "LlamaFlashAttention2"):
        setattr(llama_mod, "LlamaFlashAttention2", getattr(llama_mod, "LlamaAttention", None))


def _install_cache_compat_shim() -> None:
    """Restore ``DynamicCache.seen_tokens`` removed after transformers 4.46.

    Same value as ``get_seq_length()`` (used one line above the breakage in
    modeling_deepseekocr.py); read-only property, no behavior change beyond
    un-breaking generate(). No-op when the attribute exists.
    """
    try:
        from transformers import DynamicCache
    except ImportError:
        return
    if not hasattr(DynamicCache, "seen_tokens"):

        def _seen(self: Any) -> int:
            return int(self.get_seq_length())

        setattr(DynamicCache, "seen_tokens", property(_seen))
    if not hasattr(DynamicCache, "get_max_length"):
        # 4.57 caches grow unboundedly: None disables the mask-cropping
        # branch, which is exactly correct when there is no maximum.

        def _maxlen(self: object) -> None:
            return None

        setattr(DynamicCache, "get_max_length", _maxlen)
    if not hasattr(DynamicCache, "get_usable_length"):
        # Old semantics: cached length + incoming length (no sliding
        # window in the unbounded cache: the sum is always usable).

        def _usable(self: Any, new_seq_length: int) -> int:
            return int(self.get_seq_length()) + int(new_seq_length)

        setattr(DynamicCache, "get_usable_length", _usable)


class DeepSeekOcrDriver:
    """DeepSeek-OCR behind the VlmDriver contract (proofread-only).

    The torch runtime lives in ``deepseek_worker`` (a resident subprocess
    speaking JSON-lines over stdio): a CUDA segfault or OOM inside
    ``model.infer`` kills only the worker, the 6.7 GB weights are released by
    killing one process, and a dead worker is restarted once per page before
    the failure surfaces. ``worker_cmd`` is injectable for tests.
    """

    name = "deepseek-ocr"
    measured_boxes = False
    #: A page inference after a warm model load; the first request also pays
    #: any remaining import cost, so the bound is generous.
    REQUEST_TIMEOUT_S = 600.0

    def __init__(self, worker_cmd: list[str] | None = None) -> None:
        self._worker_cmd = worker_cmd or [
            sys.executable,
            "-m",
            "ubt.adapters.pdf.vlm.drivers.deepseek_worker",
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
        """Start (or return the live) worker process; gate model-code first."""
        from ubt.core.config import UBTConfig

        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"deepseek-ocr requires torch and transformers: {exc}. "
                "Install them or choose another driver (rapidocr, sidecar)."
            ) from exc

        if not UBTConfig.from_env().vlm_trust_remote_code:
            # Fail closed before the worker (and its snapshot_download of
            # executable model code) ever starts: this checkpoint's custom
            # architecture only loads via trust_remote_code.
            raise RuntimeError(
                "deepseek-ocr needs trust_remote_code to load its custom "
                "architecture, and UBT_VLM_TRUST_REMOTE_CODE=false refuses "
                "model-supplied code. Restore a vetted snapshot under HF_HOME "
                "and unset the flag, or choose a local engine (rapidocr)."
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
                # stderr inherited: HF download progress and the worker's
                # actionable load-failure guidance must reach the operator.
                stderr=None,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=subprocess_env(),
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start deepseek-ocr worker: {exc}") from exc
        self._proc = proc
        return proc

    def _close_worker(self, kill_now: bool = False) -> None:
        """Tear the worker down; ``kill_now`` for sessions that cannot talk.

        A wedged worker stops draining stdin and may already have a full
        pipe — the graceful ``quit`` line (and any flush of it) would then
        block right here, so an unresponsive session is killed outright.
        """
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
                # TimeoutExpired is not an OSError: suppressing only OSError let a
                # worker wedged in a native CUDA call raise out of here *after*
                # self._proc was already cleared, orphaning it while the next page
                # spawned a second 6.7 GB load.
                with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                    proc.wait(timeout=5)
        finally:
            # The child's stdin/stdout are OS pipes; terminate/wait reaps the
            # process but leaves the TextIOWrappers open, leaking two fds per
            # teardown (and a GC'd-transport ResourceWarning). Close them here,
            # on every path including the kill_now early return.
            for pipe in (proc.stdin, proc.stdout):
                if pipe is not None:
                    with contextlib.suppress(OSError, ValueError):
                        pipe.close()

    def _write_line(self, proc: subprocess.Popen[str], line: str, deadline: float) -> None:
        """Write one request line under the shared deadline.

        A wedged-but-alive worker (the CUDA/native hangs this subprocess
        wall exists for) stops draining stdin; a multi-MB base64 page then
        blocks inside write() once the 64 KB pipe fills, and a stdout-only
        timeout could never fire. The daemon thread is freed by the
        terminate/kill in _close_worker.
        """
        error: list[BaseException] = []

        def _writer() -> None:
            try:
                proc.stdin.write(line)  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
            except BaseException as exc:  # surfaced to the caller
                error.append(exc)

        thread = threading.Thread(target=_writer, daemon=True)
        thread.start()
        thread.join(max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            raise TimeoutError("deepseek-ocr worker stopped draining stdin")
        if error:
            raise error[0]

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One JSON-lines round trip; restart the worker once per request.

        Protocol desync (unparseable line, stale id) and pipe deaths mean the
        session is untrustworthy: tear down and retry on a fresh worker. A
        second failure raises -- silently respawning per page would hide a
        broken environment behind endless slow retries.
        """
        for attempt in (1, 2):
            proc = self._ensure_loaded()
            self._req_seq += 1
            req_id = self._req_seq
            if proc.stdin is None or proc.stdout is None:
                raise RuntimeError("deepseek-ocr worker pipes unavailable")
            # Armed before the write: a hung worker must time out on the way
            # IN (full pipe), not only while waiting for the reply.
            deadline = time.monotonic() + self.REQUEST_TIMEOUT_S
            try:
                self._write_line(
                    proc,
                    json.dumps({"id": req_id, **payload}, ensure_ascii=False) + "\n",
                    deadline,
                )
            except TimeoutError:
                # Wedged child: graceful shutdown would block on the same
                # full pipe, so kill it (before respawning, and before the
                # give-up raise so no wedged resident is left behind).
                if attempt == 2:
                    self._close_worker(kill_now=True)
                    raise RuntimeError("deepseek-ocr worker pipe died or wedged") from None
                self._close_worker(kill_now=True)
                continue
            except OSError:
                if attempt == 2:
                    self._close_worker(kill_now=True)
                    raise RuntimeError("deepseek-ocr worker pipe died or wedged") from None
                self._close_worker()
                continue
            line = ""
            selector = selectors.DefaultSelector()
            try:
                fd = proc.stdout.fileno()
                # ``readline`` on the text wrapper is not bounded by selector
                # readiness: a worker can write a partial JSON line and then
                # wedge, leaving readline blocked forever. Read the raw fd in
                # non-blocking chunks under the same request deadline.
                os.set_blocking(fd, False)
                selector.register(fd, selectors.EVENT_READ)
                while b"\n" not in self._stdout_buffer:
                    remaining = max(0.0, deadline - time.monotonic())
                    if remaining <= 0 or not selector.select(remaining):
                        logger.warning(
                            "deepseek-ocr worker timed out after %.0fs",
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
                    raise RuntimeError("deepseek-ocr worker crashed or timed out serving a page")
                self._close_worker()
                continue
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                if attempt == 2:
                    raise RuntimeError("deepseek-ocr worker sent an unparseable reply") from None
                self._close_worker()
                continue
            if not isinstance(reply, dict) or reply.get("id") != req_id:
                if attempt == 2:
                    raise RuntimeError("deepseek-ocr worker reply desynced")
                self._close_worker()
                continue
            if "error" in reply:
                # Session healthy (the envelope decoded and matched); this
                # page's inference failed. Surface it, keep the worker up.
                raise RuntimeError(f"deepseek-ocr page inference failed: {reply['error']}")
            return reply
        raise AssertionError("unreachable")

    def recognize(
        self,
        image: Any,
        page_size_pt: tuple[float, float],
        scale: float,
        rotation: int = 0,
    ) -> PageTranscript:
        # ``page_size_pt``, ``scale`` and ``rotation`` are geometry inputs, and
        # this driver measures no geometry: it emits text with no boxes, and
        # anchoring refuses it in recognition mode rather than guessing a page
        # of boxes out of a language model.
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        reply = self._request({"image_b64": base64.b64encode(buf.getvalue()).decode("ascii")})
        text = str(reply.get("text", ""))
        lines = tuple(
            VlmLine(text=ln.strip(), reading_index=i)
            for i, ln in enumerate(text.splitlines())
            if ln.strip()
        )
        return PageTranscript(lines=lines, engine=self.name, measured_boxes=False)

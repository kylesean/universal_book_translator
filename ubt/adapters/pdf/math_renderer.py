"""Proven-engine display-math rendering.

The home-grown LaTeX->Typst converter is replaced, for display formulas, by a
real typesetting engine: MathJax (the same TeX implementation used across
publishing and the web) renders the OCR LaTeX to SVG, which Typst embeds as a
vector image. Rendering is delegated, verification stays ours: the same
response carries a rasterization of the SVG so the existing formula witness
can compare it against the source crop, and any failure degrades to the source
graphic instead of shipping a converted guess.

Dependency discipline: Node and the pinned ``scripts/mathjax`` packages are an
optional extra. ``available`` is a pure probe; when it is False the caller
keeps the legacy Typst path, so a machine without Node behaves exactly as
before this module existed.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ubt.core.env import subprocess_env
from ubt.core.fs_perms import restrict_dir_to_owner

logger = logging.getLogger(__name__)

# Repo-root relative script; resolved from this file so editable installs work.
_SCRIPT_REL = Path("scripts") / "mathjax" / "render.mjs"
_NODE_MODULES = ("mathjax-full",)
_RENDER_TIMEOUT_S = 60.0
_PROBE_TIMEOUT_S = 20.0


# Cache directory for SVG/PNG artifacts; deterministic names keep repeated
# runs free and let _stage_image_assets copy stable references. Lives under
# the per-user cache root (the same ``UBT_CACHE_DIR`` convention svg_diagram's
# probe cache uses, with the XDG default) — never a predictable world-writable
# /tmp path, which other users on a shared machine could pre-create or
# symlink. Still cross-process, so one run's renders serve the next.
def _ubt_cache_root() -> Path:
    env = os.environ.get("UBT_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "ubt"
    return Path.home() / ".cache" / "ubt"


MATH_CACHE_DIR = _ubt_cache_root() / "math_svg"


@dataclass(frozen=True)
class MathRender:
    """One engine render: SVG vector plus an optional rasterization."""

    ok: bool
    svg: str | None = None
    png: bytes | None = None
    width: float | None = None
    height: float | None = None
    error: str | None = None


def _repo_root() -> Path:
    # ubt/adapters/pdf/math_renderer.py -> repository root
    return Path(__file__).resolve().parents[3]


class MathjaxRenderer:
    """Persistent Node renderer speaking newline-delimited JSON.

    One process per reconstructor (threaded export), one request at a time,
    restart-on-failure. Never raises: every failure is an ``ok=False`` result.
    """

    def __init__(
        self,
        node_binary: str | None = None,
        script: Path | None = None,
        timeout_s: float = _RENDER_TIMEOUT_S,
    ) -> None:
        self.node_binary = node_binary or shutil.which("node") or "node"
        self.script = script or (_repo_root() / _SCRIPT_REL)
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._version: str | None = None
        self._available: bool | None = None
        self._cache_version: str | None = None

    # -- lifecycle ---------------------------------------------------------
    def available(self) -> bool:
        """True only when Node, the script and its pinned deps all exist."""
        if self._available is not None:
            return self._available
        ok = (
            Path(self.node_binary).exists() or shutil.which(self.node_binary) is not None
        ) and self.script.is_file()
        if ok:
            for dep in _NODE_MODULES:
                if not (self.script.parent / "node_modules" / dep).exists():
                    # The local pinned install is a hard requirement: the
                    # script's require() does not resolve a global
                    # mathjax-full, so availability (and the version probe
                    # behind it) stays False without it, falling back to native Typst math.
                    ok = False
                    break
        self._available = ok
        return ok

    def version(self) -> str | None:
        """Pinned MathJax version, or None when the probe cannot run."""
        if self._version is not None:
            return self._version
        if not self.available():
            return None
        with self._lock:
            proc = self._ensure_proc_locked()
            if proc is None:
                return None
            reply = self._request_locked({"cmd": "probe"}, _PROBE_TIMEOUT_S)
        if reply and reply.get("ok"):
            self._version = str(reply.get("mathjax") or "unknown")
        return self._version

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    # -- rendering ---------------------------------------------------------
    def render(
        self,
        latex: str,
        *,
        tag: str | None = None,
        display: bool = True,
        png_width: int | None = None,
    ) -> MathRender:
        """Render LaTeX to SVG (and PNG when ``png_width`` is given)."""
        text = (latex or "").strip()
        if not text:
            return MathRender(ok=False, error="empty latex")
        if not self.available():
            return MathRender(ok=False, error="mathjax renderer unavailable")

        cached = self._read_cache(text, tag, png_width, display)
        if cached is not None:
            return cached

        request: dict[str, object] = {
            "id": _cache_key(text, tag, png_width, display, self._cache_version_key()),
            "latex": text,
        }
        if tag:
            request["tag"] = tag
        request["display"] = display
        if png_width:
            request["png_width"] = int(png_width)

        with self._lock:
            proc = self._ensure_proc_locked()
            if proc is None:
                return MathRender(ok=False, error="node renderer failed to start")
            reply = self._request_locked(request, self.timeout_s)
        if reply is None:
            return MathRender(ok=False, error="renderer timeout or crash")
        if not reply.get("ok"):
            return MathRender(ok=False, error=str(reply.get("error") or "render failed"))

        png_b64 = reply.get("png_b64")
        result = MathRender(
            ok=True,
            svg=reply.get("svg") or None,
            png=base64.b64decode(png_b64) if isinstance(png_b64, str) and png_b64 else None,
            width=_as_float(reply.get("width")),
            height=_as_float(reply.get("height")),
        )
        self._write_cache(text, tag, png_width, display, result)
        return result

    # -- internals ---------------------------------------------------------
    def _ensure_proc_locked(self) -> subprocess.Popen[str] | None:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        try:
            self._proc = subprocess.Popen(
                [self.node_binary, str(self.script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                # H-2: node needs PATH/HOME; it has no use for the LLM keys.
                env=subprocess_env(),
            )
        except OSError as exc:
            logger.warning("MathJax renderer could not start: %s", exc)
            self._proc = None
            self._available = False
        return self._proc

    def _request_locked(
        self, request: dict[str, object], timeout_s: float
    ) -> dict[str, Any] | None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            return None
        try:
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except OSError:
            self._close_locked()
            return None

        # A selector only reports *readiness*; ``readline()`` itself still blocks
        # until a newline arrives, so a Node worker that writes a partial line
        # and then wedges (OOM/native hang) blocks this thread forever and the
        # documented deadline never fires. Bound the read with a worker thread;
        # ``_close_locked`` closes the pipe and lets the daemon thread finish.
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        line_box: list[str | None] = [None]
        error_box: list[BaseException] = []

        def _read_line() -> None:
            try:
                assert proc.stdout is not None
                line_box[0] = proc.stdout.readline()
            except BaseException as read_exc:  # noqa: BLE001 - surfaced below
                error_box.append(read_exc)

        reader = threading.Thread(target=_read_line, daemon=True, name="mathjax-read")
        reader.start()
        reader.join(max(0.0, deadline - time.monotonic()))
        if reader.is_alive():
            logger.warning("MathJax renderer timed out after %.0fs", timeout_s)
            self._close_locked()
            return None
        if error_box or not line_box[0]:
            self._close_locked()
            return None
        line = line_box[0]

        if not line:
            self._close_locked()
            return None
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("MathJax renderer returned non-JSON output")
            # Reap the process: leaving it alive means the next request reads
            # this same stale/garbled line, permanently desyncing request from
            # reply until an unrelated timeout trips.
            self._close_locked()
            return None
        return parsed if isinstance(parsed, dict) else None

    def _close_locked(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        for pipe in (proc.stdin, proc.stdout):
            try:
                if pipe is not None:
                    pipe.close()
            except OSError:
                pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            with contextlib.suppress(OSError):
                proc.kill()

    # -- cache -------------------------------------------------------------
    def _cache_version_key(self) -> str:
        """Version token for the on-disk cache, probed once per renderer.

        The pinned MathJax version is part of the key, so a dependency upgrade
        cannot serve stale SVGs from the shared cache directory. The probe is a
        single round trip per renderer instance; later calls reuse it.
        """
        if self._cache_version is None:
            self._cache_version = self.version() or "unknown"
        return self._cache_version

    @staticmethod
    def _cache_paths(key: str) -> tuple[Path, Path, Path]:
        return (
            MATH_CACHE_DIR / f"{key}.json",
            MATH_CACHE_DIR / f"{key}.svg",
            MATH_CACHE_DIR / f"{key}.png",
        )

    def _read_cache(
        self, latex: str, tag: str | None, png_width: int | None, display: bool = True
    ) -> MathRender | None:
        key = _cache_key(latex, tag, png_width, display, self._cache_version_key())
        meta, svg_path, png_path = self._cache_paths(key)
        try:
            if not meta.is_file() or not svg_path.is_file():
                return None
            info = json.loads(meta.read_text(encoding="utf-8"))
            png = png_path.read_bytes() if png_path.is_file() else None
            return MathRender(
                ok=True,
                svg=svg_path.read_text(encoding="utf-8"),
                png=png,
                width=_as_float(info.get("width")),
                height=_as_float(info.get("height")),
            )
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(
        self,
        latex: str,
        tag: str | None,
        png_width: int | None,
        display: bool,
        result: MathRender,
    ) -> None:
        if not result.ok or not result.svg:
            return
        key = _cache_key(latex, tag, png_width, display, self._cache_version_key())
        meta, svg_path, png_path = self._cache_paths(key)
        try:
            restrict_dir_to_owner(MATH_CACHE_DIR)
            # Atomic per file: a concurrent reader otherwise sees a half-written
            # SVG (empty/truncated) and embeds it in the PDF.
            _atomic_write(svg_path, result.svg.encode("utf-8"))
            if result.png is not None:
                _atomic_write(png_path, result.png)
            _atomic_write(
                meta,
                json.dumps({"width": result.width, "height": result.height}).encode("utf-8"),
            )
        except OSError as exc:
            logger.debug("MathJax cache write skipped: %s", exc)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file + rename (crash/concurrency safe)."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def _cache_key(
    latex: str,
    tag: str | None,
    png_width: int | None,
    display: bool = True,
    version: str = "unknown",
) -> str:
    payload = f"v4\x1f{version}\x1f{display}\x1f{latex}\x1f{tag or ''}\x1f{png_width or 0}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

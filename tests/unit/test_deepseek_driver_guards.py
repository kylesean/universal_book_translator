"""DeepSeek-OCR driver guards: compat shims + the trust_remote_code gate.

The shims patch ``transformers`` module attributes at load time; without GPU
or torch they are pure attribute surgery, so fake modules in ``sys.modules``
cover exactly what they do — and what they must NOT do when the library is
absent or newer than the breakage.
"""

import sys
import types
from pathlib import Path

import pytest

from ubt.adapters.pdf.vlm.drivers.deepseek_driver import (
    DeepSeekOcrDriver,
    _install_cache_compat_shim,
    _install_llama_flash_compat_shim,
)


class _FakeDynamicCache:
    def get_seq_length(self) -> int:
        return 7


def _fake_transformers(
    monkeypatch: pytest.MonkeyPatch, *, with_cache: bool = True
) -> types.ModuleType:
    transformers = types.ModuleType("transformers")
    if with_cache:
        transformers.DynamicCache = _FakeDynamicCache  # type: ignore[attr-defined]
    models = types.ModuleType("transformers.models")
    llama_pkg = types.ModuleType("transformers.models.llama")
    modeling = types.ModuleType("transformers.models.llama.modeling_llama")

    class LlamaAttention:
        pass

    modeling.LlamaAttention = LlamaAttention  # type: ignore[attr-defined]
    llama_pkg.modeling_llama = modeling  # type: ignore[attr-defined]
    models.llama = llama_pkg  # type: ignore[attr-defined]
    transformers.models = models  # type: ignore[attr-defined]
    for name, module in (
        ("transformers", transformers),
        ("transformers.models", models),
        ("transformers.models.llama", llama_pkg),
        ("transformers.models.llama.modeling_llama", modeling),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return transformers


def _fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``torch``/``transformers`` so ``_ensure_loaded``'s import gate passes.

    The worker protocol is the subject of these tests; the two imports are
    existence checks (``# noqa: F401``) before the subprocess starts, so a stub
    is the honest boundary and keeps the cases running without the ~2 GB torch
    stack — which the ``dev`` extra intentionally omits, so a real
    ``importorskip`` would silently drop the worker coverage from CI.
    """
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))


def test_llama_flash_shim_aliases_and_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_transformers(monkeypatch)
    modeling = sys.modules["transformers.models.llama.modeling_llama"]
    _install_llama_flash_compat_shim()
    assert modeling.LlamaFlashAttention2 is modeling.LlamaAttention
    sentinel = modeling.LlamaFlashAttention2
    _install_llama_flash_compat_shim()  # second run must not re-alias
    assert modeling.LlamaFlashAttention2 is sentinel


def test_llama_flash_shim_keeps_existing_class(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_transformers(monkeypatch)
    modeling = sys.modules["transformers.models.llama.modeling_llama"]

    class RealFlash:
        pass

    modeling.LlamaFlashAttention2 = RealFlash  # type: ignore[attr-defined]
    _install_llama_flash_compat_shim()
    assert modeling.LlamaFlashAttention2 is RealFlash


def test_llama_flash_shim_is_noop_without_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "transformers",
        "transformers.models",
        "transformers.models.llama",
        "transformers.models.llama.modeling_llama",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    _install_llama_flash_compat_shim()  # must not raise


def test_cache_shim_restores_removed_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_transformers(monkeypatch)
    cache = _FakeDynamicCache()
    assert not hasattr(_FakeDynamicCache, "seen_tokens")
    _install_cache_compat_shim()
    assert cache.seen_tokens == 7  # type: ignore[attr-defined]
    assert cache.get_max_length() is None  # type: ignore[attr-defined]
    assert cache.get_usable_length(3) == 10  # type: ignore[attr-defined]
    seen_prop = _FakeDynamicCache.seen_tokens  # type: ignore[attr-defined]
    _install_cache_compat_shim()  # idempotent: property object unchanged
    assert _FakeDynamicCache.seen_tokens is seen_prop  # type: ignore[attr-defined]


def test_cache_shim_is_noop_without_dynamic_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_transformers(monkeypatch, with_cache=False)
    _install_cache_compat_shim()  # must not raise


def test_trust_remote_code_gate_refuses_model_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UBT_VLM_TRUST_REMOTE_CODE=false must fail before any download/import."""
    monkeypatch.setenv("UBT_VLM_TRUST_REMOTE_CODE", "false")
    _fake_runtime(monkeypatch)
    driver = DeepSeekOcrDriver()
    with pytest.raises(RuntimeError, match="trust_remote_code"):
        driver._ensure_loaded()
    # The refusal happens before the worker (and its snapshot_download of
    # model code) ever starts: nothing is fetched, no process is spawned.
    assert driver._proc is None


# --- worker-client behavior (process wall around the torch runtime) ---------

_FAKE_WORKER = """
import json, sys, pathlib
marker = pathlib.Path(sys.argv[1])
if not marker.exists():
    marker.write_text("first")
    sys.exit(1)  # simulate a CUDA crash at load, before any reply
for line in sys.stdin:
    req = json.loads(line)
    if req.get("cmd") == "quit":
        break
    sys.stdout.write(json.dumps({"id": req.get("id"), "text": "line one\\nline two"}) + "\\n")
    sys.stdout.flush()
"""

_FAKE_OK = """
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    if req.get("cmd") == "quit":
        break
    sys.stdout.write(json.dumps({"id": req.get("id"), "text": "alpha\\nbeta\\n"}) + "\\n")
    sys.stdout.flush()
"""

_FAKE_DESYNC = """
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    sys.stdout.write(json.dumps({"id": -4242, "text": "stale"}) + "\\n")
    sys.stdout.flush()
"""


class _FakeImage:
    def save(self, fp: object, format: str | None = None) -> None:  # noqa: A002
        assert hasattr(fp, "write")
        fp.write(b"png-bytes")


def _driver(
    tmp_path: Path, script_name: str, body: str, extra_args: list[str] | None = None
) -> DeepSeekOcrDriver:
    script = tmp_path / script_name
    script.write_text(body, encoding="utf-8")
    cmd = [sys.executable, str(script), *(extra_args or [])]
    return DeepSeekOcrDriver(worker_cmd=cmd)


def test_recognize_runs_through_resident_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runtime(monkeypatch)
    driver = _driver(tmp_path, "fake_ok_worker.py", _FAKE_OK)
    try:
        transcript = driver.recognize(_FakeImage(), (612.0, 792.0), 2.0)
        pid = driver._proc.pid if driver._proc else None
        transcript2 = driver.recognize(_FakeImage(), (612.0, 792.0), 2.0)
        assert [ln.text for ln in transcript.lines] == ["alpha", "beta"]
        assert transcript.engine == "deepseek-ocr" and not transcript.measured_boxes
        assert transcript2.lines == transcript.lines
        assert driver._proc is not None and driver._proc.pid == pid  # worker reused
    finally:
        driver._close_worker()


def test_close_worker_releases_the_worker_pipe_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Terminate+wait reaps the child, but its stdin/stdout are OS pipes: the
    TextIOWrappers stay open, so every worker teardown leaks two fds (and the
    GC'd-transport ResourceWarning the full suite trips on). ``_close_worker``
    must close them itself, not wait for the interpreter to do it at exit."""
    _fake_runtime(monkeypatch)
    driver = _driver(tmp_path, "fake_ok_worker.py", _FAKE_OK)
    driver.recognize(_FakeImage(), (612.0, 792.0), 2.0)
    proc = driver._proc
    assert proc is not None
    assert proc.stdin is not None and proc.stdout is not None

    driver._close_worker()

    assert proc.stdin.closed, "worker stdin pipe leaked"
    assert proc.stdout.closed, "worker stdout pipe leaked"


def test_worker_crash_restarts_once_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runtime(monkeypatch)
    marker = tmp_path / "crash-marker"
    driver = _driver(tmp_path, "crash_worker.py", _FAKE_WORKER, extra_args=[str(marker)])
    try:
        transcript = driver.recognize(_FakeImage(), (612.0, 792.0), 2.0)
        assert [ln.text for ln in transcript.lines] == ["line one", "line two"]
        assert marker.exists()  # first process died; second answered
    finally:
        driver._close_worker()


def test_persistent_desync_raises_instead_of_looping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_runtime(monkeypatch)
    driver = _driver(tmp_path, "desync_worker.py", _FAKE_DESYNC)
    try:
        with pytest.raises(RuntimeError, match="desynced"):
            driver.recognize(_FakeImage(), (612.0, 792.0), 2.0)
    finally:
        driver._close_worker()


def test_wedged_worker_times_out_on_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker wedged before stdin draining must not hang the driver inside
    write() once the OS pipe fills: REQUEST_TIMEOUT_S has to arm the write as
    well as the reply read, or one CUDA-native stall stalls the whole export
    forever (the exact failure class the worker boundary exists to contain)."""
    _FAKE_WEDGED = "import time\n\ntime.sleep(300)\n"
    _fake_runtime(monkeypatch)
    driver = _driver(tmp_path, "wedged_worker.py", _FAKE_WEDGED)
    driver.REQUEST_TIMEOUT_S = 0.5
    try:
        with pytest.raises(RuntimeError, match="died or wedged"):
            driver._request({"image": "x" * 2_000_000})  # larger than the 64 KB pipe
    finally:
        driver._close_worker()


def test_close_worker_kill_path_survives_a_wait_timeout(tmp_path: Path) -> None:
    """``subprocess.TimeoutExpired`` is not an ``OSError``.

    Suppressing only OSError let it escape ``_close_worker`` *after* the driver
    had already cleared ``self._proc``, orphaning a worker wedged in a native
    CUDA call while the next page spawned a second model load.
    """
    import subprocess as _sp

    class _Wedged:
        stdin = None
        # A real Popen always exposes ``stdout`` (None when not piped); the
        # teardown closes both pipes, so the double must model the interface.
        stdout = None

        def poll(self) -> int | None:
            return None

        def kill(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            raise _sp.TimeoutExpired("worker", timeout if timeout is not None else 0.0)

    driver = _driver(tmp_path, "wedged_worker.py", "import time\n\ntime.sleep(300)\n")
    driver._proc = _Wedged()  # type: ignore[assignment]
    driver._close_worker(kill_now=True)
    assert driver._proc is None

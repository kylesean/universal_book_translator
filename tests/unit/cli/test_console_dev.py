"""`ubt console --dev` must report the port it really serves on, and clean up.

Two defects, both measured on a real run rather than reasoned about:

* **The URL was hardcoded to 3000.** ``web/vite.config.ts`` asks for 3000, but
  Vite silently takes the next free port when it is busy. Measured here: a
  ``qoder-proxy`` process held 3000, Vite moved to 3001, and the banner still
  printed ``http://localhost:3000/`` — so the operator (and ``webbrowser.open``)
  landed on an unrelated service that answered 200. The banner now reports the
  port Vite actually bound.

* **SIGTERM orphaned Vite.** ``atexit`` and the ``finally`` cleanup only run if
  the interpreter unwinds, and Python's default SIGTERM disposition kills the
  process outright. Measured: after ``kill -TERM <console>`` the parent was gone
  in under a second and the Vite child kept serving port 3001 for the whole
  observation window, every retry. Ctrl+C was fine only because
  ``KeyboardInterrupt`` is an exception. A handler now turns SIGTERM/SIGHUP into
  the same unwind.
"""

from __future__ import annotations

import io
import signal
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

pytestmark = pytest.mark.fast


class _FakeProc:
    """A ``Popen`` stand-in whose stdout yields ``lines`` then blocks/ends.

    Declared as a ``Protocol``-compatible duck type rather than a ``Popen``
    subclass: ``Popen.__init__`` spawns a process, and these tests never want
    one. The helpers under test only touch ``stdout`` and ``poll``.
    """

    def __init__(self, lines: list[str]) -> None:
        self.stdout = io.StringIO("".join(lines))

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        """No child to stop."""


def _url(proc: _FakeProc, requested_port: int = 3000) -> str:
    """``_dev_server_url`` for a fake process (narrowing is the test's job)."""
    from ubt.cli.commands.console import _dev_server_url

    return _dev_server_url(cast("subprocess.Popen[str]", proc), requested_port)


def test_the_reported_url_uses_the_port_vite_actually_bound() -> None:
    """The regression: a taken 3000 pushed Vite to 3001, the banner said 3000."""
    proc = _FakeProc(
        [
            "\n  VITE v8.3.2  ready in 228 ms\n\n",
            "  ➜  Local:   http://localhost:3001/\n",
            "  ➜  Network: use --host to expose\n",
        ]
    )
    assert _url(proc) == "http://localhost:3001/"


def test_the_url_is_read_from_the_configured_port_when_it_was_free() -> None:
    proc = _FakeProc(["  ➜  Local:   http://localhost:3000/\n"])
    assert _url(proc) == "http://localhost:3000/"


def test_an_unreadable_banner_falls_back_to_the_requested_port() -> None:
    """A wrapper that swallows stdout must not hang the console startup."""
    proc = _FakeProc([])  # EOF immediately: Vite exited or said nothing.
    assert _url(proc) == "http://localhost:3000/"


def test_the_command_banner_prints_the_port_vite_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The wiring, not just the helper: the banner must carry the real URL.

    The helper tests above would all pass with ``console_command`` still
    printing a hardcoded ``http://localhost:3000/`` — which is exactly the bug.
    So drive the real command and read the banner it emits.
    """
    import ubt.cli.commands.console as console_mod

    web_dir = tmp_path / "web"
    (web_dir / "node_modules" / ".bin").mkdir(parents=True)
    (web_dir / "package.json").write_text("{}")
    for tool in ("vite", "tsc"):
        (web_dir / "node_modules" / ".bin" / tool).write_text("")

    class _Toolchain:
        package_manager = "pnpm"

        def dev_argv(self) -> list[str]:
            return ["vite"]

        def local(self, tool: str) -> Path | None:
            return web_dir / "node_modules" / ".bin" / tool

    monkeypatch.setattr(console_mod, "detect_js_toolchain", lambda _d: _Toolchain())
    monkeypatch.setattr(console_mod, "_repo_root", lambda: tmp_path)

    class _Proc(_FakeProc):
        def terminate(self) -> None:  # pragma: no cover - cleanup path
            pass

    started: list[_Proc] = []

    def _popen(argv: object, **kwargs: object) -> _Proc:
        proc = _Proc(["  ➜  Local:   http://localhost:3001/\n"])
        started.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _popen)
    # The signal handler is installed on the *real* signal table; record the
    # call instead so this test asserts the wiring without touching it.
    forwarded: list[object] = []
    monkeypatch.setattr(console_mod, "_forward_signals_to", forwarded.append)
    # The server would block forever; the banner is printed before it runs.
    # Patched on the module object, not by dotted path: ``ubt.api.app`` resolves
    # to the FastAPI *instance* (the package re-exports ``app``), not the module.
    import ubt.api.app as app_module

    monkeypatch.setattr(app_module, "run_server", lambda **_: (_ for _ in ()).throw(SystemExit(0)))

    with pytest.raises(SystemExit):
        console_mod.console_command(dev=True, no_browser=True)

    out = capsys.readouterr().out
    assert "http://localhost:3001/" in out, out
    assert "localhost:3000" not in out, out
    # The mode label used to hardcode "port 3000" while the URL line beside it
    # said 3001 — two contradictory statements in one panel.
    assert "Dev Mode (http://localhost:3001)" in out, out
    # Without this the Vite child outlives a `kill -TERM` on the console.
    assert forwarded == started, "the dev child was started but its signals were not wired"


def test_the_startup_banner_is_echoed_not_swallowed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Vite's own banner must reach the operator.

    Reading the bound port consumes Vite's output line by line and stops at the
    ``Local:`` line. The first version of this fix returned without echoing, so
    the ``VITE v8.3.2 ready`` and ``➜ Local:`` lines vanished — the operator saw
    a console banner and no evidence the dev server had started at all.
    """
    proc = _FakeProc(
        [
            "\n  VITE v8.3.2  ready in 228 ms\n\n",
            "  ➜  Local:   http://localhost:3001/\n",
        ]
    )
    assert _url(proc) == "http://localhost:3001/"
    out = capsys.readouterr().out
    assert "VITE v8.3.2  ready in 228 ms" in out, out
    assert "➜  Local:   http://localhost:3001/" in out, out


def test_vites_output_still_reaches_the_console(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reading the port must not swallow the HMR log ``--dev`` exists for."""
    proc = _FakeProc(["  ➜  Local:   http://localhost:3001/\n", "[vite] hmr update /src/App.tsx\n"])
    _url(proc)

    # The pump runs on a daemon thread; give it a moment to drain the rest.
    import time

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if "[vite] hmr update" in capsys.readouterr().out:
            return
        time.sleep(0.02)
    pytest.fail("Vite's own output never reached the console")


def test_sigterm_is_turned_into_an_unwind() -> None:
    """The handler must raise, not return: ``finally``/``atexit`` depend on it.

    Asserted on the *disposition* and on the exception rather than by signalling
    the test process, which would take the runner down with it.
    """
    from ubt.cli.commands import console as console_mod

    original = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        console_mod._forward_signals_to(_FakeProc([]))  # type: ignore[arg-type]
        for sig in (signal.SIGTERM, signal.SIGHUP):
            handler = signal.getsignal(sig)
            assert callable(handler), f"{sig.name} still has the default disposition"
            with pytest.raises(SystemExit) as excinfo:
                handler(sig, None)
            # 128 + N is the shell convention for "died from signal N".
            assert excinfo.value.code == 128 + int(sig)
    finally:
        for sig, previous in original.items():
            signal.signal(sig, previous)


def test_the_dev_child_is_not_orphaned_by_a_terminating_signal(tmp_path: Path) -> None:
    """End-to-end: SIGTERM to the parent leaves no child behind.

    The reproduction used ``sleep`` in place of Vite — the point under test is
    the signal plumbing, not the dev server, and a real Vite takes seconds to
    boot and a port to bind.
    """
    child_pid_file = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    repo_root = Path(__file__).resolve().parents[3]
    script.write_text(
        f"""import subprocess, sys, time
from pathlib import Path
sys.path.insert(0, {str(repo_root)!r})
from ubt.cli.commands.console import _forward_signals_to
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
Path({str(child_pid_file)!r}).write_text(str(child.pid))
_forward_signals_to(child)
try:
    time.sleep(120)
finally:
    child.terminate()
    child.wait()
"""
    )
    parent = subprocess.Popen([sys.executable, str(script)])
    try:
        import time

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not child_pid_file.exists():
            time.sleep(0.02)
        assert child_pid_file.exists(), "the probe never started its child"
        child_pid = int(child_pid_file.read_text())

        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=15)

        # The child must be gone. ``wait`` alone would not prove it: the point
        # is that ``finally`` ran and terminated it.
        for _ in range(100):
            if not _alive(child_pid):
                break
            time.sleep(0.05)
        assert not _alive(child_pid), f"child {child_pid} outlived the parent"
    finally:
        if parent.poll() is None:  # pragma: no cover - the test failed
            parent.kill()
            parent.wait()


def _alive(pid: int) -> bool:
    """Whether ``pid`` is still running (zombies count as gone)."""
    try:
        with Path(f"/proc/{pid}/stat").open(encoding="utf-8") as handle:
            state = handle.read().split(")")[-1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def test_a_platform_without_sighup_still_installs_the_sigterm_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows has no SIGHUP; --dev there must not crash installing handlers.

    The loop reads the constant through ``getattr`` at call time, so deleting
    the attribute is the platform simulation — and the assertion is that the
    SIGTERM handler, the one that actually reaps the Vite child, still lands.
    """
    from ubt.cli.commands import console as console_mod

    monkeypatch.delattr(signal, "SIGHUP")
    original = signal.getsignal(signal.SIGTERM)
    try:
        console_mod._forward_signals_to(_FakeProc([]))  # type: ignore[arg-type]
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        with pytest.raises(SystemExit):
            handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, original)

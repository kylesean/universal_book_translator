"""Console subcommand to launch the UBT Operator Console Web UI."""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from ubt.cli.jstoolchain import JsToolchainError, detect_js_toolchain

console = Console()

#: The port ``web/vite.config.ts`` asks for. Vite silently moves to the next free
#: port when it is taken, so this is the *requested* port, not necessarily the
#: one it serves on — :func:`_dev_server_url` reads the real one back.
VITE_REQUESTED_PORT = 3000
#: How long to wait for Vite to print its listening address before giving up and
#: reporting the requested port instead. Vite is ready in a few hundred ms warm
#: and a few seconds cold; the wait only bounds the "is it up yet" question.
VITE_START_TIMEOUT_S = 20.0


#: What Vite prints when it has bound a port. Matching the *port* rather than
#: parsing the whole banner keeps this working across Vite versions: the line
#: has been ``➜  Local:   http://localhost:5173/`` and ``http://localhost:3001/``
#: with different arrow glyphs and ANSI colouring.
_VITE_READY_RE = re.compile(r"http://(?:localhost|127\.0\.0\.1|\[::1\]):(\d+)/")


def _dev_server_url(proc: subprocess.Popen[str], requested_port: int) -> str:
    """The URL the Vite dev server is actually serving on.

    Vite moves to the next free port when the configured one is taken (measured
    here: ``qoder-proxy`` holding 3000 pushed the console to 3001). Printing the
    configured port regardless sent the operator — and ``webbrowser.open`` — to
    whatever *else* owns 3000, which then answered as a working page. So read
    the bound port back out of Vite's own startup line.

    Falls back to ``requested_port`` when the line cannot be read in time (a
    non-TTY stream, a wrapper that swallows stdout): a slightly wrong URL beats
    hanging, and the caller prints the same line the operator can read.

    Every line consumed here is echoed before being parsed. Vite's banner spans
    several lines (``VITE v8.3.2 ready``, ``➜ Local:``, ``➜ Network:``) and this
    function stops reading at the port, so echoing only what it *returns* would
    drop the ``Local:`` line and the version line — the first run of this fix
    lost Vite's whole startup banner that way.
    """
    if proc.stdout is None:
        return f"http://localhost:{requested_port}/"
    deadline = time.monotonic() + VITE_START_TIMEOUT_S
    try:
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                break  # Vite exited; the caller reports that separately.
            sys.stdout.write(line)
            sys.stdout.flush()
            match = _VITE_READY_RE.search(line)
            if match is not None:
                return f"http://localhost:{match.group(1)}/"
        return f"http://localhost:{requested_port}/"
    finally:
        # Whatever the outcome, the rest of Vite's output keeps flowing: an
        # unread pipe fills up and blocks the child, and hiding the HMR log
        # would take away the main reason to run ``--dev``.
        _drain_stdout(proc)


def _drain_stdout(proc: subprocess.Popen[str]) -> None:
    """Keep echoing Vite's output so its HMR log is not swallowed by the pipe.

    ``stdout=PIPE`` is what makes the bound port readable, but an unread pipe is
    a deadlock waiting for a full buffer — and hiding Vite's own "page reload"
    lines would take away the main reason to run ``--dev``. One daemon thread
    forwards every line to the console for the process's lifetime.
    """
    if proc.stdout is None:
        return

    def _pump() -> None:
        stream = proc.stdout
        if stream is None:  # pragma: no cover - narrowed for the type checker
            return
        for line in stream:
            sys.stdout.write(line)
            sys.stdout.flush()

    threading.Thread(target=_pump, name="vite-stdout", daemon=True).start()


def _forward_signals_to(proc: subprocess.Popen[str]) -> None:
    """Make a terminating signal stop the dev server too.

    ``Popen.terminate`` in a ``finally`` (and the ``atexit`` hook) only runs if
    the interpreter unwinds. A bare ``kill -TERM <console>`` does not: Python's
    default SIGTERM disposition kills the process outright, so the Vite child is
    orphaned and keeps port 3001 (measured: it survived for the full 6s the
    check waited, and again on every retry). Ctrl+C worked only because
    ``KeyboardInterrupt`` *is* an exception.

    Installing a handler turns SIGTERM/SIGHUP into that same unwind, so the
    existing cleanup path runs. ``SIGINT`` is left alone: uvicorn installs its
    own handler and raises ``KeyboardInterrupt``.
    """
    for signum in (signal.SIGTERM, signal.SIGHUP):

        def _handler(received: int, frame: object) -> None:
            # Raise as the exception the ``finally`` already handles. Raising
            # from a handler unwinds the stack at the interrupted point, which
            # for a blocking ``uvicorn.run`` is inside its serve loop.
            raise SystemExit(128 + int(received))

        with contextlib.suppress(ValueError, OSError):  # not the main thread
            signal.signal(signum, _handler)


def _repo_root() -> Path:
    """The repository root: this file is ``<root>/ubt/cli/commands/console.py``.

    A function rather than an inline expression so a test can point the command
    at a temporary tree without rewriting ``__file__`` (which the module-level
    imports have already resolved).
    """
    return Path(__file__).resolve().parent.parent.parent.parent


def console_command(
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Bind host for console server"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Bind port for console server"),
    ] = 8000,
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Do not automatically open default web browser"),
    ] = False,
    dev: Annotated[
        bool,
        typer.Option(
            "--dev",
            help=(
                "Launch in Vite HMR development mode (starts the Vite dev server, "
                f"which asks for port {VITE_REQUESTED_PORT} and moves to the next free one)"
            ),
        ),
    ] = False,
) -> None:
    """Launch the UBT Operator Console & Review Workbench."""
    from ubt import __version__
    from ubt.api.app import run_server
    from ubt.api.security import _no_auth_allowed, _strict_auth_from_env

    # For local desktop console on loopback, allow local open authentication by
    # default -- but never override a deployment that promised authentication.
    # ``_require_api_key_gate`` checks strict mode before this override, so
    # setting it in a strict deployment would only produce a confusing refusal
    # after the console has printed its banner; leave it unset there.
    if (
        host in ("127.0.0.1", "localhost", "::1")
        and not os.getenv("UBT_API_KEY")
        and not _no_auth_allowed()
        and not _strict_auth_from_env()
    ):
        os.environ["UBT_ALLOW_NO_AUTH"] = "1"

    repo_root = _repo_root()
    web_dir = repo_root / "web"
    static_dir = Path(__file__).resolve().parent.parent.parent / "api" / "static"
    index_html = static_dir / "index.html"

    vite_proc: subprocess.Popen[str] | None = None

    if dev:
        # Development mode: Launch Vite with Hot Module Replacement (HMR)
        if not (web_dir / "package.json").exists():
            console.print("[bold red]❌ Cannot start --dev: web/package.json not found.[/]")
            raise typer.Exit(1)

        # Probe rather than assume. This used to run a bare ``pnpm dev``, so a
        # machine without pnpm got a raw ``FileNotFoundError`` traceback instead of
        # being told what to install — even though ``web/node_modules/.bin/vite``
        # was already present and needs no package manager at all.
        toolchain = detect_js_toolchain(web_dir)
        try:
            dev_argv = toolchain.dev_argv()
        except JsToolchainError as exc:
            console.print(f"[bold red]❌ {escape(str(exc))}[/]")
            raise typer.Exit(1) from exc

        runner = (
            "Vite (local node_modules)"
            if toolchain.local("vite")
            else str(toolchain.package_manager)
        )
        console.print(f"[bold cyan]🚀 Starting Vite frontend dev server via {runner}...[/]")
        try:
            # ``stdout=PIPE`` so the bound port can be read back: Vite moves to
            # the next free port when the configured one is taken, and the
            # console must report the port it actually serves on. The dev
            # server is a third-party node toolchain: it gets the sanitized
            # environment, not the operator's credentials.
            from ubt.core.env import subprocess_env

            vite_proc = subprocess.Popen(
                dev_argv,
                cwd=web_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=subprocess_env(),
            )
        except OSError as exc:
            console.print(
                f"[bold red]❌ Could not start the Vite dev server: {escape(str(exc))}[/]"
            )
            raise typer.Exit(1) from exc
        _forward_signals_to(vite_proc)
        target_url = _dev_server_url(vite_proc, VITE_REQUESTED_PORT)

        def _cleanup_vite(*_: object) -> None:
            if vite_proc and vite_proc.poll() is None:
                vite_proc.terminate()

        # Ensure Vite terminates on exit
        import atexit

        atexit.register(_cleanup_vite)
    else:
        # Production mode: Serve compiled SPA from static_dir
        if not index_html.exists():
            console.print(
                "[bold yellow]⚠️  Static web console assets not found in ubt/api/static.[/]"
            )
            console.print("[dim]Building assets via scripts/build_console_assets.py...[/]")
            try:
                script = repo_root / "scripts" / "build_console_assets.py"
                if script.exists():
                    from ubt.core.env import subprocess_env

                    subprocess.run(
                        [sys.executable, str(script)],
                        check=True,
                        env=subprocess_env(),
                    )
                else:
                    console.print(
                        "[bold red]❌ Cannot auto-build: scripts/build_console_assets.py not found.[/]"
                    )
            except subprocess.CalledProcessError as exc:
                # The script prints its own diagnosis (it now probes for a
                # toolchain instead of assuming pnpm); echoing the exception alone
                # hid the reason behind "returned non-zero exit status 1".
                console.print(
                    "[bold red]❌ Failed to build the web console assets "
                    f"(exit {exc.returncode}); see the message above.[/]"
                )
                console.print(
                    "[dim]The console can still start once ubt/api/static exists. "
                    "Build it with: python scripts/build_console_assets.py[/dim]"
                )
            except Exception as exc:
                console.print(f"[bold red]❌ Failed to build static assets: {exc}[/]")

        target_url = f"http://{host}:{port}/"

    mode_label = (
        # The port comes from the URL Vite actually bound: the label said
        # "port 3000" while the Web UI line beside it said 3001.
        f"[bold yellow]Vite HMR Dev Mode ({target_url.rstrip('/')})[/bold yellow]"
        if dev
        else "[bold green]Production Embedded SPA[/bold green]"
    )

    console.print(
        Panel.fit(
            f"[bold cyan]UBT Operator Console v{__version__}[/bold cyan] — {mode_label}\n"
            f"[green]Compiler Cockpit & Quality Gate Workbench[/green]\n\n"
            f"• Web UI URL:  [bold underline white]{target_url}[/bold underline white]\n"
            f"• Backend API: [dim]http://{host}:{port}/docs[/dim]\n"
            f"• Press [bold red]Ctrl+C[/bold red] to stop.",
            title="[bold blue]Universal Book Translator[/bold blue]",
            border_style="blue",
        )
    )

    if not no_browser:

        def _open_browser() -> None:
            time.sleep(1.2 if dev else 0.8)
            webbrowser.open(target_url)

        thread = threading.Thread(target=_open_browser, daemon=True)
        thread.start()

    try:
        run_server(host=host, port=port)
    finally:
        if vite_proc and vite_proc.poll() is None:
            vite_proc.terminate()

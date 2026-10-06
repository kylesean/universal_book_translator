"""Console subcommand to launch the UBT Operator Console Web UI."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()


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
            "--dev", help="Launch in Vite HMR development mode (starts pnpm dev on port 3000)"
        ),
    ] = False,
) -> None:
    """Launch the UBT Operator Console & Review Workbench."""
    from ubt import __version__
    from ubt.api.app import run_server

    # For local desktop console on loopback, allow local open authentication by default
    if (
        host in ("127.0.0.1", "localhost", "::1")
        and not os.getenv("UBT_API_KEY")
        and not os.getenv("UBT_ALLOW_NO_AUTH")
    ):
        os.environ["UBT_ALLOW_NO_AUTH"] = "1"

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    web_dir = repo_root / "web"
    static_dir = Path(__file__).resolve().parent.parent.parent / "api" / "static"
    index_html = static_dir / "index.html"

    vite_proc: subprocess.Popen[bytes] | None = None

    if dev:
        # Development mode: Launch Vite with Hot Module Replacement (HMR)
        if not (web_dir / "package.json").exists():
            console.print("[bold red]❌ Cannot start --dev: web/package.json not found.[/]")
            raise typer.Exit(1)

        console.print("[bold cyan]🚀 Starting Vite frontend dev server (pnpm dev)...[/]")
        vite_proc = subprocess.Popen(["pnpm", "dev"], cwd=web_dir)
        target_url = "http://localhost:3000/"

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
                    subprocess.run([sys.executable, str(script)], check=True)
                else:
                    console.print(
                        "[bold red]❌ Cannot auto-build: scripts/build_console_assets.py not found.[/]"
                    )
            except Exception as exc:
                console.print(f"[bold red]❌ Failed to build static assets: {exc}[/]")

        target_url = f"http://{host}:{port}/"

    mode_label = (
        "[bold yellow]Vite HMR Dev Mode (port 3000)[/bold yellow]"
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

"""A ``--json`` flag must never be printed through a presentation layer.

``test_json_output_contract.py`` proves the currently-known call sites are
correct. This file stops a *new* one from being added: it walks the CLI
package's own source and fails on the shape that caused the release blocker.

The rule is deliberately structural rather than behavioural. A behavioural test
only catches the command whose payload happens to contain a long token or a
bracketed title, so a regression in a command nobody exercises stays hidden
until a user with a long filename hits it. Reading the source catches the
mistake at the moment it is written.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

CLI_ROOT = Path(__file__).resolve().parents[3] / "ubt" / "cli"


def _json_emitting_rich_calls(path: Path) -> list[tuple[int, str, list[str]]]:
    """Return ``(line, attribute, keywords)`` for Rich calls printing JSON."""
    source = path.read_text(encoding="utf-8")
    found: list[tuple[int, str, list[str]]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
            continue
        if func.value.id != "console" or func.attr not in {"print", "print_json"}:
            continue
        segment = ast.get_source_segment(source, node) or ""
        if "json.dump" not in segment and "dumps" not in segment:
            continue
        found.append((node.lineno, func.attr, sorted(k.value for k in node.keywords if k.arg)))
    return found


@pytest.mark.parametrize(
    "module",
    sorted(p for p in CLI_ROOT.rglob("*.py")),
    ids=lambda p: str(p.relative_to(CLI_ROOT)),
)
def test_no_module_prints_json_through_the_rich_console(module: Path) -> None:
    """Structured output goes through ``ubt.cli.ui.emit_json``, not Rich.

    Rich lays text out for a human: it keeps a width budget, so a token wider
    than the terminal is cut; and it interprets ``[bold]``-shaped text as
    style, so it deletes it. Either behaviour silently corrupts a document a
    caller is about to ``json.loads``.
    """
    violations = _json_emitting_rich_calls(module)

    assert violations == [], (
        f"{module} prints JSON through Rich at line(s) "
        f"{[v[0] for v in violations]}; use ubt.cli.ui.emit_json(payload) instead"
    )


def test_the_scanner_would_actually_catch_a_violation(tmp_path: Path) -> None:
    """Guard the guard: the detector must fire on the shape it forbids."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from rich.console import Console\n"
        "import json\n"
        "console = Console()\n"
        "def f(data):\n"
        "    console.print(json.dumps(data, indent=2))\n"
        "    console.print_json(json.dumps(data))\n"
        "    console.print('plain human text')\n",
        encoding="utf-8",
    )

    found = _json_emitting_rich_calls(sample)

    assert len(found) == 2, found


def test_the_shared_emitter_exists_and_is_exported() -> None:
    """The replacement must be importable from one obvious place."""
    from ubt.cli.ui import emit_json

    assert callable(emit_json)


def test_no_module_still_imports_print_json_for_structured_output() -> None:
    """``console.print_json`` is a presentation helper, not an output contract.

    It happens to escape correctly today, but it exists to pretty-print for a
    human: it can colour, it can re-wrap, and a future Rich release is free to
    change either. Structured output should not depend on that.
    """
    offenders = [
        module.relative_to(CLI_ROOT)
        for module in CLI_ROOT.rglob("*.py")
        if "console.print_json(" in module.read_text(encoding="utf-8")
    ]

    assert offenders == [], f"{offenders} should use ubt.cli.ui.emit_json instead"

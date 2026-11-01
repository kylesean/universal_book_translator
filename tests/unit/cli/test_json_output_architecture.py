from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

CLI_ROOT = Path(__file__).resolve().parents[3] / "ubt" / "cli"


def _json_emitting_rich_calls(path: Path) -> list[tuple[int, str, list[str]]]:
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
        found.append(
            (
                node.lineno,
                func.attr,
                sorted(str(k.value) for k in node.keywords if k.arg is not None),
            )
        )
    return found


@pytest.mark.parametrize(
    "module",
    sorted(p for p in CLI_ROOT.rglob("*.py")),
    ids=lambda p: str(p.relative_to(CLI_ROOT)),
)
def test_no_module_prints_json_through_the_rich_console(module: Path) -> None:
    violations = _json_emitting_rich_calls(module)

    assert violations == [], (
        f"{module} prints JSON through Rich at line(s) "
        f"{[v[0] for v in violations]}; use ubt.cli.ui.emit_json(payload) instead"
    )


def test_the_scanner_would_actually_catch_a_violation(tmp_path: Path) -> None:
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
    from ubt.cli.ui import emit_json

    assert callable(emit_json)


def test_no_module_still_imports_print_json_for_structured_output() -> None:
    offenders = [
        module.relative_to(CLI_ROOT)
        for module in CLI_ROOT.rglob("*.py")
        if "console.print_json(" in module.read_text(encoding="utf-8")
    ]

    assert offenders == [], f"{offenders} should use ubt.cli.ui.emit_json instead"

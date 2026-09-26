"""Introspect the ``UBTConfig`` surface: every field, its env var(s), its value.

The env-var list is derived from the model schema, so a field cannot be added
without appearing here. This is the live counterpart to the hand-maintained
table in ``docs/guides/USER_GUIDE.md`` (a test pins the two together), and the way to
answer "which env var sets X?" without grepping the codebase.
"""

from __future__ import annotations

import json
from typing import Any

import typer
from pydantic import SecretStr
from rich.console import Console
from rich.table import Table

from ubt.core.config import UBTConfig, env_var_names

console = Console()

#: Fields whose value must never be printed; shown as a length, not the secret.
_SECRET_FIELDS = frozenset({"api_key", "ocr_api_key", "service_api_key"})


def _render_value(field_name: str, value: Any) -> str:
    """Render a config value, redacting secrets to a length marker."""
    if field_name in _SECRET_FIELDS:
        secret = value.get_secret_value() if isinstance(value, SecretStr) else str(value or "")
        return "<unset>" if not secret else f"<set: {len(secret)} chars>"
    if isinstance(value, SecretStr):
        return "<set>" if value.get_secret_value() else "<unset>"
    return str(value)


def _render_default(field_name: str, field: Any) -> str:
    """Render a field's default without invoking dynamic factories.

    ``field.default`` is ``PydanticUndefined`` for a ``default_factory`` field,
    which printed literally; several factories are also env-dependent or
    side-effecting (base-url sniffing, opencode session id) and a secret
    factory would print a fake credential. Show a stable marker instead.
    """
    if field.is_required():
        return "<required>"
    if field.default_factory is not None:
        return "<dynamic>"
    return _render_value(field_name, field.default)


def config_command(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    set_only: bool = typer.Option(
        False, "--set-only", help="Show only fields set from the environment or a profile."
    ),
) -> None:
    """List every configuration field with its env var(s), current value and default.

    Secrets are never printed: a credential shows as ``<set: N chars>``. Use
    ``--json`` for scripting and ``--set-only`` to see just what is in effect.
    """
    config = UBTConfig.from_env()
    rows: list[dict[str, Any]] = []
    for name, field in UBTConfig.model_fields.items():
        explicitly_set = name in config.model_fields_set
        if set_only and not explicitly_set:
            continue
        rows.append(
            {
                "field": name,
                "env": env_var_names(name),
                "value": _render_value(name, getattr(config, name)),
                "default": _render_default(name, field),
                "set": explicitly_set,
            }
        )

    if json_output:
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    table = Table(title=f"UBTConfig — {len(rows)} field(s)")
    table.add_column("Field", no_wrap=True)
    table.add_column("Env var")
    table.add_column("Value")
    table.add_column("Default")
    for row in rows:
        table.add_row(
            row["field"],
            ", ".join(row["env"]),
            row["value"],
            row["default"],
            style="bold" if row["set"] else None,
        )
    console.print(table)

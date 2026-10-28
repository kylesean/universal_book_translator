"""Introspect and validate the ``UBTConfig`` surface and its domain sub-configs.

Provides:
- Sub-domain filtering (``--domain router|typst|quality|engine``) to inspect decoupled
  domain settings without wading through 100+ flat fields.
- Static configuration validation and environment diagnostics (``--validate``) verifying
  rate-limit bounds, URL schemes, storage directories, and toolchain presence.
- Secrets masking (showing length, never sensitive credential values).
- Machine-readable JSON output (``--json``) for automated CI and deployment pipelines.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
from pydantic import SecretStr
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ubt.core.config import (
    EngineConfig,
    QualityConfig,
    RouterConfig,
    TypstConfig,
    UBTConfig,
    env_var_names,
)

console = Console()

#: Fields whose value must never be printed; shown as a length, not the secret.
_SECRET_FIELDS = frozenset({"api_key", "ocr_api_key", "service_api_key", "qe_judge_api_key"})

DOMAIN_CONFIG_MODELS: dict[str, tuple[type, str]] = {
    "router": (RouterConfig, "LLM routing, providers, endpoints, and rate limits"),
    "typst": (TypstConfig, "Document rendering, Typst typesetting, and page layouts"),
    "quality": (QualityConfig, "Quality gates, MTQE, defect screening, and consistency"),
    "engine": (EngineConfig, "Pipeline execution, storage, databases, and worker queues"),
}


@dataclass(slots=True)
class ValidationFinding:
    """A single diagnostic sanity check finding."""

    domain: str
    target: str
    level: str  # "OK", "WARN", "FAIL"
    message: str


def _render_value(field_name: str, value: Any) -> str:
    """Render a config value, redacting secrets to a length marker."""
    if field_name in _SECRET_FIELDS:
        secret = value.get_secret_value() if isinstance(value, SecretStr) else str(value or "")
        return "<unset>" if not secret else f"<set: {len(secret)} chars>"
    if field_name == "extra_headers":
        # These headers are forwarded to the outbound provider and commonly
        # carry a bearer/API token. Names are safe to show; values are not.
        if not value:
            return "<unset>"
        return f"<{len(value)} header(s): {', '.join(sorted(value))}> (values redacted)"
    if isinstance(value, SecretStr):
        return "<set>" if value.get_secret_value() else "<unset>"
    return str(value)


def _render_default(field_name: str, field: Any) -> str:
    """Render a field's default without invoking dynamic factories."""
    if field.is_required():
        return "<required>"
    if field.default_factory is not None:
        return "<dynamic>"
    return _render_value(field_name, field.default)


def validate_configuration(config: UBTConfig, domain: str | None = None) -> list[ValidationFinding]:
    """Perform static sanity checks and environment validation on active configuration."""
    findings: list[ValidationFinding] = []

    # 1. Router checks
    if domain is None or domain == "router":
        parsed = urlparse(config.base_url)
        if parsed.scheme in ("http", "https"):
            findings.append(
                ValidationFinding(
                    "router", "base_url", "OK", f"Valid URL scheme: {config.base_url}"
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    "router",
                    "base_url",
                    "FAIL",
                    f"Invalid scheme for base_url: '{config.base_url}' (must be http or https)",
                )
            )

        if config.rate_limit_rpm <= config.rate_limit_max_rpm:
            findings.append(
                ValidationFinding(
                    "router",
                    "rate_limit_rpm",
                    "OK",
                    f"{config.rate_limit_rpm} <= max {config.rate_limit_max_rpm}",
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    "router",
                    "rate_limit_rpm",
                    "FAIL",
                    f"Initial RPM {config.rate_limit_rpm} exceeds max RPM {config.rate_limit_max_rpm}",
                )
            )

        if config.rate_limit_tpm <= config.rate_limit_max_tpm:
            findings.append(
                ValidationFinding(
                    "router",
                    "rate_limit_tpm",
                    "OK",
                    f"{config.rate_limit_tpm} <= max {config.rate_limit_max_tpm}",
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    "router",
                    "rate_limit_tpm",
                    "FAIL",
                    f"Initial TPM {config.rate_limit_tpm} exceeds max TPM {config.rate_limit_max_tpm}",
                )
            )

        if config.api_timeout > 0:
            findings.append(
                ValidationFinding("router", "api_timeout", "OK", f"{config.api_timeout}s timeout")
            )
        else:
            findings.append(
                ValidationFinding(
                    "router",
                    "api_timeout",
                    "FAIL",
                    f"api_timeout must be > 0, got {config.api_timeout}",
                )
            )

    # 2. Typst checks
    if domain is None or domain == "typst":
        typst_path = shutil.which("typst")
        if typst_path:
            findings.append(
                ValidationFinding("typst", "compiler", "OK", f"Typst binary found at {typst_path}")
            )
        else:
            findings.append(
                ValidationFinding(
                    "typst",
                    "compiler",
                    "WARN",
                    "Typst binary not found in PATH; PDF render and preflight gates will be skipped",
                )
            )

        if config.dual_mode in ("auto", "mono", "bilingual", "side_by_side", "interleaved"):
            findings.append(
                ValidationFinding(
                    "typst", "dual_mode", "OK", f"Valid dual mode: {config.dual_mode}"
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    "typst", "dual_mode", "FAIL", f"Unknown dual_mode: {config.dual_mode}"
                )
            )

        if config.pdf_optimize_profile in ("lossless", "ebook", "screen"):
            findings.append(
                ValidationFinding(
                    "typst",
                    "pdf_optimize_profile",
                    "OK",
                    f"Valid PDF optimization profile: {config.pdf_optimize_profile}",
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    "typst",
                    "pdf_optimize_profile",
                    "FAIL",
                    f"Unknown PDF optimization profile: {config.pdf_optimize_profile}",
                )
            )

    # 3. Quality checks
    if domain is None or domain == "quality":
        if 0.0 <= config.qe_threshold <= 1.0:
            findings.append(
                ValidationFinding(
                    "quality",
                    "qe_threshold",
                    "OK",
                    f"Threshold {config.qe_threshold} in [0.0, 1.0]",
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    "quality",
                    "qe_threshold",
                    "FAIL",
                    f"qe_threshold must be in [0.0, 1.0], got {config.qe_threshold}",
                )
            )

        if config.max_repair_rounds >= 0:
            findings.append(
                ValidationFinding(
                    "quality",
                    "max_repair_rounds",
                    "OK",
                    f"Max repair rounds: {config.max_repair_rounds}",
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    "quality",
                    "max_repair_rounds",
                    "FAIL",
                    f"max_repair_rounds must be >= 0, got {config.max_repair_rounds}",
                )
            )

    # 4. Engine checks
    if domain is None or domain == "engine":
        db_path = Path(config.db_dir)
        try:
            parent = db_path.parent if not db_path.exists() else db_path
            if parent.exists() and os.access(parent, os.W_OK):
                findings.append(
                    ValidationFinding(
                        "engine", "db_dir", "OK", f"Ledger storage writable at {db_path}"
                    )
                )
            else:
                findings.append(
                    ValidationFinding(
                        "engine",
                        "db_dir",
                        "WARN",
                        f"Ledger storage directory parent {parent} is not writable or absent",
                    )
                )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    "engine", "db_dir", "FAIL", f"Failed to check db_dir {db_path}: {exc}"
                )
            )

        if config.max_concurrency > 0:
            findings.append(
                ValidationFinding(
                    "engine", "max_concurrency", "OK", f"Concurrency pool: {config.max_concurrency}"
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    "engine",
                    "max_concurrency",
                    "FAIL",
                    f"max_concurrency must be > 0, got {config.max_concurrency}",
                )
            )

        if config.document_mode in ("auto", "paper", "book"):
            findings.append(
                ValidationFinding(
                    "engine", "document_mode", "OK", f"Valid document mode: {config.document_mode}"
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    "engine",
                    "document_mode",
                    "FAIL",
                    f"Unknown document_mode: {config.document_mode}",
                )
            )

    return findings


def config_command(
    domain: str | None = typer.Option(
        None,
        "--domain",
        "-d",
        help="Filter configuration by sub-domain ('router', 'typst', 'quality', 'engine').",
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        "-v",
        help="Run static configuration validation and sanity checks.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    set_only: bool = typer.Option(
        False, "--set-only", help="Show only fields set from the environment or a provider block."
    ),
) -> None:
    """List and validate configuration settings by sub-domain.

    Secrets are never printed: credentials show as ``<set: N chars>``.
    Use ``--domain [router|typst|quality|engine]`` to inspect a specific sub-system.
    Use ``--validate`` to perform preflight health and sanity diagnostics.
    """
    domain_norm = domain.strip().lower() if domain else None
    if domain_norm and domain_norm not in DOMAIN_CONFIG_MODELS:
        allowed = ", ".join(f"'{k}'" for k in sorted(DOMAIN_CONFIG_MODELS))
        typer.echo(
            f"Error: Unknown domain '{domain}'. Allowed domains are: {allowed}.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        config = UBTConfig.from_env()
    except Exception as exc:
        if validate:
            if json_output:
                typer.echo(
                    json.dumps(
                        {
                            "valid": False,
                            "findings": [
                                {
                                    "domain": domain_norm or "config",
                                    "target": "schema",
                                    "level": "FAIL",
                                    "message": f"Configuration schema validation failed: {exc}",
                                }
                            ],
                        },
                        indent=2,
                    )
                )
            else:
                table = Table(
                    title=f"UBT Configuration Validation{f' ({domain_norm})' if domain_norm else ''}"
                )
                table.add_column("Domain", style="cyan", no_wrap=True)
                table.add_column("Target", no_wrap=True)
                table.add_column("Status", no_wrap=True)
                table.add_column("Details")
                table.add_row(
                    domain_norm or "config",
                    "schema",
                    "[bold red]FAIL[/bold red]",
                    str(exc),
                )
                console.print(table)
                console.print(
                    Panel(
                        "[bold red]Configuration validation failed. Resolve schema errors above.[/bold red]"
                    )
                )
            raise typer.Exit(code=1) from None
        typer.echo(f"Error loading configuration: {exc}", err=True)
        raise typer.Exit(code=1) from None

    # Diagnostic validation mode
    if validate:
        findings = validate_configuration(config, domain=domain_norm)
        has_failure = any(f.level == "FAIL" for f in findings)

        if json_output:
            data = [
                {
                    "domain": f.domain,
                    "target": f.target,
                    "level": f.level,
                    "message": f.message,
                }
                for f in findings
            ]
            typer.echo(json.dumps({"valid": not has_failure, "findings": data}, indent=2))
            if has_failure:
                raise typer.Exit(code=1)
            return

        table = Table(
            title=f"UBT Configuration Validation{f' ({domain_norm})' if domain_norm else ''}"
        )
        table.add_column("Domain", style="cyan", no_wrap=True)
        table.add_column("Target", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Details")

        for f in findings:
            if f.level == "OK":
                status_style = "[bold green]OK[/bold green]"
            elif f.level == "WARN":
                status_style = "[bold yellow]WARN[/bold yellow]"
            else:
                status_style = "[bold red]FAIL[/bold red]"
            table.add_row(f.domain, f.target, status_style, f.message)

        console.print(table)
        if has_failure:
            console.print(
                Panel(
                    "[bold red]Configuration validation failed. Resolve FAIL items above.[/bold red]"
                )
            )
            raise typer.Exit(code=1)
        console.print(Panel("[bold green]Configuration passed sanity checks.[/bold green]"))
        return

    # Filter fields by domain if specified
    allowed_field_names: set[str] | None = None
    domain_title_suffix = ""
    if domain_norm:
        sub_cls, domain_desc = DOMAIN_CONFIG_MODELS[domain_norm]
        allowed_field_names = set(sub_cls.model_fields.keys())
        domain_title_suffix = f" [{domain_norm}: {domain_desc}]"

    rows: list[dict[str, Any]] = []
    for name, field in UBTConfig.model_fields.items():
        if allowed_field_names is not None and name not in allowed_field_names:
            continue
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

    table = Table(title=escape(f"UBTConfig{domain_title_suffix} — {len(rows)} field(s)"))
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

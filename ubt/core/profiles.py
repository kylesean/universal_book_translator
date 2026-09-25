"""Provider profile loading from TOML configuration files.

Search order:
1. Explicit file path passed by caller (if provided)
2. Local directory: ``./ubt.toml``
3. User config directory: ``~/.config/ubt/config.toml``
4. User home directory: ``~/.ubt/config.toml``

Uses standard-library ``tomllib`` (Python 3.11+) with zero extra dependencies.
"""

from __future__ import annotations

import logging
import os
import re
import tomllib
from pathlib import Path
from typing import Any

from ubt.core.exceptions import UBTError

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_LOCATIONS: tuple[Path, ...] = (
    Path("ubt.toml"),
    Path.home() / ".config" / "ubt" / "config.toml",
    Path.home() / ".ubt" / "config.toml",
)


class ProfileNotFoundError(UBTError):
    """Raised when a requested provider profile does not exist in configuration files."""


def find_config_file(custom_path: Path | str | None = None) -> Path | None:
    """Find the first existing configuration file in search order."""
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Configuration file not found: {custom_path}")

    for candidate in DEFAULT_CONFIG_LOCATIONS:
        try:
            resolved = candidate.expanduser().resolve()
            if resolved.is_file():
                return resolved
        except (OSError, RuntimeError):
            continue
    return None


def _expand_env_vars(val: Any) -> Any:
    if isinstance(val, str):

        def _repl(m: re.Match[str]) -> str:
            var = m.group(1)
            default = m.group(3) if m.group(2) else ""
            return os.environ.get(var, default)

        return re.sub(r"\$\{([A-Za-z0-9_]+)(:-([^}]*))?\}", _repl, val)
    return val


def load_all_profiles(custom_path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Load all provider profiles defined in the configuration file.

    Profiles are expected under the table ``[profiles.<profile_name>]``.
    Returns an empty dict if no configuration file is found.
    """
    config_file = find_config_file(custom_path)
    if not config_file:
        return {}

    try:
        with config_file.open("rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        logger.warning("Failed to parse configuration file %s: %s", config_file, exc)
        return {}

    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        return {}
    res: dict[str, dict[str, Any]] = {}
    for k, v in profiles.items():
        if isinstance(v, dict):
            expanded_v = {str(prop): _expand_env_vars(val) for prop, val in v.items()}
            res[str(k)] = expanded_v
    return res


def load_provider_profile(name: str, custom_path: Path | str | None = None) -> dict[str, Any]:
    """Load a specific provider profile by name.

    Raises:
        ProfileNotFoundError: If the profile is not found or config file does not exist.
    """
    profiles = load_all_profiles(custom_path)
    if name not in profiles:
        available = list(profiles.keys())
        msg = (
            f"Provider profile '{name}' not found. "
            f"Available profiles: {available if available else 'none'}"
        )
        raise ProfileNotFoundError(msg)
    return profiles[name]

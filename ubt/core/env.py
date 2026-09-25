"""Host-environment probes and the environment handed to child processes.

Lives in ``core`` (not the adapters) so callers such as the TUI can probe without
importing a PDF adapter — ``docling_adapter`` pulled in the whole Docling/torch
import graph just to answer "is there a GPU". The probe itself is pure torch and
must stay import-cheap, which is also why :func:`subprocess_env` is stdlib-only.
"""

from __future__ import annotations

import os

#: Substrings that mark an environment variable as a credential.
_SECRET_ENV_MARKERS = ("API_KEY", "APIKEY", "ACCESS_KEY", "PRIVATE_KEY", "SECRET")

#: Suffixes that mark an environment variable as a credential.
_SECRET_ENV_SUFFIXES = ("_SECRET", "_PASSWORD", "_PASSWD", "_CREDENTIALS", "_TOKEN")

#: Credentials a *child* legitimately needs for its own model download.
#: Stripping these would break the COMET scorer on gated weights without
#: protecting anything the parent holds for itself.
_SECRET_ENV_ALLOWED = frozenset({"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"})


def _is_secret_env_name(name: str) -> bool:
    """Whether ``name`` looks like a credential this process should not leak."""
    upper = name.upper()
    if upper in _SECRET_ENV_ALLOWED:
        return False
    return any(marker in upper for marker in _SECRET_ENV_MARKERS) or upper.endswith(
        _SECRET_ENV_SUFFIXES
    )


def subprocess_env() -> dict[str, str]:
    """``os.environ`` minus credential-shaped variables.

    The external binaries this project drives (typst, node/MathJax, pandoc,
    pdftocairo, the COMET scorer) inherit the full environment today, keys
    included, so a poisoned dependency or toolchain plugin can read the LLM
    credentials the parent holds, preventing credential leakage to
    untrusted child processes or plugin toolchains.

    A denylist — not an allowlist — is deliberate: ``PATH``, ``HOME``,
    ``XDG_*``, fontconfig, ``HF_HOME`` and ``TORCH_*`` must survive or the
    child stops working, and minimizing the environment wholesale breaks real
    functionality (typst fonts, node startup, COMET weight caching). Pass the
    result as ``env=`` to every spawn.
    """
    return {name: value for name, value in os.environ.items() if not _is_secret_env_name(name)}


def has_accelerator() -> bool:
    """Check if a PyTorch CUDA or Apple MPS accelerator is available."""
    try:
        import torch

        return bool(
            torch.cuda.is_available()
            or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        )
    except Exception:
        return False

"""``subprocess_env``: the credential denylist handed to every external binary.

Every external binary (typst/node/pandoc/pdftocairo/the COMET scorer) inherits
the parent environment, so the spawn sites pass ``env=subprocess_env()``
instead. The two properties that matter are *secrets are stripped* and *the
child still works* — an allowlist would satisfy the first by breaking the
second, which is why these tests pin both directions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from ubt.core.env import subprocess_env

SAMPLE = {
    "PATH": "/usr/bin",
    "HOME": "/home/tester",
    "XDG_CONFIG_HOME": "/home/tester/.config",
    "HF_HOME": "/home/tester/.cache/huggingface",
    "TORCH_HOME": "/home/tester/.cache/torch",
    "UBT_DB_DIR": ".ubt/ledgers",
    "OPENAI_API_KEY": "sk-parent-secret",
    "ANTHROPIC_API_KEY": "sk-ant-parent-secret",
    "UBT_LLM_API_KEY": "parent-secret",
    "AWS_ACCESS_KEY_ID": "AKIA-parent",
    "GITLAB_PRIVATE_KEY": "pk",
    "DB_PASSWORD": "pw",
    "APP_PASSWD": "pw",
    "OAUTH_CLIENT_SECRET": "cs",
    # `*SECRET_KEY` used to match neither the markers nor the `_SECRET` suffix,
    # so a Django/Rails-style key reached every child spawn.
    "DJANGO_SECRET_KEY": "cs",
    "GH_CREDENTIALS": "cred",
    "SLACK_API_TOKEN": "xoxp-parent",
    "HF_TOKEN": "hf-legit-child-token",
    "HUGGING_FACE_HUB_TOKEN": "hf-legit-child-token",
}


@pytest.fixture
def sample_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", dict(SAMPLE))


def test_credential_shaped_names_are_stripped(sample_env: None) -> None:
    env = subprocess_env()
    for name, value in SAMPLE.items():
        if "parent-secret" in value or value in {
            "AKIA-parent",
            "pk",
            "pw",
            "cs",
            "cred",
            "xoxp-parent",
        }:
            assert name not in env, f"{name} leaked its value to a child process"


def test_functional_names_survive(sample_env: None) -> None:
    env = subprocess_env()
    for name in ("PATH", "HOME", "XDG_CONFIG_HOME", "HF_HOME", "TORCH_HOME", "UBT_DB_DIR"):
        assert env.get(name) == SAMPLE[name], f"{name} is what makes the child usable"


def test_secret_key_shaped_names_are_stripped(sample_env: None) -> None:
    """``*SECRET_KEY`` is a credential too.

    The markers covered ``API_KEY``/``ACCESS_KEY``/``PRIVATE_KEY`` and the
    suffixes covered ``_SECRET``, so ``DJANGO_SECRET_KEY`` — a ``_KEY`` name that
    contains but does not end with ``SECRET`` — slipped through to typst, node,
    pandoc and the COMET scorer.
    """
    env = subprocess_env()
    assert "DJANGO_SECRET_KEY" not in env
    assert "APP_SECRET_KEY" not in env
    # The allowlist still wins: the scorer needs its own gated-weights token.
    assert env.get("HF_TOKEN") == SAMPLE["HF_TOKEN"]


def test_hf_download_tokens_are_allowed_for_the_child(sample_env: None) -> None:
    """Gated COMET weights download inside the scorer; stripping breaks it."""
    env = subprocess_env()
    assert env.get("HF_TOKEN") == SAMPLE["HF_TOKEN"]
    assert env.get("HUGGING_FACE_HUB_TOKEN") == SAMPLE["HUGGING_FACE_HUB_TOKEN"]


def test_matching_is_case_insensitive_and_value_agnostic(
    sample_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(os.environ, "openai_api_key", "lowercase-name")
    assert "openai_api_key" not in subprocess_env()


def test_returns_a_copy_not_os_environ_itself(sample_env: None) -> None:
    env: Any = subprocess_env()
    env["INJECTED"] = "x"
    assert "INJECTED" not in os.environ


#: ``subprocess`` entry points that spawn a child process.
_SPAWN_FUNCS = frozenset({"run", "Popen", "call", "check_output", "check_call"})


def test_every_spawn_site_passes_a_scrubbed_env() -> None:
    """The denylist is only worth having if *every* spawn uses it.

    An AST scan, not a grep: ``env=subprocess_env()`` sits on its own line in
    these calls, so a line-based search silently passes a site that dropped it.
    """
    import ast

    repo_root = Path(__file__).resolve().parent.parent.parent
    offenders: list[str] = []
    for path in sorted((repo_root / "ubt").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr in _SPAWN_FUNCS
                and "env" not in {kw.arg for kw in node.keywords}
            ):
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
    assert not offenders, f"subprocess spawn without env=subprocess_env(): {offenders}"

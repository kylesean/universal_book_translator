"""The sensitive-path deny list enumerates the credential stores it claims to.

F1 regression: the list covered ``.ssh``/``.aws``/``.docker``/``.kube`` but
omitted the tool-level credential stores (``.npmrc`` publish tokens,
``.pypirc`` upload tokens, ``.wgetrc``/``.curlrc``), the Azure/GCloud CLI
token caches, browser profiles outside ``.config`` (``.mozilla``), and the
remaining SSH key files — all readable by the submit/MCP surfaces on a
home-wide sandbox.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.fs_perms import (
    SENSITIVE_FILENAME_PARTS,
    is_sensitive_path_part,
)

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    "part",
    [
        # Already covered and must stay covered (regression guard on the base set).
        ".ssh",
        ".aws",
        ".docker",
        ".kube",
        ".gnupg",
        ".netrc",
        ".git-credentials",
        "id_rsa",
        "id_ed25519",
        # F1 additions: tool credential stores.
        ".npmrc",
        ".pypirc",
        ".my.cnf",
        ".wgetrc",
        ".curlrc",
        ".htpasswd",
        ".gitcookies",
        # F1 additions: cloud CLI credential caches.
        ".azure",
        ".gcloud",
        # F1 additions: browser profiles outside the (already denied) .config.
        ".mozilla",
        # F1 additions: the remaining SSH private keys and their .pub halves.
        "id_dsa",
        "id_ecdsa",
        "id_rsa.pub",
        "id_dsa.pub",
        "id_ecdsa.pub",
        "id_ed25519.pub",
    ],
)
def test_credential_store_names_are_denied(part: str) -> None:
    assert part in SENSITIVE_FILENAME_PARTS
    assert is_sensitive_path_part(part)


def test_the_whole_config_tree_is_denied_so_its_credential_stores_are_covered() -> None:
    # .config/gcloud, .config/chromium, .config/google-chrome are covered by
    # the ".config" entry; assert they are NOT re-listed as duplicates.
    assert ".config" in SENSITIVE_FILENAME_PARTS
    assert not any(part.startswith(".config/") for part in SENSITIVE_FILENAME_PARTS)


@pytest.mark.parametrize(
    "part",
    [".SSH", ".Npmrc", "ID_RSA", "ID_ED25519.PUB", ".My.CNF"],
)
def test_matching_is_casefolded(part: str) -> None:
    assert is_sensitive_path_part(part)


@pytest.mark.parametrize(
    "part",
    [".env.production", ".envrc", ".env.local"],
)
def test_the_dynamic_dotenv_rule_still_applies(part: str) -> None:
    assert is_sensitive_path_part(part)


@pytest.mark.parametrize(
    "part",
    ["books", "chapter1.md", "notenv.txt"],
)
def test_ordinary_names_still_pass(part: str) -> None:
    assert not is_sensitive_path_part(part)


def test_resolve_secure_path_refuses_a_home_npmrc(tmp_path: Path) -> None:
    """The deny list still wins over an explicit allowlist, for the new entries too."""
    from fastapi import HTTPException

    from ubt.api.security import resolve_secure_path
    from ubt.core.config import UBTConfig

    (tmp_path / ".npmrc").write_text("//registry.example/:_authToken=secret", encoding="utf-8")
    config = UBTConfig(db_dir=tmp_path / "db", allowed_dirs=str(tmp_path))
    with pytest.raises(HTTPException) as err:
        resolve_secure_path(tmp_path / ".npmrc", config=config)
    assert err.value.status_code == 403
    assert "sensitive" in str(err.value.detail)

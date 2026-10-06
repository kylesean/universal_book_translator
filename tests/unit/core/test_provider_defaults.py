"""The [defaults] TOML block accepts run-level UBTConfig fields like db_dir."""

from __future__ import annotations

import pytest

from ubt.core.providers import PROVIDER_ALLOWED_KEYS, ProviderConfigError, _validate_block

pytestmark = pytest.mark.fast


def test_db_dir_is_allowed_in_a_toml_block() -> None:
    # db_dir is a real UBTConfig field (--db-dir / UBT_DB_DIR); it must be
    # settable from the config file too, or the same key is settable on every
    # surface except TOML.
    assert "db_dir" in PROVIDER_ALLOWED_KEYS
    assert _validate_block("[defaults]", {"db_dir": "/tmp/ledgers"}) == {"db_dir": "/tmp/ledgers"}


def test_an_unknown_key_is_still_rejected() -> None:
    with pytest.raises(ProviderConfigError):
        _validate_block("[defaults]", {"not_a_real_field": "x"})

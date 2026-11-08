"""The shipped example config must parse with the real loader.

The sample once drifted into a fictional schema — ``[project]``/``[model]``/
``[render]`` tables the document model rejects, plus a usage line naming a
``--config`` flag the CLI never had. A first-time user copying it hit loud
rejections for settings that looked canonical. This test pins the example to
the actual TOML contract so schema drift fails CI instead of the newcomer.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "examples" / "sample_book" / "ubt_sample_config.toml"


def _secret(value: object) -> str:
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)


def test_the_sample_config_parses_and_selects_its_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.config import UBTConfig

    (tmp_path / "ubt.toml").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-sample-test")

    config = UBTConfig.from_env()

    assert config.provider == "deepseek"
    assert config.base_url == "https://api.deepseek.com/v1"
    assert config.draft_model == "deepseek-chat"
    assert _secret(config.api_key) == "sk-sample-test"


def test_an_unresolved_credential_reference_leaves_no_usable_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """${DEEPSEEK_API_KEY} unset expands to empty — never a silently usable key.

    The run then has no credential and ``require_api_key`` fails loudly at
    launch, which is the honest behaviour: a sample whose variable is not set
    must not look configured.
    """
    from ubt.core.config import UBTConfig, has_real_credential

    (tmp_path / "ubt.toml").write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    config = UBTConfig.from_env()

    assert _secret(config.api_key) == ""
    assert has_real_credential(config) is False

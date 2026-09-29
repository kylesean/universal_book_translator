"""Cross-field config validation and endpoint invariants.

A repair mode with a zero budget, best-of-n rerank driven by a heuristic score
that cannot rank, and the rule that the protocol comes only from explicit
configuration. The flat field surface (and its ``UBT_<FIELD>`` env names / CLI
flags) is canonical; there used to be four frozen ``.route`` / ``.qe`` /
``.render`` / ``.pdf`` views over it, which nothing in ``ubt/`` ever read -- 41
duplicated field declarations plus a test that only proved the mirror still
reflected. Deleted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from ubt.core.config import UBTConfig
from ubt.core.job_options import apply_config_overrides


def test_consistency_repair_requires_positive_budget() -> None:
    with pytest.raises(ValueError, match="consistency_max_repairs"):
        UBTConfig(consistency_enforce="repair", consistency_max_repairs=0)
    # Detect-only and a bounded repair budget are both valid.
    assert UBTConfig(consistency_enforce="report", consistency_max_repairs=0)
    assert UBTConfig(consistency_enforce="repair", consistency_max_repairs=5)
    assert UBTConfig(consistency_enforce="off", consistency_max_repairs=0)


def test_rerank_with_heuristic_engine_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="ubt.core.config"):
        cfg = UBTConfig(rerank_k=3, qe_engine="heuristic")
    assert cfg.rerank_k == 3
    assert "rerank_k=3 has no effect" in caplog.text

    # A neural runner can rank, so the warning must not fire.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="ubt.core.config"):
        UBTConfig(rerank_k=3, qe_engine="subprocess")
    assert "no effect" not in caplog.text


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "127.0.0.1:11434/v1",  # the classic: a URL with the scheme left off
        "file:///tmp/provider-key",
        "ftp://host.example/v1",
        "http://::1:11434/v1",  # bracketless IPv6 — urlsplit reports no host
    ],
)
def test_base_url_must_name_an_http_target(bad: str) -> None:
    """The base URL decides where the API key goes, so a typo fails at parse.

    It used to survive validation and surface as an httpx error on the first
    request — after preflight had already estimated a cost — or, for a
    non-http scheme, as a silent no-op.
    """
    with pytest.raises(ValidationError, match="base_url"):
        UBTConfig(base_url=bad)


def test_acceptable_base_urls_still_parse() -> None:
    """The shapes real deployments use must not be caught by the guard."""
    assert UBTConfig(base_url="http://[::1]:11434/v1").base_url == "http://[::1]:11434/v1"
    assert UBTConfig(base_url="http://127.0.0.1:11434/v1").base_url == "http://127.0.0.1:11434/v1"


@pytest.mark.parametrize(
    "raw",
    [
        "https://generativelanguage.googleapis.com",
        "https://generativelanguage.googleapis.com/v1beta",
        "https://api.anthropic.com",
        "https://opencode.ai/zen/go/v1",
    ],
)
def test_a_hostname_never_selects_the_protocol(raw: str) -> None:
    """The endpoint says *where* to call; only configuration says *which wire*.

    These hosts used to re-route the protocol by sniffing (``api.anthropic.com``
    -> anthropic, the Gemini host -> an OpenAI-compat rewrite). A provider is
    just an endpoint that speaks one of four protocols, so an unset ``api_mode``
    stays at the default no matter the host.
    """
    assert UBTConfig(base_url=raw).api_mode == "openai-chat"


def test_a_model_name_never_selects_the_protocol() -> None:
    """``muse-`` used to force the responses wire; the model picks no protocol."""
    assert UBTConfig(draft_model="muse-spark-1.3-contributor").api_mode == "openai-chat"


def test_an_override_does_not_move_the_protocol() -> None:
    """``apply_config_overrides`` must not re-derive a protocol from a new model."""
    base = UBTConfig(draft_model="gpt-4o")
    cfg = apply_config_overrides(
        base,
        {"draft_model": "muse-spark-1.3-contributor", "base_url": "https://api.anthropic.com"},
    )
    assert cfg.api_mode == "openai-chat"
    assert (
        cfg.api_mode
        == UBTConfig.from_env(
            draft_model="muse-spark-1.3-contributor", base_url="https://api.anthropic.com"
        ).api_mode
    )


def test_explicit_api_mode_is_honoured() -> None:
    cfg = apply_config_overrides(
        UBTConfig(draft_model="gpt-4o"),
        {"api_mode": "openai-responses"},
    )
    assert cfg.api_mode == "openai-responses"


def test_programmatic_allowed_dir_not_overridden_by_ambient_allowed_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit programmatic allowed_dir must not be silently dropped by ambient UBT_ALLOWED_DIRS."""
    env_dir = tmp_path / "env_allowed"
    env_dir.mkdir()
    prog_dir = tmp_path / "prog_allowed"
    prog_dir.mkdir()

    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(env_dir))
    cfg = UBTConfig(allowed_dir=str(prog_dir))

    # The programmatic allowed_dir must be included in allowed bases
    bases = cfg.allowed_base_dirs()
    assert prog_dir.resolve() in bases

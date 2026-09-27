"""Cross-field config validation and endpoint-derived invariants.

A repair mode with a zero budget, best-of-n rerank driven by a heuristic score
that cannot rank, and the endpoint/model -> ``api_mode`` derivation. The flat
field surface (and its ``UBT_<FIELD>`` env names / CLI flags) is canonical;
there used to be four frozen ``.route`` / ``.qe`` / ``.render`` / ``.pdf`` views
over it, which nothing in ``ubt/`` ever read -- 41 duplicated field declarations
plus a test that only proved the mirror still reflected. Deleted.
"""

from __future__ import annotations

import logging

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
    # The gemini shortcut is normalized before the check runs.
    assert UBTConfig(base_url="gemini").base_url.startswith("https://generativelanguage.")


@pytest.mark.parametrize(
    "raw",
    [
        "gemini",
        "https://generativelanguage.googleapis.com",
        "https://generativelanguage.googleapis.com/v1beta",
    ],
)
def test_gemini_endpoint_is_normalized_to_the_openai_route(raw: str) -> None:
    """Every Gemini spelling must land on the ``/v1beta/openai`` compatibility route."""
    assert (
        UBTConfig(base_url=raw).base_url
        == "https://generativelanguage.googleapis.com/v1beta/openai"
    )


def test_anthropic_endpoint_selects_anthropic_mode() -> None:
    # The host decides the wire; an explicit api_mode still wins.
    assert UBTConfig(base_url="https://api.anthropic.com").api_mode == "anthropic"
    assert UBTConfig(base_url="https://api.anthropic.com", api_mode="chat").api_mode == "chat"


def test_muse_model_selects_responses_mode() -> None:
    assert UBTConfig(draft_model="muse-spark-1.3-contributor").api_mode == "responses"


def test_api_mode_rederives_after_a_draft_override() -> None:
    """A request override that moves off a muse- model must drop responses mode.

    ``apply_config_overrides`` assigns one field at a time, so the ``responses``
    api_mode derived from a muse- draft used to latch in ``model_fields_set`` and
    block re-derivation, making the request path disagree with ``from_env``.
    """
    base = UBTConfig(draft_model="muse-spark-1.3-contributor")
    assert base.api_mode == "responses"

    via_request = apply_config_overrides(base, {"draft_model": "gpt-4o"})
    assert via_request.api_mode == "chat"
    assert via_request.api_mode == UBTConfig.from_env(draft_model="gpt-4o").api_mode


def test_api_mode_rederives_after_an_endpoint_override() -> None:
    """Moving the endpoint to Anthropic must re-select anthropic mode."""
    base = UBTConfig(draft_model="muse-spark-1.3-contributor")
    assert base.api_mode == "responses"

    cfg = apply_config_overrides(
        base, {"draft_model": "gpt-4o", "base_url": "https://api.anthropic.com"}
    )
    assert cfg.api_mode == "anthropic"


def test_explicit_api_mode_survives_derivation() -> None:
    cfg = apply_config_overrides(
        UBTConfig(draft_model="muse-spark-1.3-contributor"),
        {"draft_model": "gpt-4o", "api_mode": "responses"},
    )
    assert cfg.api_mode == "responses"

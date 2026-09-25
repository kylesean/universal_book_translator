"""Cross-field config validation: the two invariants that used to be silent no-ops.

A repair mode with a zero budget, and best-of-n rerank driven by a heuristic
score that cannot rank. The flat field surface (and its ``UBT_<FIELD>`` env
names / CLI flags) is canonical; there used to be four frozen ``.route`` /
``.qe`` / ``.render`` / ``.pdf`` views over it, which nothing in ``ubt/`` ever
read -- 41 duplicated field declarations plus a test that only proved the
mirror still reflected. Deleted.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from ubt.core.config import UBTConfig


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
    # The profile shortcut is normalized before the check runs.
    assert UBTConfig(base_url="gemini").base_url.startswith("https://generativelanguage.")

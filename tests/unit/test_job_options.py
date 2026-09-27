"""``ubt.core.job_options`` is the single owner of the request-key contract.

This is its canonical test file (AGENTS.md "One Behavior, One Home"). That
module decides which request keys become ``UBTConfig`` overrides — a key is
recognised exactly when it names a config field or a ``RUN_ONLY_KEYS`` entry —
and it exists precisely because hand-maintained per-surface copies drift. What
it cannot police from inside is whether each *shell* actually carries the keys.

The CLI builds its request mapping inline and has always been able to set every
engine knob. The two *payload-gated* shells are allowlists, so a knob missing
from either is a knob that shell's caller simply cannot ask for:

- the REST ``JobSubmitRequest`` model (``extra="forbid"`` turns an unknown key
  into a 422 before a job exists), and
- the MCP ``ubt_translate_book`` signature (its parameters *are* the payload).

Both were found narrow: the REST model carried 19 of the CLI's 40+
keys and the MCP tool carried 17, so neither could cap spend, raise the
concurrency ceiling, choose the OCR engine or pin the formula policy. These
assertions are the guard against a third occurrence.
"""

from __future__ import annotations

import inspect

from ubt.api.models import JobSubmitRequest
from ubt.core.config import UBTConfig
from ubt.core.job_options import RUN_ONLY_KEYS
from ubt.mcp.server import ubt_translate_book

#: Knobs deliberately exposed on the REST surface only, with the reason.
#: Empty today: every config knob the REST payload accepts is also an MCP
#: parameter (``priority`` is queue scheduling, not a ``UBTConfig`` field, so it
#: is not part of this comparison).
_API_ONLY_EXCEPTIONS: dict[str, str] = {}

#: Config knobs deliberately exposed on the MCP surface only, with the reason.
_MCP_ONLY_EXCEPTIONS: dict[str, str] = {
    # Server-owned storage: the REST service picks its own ledger directory
    # from config, while the MCP server is a local stdio process whose operator
    # may legitimately point it elsewhere (under its own path sandbox).
    "db_dir": "server-owned on REST; operator-owned on a local stdio server",
}


def _config_knobs(names: set[str]) -> set[str]:
    """The subset of ``names`` the shared mapping would actually apply."""
    return {name for name in names if name in UBTConfig.model_fields}


def _api_params() -> set[str]:
    return set(JobSubmitRequest.model_fields)


def _mcp_params() -> set[str]:
    return set(inspect.signature(ubt_translate_book).parameters)


def test_rest_and_mcp_expose_the_same_config_knobs() -> None:
    api = _config_knobs(_api_params())
    mcp = _config_knobs(_mcp_params())

    api_only = sorted(api - mcp - set(_API_ONLY_EXCEPTIONS))
    assert api_only == [], (
        "UBTConfig knobs the REST payload accepts but the MCP tool cannot: "
        f"{api_only} — add them to ubt/mcp/server.py::ubt_translate_book, or "
        "record the reason in _API_ONLY_EXCEPTIONS"
    )

    mcp_only = sorted(mcp - api - set(_MCP_ONLY_EXCEPTIONS))
    assert mcp_only == [], (
        "UBTConfig knobs the MCP tool accepts but the REST payload cannot: "
        f"{mcp_only} — add them to ubt/api/models.py::JobSubmitRequest, or "
        "record the reason in _MCP_ONLY_EXCEPTIONS"
    )


def test_both_surfaces_carry_every_run_only_key() -> None:
    """``RUN_ONLY_KEYS`` is the other half of the shared contract.

    These keys never touch ``UBTConfig`` — they are consumed by
    ``PipelineOrchestrator.run`` — so the config-knob comparison cannot see
    them. ``start_chapter``/``max_chapters`` were exactly the case that slipped
    through: neither shell could scope a run to a chapter window.
    """
    for surface, params in (("REST", _api_params()), ("MCP", _mcp_params())):
        missing = sorted(set(RUN_ONLY_KEYS) - params)
        assert missing == [], f"{surface} payload cannot carry run-only keys: {missing}"


def test_the_exception_lists_have_not_gone_stale() -> None:
    """An exception for a knob that is now on both surfaces is a lie.

    Without this the lists rot into "reasons nobody re-checked", and the drift
    they document returns.
    """
    api = _config_knobs(_api_params())
    mcp = _config_knobs(_mcp_params())

    for knob, reason in _API_ONLY_EXCEPTIONS.items():
        assert reason, knob
        assert knob in api, f"{knob!r} is no longer a REST config knob"
        assert knob not in mcp, f"{knob!r} is now on MCP; drop the exception"
    for knob, reason in _MCP_ONLY_EXCEPTIONS.items():
        assert reason, knob
        assert knob in mcp, f"{knob!r} is no longer an MCP config knob"
        assert knob not in api, f"{knob!r} is now on REST; drop the exception"


def test_fresh_none_does_not_override_environment() -> None:
    """REST's concrete ``fresh=False`` silently erased ``UBT_FRESH``; None must be skipped."""
    from ubt.core.job_options import overrides_from_request

    assert "fresh" not in overrides_from_request({"input_path": "x.pdf", "fresh": None})
    explicit = overrides_from_request({"input_path": "x.pdf", "fresh": True})
    assert explicit["fresh"] is True


def test_overrides_apply_shared_adaptive_dual_mode() -> None:
    """Profile/engine-aware dual_mode default must be identical on every surface.

    Only the CLI applied it before, so an academic paper rendered bilingual
    through the API/MCP and monolingual through the CLI.
    """
    from ubt.core.job_options import overrides_from_request

    assert overrides_from_request({"profile": "paper"})["dual_mode"] == "monolingual"
    assert overrides_from_request({"render_engine": "rigid"})["dual_mode"] == "monolingual"
    # General on reflow stays unset (follows config / UBT_DUAL_MODE).
    assert "dual_mode" not in overrides_from_request({"profile": "general"})
    # An explicit choice is never overridden.
    assert (
        overrides_from_request({"profile": "paper", "dual_mode": "facing"})["dual_mode"] == "facing"
    )


def test_validate_request_enums_rejects_unknown_values() -> None:
    """Shared enum validation: every surface gets the same upfront error.

    CLI (typer Literal) and REST (pydantic) already rejected these; MCP took
    bare strings and only failed the job inside its background task.
    """
    import pytest

    from ubt.core.exceptions import UBTError
    from ubt.core.job_options import overrides_from_request, validate_request_enums

    # Known vocabulary (including the config-only 'publication' engine) passes.
    validate_request_enums(
        {
            "render_engine": "publication",
            "preset": "fast",
            "qe_engine": "comet",
            "dual_mode": "auto",
        }
    )
    with pytest.raises(UBTError, match="Invalid render_engine"):
        validate_request_enums({"render_engine": "bogus"})
    with pytest.raises(UBTError, match="Invalid preset"):
        overrides_from_request({"preset": "turbo"})
    with pytest.raises(UBTError, match="Invalid ocr_mode"):
        overrides_from_request({"ocr_mode": "telepathy"})

"""Every job-option surface must expose the same engine knobs.

The console reached a state where ``render_mode`` existed in the CLI and the
engine but in neither the REST model nor the MCP tool, so the web UI had no way
to select the semantic reflow path that the rest of this work productized. The
bug was invisible to the engine tests — they call the pipeline directly and
never cross the wire.

This test closes that class of bug by asserting the three surfaces agree on the
knobs a user can set, rather than asserting any single surface's contents.
"""

from __future__ import annotations

import inspect

import pytest

from ubt.api.models import JobSubmitRequest
from ubt.core.exceptions import UBTError
from ubt.core.job_options import (
    _ENUM_REQUEST_FIELDS,
    overrides_from_request,
    validate_request_enums,
)

pytestmark = pytest.mark.fast

#: The mode knobs a user may set from any surface. Deliberately short: the
#: converged design puts exactly two decisions in front of a user — what the
#: reader receives (``dual_mode``) and how much effort to spend (``preset``) —
#: plus ``cover_mode``, which is a clear, low-frequency layout choice. Every
#: other mode field is an engine detail set by a preset or the operator's
#: ``UBT_*`` environment.
_USER_MODE_KNOBS = ("dual_mode", "preset", "cover_mode")

#: Engine details that must NOT reappear on a user surface. Each one is either
#: inert (measured: ``formula_render`` changes no behavior), already implied by
#: the renderer (``table_render_mode``), or a preset's business
#: (``exec_mode``/``prompt_strategy``/``qe_engine``/``formula_enrichment``).
_INTERNAL_KNOBS = (
    "exec_mode",
    "prompt_strategy",
    "qe_engine",
    "table_render_mode",
    "formula_enrichment",
    "formula_render",
    "formula_mode",
    "math_backend",
)


@pytest.mark.parametrize("knob", _INTERNAL_KNOBS)
def test_an_internal_knob_is_absent_from_every_user_surface(knob: str) -> None:
    """An engine detail must not be offered as a choice.

    Offering one is not neutral: it multiplies the documented/tested surface and
    invites a user to set a value the renderer ignores. ``formula_render`` was
    the clearest case — all three of its values produced identical behavior.
    """
    from ubt.core.config import UBTConfig

    assert knob not in JobSubmitRequest.model_fields, f"{knob} is back on REST"
    assert knob not in _mcp_tool_signature_names(), f"{knob} is back on MCP"
    assert knob not in _ENUM_REQUEST_FIELDS, f"{knob} is back in the enum validator"
    # It may still exist as a config field (presets and env set it) …
    if knob in UBTConfig.model_fields:
        # … but then it must be reachable by environment, or a preset is the
        # only way to change it and the field is effectively frozen.
        from ubt.core.config import env_var_names

        assert env_var_names(knob), f"{knob} has no env escape hatch"


def _mcp_tool_signature_names() -> set[str]:
    """The parameter names the MCP translation tool declares.

    Read from the source rather than by importing the FastMCP decorator: the
    tool is registered through a framework object whose wrapped function is not
    the one defined in the module.
    """
    from ubt.mcp import server

    source = inspect.getsource(server)
    marker = "async def ubt_translate_book("
    assert marker in source, "MCP translate tool was renamed; update this test"
    body = source.split(marker, 1)[1]
    # The signature ends at the first closing paren at column 0.
    signature = body.split("\n) ->", 1)[0]
    return {
        line.strip().split(":", 1)[0]
        for line in signature.splitlines()
        if ":" in line and not line.strip().startswith("#")
    }


@pytest.mark.parametrize("knob", _USER_MODE_KNOBS)
def test_the_rest_model_exposes_the_knob(knob: str) -> None:
    """A knob absent from the request model cannot arrive over REST at all."""
    assert knob in JobSubmitRequest.model_fields


@pytest.mark.parametrize("knob", _USER_MODE_KNOBS)
def test_the_mcp_tool_exposes_the_knob(knob: str) -> None:
    """MCP builds its payload from an explicit dict; a missing arg is a no-op."""
    assert knob in _mcp_tool_signature_names()


@pytest.mark.parametrize("knob", _USER_MODE_KNOBS)
def test_the_knob_reaches_the_engine_config(knob: str) -> None:
    """The override mapper must translate the request key into engine state.

    ``overrides_from_request`` filters plain keys against
    ``UBTConfig.model_fields``, so a request key with no matching config field is
    dropped without an error. ``preset`` is the exception: it is not a config
    field at all, it is *expanded* into config fields by
    ``resolve_engine_params`` before the filter runs.
    """
    from ubt.core.config import UBTConfig

    value = _ENUM_REQUEST_FIELDS[knob][0]
    if knob == "preset":
        overrides = overrides_from_request({knob: value})
        assert overrides, "a preset must expand into engine parameters"
        assert all(key in UBTConfig.model_fields for key in overrides)
        return
    assert knob in UBTConfig.model_fields
    overrides = overrides_from_request({knob: value})
    assert knob in overrides


@pytest.mark.parametrize("knob", _USER_MODE_KNOBS)
def test_an_out_of_vocabulary_value_is_rejected_up_front(knob: str) -> None:
    """A bad value must fail before a job id is handed out.

    MCP accepts bare strings, so without the shared validator it returned a
    job_id and the job failed later in the background — the caller saw success.
    """
    with pytest.raises(UBTError, match=knob):
        validate_request_enums({knob: "definitely-not-a-real-value"})

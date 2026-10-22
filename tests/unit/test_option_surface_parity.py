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

import ast
import dataclasses
import inspect
from pathlib import Path

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
#: reader receives (``dual_mode``) and how much effort to spend (``preset``).
#: Every other mode field is an engine detail set by a preset or the operator's
#: ``UBT_*`` environment.
_USER_MODE_KNOBS = ("dual_mode", "preset")

#: Engine details that must NOT reappear on a user surface. Each one is either
#: inert (measured: ``formula_render`` and ``cover_mode`` change no behavior),
#: already implied by the renderer (``table_render_mode``), or a preset's
#: business (``exec_mode``/``prompt_strategy``/``qe_engine``/
#: ``formula_enrichment``).
_INTERNAL_KNOBS = (
    "exec_mode",
    "prompt_strategy",
    "qe_engine",
    "table_render_mode",
    "formula_enrichment",
    "formula_crop_margin",
    "toc_mode",
    "formula_render",
    "formula_mode",
    "math_backend",
    "cover_mode",
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


# --------------------------------------------------------------------------- #
# Knobs deleted for being inert, not merely internal
# --------------------------------------------------------------------------- #


def test_cover_mode_is_gone_from_the_config_too() -> None:
    """``cover_mode`` was never implemented — delete it, do not hide it.

    Every other knob in ``_INTERNAL_KNOBS`` is a real engine decision moved to a
    preset; ``cover_mode`` was different. It was plumbed through the CLI, REST
    and MCP surfaces and stored on the ``RenderPlan``, but nothing ever read it:
    ``TypstDocumentRenderer`` does not accept it, and end-to-end runs with
    ``auto`` / ``always`` / ``never`` produced byte-identical text layers and
    identical page counts. A user-visible switch that changes nothing is worse
    than an absent one, because the help text claims otherwise.
    """
    from ubt.core.config import UBTConfig
    from ubt.core.ir.render_plan import RenderPlan

    assert "cover_mode" not in UBTConfig.model_fields
    assert "cover_mode" not in {f.name for f in dataclasses.fields(RenderPlan)}


def test_the_dead_overlay_table_mode_is_not_accepted() -> None:
    """``table_render_mode='overlay'`` named a render that no longer exists.

    Its whole meaning was "translate simple tables, leave complex ones on the
    source page". There is no source page: a kept table is simply absent from
    the rebuilt document. Accepting the value silently reinterpreted a stale
    config as ``auto``, so a user who had set it got translated tables they had
    asked to leave alone — a rejected value is the honest outcome.
    """
    import typing

    from ubt.core.config import TableRenderMode

    assert "overlay" not in typing.get_args(TableRenderMode)

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UBTConfig_from_env(env={"UBT_TABLE_RENDER_MODE": "overlay"})


def UBTConfig_from_env(*, env: dict[str, str]) -> object:
    """Build a config with ``env`` applied, restoring the process env after."""
    import os

    from ubt.core.config import UBTConfig

    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return UBTConfig.from_env()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_the_render_metadata_keys_have_a_writer() -> None:
    """Every artifact metadata key must be *written* by something.

    ``render_engine_effective`` and ``render_padding_pages`` were carried in
    ``ARTIFACT_METADATA_KEYS`` after their writers were deleted with the overlay
    engine — so the set advertised passthrough for keys the pipeline never
    produces. This guard is cheap and catches the next one.

    The check is AST-based on purpose. The earlier version shelled out to
    ``grep`` for the quoted key, so a *reader* — or even a comment — satisfied
    it, which is how ``chapter_translations`` stayed listed while nothing has
    ever written it.
    """
    from ubt.core.ir.run_metadata import ARTIFACT_METADATA_KEYS

    root = Path(__file__).resolve().parents[2]
    written: dict[str, list[str]] = {key: [] for key in ARTIFACT_METADATA_KEYS}

    for path in sorted((root / "ubt").rglob("*.py")):
        if path.name == "run_metadata.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for key in _metadata_keys_written(node):
                if key in written:
                    written[key].append(str(path.relative_to(root)))

    # ``chapter_translations`` is the one deliberate exception: it is a
    # *caller-supplied* contract (the bible stage reads it to seed chapter
    # titles), never produced by this pipeline, so it has no writer here.
    writer_optional = {"chapter_translations"}
    missing = sorted(
        key for key, paths in written.items() if not paths and key not in writer_optional
    )
    assert missing == [], f"declared artifact metadata key(s) with no writer: {missing}"


def _metadata_keys_written(node: ast.AST) -> list[str]:
    """The metadata keys a node writes.

    Recognises the three shapes the pipeline uses: ``metadata["key"] = ...``,
    a ``{"key": ...}`` literal assigned to a name containing ``metadata``, and a
    ``metadata={...}`` keyword argument (which may declare several keys).
    """
    keys: list[str] = []
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
                and _is_metadata_reference(target.value)
            ):
                keys.append(target.slice.value)
        if any(isinstance(t, ast.Name) and "metadata" in t.id for t in node.targets):
            keys.extend(_str_keys(node.value))
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        target = node.target
        if isinstance(target, ast.Name) and "metadata" in target.id:
            keys.extend(_str_keys(node.value))
    if isinstance(node, ast.keyword) and node.arg == "metadata":
        keys.extend(_str_keys(node.value))
    return keys


def _str_keys(node: ast.AST) -> list[str]:
    """Every string key of a dict literal."""
    if not isinstance(node, ast.Dict):
        return []
    return [
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]


def _is_metadata_reference(node: ast.AST) -> bool:
    """Whether an expression names a metadata mapping (``*.metadata``)."""
    return (isinstance(node, ast.Attribute) and node.attr == "metadata") or (
        isinstance(node, ast.Name) and "metadata" in node.id
    )


def _cli_translate_option_names() -> set[str]:
    """The parameter names ``ubt translate`` accepts, read from its source.

    AST rather than ``typer.main.get_command``: the command is registered by a
    decorator on a plain function, and the decorator's return value is not the
    ``Typer`` instance the public API wants. Reading the signature also keeps
    this test free of Typer's internals.
    """
    import ast

    from ubt.cli.commands import translate as translate_module

    tree = ast.parse(inspect.getsource(translate_module))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "translate":
            args = node.args
            return {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    raise AssertionError("ubt.cli.commands.translate.translate was renamed")


def test_a_layout_knob_offered_on_one_surface_is_offered_on_all_three() -> None:
    """The three entry points must expose the same *user* knobs.

    The earlier version of this file claimed "the three surfaces agree" while
    reading no CLI at all and comparing two knobs one directionally. Measured
    then: ``layout_columns`` was accepted by REST (``JobSubmitRequest``) and by
    the CLI (``--columns``) but absent from the MCP tool, so an agent could not
    set a layout the other two doors offered.

    The knob list is deliberately explicit — a *new* user knob is added here in
    the same commit that adds it to the surfaces, which is the point: the test
    fails until all three agree.
    """
    from ubt.api.models import JobSubmitRequest

    shared = ("layout_columns", "dual_mode", "preset", "pages")
    rest = set(JobSubmitRequest.model_fields)
    mcp = _mcp_tool_signature_names()
    cli = _cli_translate_option_names()

    missing_rest = [k for k in shared if k not in rest]
    missing_mcp = [k for k in shared if k not in mcp]
    # The CLI spells ``layout_columns`` as ``--columns``; the Typer name is
    # ``columns``. That mapping is asserted here so a rename cannot silently
    # pass.
    cli_expected = {
        "layout_columns": "columns",
        "dual_mode": "dual_mode",
        "preset": "preset",
        "pages": "pages",
    }
    missing_cli = [k for k in shared if cli_expected[k] not in cli]

    assert not missing_rest, f"REST is missing {missing_rest}"
    assert not missing_mcp, f"MCP is missing {missing_mcp}"
    assert not missing_cli, f"CLI is missing {missing_cli}"

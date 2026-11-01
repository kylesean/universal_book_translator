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

_USER_MODE_KNOBS = ("dual_mode", "preset")

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
    from ubt.core.config import UBTConfig

    assert knob not in JobSubmitRequest.model_fields, f"{knob} is back on REST"
    assert knob not in _mcp_tool_signature_names(), f"{knob} is back on MCP"
    assert knob not in _ENUM_REQUEST_FIELDS, f"{knob} is back in the enum validator"
    if knob in UBTConfig.model_fields:
        from ubt.core.config import env_var_names

        assert env_var_names(knob), f"{knob} has no env escape hatch"


def _mcp_tool_signature_names() -> set[str]:
    from ubt.mcp import server

    source = inspect.getsource(server)
    marker = "async def ubt_translate_book("
    assert marker in source, "MCP translate tool was renamed; update this test"
    body = source.split(marker, 1)[1]
    signature = body.split("\n) ->", 1)[0]
    return {
        line.strip().split(":", 1)[0]
        for line in signature.splitlines()
        if ":" in line and not line.strip().startswith("#")
    }


@pytest.mark.parametrize("knob", _USER_MODE_KNOBS)
def test_the_rest_model_exposes_the_knob(knob: str) -> None:
    assert knob in JobSubmitRequest.model_fields


@pytest.mark.parametrize("knob", _USER_MODE_KNOBS)
def test_the_mcp_tool_exposes_the_knob(knob: str) -> None:
    assert knob in _mcp_tool_signature_names()


@pytest.mark.parametrize("knob", _USER_MODE_KNOBS)
def test_the_knob_reaches_the_engine_config(knob: str) -> None:
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
    with pytest.raises(UBTError, match=knob):
        validate_request_enums({knob: "definitely-not-a-real-value"})


def test_cover_mode_is_gone_from_the_config_too() -> None:
    from ubt.core.config import UBTConfig
    from ubt.core.ir.render_plan import RenderPlan

    assert "cover_mode" not in UBTConfig.model_fields
    assert "cover_mode" not in {f.name for f in dataclasses.fields(RenderPlan)}


def test_the_dead_overlay_table_mode_is_not_accepted() -> None:
    import typing

    from ubt.core.config import TableRenderMode

    assert "overlay" not in typing.get_args(TableRenderMode)

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UBTConfig_from_env(env={"UBT_TABLE_RENDER_MODE": "overlay"})


def UBTConfig_from_env(*, env: dict[str, str]) -> object:
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

    writer_optional = {"chapter_translations"}
    missing = sorted(
        key for key, paths in written.items() if not paths and key not in writer_optional
    )
    assert missing == [], f"declared artifact metadata key(s) with no writer: {missing}"


def _metadata_keys_written(node: ast.AST) -> list[str]:
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
    if not isinstance(node, ast.Dict):
        return []
    return [
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]


def _is_metadata_reference(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "metadata") or (
        isinstance(node, ast.Name) and "metadata" in node.id
    )


def _cli_translate_option_names() -> set[str]:
    import ast

    from ubt.cli.commands import translate as translate_module

    tree = ast.parse(inspect.getsource(translate_module))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "translate":
            args = node.args
            return {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    raise AssertionError("ubt.cli.commands.translate.translate was renamed")


def test_a_layout_knob_offered_on_one_surface_is_offered_on_all_three() -> None:
    from ubt.api.models import JobSubmitRequest

    shared = ("layout_columns", "dual_mode", "preset", "pages")
    rest = set(JobSubmitRequest.model_fields)
    mcp = _mcp_tool_signature_names()
    cli = _cli_translate_option_names()

    missing_rest = [k for k in shared if k not in rest]
    missing_mcp = [k for k in shared if k not in mcp]
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

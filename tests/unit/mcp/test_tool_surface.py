"""The MCP tool surface: the two discovery/repair tools and resubmit parity.

Three gaps closed (2026-10-14 review, W4-6):

1. ``ubt_re_export`` — a human edit (``ubt_edit_segment``) lands in the
   ledger but the delivered file was rendered before it; the REST surface
   had a re-export endpoint and the MCP surface had none, so an agent could
   edit and then had no way to make the fix reach the download.
2. ``ubt_list_jobs`` — an agent could only ask about a job id it already
   knew. The console's queue table reads the per-job ledgers on disk; the
   MCP surface now does too.
3. Resubmit parity — REST treats a live job id as an idempotent resubmit
   (returns the existing job); MCP raised, so a retry after a timeout gave
   the agent an error it could not act on.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ubt.api.models import JobSubmitRequest
from ubt.core.engine.job_queue import JobStatus
from ubt.mcp.server import mcp, ubt_translate_book

pytestmark = pytest.mark.fast


def _tool_names() -> list[str]:
    return sorted(tool.name for tool in asyncio.run(mcp.list_tools()))


def test_the_discovery_and_repair_tools_are_registered() -> None:
    names = _tool_names()
    assert "ubt_re_export" in names
    assert "ubt_list_jobs" in names
    # The original eight stay registered.
    for original in (
        "ubt_translate_book",
        "ubt_job_status",
        "ubt_cancel_job",
        "ubt_inspect_book",
        "ubt_assess_book",
        "ubt_list_issues",
        "ubt_edit_segment",
        "ubt_doctor",
    ):
        assert original in names
    for agent_tool in (
        "ubt_agent_prepare",
        "ubt_agent_claim",
        "ubt_agent_submit",
        "ubt_agent_finish",
    ):
        assert agent_tool in names


def test_every_tool_carries_a_description() -> None:
    """A tool the model cannot read is a tool it cannot use correctly."""
    for tool in asyncio.run(mcp.list_tools()):
        assert tool.description and tool.description.strip(), tool.name


def test_every_tool_declares_annotations_and_agent_tools_are_typed() -> None:
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    for tool in tools.values():
        assert tool.annotations is not None, tool.name
    for name in (
        "ubt_agent_prepare",
        "ubt_agent_claim",
        "ubt_agent_submit",
        "ubt_agent_finish",
    ):
        assert tools[name].output_schema, name
        assert tools[name].annotations is not None


def test_agent_tools_have_bounded_input_schemas() -> None:
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    prepare = tools["ubt_agent_prepare"].input_schema["properties"]
    assert prepare["package_size"]["maximum"] == 64
    assert prepare["lease_seconds"]["minimum"] == 30
    submit = tools["ubt_agent_submit"].input_schema["properties"]
    assert "lease_token" in submit
    assert "blocks" in submit
    assert submit["blocks"]["maxItems"] == 64
    defs = tools["ubt_agent_submit"].input_schema.get("$defs", {})
    assert defs["AgentBlockSubmission"]["additionalProperties"] is False


def test_the_new_tools_document_their_arguments() -> None:
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    re_export = tools["ubt_re_export"]
    assert "job_id" in re_export.input_schema.get("properties", {})
    assert "job_id" in re_export.input_schema.get("required", [])
    list_jobs = tools["ubt_list_jobs"]
    assert "db_dir" in list_jobs.input_schema.get("properties", {})


class _Record:
    """Just enough of a JobRecord for the resubmit guard to read."""

    def __init__(self, job_id: str, request: JobSubmitRequest, status: str) -> None:
        self.job_id = job_id
        self.request = request
        self.status = status


def _with_existing_record(monkeypatch: pytest.MonkeyPatch, record: _Record | None) -> None:
    import ubt.mcp.server as server

    if record is None:
        monkeypatch.setattr(server._MANAGER, "get_job", lambda _jid: None, raising=True)
    else:
        monkeypatch.setattr(server._MANAGER, "get_job", lambda _jid: record, raising=True)


async def _call(**kwargs: Any) -> dict[str, Any]:
    return await ubt_translate_book(**kwargs)


@pytest.mark.asyncio
async def test_a_repeated_submit_with_the_same_arguments_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resubmit contract now compares the full normalized request."""
    source = tmp_path / "book.md"
    source.write_text("# t\n\nhello\n", encoding="utf-8")
    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(tmp_path))
    request = JobSubmitRequest(
        input_path=str(source.resolve()),
        target_lang="zh",
        source_lang="en",
        profile="auto",
        job_id="mcp-idem-1",
    )
    _with_existing_record(monkeypatch, _Record("mcp-idem-1", request, JobStatus.RUNNING.value))
    result = await _call(input_path=str(source), job_id="mcp-idem-1")
    assert result["reused"] is True
    assert result["job_id"] == "mcp-idem-1"


@pytest.mark.asyncio
async def test_a_repeated_submit_with_changed_arguments_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.md"
    source.write_text("# t\n\nhello\n", encoding="utf-8")
    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(tmp_path))
    request = JobSubmitRequest(
        input_path=str(source.resolve()),
        target_lang="zh",
        source_lang="en",
        profile="auto",
        job_id="mcp-idem-2",
    )
    _with_existing_record(monkeypatch, _Record("mcp-idem-2", request, JobStatus.RUNNING.value))
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="different arguments: target_lang"):
        await _call(input_path=str(source), job_id="mcp-idem-2", target_lang="ja")

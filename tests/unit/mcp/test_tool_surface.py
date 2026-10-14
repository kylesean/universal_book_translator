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

import pytest

from ubt.mcp.server import mcp

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


def test_every_tool_carries_a_description() -> None:
    """A tool the model cannot read is a tool it cannot use correctly."""
    for tool in asyncio.run(mcp.list_tools()):
        assert tool.description and tool.description.strip(), tool.name


def test_the_new_tools_document_their_arguments() -> None:
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    re_export = tools["ubt_re_export"]
    assert "job_id" in re_export.input_schema.get("properties", {})
    assert "job_id" in re_export.input_schema.get("required", [])
    list_jobs = tools["ubt_list_jobs"]
    assert "db_dir" in list_jobs.input_schema.get("properties", {})

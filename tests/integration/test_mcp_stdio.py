"""Official MCP stdio client integration: initialize, list, call, structured output."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

pytestmark = pytest.mark.fast


@pytest.mark.asyncio
async def test_official_stdio_client_lists_and_calls_agent_tools(tmp_path: Path) -> None:
    source = tmp_path / "book.md"
    source.write_text("# Title\n\nHello world.\n", encoding="utf-8")
    env = {
        **os.environ,
        "UBT_DB_DIR": str(tmp_path / "db"),
        "UBT_ALLOWED_DIRS": str(tmp_path),
        "UBT_VISUAL_JUDGE_ENABLED": "false",
        "UBT_OCR_MODE": "off",
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ubt.mcp"],
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    # No sampling callback/capability is supplied. Agent mode must work through
    # ordinary tool calls only.
    async with Client(params) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        assert {
            "ubt_agent_prepare",
            "ubt_agent_claim",
            "ubt_agent_submit",
            "ubt_agent_finish",
        } <= names
        prepare_tool = next(tool for tool in listed.tools if tool.name == "ubt_agent_prepare")
        assert prepare_tool.output_schema
        assert prepare_tool.annotations is not None

        prepared = await client.call_tool(
            "ubt_agent_prepare",
            {
                "input_path": str(source),
                "output_path": str(tmp_path / "translated.md"),
                "job_id": "stdio_agent",
                "db_dir": str(tmp_path / "db"),
            },
        )
        assert not prepared.is_error
        assert prepared.structured_content is not None
        assert prepared.structured_content["job_id"] == "stdio_agent"
        assert prepared.structured_content["model_calls"] == 0

        claimed = await client.call_tool(
            "ubt_agent_claim",
            {"job_id": "stdio_agent", "db_dir": str(tmp_path / "db")},
        )
        assert not claimed.is_error
        assert claimed.structured_content is not None
        package = claimed.structured_content["package"]
        assert package["blocks"]

        submitted = await client.call_tool(
            "ubt_agent_submit",
            {
                "job_id": "stdio_agent",
                "package_id": package["package_id"],
                "lease_token": package["lease_token"],
                "blocks": [
                    {
                        "block_id": block["block_id"],
                        "translation": (
                            block["masked_source"]
                            .replace("Title", "标题")
                            .replace("Hello world.", "你好世界。")
                        ),
                    }
                    for block in package["blocks"]
                ],
                "db_dir": str(tmp_path / "db"),
            },
        )
        assert not submitted.is_error
        assert submitted.structured_content is not None
        assert submitted.structured_content["accepted_blocks"] == len(package["blocks"])

        # Invalid input is an MCP error, not a protocol crash.
        invalid = await client.call_tool(
            "ubt_agent_prepare",
            {"input_path": str(source), "package_size": 999},
        )
        assert invalid.is_error

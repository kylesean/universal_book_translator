"""Automated verification of the three production quality red lines (Quality Gates).

Red Alert 1: Unclosed <img> or unescaped quote corruptions must be flagged/blocked.
Red Alert 2: flow_id="sidebar_aside" must never pollute main_story context window.
Red Alert 3: ubt/core/ must contain 0 print() calls and 0 imports of click/fastapi/typer.
"""

import ast
from pathlib import Path

from ubt.core.ir.models import BlockType, FlowID, IRBlock
from ubt.core.memory.neighbor_window import NeighborContextBuilder
from ubt.core.validators.html_delta import HTMLDeltaValidator


def test_red_alert_1_unclosed_img_or_unescaped_quotes_detected() -> None:
    """Red Alert 1: Corrupted HTML tags/attributes must be caught by HTMLDeltaValidator."""
    validator = HTMLDeltaValidator()

    # Source has valid image tag
    source = '<p>Here is an illustration: <img src="fig1.jpg" alt="A lovely landscape"/></p>'

    # Translation 1: Unclosed <img> tag or missing tag
    unclosed_target = '<p>这是插图：<img src="fig1.jpg" alt="美丽的风景"</p>'
    res1 = validator.validate(source, unclosed_target)
    assert not res1.is_valid
    assert res1.error_code == "HTML_DELTA_MISMATCH"

    # Translation 2: Corrupted unescaped inner quotes destroying HTML attributes
    corrupted_attr_target = '<p>这是插图：<img src="fig1.jpg" alt="他说"你好"世界"/></p>'
    res2 = validator.validate(source, corrupted_attr_target)
    assert not res2.is_valid
    assert res2.error_code == "HTML_DELTA_MISMATCH"
    assert "malformed_tags" in res2.details

    # Translation 3: Deleted image tag (missing image)
    missing_img_target = "<p>这是插图：[图片被删除]</p>"
    res3 = validator.validate(source, missing_img_target)
    assert not res3.is_valid
    assert res3.error_code == "HTML_DELTA_MISMATCH"
    assert "missing_html" in res3.details.get("image_diff", {})


def test_red_alert_2_sidebar_flow_never_pollutes_main_story_sliding_window() -> None:
    """Red Alert 2: Non-main flows must never contaminate the main story sliding context window."""
    blocks = [
        IRBlock(
            id="blk_01",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Main Story Paragraph 1: The expedition began at dawn.",
            target_text="正文第1段：探险队在黎明出发。",
        ),
        IRBlock(
            id="blk_02",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text="Main Story Paragraph 2: Navigating through the dense fog.",
            target_text="正文第2段：穿行在浓雾之中。",
        ),
        IRBlock(
            id="blk_03",
            flow_id=FlowID.SIDEBAR_ASIDE,
            spine_index=3,
            block_type=BlockType.NARRATIVE,
            source_text="Sidebar 1: Geological formation of the mountain range.",
            target_text="边栏1：山脉的地质构造历史。",
        ),
        IRBlock(
            id="blk_04",
            flow_id=FlowID.SIDEBAR_ASIDE,
            spine_index=4,
            block_type=BlockType.NARRATIVE,
            source_text="Sidebar 2: Equipment checklist for alpine explorers.",
            target_text="边栏2：高山探险装备清单。",
        ),
        IRBlock(
            id="blk_05",
            flow_id=FlowID.MAIN_STORY,
            spine_index=5,
            block_type=BlockType.NARRATIVE,
            source_text="Main Story Paragraph 3: They finally reached the ridge summit.",
            target_text="正文第3段：他们终于登上了山脊之巅。",
        ),
    ]

    builder = NeighborContextBuilder()

    # Context for blk_05 (MAIN_STORY): Must only pull blk_02, never blk_03 or blk_04.
    # The preceding neighbour contributes its finished translation (that is the
    # prose the model is continuing), so isolation is asserted on the text of the
    # right paragraph rather than on its language -- and exactly one side of each
    # paragraph is present, never both.
    main_ctx = builder.extract_from_blocks(blocks[4], blocks)
    assert "正文第2段" in main_ctx
    assert "Main Story Paragraph 2" not in main_ctx
    assert "Sidebar 1" not in main_ctx
    assert "Sidebar 2" not in main_ctx
    assert "地质构造" not in main_ctx
    assert "装备清单" not in main_ctx

    # Context for blk_04 (SIDEBAR_ASIDE): Must only pull blk_03, never blk_01, blk_02 or blk_05
    sidebar_ctx = builder.extract_from_blocks(blocks[3], blocks)
    assert "边栏1" in sidebar_ctx
    assert "Sidebar 1" not in sidebar_ctx
    assert "Main Story Paragraph 1" not in sidebar_ctx
    assert "Main Story Paragraph 2" not in sidebar_ctx
    assert "Main Story Paragraph 3" not in sidebar_ctx
    assert "探险队在黎明出发" not in sidebar_ctx


def test_red_alert_3_core_purity_ast_scanner() -> None:
    """Red Alert 3: Static AST inspection ensuring ubt/core/ has 0 print() and 0 UI/API framework imports."""
    core_dir = Path(__file__).resolve().parent.parent.parent / "ubt" / "core"
    assert core_dir.exists() and core_dir.is_dir()

    forbidden_modules = {"click", "fastapi", "starlette", "typer"}
    forbidden_calls = {"print"}

    violations: list[str] = []

    for py_file in core_dir.rglob("*.py"):
        code = py_file.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(py_file))

        for node in ast.walk(tree):
            # Check 1: Forbidden imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pkg = alias.name.split(".")[0]
                    if pkg in forbidden_modules:
                        violations.append(
                            f"{py_file.name}:{node.lineno} imports forbidden package '{pkg}'"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    pkg = node.module.split(".")[0]
                    if pkg in forbidden_modules:
                        violations.append(
                            f"{py_file.name}:{node.lineno} imports from forbidden package '{pkg}'"
                        )

            # Check 2: Forbidden print() calls
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in forbidden_calls
            ):
                violations.append(
                    f"{py_file.name}:{node.lineno} calls forbidden builtin '{node.func.id}()'"
                )

    assert not violations, "Core architecture purity violated:\n" + "\n".join(violations)

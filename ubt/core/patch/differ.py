"""Semantic AST differencing engine for document revisions."""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.core.ir.models import IRBlock
from ubt.core.patch.models import BlockDiffItem, PatchBlockStatus, PatchSummary

_WHITESPACE_RE = re.compile(r"\s+")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def normalize_source_text(text: str | None) -> str:
    """Normalize text for whitespace-invariant comparison."""
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def estimate_block_tokens(text: str | None) -> int:
    """Estimate token volume for a block's source text."""
    if not text:
        return 0
    clean = text.strip()
    if not clean:
        return 0

    cjk_count = len(_CJK_CHAR_RE.findall(clean))
    non_cjk = _CJK_CHAR_RE.sub(" ", clean)
    words = non_cjk.split()
    estimated = int(len(words) * 1.3) + cjk_count
    return max(1, estimated)


class DocumentSemanticDiffer:
    """Differential comparison engine between base and revised book versions."""

    def __init__(self, min_similarity: float = 0.40) -> None:
        self.min_similarity = max(0.0, min(1.0, min_similarity))

    def diff_blocks(
        self,
        base_blocks: Sequence[IRBlock],
        revised_blocks: Sequence[IRBlock],
        *,
        base_job_id: str | None = None,
        base_source: str = "",
        revised_source: str = "",
    ) -> PatchSummary:
        """Compute fine-grained semantic diff between two block sequences."""
        if not base_blocks and not revised_blocks:
            return PatchSummary(
                base_job_id=base_job_id,
                base_source=base_source,
                revised_source=revised_source,
            )

        if not base_blocks:
            items = [
                BlockDiffItem(
                    status=PatchBlockStatus.ADDED,
                    revised_block_id=b.id,
                    spine_index=b.spine_index,
                    block_type=b.block_type,
                    revised_source_text=b.source_text,
                    diff_reason="Newly added in revised edition",
                )
                for b in revised_blocks
            ]
            return PatchSummary(
                base_job_id=base_job_id,
                base_source=base_source,
                revised_source=revised_source,
                total_base_blocks=0,
                total_revised_blocks=len(revised_blocks),
                added_count=len(revised_blocks),
                diff_items=items,
            )

        if not revised_blocks:
            items = [
                BlockDiffItem(
                    status=PatchBlockStatus.DELETED,
                    base_block_id=b.id,
                    spine_index=b.spine_index,
                    block_type=b.block_type,
                    base_source_text=b.source_text,
                    base_target_text=b.target_text,
                    base_status=b.status,
                    diff_reason="Deleted from base edition",
                )
                for b in base_blocks
            ]
            return PatchSummary(
                base_job_id=base_job_id,
                base_source=base_source,
                revised_source=revised_source,
                total_base_blocks=len(base_blocks),
                total_revised_blocks=0,
                deleted_count=len(base_blocks),
                diff_items=items,
            )

        base_keys = [
            f"{b.block_type.value}:{normalize_source_text(b.source_text)}" for b in base_blocks
        ]
        rev_keys = [
            f"{b.block_type.value}:{normalize_source_text(b.source_text)}" for b in revised_blocks
        ]

        matcher = difflib.SequenceMatcher(None, base_keys, rev_keys, autojunk=False)
        diff_items: list[BlockDiffItem] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2), strict=True):
                    base_b = base_blocks[i]
                    rev_b = revised_blocks[j]
                    diff_items.append(
                        BlockDiffItem(
                            status=PatchBlockStatus.UNCHANGED,
                            base_block_id=base_b.id,
                            revised_block_id=rev_b.id,
                            spine_index=rev_b.spine_index,
                            block_type=rev_b.block_type,
                            base_source_text=base_b.source_text,
                            revised_source_text=rev_b.source_text,
                            similarity=1.0,
                            base_target_text=base_b.target_text,
                            base_status=base_b.status,
                            diff_reason="Exact match: 100% reusable",
                        )
                    )
            elif tag == "delete":
                for i in range(i1, i2):
                    base_b = base_blocks[i]
                    diff_items.append(
                        BlockDiffItem(
                            status=PatchBlockStatus.DELETED,
                            base_block_id=base_b.id,
                            spine_index=base_b.spine_index,
                            block_type=base_b.block_type,
                            base_source_text=base_b.source_text,
                            base_target_text=base_b.target_text,
                            base_status=base_b.status,
                            diff_reason="Removed in revised edition",
                        )
                    )
            elif tag == "insert":
                for j in range(j1, j2):
                    rev_b = revised_blocks[j]
                    diff_items.append(
                        BlockDiffItem(
                            status=PatchBlockStatus.ADDED,
                            revised_block_id=rev_b.id,
                            spine_index=rev_b.spine_index,
                            block_type=rev_b.block_type,
                            revised_source_text=rev_b.source_text,
                            diff_reason="Newly added block",
                        )
                    )
            elif tag == "replace":
                sub_base = list(enumerate(base_blocks[i1:i2], start=i1))
                sub_rev = list(enumerate(revised_blocks[j1:j2], start=j1))

                matches: list[tuple[float, int, int]] = []
                for b_idx, base_b in sub_base:
                    norm_base = normalize_source_text(base_b.source_text)
                    for r_idx, rev_b in sub_rev:
                        norm_rev = normalize_source_text(rev_b.source_text)
                        sim = difflib.SequenceMatcher(None, norm_base, norm_rev).ratio()
                        if base_b.block_type == rev_b.block_type:
                            sim = min(1.0, sim + 0.05)
                        if sim >= self.min_similarity:
                            matches.append((sim, b_idx, r_idx))

                matches.sort(key=lambda x: x[0], reverse=True)
                used_base: set[int] = set()
                used_rev: set[int] = set()
                paired: dict[int, tuple[int, float]] = {}

                for sim, b_idx, r_idx in matches:
                    if b_idx not in used_base and r_idx not in used_rev:
                        used_base.add(b_idx)
                        used_rev.add(r_idx)
                        paired[r_idx] = (b_idx, sim)

                for r_idx, rev_b in sub_rev:
                    if r_idx in paired:
                        b_idx, sim = paired[r_idx]
                        base_b = base_blocks[b_idx]
                        diff_items.append(
                            BlockDiffItem(
                                status=PatchBlockStatus.MODIFIED,
                                base_block_id=base_b.id,
                                revised_block_id=rev_b.id,
                                spine_index=rev_b.spine_index,
                                block_type=rev_b.block_type,
                                base_source_text=base_b.source_text,
                                revised_source_text=rev_b.source_text,
                                similarity=round(sim, 3),
                                base_target_text=base_b.target_text,
                                base_status=base_b.status,
                                diff_reason=f"Modified block (similarity: {sim:.2f})",
                            )
                        )
                    else:
                        diff_items.append(
                            BlockDiffItem(
                                status=PatchBlockStatus.ADDED,
                                revised_block_id=rev_b.id,
                                spine_index=rev_b.spine_index,
                                block_type=rev_b.block_type,
                                revised_source_text=rev_b.source_text,
                                diff_reason="Newly added block in modified section",
                            )
                        )

                for b_idx, base_b in sub_base:
                    if b_idx not in used_base:
                        diff_items.append(
                            BlockDiffItem(
                                status=PatchBlockStatus.DELETED,
                                base_block_id=base_b.id,
                                spine_index=base_b.spine_index,
                                block_type=base_b.block_type,
                                base_source_text=base_b.source_text,
                                base_target_text=base_b.target_text,
                                base_status=base_b.status,
                                diff_reason="Removed from modified section",
                            )
                        )

        unchanged = sum(1 for item in diff_items if item.status == PatchBlockStatus.UNCHANGED)
        modified = sum(1 for item in diff_items if item.status == PatchBlockStatus.MODIFIED)
        added = sum(1 for item in diff_items if item.status == PatchBlockStatus.ADDED)
        deleted = sum(1 for item in diff_items if item.status == PatchBlockStatus.DELETED)

        total_revised = len(revised_blocks)
        reuse_ratio = unchanged / max(total_revised, 1)

        saved_tokens = sum(
            estimate_block_tokens(item.revised_source_text or "")
            for item in diff_items
            if item.status == PatchBlockStatus.UNCHANGED
        )
        saved_cost = round((saved_tokens / 1000.0) * 0.003, 4)

        return PatchSummary(
            base_job_id=base_job_id,
            base_source=base_source,
            revised_source=revised_source,
            total_base_blocks=len(base_blocks),
            total_revised_blocks=total_revised,
            unchanged_count=unchanged,
            modified_count=modified,
            added_count=added,
            deleted_count=deleted,
            reuse_ratio=round(reuse_ratio, 4),
            estimated_tokens_saved=saved_tokens,
            estimated_cost_saved_usd=saved_cost,
            diff_items=diff_items,
        )

    def render_diff_terminal(self, summary: PatchSummary, console: Console | None = None) -> None:
        """Render Rich terminal dashboard of diff summary."""
        con = console or Console()

        pct = f"{summary.reuse_ratio * 100:.1f}%"
        stat_text = (
            f"[bold]Base Source:[/] {summary.base_source or summary.base_job_id or 'unknown'}\n"
            f"[bold]Revised Source:[/] {summary.revised_source or 'unknown'}\n\n"
            f"[green]● Unchanged (100% Reused):[/] {summary.unchanged_count} blocks\n"
            f"[yellow]● Modified (Delta Draft):[/] {summary.modified_count} blocks\n"
            f"[cyan]● Newly Added:[/] {summary.added_count} blocks\n"
            f"[red]● Deleted:[/] {summary.deleted_count} blocks\n\n"
            f"[bold green]Asset Reuse Rate:[/] {pct}\n"
            f"[bold green]Estimated Tokens Saved:[/] ~{summary.estimated_tokens_saved:,} tokens\n"
            f"[bold green]Estimated Cost Saved:[/] ~${summary.estimated_cost_saved_usd:.4f} USD"
        )
        con.print(
            Panel(
                stat_text,
                title="[bold blue]UBT Book Semantic Diff & Patch Summary[/]",
                border_style="blue",
            )
        )

        table = Table(title="Sample Block Diff Matrix (First 15 Delta Items)", expand=True)
        table.add_column("Status", justify="center", width=12)
        table.add_column("Spine", justify="right", width=6)
        table.add_column("Type", justify="center", width=10)
        table.add_column("Similarity", justify="right", width=10)
        table.add_column("Source Text Excerpt", justify="left")

        displayed = 0
        for item in summary.diff_items:
            if item.status == PatchBlockStatus.UNCHANGED and displayed >= 5:
                continue
            if displayed >= 15:
                break

            status_style = {
                PatchBlockStatus.UNCHANGED: "[green]UNCHANGED[/]",
                PatchBlockStatus.MODIFIED: "[yellow]MODIFIED[/]",
                PatchBlockStatus.ADDED: "[cyan]ADDED[/]",
                PatchBlockStatus.DELETED: "[red]DELETED[/]",
            }.get(item.status, str(item.status))

            sim_str = f"{item.similarity:.2f}" if item.similarity > 0 else "-"
            text_excerpt = (item.revised_source_text or item.base_source_text or "")[:60].replace(
                "\n", " "
            )

            table.add_row(
                status_style,
                str(item.spine_index),
                item.block_type.value,
                sim_str,
                text_excerpt,
            )
            displayed += 1

        con.print(table)

    def render_diff_markdown(self, summary: PatchSummary) -> str:
        """Render Markdown format report for PR/Artifact audit."""
        pct = f"{summary.reuse_ratio * 100:.1f}%"
        lines = [
            "### Universal Book Translator: Semantic Revision Patch Summary",
            "",
            f"- **Base Source**: `{summary.base_source or summary.base_job_id or 'unknown'}`",
            f"- **Revised Source**: `{summary.revised_source or 'unknown'}`",
            f"- **Total Base Blocks**: {summary.total_base_blocks}",
            f"- **Total Revised Blocks**: {summary.total_revised_blocks}",
            f"- **Unchanged Blocks (Reused)**: {summary.unchanged_count}",
            f"- **Modified Blocks (Delta)**: {summary.modified_count}",
            f"- **Added Blocks**: {summary.added_count}",
            f"- **Deleted Blocks**: {summary.deleted_count}",
            f"- **Asset Reuse Rate**: **{pct}**",
            f"- **Estimated Tokens Saved**: **~{summary.estimated_tokens_saved:,} tokens**",
            f"- **Estimated Cost Saved**: **~${summary.estimated_cost_saved_usd:.4f} USD**",
            "",
            "| Status | Spine | Block Type | Similarity | Excerpt |",
            "| :---: | :---: | :---: | :---: | :--- |",
        ]
        for item in summary.diff_items[:20]:
            excerpt = (item.revised_source_text or item.base_source_text or "")[:50].replace(
                "\n", " "
            )
            sim_str = f"{item.similarity:.2f}" if item.similarity > 0 else "-"
            lines.append(
                f"| `{item.status.value}` | {item.spine_index} | {item.block_type.value} | {sim_str} | {excerpt} |"
            )
        return "\n".join(lines)

    def render_diff_json(self, summary: PatchSummary) -> str:
        """Render JSON string output."""
        return json.dumps(summary.model_dump(), ensure_ascii=False, indent=2)

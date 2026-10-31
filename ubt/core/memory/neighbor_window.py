"""The read-only neighbour excerpts handed to the prompt beside a block.

Which blocks count as a neighbour is a flow question, decided in one place:
:meth:`NeighborContextBuilder.extract_from_blocks`. Everything else here is
presentation — trimming, labelling, and the section heading that keeps the model
from treating an excerpt as text to translate.
"""

from ubt.core.ir.models import IRBlock

DEFAULT_NEIGHBOR_CHARS = 300


class NeighborContextBuilder:
    """Assembles prompt-ready read-only neighbouring excerpts."""

    def __init__(self, neighbor_chars: int = DEFAULT_NEIGHBOR_CHARS) -> None:
        self.neighbor_chars = neighbor_chars

    def extract_excerpts(
        self,
        prev_text: str | None = None,
        next_text: str | None = None,
    ) -> dict[str, str]:
        """Extract trimmed tail from prev_text and head from next_text."""
        prev_excerpt = ""
        next_excerpt = ""

        if prev_text:
            s = prev_text.strip()
            prev_excerpt = s[-self.neighbor_chars :].strip() if len(s) > self.neighbor_chars else s

        if next_text:
            s = next_text.strip()
            next_excerpt = s[: self.neighbor_chars].strip() if len(s) > self.neighbor_chars else s

        return {
            "prev_excerpt": prev_excerpt,
            "next_excerpt": next_excerpt,
        }

    def format_prompt_block(
        self,
        prev_text: str | None = None,
        next_text: str | None = None,
    ) -> str:
        """Render a concise markdown block ready for prompt injection."""
        excerpts = self.extract_excerpts(prev_text, next_text)
        parts: list[str] = []

        if excerpts["prev_excerpt"]:
            parts.append(
                f"[READ-ONLY PRECEDING CONTEXT: DO NOT TRANSLATE OR ECHO]\n"
                f"{excerpts['prev_excerpt']}"
            )
        if excerpts["next_excerpt"]:
            parts.append(
                f"[READ-ONLY SUBSEQUENT CONTEXT: DO NOT TRANSLATE OR ECHO]\n"
                f"{excerpts['next_excerpt']}"
            )

        if not parts:
            return ""

        # A section heading, so the excerpts read as a *reference section*
        # structurally parallel to "### Source Paragraph to Translate" instead of
        # as bare paragraphs adjacent to the text to translate. The bracketed
        # per-excerpt labels were not enough on their own: chapter-3 shipped
        # blocks whose output was the neighbour excerpt translated and prepended,
        # and every one of them failed the added-content probe for that reason.
        # No information is added or removed.
        return "### Reference Context (read-only, NOT part of the task)\n\n" + "\n\n".join(parts)

    @classmethod
    def combine_chunk_neighbor_contexts(cls, first_ctx: str, last_ctx: str) -> str:
        """Combine preceding context from first block and subsequent context from last block.

        For macro-chunks, the first block's subsequent context is an internal chunk block
        (e.g. chunk[1]), while the last block's subsequent context is what follows the entire
        chunk. This helper joins the external preceding context of chunk[0] with the external
        subsequent context of chunk[-1], preventing internal chunk blocks from being mislabeled
        as read-only subsequent context.
        """
        pre_tag = "[READ-ONLY PRECEDING CONTEXT: DO NOT TRANSLATE OR ECHO]"
        sub_tag = "[READ-ONLY SUBSEQUENT CONTEXT: DO NOT TRANSLATE OR ECHO]"
        prev_part = ""
        next_part = ""

        if pre_tag in (first_ctx or ""):
            start = first_ctx.index(pre_tag)
            end = first_ctx.find(sub_tag, start) if sub_tag in first_ctx else len(first_ctx)
            prev_part = first_ctx[start:end].strip()

        if sub_tag in (last_ctx or ""):
            start = last_ctx.index(sub_tag)
            next_part = last_ctx[start:].strip()

        parts = [p for p in (prev_part, next_part) if p]
        if not parts:
            return ""
        return "### Reference Context (read-only, NOT part of the task)\n\n" + "\n\n".join(parts)

    def extract_from_blocks(
        self,
        target_block: IRBlock,
        surrounding_blocks: list[IRBlock],
        fallback_prev_text: str | None = None,
        fallback_next_text: str | None = None,
    ) -> str:
        """The one flow-isolating entry point: neighbours are drawn only from blocks
        sharing ``target_block.flow_id``, so a footnote's window never reaches into
        the body paragraph it annotates. :meth:`extract_excerpts` is flow-agnostic
        and trusts whatever text the caller hands it.
        """
        # Filter blocks strictly belonging to the identical flow_id
        same_flow = [b for b in surrounding_blocks if b.flow_id == target_block.flow_id]
        same_flow.sort(key=lambda b: b.spine_index)

        prev_text: str | None = None
        next_text: str | None = None

        for i, b in enumerate(same_flow):
            if b.id == target_block.id:
                if i > 0:
                    # What precedes the block is what the model is continuing:
                    # prefer its finished translation, fall back to source while
                    # the neighbourhood is still untranslated.
                    prev_text = same_flow[i - 1].target_text or same_flow[i - 1].source_text
                elif fallback_prev_text:
                    prev_text = fallback_prev_text
                if i + 1 < len(same_flow):
                    next_text = same_flow[i + 1].source_text
                elif fallback_next_text:
                    next_text = fallback_next_text
                break

        if prev_text is None and fallback_prev_text:
            prev_text = fallback_prev_text
        if next_text is None and fallback_next_text:
            next_text = fallback_next_text

        return self.format_prompt_block(prev_text, next_text)

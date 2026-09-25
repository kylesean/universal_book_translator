"""Bidirectional read-only neighbor sliding window generator."""

from ubt.core.ir.models import IRBlock

DEFAULT_NEIGHBOR_CHARS = 300


class NeighborContextBuilder:
    """Constructs prompt-ready read-only neighboring excerpts strictly within the same semantic flow."""

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
        # blocks whose output was the neighbour excerpt translated and prepended
        # (pdf_main#b0009/b0045/b0068/b0089/b0162 all fail the added-content
        # probe for exactly this reason). No information is added or removed.
        return "### Reference Context (read-only, NOT part of the task)\n\n" + "\n\n".join(parts)

    def extract_from_blocks(
        self,
        target_block: IRBlock,
        surrounding_blocks: list[IRBlock],
        fallback_prev_text: str | None = None,
        fallback_next_text: str | None = None,
    ) -> str:
        """Extract neighbor context ensuring strict flow isolation."""
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

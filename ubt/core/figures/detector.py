"""Detector for subfigures, composite captions, and multi-panel figure groupings."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ubt.core.figures.models import (
    FigureLayoutConfig,
    FigureLayoutMode,
    FigurePanel,
    SubFigure,
)

if TYPE_CHECKING:
    from ubt.core.ir.models import IRBlock

_SUB_MARKER_RE = re.compile(
    r"(?:^|(?<=\s)|(?<=[，。；;,：:]))"
    r"(?:"
    r"\((?P<ascii_paren>[a-zA-Z]|\d+)\)"
    r"|（(?P<cjk_paren>[a-zA-Z]|\d+|[一二三四五六七八九十]+)）"
    r"|(?P<ascii_letter_paren>[a-zA-Z])\)"
    r"|(?P<ascii_letter_colon>[a-zA-Z])[:：]"
    r")",
    re.UNICODE,
)

_CAPTION_OPENER_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|Table|Listing|Algorithm|图|表|附图|附表)\s*[\d.]+\s*[:：.|｜]?",
    re.IGNORECASE,
)

_MD_IMAGE_RE = re.compile(r"^\s*!\[(.*?)\]\((.*?)\)\s*$")

_MAX_UNLABELLED_CAPTION_CHARS = 320


def extract_main_and_subcaptions(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a composite caption into the main caption and labelled subcaptions.

    For example:
    "Figure 3: Ablation studies. (a) Learning rate decay. (b) Batch size variation."
    returns:
    ("Figure 3: Ablation studies.", [("(a)", "Learning rate decay."), ("(b)", "Batch size variation.")])
    """
    matches = list(_SUB_MARKER_RE.finditer(text))
    if not matches:
        return text.strip(), []

    main_caption = text[: matches[0].start()].strip()
    # Strip any trailing colons or dashes from main caption prefix
    main_caption = re.sub(r"[:：—–-]\s*$", "", main_caption).strip()

    subcaptions: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        raw_label = m.group(0).strip()
        # Normalize raw_label to standard (x) format if necessary
        char = (
            m.group("ascii_paren")
            or m.group("cjk_paren")
            or m.group("ascii_letter_paren")
            or m.group("ascii_letter_colon")
        )
        norm_label = f"({char})" if char else raw_label

        start_pos = m.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sub_text = text[start_pos:end_pos].strip()

        # Clean leading separator and trailing punctuation
        sub_text = re.sub(r"^[:：—–-]\s*", "", sub_text).strip()
        sub_text = re.sub(r"[;；,，]\s*$", "", sub_text).strip()

        subcaptions.append((norm_label, sub_text))

    return main_caption, subcaptions


def parse_subcaptions(text: str) -> list[tuple[str, str]]:
    """Parse and return only the (label, subcaption) pairs from a caption string."""
    _, subs = extract_main_and_subcaptions(text)
    return subs


def _is_caption_block(block: IRBlock, text: str) -> bool:
    """Determine whether an IR block serves as a figure caption."""
    flow = str(getattr(block, "flow_id", None) or "")
    if flow in ("caption", "FlowID.CAPTION"):
        return True
    if _CAPTION_OPENER_RE.match(text):
        return True
    if _SUB_MARKER_RE.search(text):
        return True
    return len(text) <= _MAX_UNLABELLED_CAPTION_CHARS


def _resolve_asset_id(block: IRBlock) -> str:
    """Extract the asset path or identifier from an IR block."""
    elem = getattr(block, "element", None)
    if elem is not None:
        asset = getattr(elem, "asset_id", None) or getattr(elem, "src", None)
        if asset:
            return str(asset).strip()
    return (block.source_text or "").strip()


def _resolve_width_pt(block: IRBlock) -> float | None:
    """Compute width in PDF points from the block bounding box if present."""
    bbox = getattr(block, "bbox", None)
    if bbox is not None:
        return float(bbox.x1 - bbox.x0)
    return None


def calculate_grid_columns(count: int, max_cols: int = 4) -> int:
    """Calculate the optimal number of grid columns for a given number of subfigures."""
    if count <= 1:
        return 1
    if count == 2:
        return min(2, max_cols)
    if count == 3:
        return min(3, max_cols)
    if count == 4:
        # 2x2 grid preferred over 1x4
        return min(2, max_cols)
    if count in (5, 6):
        # 2x3 or 3x2 grid
        return min(3, max_cols)
    return min(max(2, count // 2), max_cols)


def detect_figure_panels(
    blocks: list[IRBlock],
    config: FigureLayoutConfig | None = None,
) -> list[FigurePanel]:
    """Scan an IR block stream and group related images and captions into FigurePanels."""
    cfg = config or FigureLayoutConfig()
    panels: list[FigurePanel] = []

    pending_subs: list[SubFigure] = []
    panel_idx = 1

    def _flush_panel(
        main_caption: str = "", parsed_subs: list[tuple[str, str]] | None = None
    ) -> None:
        nonlocal panel_idx, pending_subs
        if not pending_subs:
            return

        # If subcaption labels/texts were parsed from composite caption, associate them
        if parsed_subs:
            for i, (lbl, sub_cap) in enumerate(parsed_subs):
                if i < len(pending_subs):
                    if not pending_subs[i].label:
                        pending_subs[i].label = lbl
                    if not pending_subs[i].caption:
                        pending_subs[i].caption = sub_cap

        # If auto-subfigures enabled and multiple images lack labels, generate (a), (b), ...
        if cfg.auto_subfigures and len(pending_subs) > 1:
            for i, sf in enumerate(pending_subs):
                if not sf.label:
                    char = chr(ord("a") + (i % 26))
                    sf.label = f"({char})"

        cols = calculate_grid_columns(len(pending_subs), cfg.max_columns)
        span_full = any(
            (sf.width_pt is not None and sf.width_pt >= cfg.max_full_width_threshold_pt)
            for sf in pending_subs
        )

        panel = FigurePanel(
            panel_id=f"fig-panel-{panel_idx}",
            main_caption=main_caption,
            subfigures=list(pending_subs),
            columns=cols,
            gutter_pt=cfg.gutter_pt,
            span_full_width=span_full,
            layout_mode=FigureLayoutMode.GRID if len(pending_subs) > 1 else FigureLayoutMode.AUTO,
        )
        panels.append(panel)
        panel_idx += 1
        pending_subs.clear()

    i = 0
    n = len(blocks)
    while i < n:
        b = blocks[i]
        btype = str(
            getattr(
                getattr(b, "block_type", None),
                "value",
                getattr(b, "block_type", getattr(b, "type", "")),
            )
            or ""
        ).lower()
        text = (b.target_text or b.source_text or "").strip()
        m_img = _MD_IMAGE_RE.match(text)
        if "image" in btype or m_img:
            asset = m_img.group(2).strip() if m_img else _resolve_asset_id(b)
            width = _resolve_width_pt(b)
            sf = SubFigure(
                asset=asset,
                width_pt=width,
                block_id=b.id,
            )
            pending_subs.append(sf)
            i += 1
            continue

        if pending_subs and _is_caption_block(b, text):
            # Block binds as caption to current pending subfigures
            main_cap, parsed_subs = extract_main_and_subcaptions(text)
            _flush_panel(main_caption=main_cap, parsed_subs=parsed_subs)
            i += 1
            continue

        # Encountered a non-image, non-caption block: flush any pending panel
        if pending_subs:
            _flush_panel()

        i += 1

    # End of stream flush
    if pending_subs:
        _flush_panel()

    return panels

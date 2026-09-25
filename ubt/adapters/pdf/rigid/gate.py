"""Fail-closed placement gate (prose-only, no chrome).

Pure predicate over one ledger block: chrome (headers, footers, page
numbers, policy-skipped blocks) never backfills and stays
source-visible; captions are opt-in; short header/footer-band text is
skipped; verbatim pairs need no paint. Chrome diagnostics keep their
reasons because the verbatim check runs last.
"""

from __future__ import annotations

from ubt.core.ir.models import IRBlock, LayoutRole
from ubt.core.policy.layout_policy import (
    BAND_TEXT_MAX_LEN,
    CAPTION_RE,
    FOOTER_BAND_PT,
    FOOTER_PATTERNS,
    HEADER_BAND_PT,
    PROSE_BLOCK_TYPES,
)

_CHROME_ROLES = frozenset({LayoutRole.HEADER, LayoutRole.FOOTER, LayoutRole.PAGE_NUMBER})


def skip_reason(
    block: IRBlock,
    page_h: float,
    *,
    backfill_captions: bool = True,
    translate_chrome: bool = False,
) -> str | None:
    """Fail-closed gate: body prose only, chrome never paints.

    Footers/headers/page numbers always stay source-visible — even when a
    stale ledger carries a translation for them. Backfilling chrome caused
    the title-page footer garble: a narrow strip's cover wipes the source
    while the translation cannot fit. Policy-skipped blocks (references,
    watermarks, debris stamped at ingest) likewise never paint, so frozen
    ledgers render what ingest would produce today.

    ``translate_chrome`` (config opt-in) lifts the ban for HEADER/FOOTER
    roles that actually went through translation: the block must be
    translated (no parse-time skip, no explicit policy keep) and still fits
    its own band, otherwise the typesetter fails it closed as spill.
    PAGE_NUMBER and untranslated chrome keep their reason.
    """
    if block.block_type not in PROSE_BLOCK_TYPES:
        return "non_prose"
    if block.bbox is None:
        return "no_bbox"
    src = (block.source_text or "").strip()
    if block.layout_role is LayoutRole.PAGE_NUMBER:
        return "chrome"
    # A band block whose text carries no letters is a page number in all but
    # name ("78" roled HEADER): never paintable, even under the opt-in.
    painting_chrome = (
        translate_chrome and block.layout_role in _CHROME_ROLES and any(c.isalpha() for c in src)
    )
    if block.layout_role in _CHROME_ROLES and not painting_chrome:
        return "chrome"
    if painting_chrome and (block.skip_translate or block.policy_translate is False):
        # Frozen ledgers and explicit keeps still hold their source.
        return "policy"
    if block.layout_role != LayoutRole.TITLE:
        if block.skip_translate or block.policy_translate is False:
            return "policy"
    elif block.skip_translate:
        return "policy"
    y0, y1 = block.bbox.y0, block.bbox.y1
    if CAPTION_RE.match(src) and not backfill_captions:
        return "caption"
    if FOOTER_PATTERNS.search(src):
        return "footer"
    if block.layout_role != LayoutRole.TITLE and not painting_chrome:
        if y1 > page_h - HEADER_BAND_PT and len(src) < BAND_TEXT_MAX_LEN:
            return "header_band"
        if y0 < FOOTER_BAND_PT and len(src) < BAND_TEXT_MAX_LEN:
            return "footer_band"
    # R2: verbatim pairs need no paint. When the pipeline kept the
    # source (references/verbatim policy, target == source), emitting a
    # cover would only strip pristine source text and repaint it in the
    # overlay font — readers lose nothing, the page risks everything.
    # Checked last so chrome/caption diagnostics keep their reasons.
    if src and (block.target_text or "").strip() == src:
        return "verbatim"
    return None

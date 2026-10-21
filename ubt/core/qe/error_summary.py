"""A one-line summary of a provider error, for display inside a block.

The repair stage wraps a failed draft in ``<mark class="ubt-failed-draft"
title="...">`` so the deliverable keeps the text instead of losing it. The
``title`` used to receive ``str(exception)`` verbatim, and a provider error is a
whole escaped JSON envelope — measured at ~700 characters on the reference job:

    Repair error: Model API error (500): {&quot;src&quot;:&quot;llama-swap&quot;, ... }

Two problems, both visible in the review workbench:

* **It broke the layout.** The workbench renders ``target_text`` as plain text, so
  the attribute text became content: a 828px-wide string inside a 446px column,
  pushing the whole page into horizontal scroll.
* **It said nothing useful.** The operator needs to know a repair failed and why
  in one phrase; the envelope is for the ledger, where it already is.

This module keeps the *prefix* intact — ``REPAIR_ERROR_PREFIX`` is load-bearing,
because ``ubt.core.qe.defect_taxonomy.TRANSIENT_FAILURE_PREFIXES`` decides from it
whether a resume may retry the block — and shortens only the detail that follows.
"""

from __future__ import annotations

import json
import re

from ubt.core.qe.defect_taxonomy import REPAIR_ERROR_PREFIX

#: Longest snippet kept from the provider's own message.
_MAX_DETAIL_CHARS = 120

#: A trailing JSON object, which is what llama-swap and friends append to their
#: HTTP error text. Matched from the first ``{`` that is followed by a quote, so
#: a message that merely contains braces is left alone.
_TRAILING_JSON = re.compile(r'\s*\{\s*".*\}\s*$', re.S)

#: "(500)" / "status 500" / "HTTP 503" — the one fact worth keeping verbatim.
_STATUS_CODE = re.compile(r"\b(?:status[:\s]*)?(\d{3})\b", re.I)


def _message_from_envelope(raw: str) -> str:
    """The ``message`` field inside a provider's JSON error envelope, or ``""``.

    llama-swap answers with ``{"src": ..., "error": {"message": "..."}}``; taking
    only the top-level ``message`` missed it and returned nothing, which made
    every failure read "Repair error" with no cause.
    """
    start = raw.find('{"')
    if start < 0:
        return ""
    try:
        parsed = json.loads(raw[start:])
    except (ValueError, TypeError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    inner = parsed.get("error")
    if isinstance(inner, dict):
        nested = inner.get("message")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    for key in ("message", "error", "detail"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def summarize_provider_error(exc: BaseException | str) -> str:
    """``"Provider 500: upstream command exited prematurely"``, or a short class name.

    Never raises: this runs on the error path, and a summarizer that fails there
    would replace a block's text with a stack trace.
    """
    raw = str(exc) if not isinstance(exc, str) else exc
    # Pull the provider's own sentence out of its envelope *first*: the JSON
    # body is where "upstream command exited prematurely" lives, and stripping
    # the object wholesale threw that away.
    extracted = _message_from_envelope(raw)
    text = _TRAILING_JSON.sub("", raw).strip()
    if extracted:
        text = f"{text.split('{', 1)[0].strip()} {extracted}".strip()

    status = ""
    match = _STATUS_CODE.search(raw)
    if match:
        status = f" {match.group(1)}"

    # Drop the "Model API error (500):" style lead-in; the status is re-added in
    # a canonical form so every provider reads the same way.
    text = re.sub(
        r"^(?:model\s+api\s+error|api\s+error|httpx?\w*error|provider\s+error)"
        r"\s*(?:\(?\s*\d{3}\s*\)?)?\s*:?\s*",
        "",
        text,
        flags=re.I,
    )

    # An upstream message may itself be a small JSON object; take its "message".
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            text = str(parsed.get("message") or parsed.get("error") or "")

    text = " ".join(text.split())
    if len(text) > _MAX_DETAIL_CHARS:
        text = text[: _MAX_DETAIL_CHARS - 1].rstrip() + "…"
    if not text:
        return REPAIR_ERROR_PREFIX[:-1]  # "Repair error" (no dangling colon)
    # ``REPAIR_ERROR_PREFIX`` must remain a literal prefix: retry-on-resume is
    # decided by ``str.startswith`` in ``TRANSIENT_FAILURE_PREFIXES``, so the
    # status code goes after the colon, never inside "Repair error:".
    if status:
        return f"{REPAIR_ERROR_PREFIX}{status} {text}"
    return f"{REPAIR_ERROR_PREFIX} {text}"

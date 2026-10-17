"""Read a table's data cells with a vision model, keeping Docling's structure.

Docling types a table as a geometric grid and then *fuses* rows it thinks are
one cell. Measured on arXiv 2609.20519 the failure is systematic: the benchmark
table's "Recognition Method" column stacks four model names in one visual cell,
Docling returns them joined by spaces, and the renderer shows one unreadable
blob. Seven repair helpers have grown around this (`_split_stacked_cell`,
`merge_table_continuation_fragments`, …) and each one fixes a shape the previous
one did not anticipate.

A vision reader reads the crop directly and emits an explicit grid: measured on
the same document, PaddleOCR-VL-1.6 returned 144 of 144 data cells exactly, with
merged headers and spans intact.

What this module deliberately does **not** do is trust the model with the whole
table:

* the model is a *region* reader, not a page parser — a page-sized crop makes it
  collapse several tables into one row, and on a page with several tables it
  loses rows entirely (measured);
* it misreads header text (``Traffic`` -> ``Attack``, ``Eff.`` -> ``EI.``, 2 of
  11 labels), and the header is what the translator and the leak metric key on.

So Docling keeps the header row and the column count, the model supplies the
data cells, and the merge is refused unless the shapes agree. A reader that
fails leaves Docling's table untouched — a fallback, never a regression.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Enables the table reader. Off by default: it needs a running service, and the
#: default pipeline must work on a CPU-only machine with no network.
ENABLE_ENV_VAR = "UBT_TABLE_READER"

#: The endpoint variables the existing vision-OCR channel already uses, so one
#: service configuration serves both.
ENDPOINT_ENV_VAR = "UBT_OCR_ENDPOINT"
MODEL_ENV_VAR = "UBT_OCR_MODEL"
API_KEY_ENV_VAR = "UBT_OCR_API_KEY"

#: PaddleOCR-VL's table task token. The prompt IS the task switch: the same
#: model answers "OCR:", "Formula Recognition:" and "Table Recognition:", and a
#: paraphrase is a different request.
TABLE_PROMPT = "Table Recognition:"

#: One cell of a reply: a marker and the text that follows it. The marker set is
#: PaddleOCR-VL's — a cell with text, a continuation of the cell to the left, a
#: continuation of the cell above, and a row break.
_CELL_TOKEN_RE = re.compile(r"<(fcel|ucel|lcel|nl)>([^<]*)")


def enabled() -> bool:
    """Whether the table reader is switched on and has somewhere to send work."""
    raw = os.environ.get(ENABLE_ENV_VAR, "").strip().lower()
    if raw not in ("1", "true", "on", "yes", "table", "tables"):
        return False
    return bool(os.environ.get(ENDPOINT_ENV_VAR, "").strip())


@dataclass(frozen=True)
class Grid:
    """A table read from an image: rows of cell text, ``""`` for merged spans."""

    rows: tuple[tuple[str, ...], ...]

    @property
    def columns(self) -> int:
        return max((len(row) for row in self.rows), default=0)


def parse_table_markup(markup: str) -> Grid | None:
    """Parse a ``<fcel>…<ucel>…<nl>`` reply into a rectangular grid, or ``None``.

    Each marker opens one grid position and its text runs to the next marker, so
    a row is simply its positions in order — no separate text-prefix handling,
    which is what shifted an earlier version by one column. ``<nl>`` ends a row
    (the marker that follows it opens the next one, so no empty leading cell is
    produced).

    Returned cells are ``|``-escaped the way ``table_to_markdown`` escapes them,
    so both grids can be handed to the same renderer. A reply with no cell
    markers is not a table and returns ``None``.
    """
    if not markup or ("<fcel>" not in markup and "<lcel>" not in markup):
        return None
    rows: list[list[str]] = []
    row: list[str] = []
    for marker, text in _CELL_TOKEN_RE.findall(markup):
        if marker == "nl":
            if row:
                rows.append(row)
            row = []
            continue
        # ``ucel``/``lcel`` hold a position but no text of their own: the content
        # lives in the anchor cell of the span.
        if marker in ("ucel", "lcel"):
            row.append("")
            continue
        row.append(" ".join(text.split()).replace("|", r"\|"))
    if row:
        rows.append(row)
    if not rows:
        return None
    width = max(len(r) for r in rows)
    return Grid(rows=tuple(tuple(r + [""] * (width - len(r))) for r in rows))


def merge_with_docling(docling_rows: list[list[str]], read: Grid) -> list[list[str]] | None:
    """Replace Docling's data cells with the model's, keeping Docling's header.

    Returns ``None`` — meaning "keep Docling's table" — unless the two agree on
    the column count and the model supplied at least as many body rows. Those
    two checks are what make this safe:

    * a different column count means the crop caught the wrong region or the
      model split cells differently, and a foreign grid would corrupt every
      column;
    * fewer body rows means the model collapsed rows, which is the failure it
      exists to fix — Docling's version is then no worse.

    Header rows are kept from Docling: the model misreads header text, and the
    header is what downstream translation and leak checks key on.
    """
    if not docling_rows or read.columns != len(docling_rows[0]):
        return None
    body = read.rows
    if len(body) < len(docling_rows):
        return None
    header_count = _header_row_count(docling_rows, body)
    merged: list[list[str]] = [list(row) for row in docling_rows[:header_count]]
    merged.extend(list(row) for row in body[header_count:])
    return merged


def _header_row_count(docling_rows: list[list[str]], body: tuple[tuple[str, ...], ...]) -> int:
    """How many leading rows are header, agreeing with what the model produced.

    A header row in Docling's grid is one whose model counterpart is textually
    different (the model transcribed it from the image) while the row shape is
    the same. Counting stops at the first row the two agree on, which is where
    the data starts. Capped at two rows: every header this project has seen is
    one or two rows deep, and a larger count would start discarding data.
    """
    count = 0
    for index, docling_row in enumerate(docling_rows[:2]):
        if index >= len(body):
            break
        model_row = body[index]
        if not any(cell.strip() for cell in model_row):
            break
        docling_text = " ".join(" ".join(cell.split()) for cell in docling_row)
        model_text = " ".join(" ".join(cell.split()) for cell in model_row)
        if docling_text == model_text:
            break
        count += 1
    return count


def _build_request(image: Any) -> tuple[str, dict[str, Any], dict[str, str]] | None:
    """The chat request for one table crop, or ``None`` when unconfigured."""
    import base64
    import io

    endpoint = os.environ.get(ENDPOINT_ENV_VAR, "").strip()
    if not endpoint:
        return None
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint.rstrip('/')}/chat/completions"
    model = os.environ.get(MODEL_ENV_VAR, "").strip()
    if not model:
        logger.warning(
            "%s is set but %s is empty: table reader skipped", ENABLE_ENV_VAR, MODEL_ENV_VAR
        )
        return None

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(buf.getvalue()).decode("ascii")
                        },
                    },
                    {"type": "text", "text": TABLE_PROMPT},
                ],
            }
        ],
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return endpoint, payload, headers


def read_table_image_sync(image: Any) -> Grid | None:
    """Send a table crop to the configured reader and parse its reply.

    Any failure (no endpoint, HTTP error, unparsable reply) returns ``None`` so
    the caller keeps Docling's table: a vision read is an upgrade, and a failed
    upgrade must not remove content. Blocking on purpose — the extraction path
    runs inside a worker thread, so restructuring the parser into ``async``
    would buy nothing.
    """
    import httpx

    request = _build_request(image)
    if request is None:
        return None
    endpoint, payload, headers = request
    try:
        with httpx.Client(timeout=float(os.environ.get("UBT_OCR_TIMEOUT", "300"))) as client:
            response = client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        content = str(body["choices"][0]["message"]["content"])
    except Exception as exc:  # noqa: BLE001 - a failed upgrade keeps Docling's table
        logger.warning("table reader failed, keeping Docling's cells: %s", exc)
        return None
    return parse_table_markup(content)

"""Deterministic test doubles that stand in for a translation model.

These are not fixtures that happen to return text: each one makes promises the
pipelines' own gates then measure against. :class:`TokenEchoMockProvider` is the
offline end-to-end baseline double, and its contract lives in its docstring —
when a baseline fails, read the contract first, because the double is supposed
to behave like a competent model and the gate is supposed to be right.
"""

import re

from ubt.core.qe.fast_pass import grid_columns
from ubt.core.qe.omission import count_sentences, identifier_terms
from ubt.core.router.provider import MockModelProvider
from ubt.core.validators.math_guard import target_missing_math_delimiters

_TOKEN_ECHO_RE = re.compile(r"⟦[A-Z_]+_\d+(?:-[0-9a-z]{3})?⟧")
# Same numeric token shape as NumericConsistencyValidator._NUM.
_NUM_ECHO_RE = re.compile(r"\d[\d,.\-–—/]*\d|\d")
# Inline/display math the structural gate counts. The draft prompt carries it
# masked as ⟦MATH_n⟧ (echoed via _TOKEN_ECHO_RE); the repair prompt carries it
# raw, so it must be echoed verbatim too.
_MATH_ECHO_RE = re.compile(r"\$\$[^$]+\$\$|\$[^$\n]+\$")
# Column-rotation labels for rendered grid cells. The cell text carries no
# meaning by design; what the gate inspects is the grid *shape*.
_CELL_LABELS = ("组件", "维度", "性质", "取值", "附注", "备注")
_CN_DIGITS = "零一二三四五六七八九"


def _cjk_counter(n: int) -> str:
    """Render ``n`` with CJK digits.

    Positional, not idiomatic (24 -> 二四): the point is a globally unique
    suffix that stays out of the ASCII numeral space, so the rendered cells can
    never be mistaken for source figures by the numeric-consistency gate.
    """
    return "".join(_CN_DIGITS[int(ch)] for ch in str(n))


def _cell_tokens(cell: str) -> list[str]:
    """Opaque/verbatim payloads inside one table cell that must survive."""
    return [
        *_MATH_ECHO_RE.findall(cell),
        *sorted(identifier_terms(cell)),
        *_TOKEN_ECHO_RE.findall(cell),
        *_NUM_ECHO_RE.findall(cell),
    ]


def render_grid(source: str) -> str:
    """Re-emit ``source``'s markdown grid with an identical row/column shape.

    The offline baselines must clear the structural gate on their own merits,
    and that gate refuses a source table flattened into prose
    (``FastPassFilter`` "Table dropped", MQM critical). Rendering through the
    gate's *own* predicate (:func:`grid_columns`) is what keeps the two in
    lockstep: column counts are equal by construction, so a baseline failure on
    a table block means the pipeline lost the table, not that the double
    mistranslated it.
    """
    rows: list[str] = []
    counter = 0
    for line in source.splitlines():
        if grid_columns(line) is None:
            continue
        cells: list[str] = []
        for column, raw in enumerate(line.strip().strip("|").split("|")):
            counter += 1
            label = f"{_CELL_LABELS[column % len(_CELL_LABELS)]}{_cjk_counter(counter)}"
            cells.append(" ".join([label, *_cell_tokens(raw)]))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


class TokenEchoMockProvider(MockModelProvider):
    """Deterministic mock that behaves like a *competent* model.

    Five contracts the offline baselines depend on, all enforced in
    :meth:`generate`:

    1. Every mapped term the source paragraph contains contributes to the target
       (no first-match-wins dropout when a paragraph mentions two terms).
    2. The target tracks the source's shape: roughly one sentence per source
       sentence and a length inside the en->zh pair bounds (0.2..1.5), so the
       omission and length-ratio gates measure the pipeline, not the fixture.
    3. Opaque placeholders (``⟦MATH_1⟧``), math, identifiers and numbers from
       the source are echoed back; production masking/unmasking and the
       omission/numeric gates assume a competent model returns them, so a mock
       that drops them would test mock inadequacy instead of pipeline behavior.
       Position is not preserved (appended at end); real LLMs keep position.
    4. A markdown table in the source stays a markdown table with the same grid
       shape (see :func:`render_grid`); a mock that flattened it would trip the
       structural gate and the baseline would be measuring its own fixture.
    5. Flattened math residue is re-delimited in ``$...$``, decided by the gate's
       own predicate rather than a copy of its regex.

    ``tests/unit/test_mock_provider_contract.py`` guards these against the
    product's own predicates so a gate tightening shows up there rather than as
    an unexplained red CI.
    """

    #: One numbered sentence of filler. The index matters: FastPass treats a
    #: 4-20 char unit repeated 4+ times as a hallucination loop, so identical
    #: fillers would fail the very gate the baseline exercises.
    _UNIT = "离线基线占位译文第 {} 句。"
    #: Filler for blocks too short to fit :data:`_UNIT` under the length cap.
    _MICRO = "占位 {}。"
    #: Flattened math residue the 3c gate wants re-delimited (``psi_pert``,
    #: ``V_th``, ``F 1``, Greek). Wrapped in ``$...$`` when the source carries
    #: it and the target has no span.
    _MATH_RESIDUE_RE = re.compile(
        r"[A-Za-z]+_[A-Za-z0-9{]+"
        r"|[A-Za-z]\s*[_^]\s*[A-Za-z0-9{]"
        r"|\b[A-Z]\s+\d"
        r"|[\u0370-\u03ff]"
    )
    #: Headings that carry the source in the draft and repair prompts. The
    #: repair path uses a different one, and missing it made every repair
    #: candidate look truncated to the length gate.
    _SOURCE_MARKERS = (
        "### Source Paragraph to Translate\n",
        "### Original Source\n",
    )
    _SOURCE_STOPS = ("\n\n### ", "\n\nTranslate only the text")

    @classmethod
    def _extract_source(cls, prompt: str) -> str | None:
        """Return the source paragraph embedded in a draft/repair prompt."""
        for marker in cls._SOURCE_MARKERS:
            if marker not in prompt:
                continue
            tail = prompt.split(marker, 1)[1]
            stops = [tail.find(stop) for stop in cls._SOURCE_STOPS]
            stops = [index for index in stops if index != -1]
            return tail[: min(stops)] if stops else tail
        return None

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        # Call super() for its side effect (call_history); the competent text is
        # built below rather than reusing the first-match-wins mock response.
        await super().generate(
            prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

        source = self._extract_source(prompt)
        scope = source if source is not None else prompt

        # Competent-model contract #1 — no per-term dropout: every mapped term
        # the *source paragraph* contains contributes. Scoping to the source
        # (not the whole prompt) matters: the prompt also carries the glossary
        # table, and matching against that stitched every term into every block.
        matched = [value for key, value in self._custom.items() if key in scope]

        # Contract #4 — a source grid stays a grid, rendered by the gate's own
        # predicate. Only when a source paragraph was actually carved out of the
        # prompt: the prompt scaffolding itself contains a glossary table, and
        # echoing that into every block would ship a table in every paragraph.
        grid = render_grid(source) if source is not None else ""

        # Contract #3 — fidelity on exactly the tokens the deterministic gates
        # require: source identifiers (camelCase / ALL_CAPS / digit-bearing runs,
        # via the omission gate's own definition), display/inline math, opaque
        # placeholders (``⟦MATH_1⟧``) and numbers. A competent model never drops
        # these; the placement is appended rather than preserved, which real
        # models do better but which no gate here requires. Computed before the
        # body because the length budget must reserve room for them — cell
        # payloads already inside ``grid`` are dropped from the tail so the
        # figures are not doubled.
        extras = [
            token
            for token in dict.fromkeys(
                [
                    *_MATH_ECHO_RE.findall(scope),
                    *sorted(identifier_terms(scope)),
                    *_TOKEN_ECHO_RE.findall(scope),
                    *_NUM_ECHO_RE.findall(scope),
                ]
            )
            if token not in grid
        ]
        extras_text = f" {' '.join(extras)}" if extras else ""

        if source is None:
            # No source paragraph to size against (non-draft prompt).
            text = ("".join(matched) if matched else self._default) + extras_text
        else:
            src = source.strip()
            src_len = max(1, len(src))
            mandatory_len = len(grid) + len(extras_text)
            # The length gate accepts (0.2, 1.5) x source; target ~0.35x and cap
            # at 1.2x so the gate measures the pipeline, not the fixture. The
            # mandatory extras count against both bounds: on a short all-caps
            # title they ARE the target ("THE END" -> "END THE", ratio 1.0).
            cap = int(src_len * 1.2)
            body_cap = max(0, cap - mandatory_len)
            body_floor = max(0, int(src_len * 0.35) - mandatory_len)
            # The filler has to fit under the body cap: tiny blocks take micro.
            unit = self._UNIT if len(self._UNIT.format(0)) <= body_cap else self._MICRO
            body = ""
            for value in matched:
                if len(body) + len(value) <= body_cap:
                    body += value
            if not body and len(self._default) <= body_cap:
                body = self._default
            # Contract #2a — sentence parity: the omission gate refuses a target
            # with fewer than half the source's sentences, so a competent
            # translation keeps roughly the paragraph's sentence structure.
            sentences = count_sentences(src)
            index = 0
            while count_sentences(body) < sentences:
                candidate = unit.format(index + 1)
                if len(body) + len(candidate) > body_cap:
                    # Fall back to the shorter filler before giving up: a short
                    # multi-sentence source ('"Wow! wow! wow!"') cannot fit the
                    # long unit under the length cap, and breaking here made the
                    # mock violate contract #2 and trip the omission gate on its
                    # own fixture.
                    candidate = self._MICRO.format(index + 1)
                    if len(body) + len(candidate) > body_cap:
                        break
                index += 1
                body += candidate
            # Contract #2b — length plausibility (>0.2x source, <1.5x).
            while len(body) < body_floor:
                index += 1
                body += unit.format(index)
            text = body + extras_text
            if grid:
                # The table gets its own line, and comes last: a grid row that
                # shares a line with prose (or with a trailing echo token) reads
                # as one extra column to ``grid_columns``.
                text += f"\n{grid}"
            if not text.strip():
                text = unit.format(1)

        # Contract #5 — undelimited math: when the source carries flattened math
        # signals that would render as prose, re-emit them inside ``$...$``.
        # The gate's own predicate decides, so fixture and pipeline agree.
        if source is not None and target_missing_math_delimiters(scope, text):
            residue = "".join(
                f" ${token.strip()}$" for token in self._MATH_RESIDUE_RE.findall(scope)[:2]
            )
            text += f"\n{residue}" if grid else residue
        return text


__all__ = ["TokenEchoMockProvider", "render_grid"]

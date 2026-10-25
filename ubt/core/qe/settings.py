"""Resolving the quality-estimation gate a run will actually use.

Three surfaces answer "is this run scored, and by what?" — ``ubt doctor``, the
console's QE card, and the pipeline itself — and each used to derive it
independently. They disagreed in a way that mattered: the doctor printed the
translator's ``repair_model`` as the judge while a ``qe_judge_provider`` block
answered with its own model, and the console reported "QE not measured" with no
hint about which switch was off.

**``qe_judge_enabled`` is the switch that scores anything.** ``qe_engine`` names
the intended runner but does not select one: the pipeline wraps the heuristic
runner in :class:`TieredQERunner` whenever ``qe_judge_enabled`` is true, and only
*logs a warning* when ``qe_engine = "tiered"`` was written without it. So
``qe_engine = "tiered"`` alone is a no-op, and ``qe_judge_enabled = true`` alone
does score — a distinction the documentation states but no surface enforced.

:func:`is_qe_measurable` therefore keys on the judge, not the label. The console
card writes both keys together so the no-op combination is unreachable from the
UI; a hand-edited config that still has it gets a warning from the engine.
"""

from __future__ import annotations

from typing import Any

#: ``[defaults]`` keys the console may write for the quality-estimation gate.
#:
#: One list, imported by both the writer (:mod:`ubt.api.provider_editor`) and the
#: reader (:mod:`ubt.api.qe_settings`): a field the form can display but not save,
#: or save but not display, is a control that lies.
#:
#: Deliberately *not* ``qe_judge_base_url`` / ``qe_judge_api_key`` /
#: ``qe_judge_api_mode``. A judge endpoint belongs in a ``[providers.<name>]``
#: block, which the console already edits and which resolves credentials through
#: the same ``${VAR}`` path as every other block — offering a second, ad-hoc
#: endpoint form here would duplicate that and give the console a second place to
#: write a credential.
QE_WRITABLE_FIELDS: tuple[str, ...] = (
    "qe_engine",
    "qe_judge_enabled",
    "qe_judge_provider",
    "qe_judge_model",
    "qe_threshold",
    "qe_judge_pass_sample",
    "qe_judge_sentences",
)


def resolve_judge_model(config: Any) -> str | None:
    """The model id the judge will actually call, or ``None`` when unset.

    Three sources, in the order the pipeline resolves them
    (``EngineOrchestrator._build_judge_fn``):

    1. ``qe_judge_provider`` — a ``[providers.<name>]`` block. That block carries
       its own ``repair_model`` / ``draft_model``, and it is the one that answers.
       Checking only ``qe_judge_model or repair_model`` priced a different name
       than the run would bill (measured: a local ``gemma4-e4b`` sent to a
       different gateway answered ``404 no such model`` on every judge call).
    2. ``qe_judge_model`` — a model on the translation endpoint.
    3. ``repair_model`` — the implicit default.

    A broken provider block is not reported here: it has its own doctor check,
    and raising from a status surface would turn a diagnosable config into a
    crashed page.
    """
    judge: str | None = config.qe_judge_model or config.repair_model
    provider = getattr(config, "qe_judge_provider", None)
    if provider:
        from ubt.core.providers import load_layer_config

        try:
            judge_cfg = load_layer_config(provider)
        except Exception:  # noqa: BLE001 - reported by the provider's own check
            return judge
        judge = str(judge_cfg.repair_model or judge_cfg.draft_model or judge or "") or judge
    return judge


def is_qe_measurable(config: Any) -> bool:
    """Whether this configuration produces a *measured* average QE.

    Two conditions, both required:

    * ``qe_judge_enabled = true`` — this is what wraps the heuristic runner. A
      ``qe_engine = "tiered"`` label on its own changes nothing but a log line.
    * a judge model — with none, the wrapper has nothing to call and every block
      keeps the deterministic verdict.

    The quality report's ``qe_measured`` is the same fact computed after the run
    (:func:`ubt.core.qe.score_policy.is_qe_scored`); this is the prediction the
    console shows *before* a job is submitted, which is why both exist.
    """
    return bool(getattr(config, "qe_judge_enabled", False)) and bool(resolve_judge_model(config))

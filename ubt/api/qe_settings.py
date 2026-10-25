"""The console's view of the quality-estimation gate.

**Why this exists.** The console showed "QE not measured (heuristic engine)" on
every completed job and offered no way to change it. The remedy was documented —
``qe_engine = "tiered"`` plus ``qe_judge_enabled = true`` plus a judge model, in
``ubt.toml [defaults]`` or as ``UBT_QE_*`` variables — but the only place to act
on it was a shell. A status that names a fix the reader cannot reach from the
screen they are on is a dead end, which is the same defect the provider panel was
built to remove.

**Why the card writes both keys.** ``qe_engine = "tiered"`` on its own is a
no-op: the pipeline selects the judge on ``qe_judge_enabled`` and only logs a
warning about the mismatch. Exposing the two as independent controls would let an
operator save a configuration that reads "tiered" and still scores nothing. The
console therefore treats them as one choice ("heuristic" → both off, "tiered" →
both on), and :func:`ubt.api.provider_editor.update_qe_settings` writes the pair
from a single request.

**Why the effective values are re-read, not echoed.** ``UBT_QE_ENGINE`` and
friends outrank the file. A save that wrote ``ubt.toml`` while a variable was
exported would report a success the next job never sees, so
:func:`describe_qe_settings` reports which fields are overridden and the card
says so instead of pretending the write took effect.
"""

from __future__ import annotations

import os
from typing import Any

from ubt.core.qe.settings import QE_WRITABLE_FIELDS, is_qe_measurable, resolve_judge_model


def _env_name(field: str) -> str:
    """The variable that outranks this field in ``UBTConfig``.

    Every one of these uses the settings source's default ``UBT_`` prefix plus
    the upper-cased field name; ``qe_judge_provider`` and ``qe_judge_sentences``
    declare that same alias explicitly, so one rule covers all seven.
    """
    return f"UBT_{field.upper()}"


def env_overrides() -> list[str]:
    """Fields whose effective value comes from the environment, not the file."""
    return [field for field in QE_WRITABLE_FIELDS if os.environ.get(_env_name(field))]


def describe_qe_settings(config: Any) -> dict[str, Any]:
    """The gate as the engine will run it, in the shape the card renders."""
    engine = str(getattr(config, "qe_engine", "heuristic") or "heuristic")
    judge_enabled = bool(getattr(config, "qe_judge_enabled", False))
    judge_provider = getattr(config, "qe_judge_provider", None)
    # ``or 0.75`` would rewrite a legitimate 0.0 threshold (fail everything) to
    # the default; only an absent value may fall back.
    raw_threshold = getattr(config, "qe_threshold", None)
    return {
        "engine": engine,
        # The switch that actually wraps the runner, reported separately from
        # the label: a hand-edited `qe_engine="tiered"` with the judge off is a
        # real state, and the card has to be able to say so.
        "judge_enabled": judge_enabled,
        "judge_provider": str(judge_provider) if judge_provider else None,
        "judge_model": resolve_judge_model(config) or "",
        # Raw, not resolved: the edit form seeds its input from this, and seeding
        # it from the resolved id would write an inherited `repair_model` back as
        # an explicit `qe_judge_model` on a save that never touched the field.
        "judge_model_explicit": (
            str(config.qe_judge_model) if getattr(config, "qe_judge_model", None) else None
        ),
        "threshold": float(0.75 if raw_threshold is None else raw_threshold),
        "judge_pass_sample": float(getattr(config, "qe_judge_pass_sample", 0.0) or 0.0),
        "judge_sentences": bool(getattr(config, "qe_judge_sentences", True)),
        # The one derived fact the card exists to state: without a judge model
        # the wrapper has nothing to call, and every block keeps its
        # deterministic verdict. Predicted here; the report's `qe_measured` is
        # the same fact measured after the run.
        "measurable": is_qe_measurable(config),
        "env_overrides": env_overrides(),
        "config_source": "ubt.toml [defaults] or UBT_QE_* environment variables",
    }

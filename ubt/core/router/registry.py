"""Capability profile resolution: declared per provider, never inferred from a name.

**Why this module no longer ships a model table.** It used to carry fourteen
built-in ``ModelProfile`` entries keyed by name patterns (``o1``, ``o3``,
``claude-3-7``, ``qwen``, ``muse-spark``, ``deepseek-r1``, ``reasoning``,
``reasoner``, ``deepseek``, ``flash``, ``pro``) and resolve a model by exact
match, then by separator-bounded substring match, then by a heuristic on the
tokens ``reasoner``/``reasoning``/``thinking``/``r1``. Four independent problems
made that design unfixable by editing the list:

1. **It answered a question nobody asked.** All fourteen entries carried
   ``prompt_strategy=hybrid`` and ``extraction_strategy=auto``, so the table
   produced exactly four distinct behaviour tuples and never once selected
   ``RICH`` or ``MINIMAL``. The screen that rendered it read as a model
   inventory while listing no configured model.

2. **Substring matching collided with unrelated vendors.** ``pro`` captured
   ``llama-3.1-pro``, ``doubao-pro``, ``hunyuan-pro`` and ``minimax-m2-pro``;
   ``flash`` captured ``glm-4.6-flash``. Because a ``pro`` match sets
   ``supports_reasoning_effort=True``, and that flag makes the router leave
   ``max_tokens`` unset, those models silently lost the output budget computed
   for them. ``r1`` captured ``tar1ff`` and ``mar1ne-embed``.

3. **It contradicted the pricing table in the same package.**
   ``ubt/resources/prices.toml`` prices ``gpt-5``, ``gpt-5-mini``,
   ``claude-sonnet-4`` and ``gemini-3``; none of them resolved to a capability
   profile. A ``claude-sonnet-4-5`` run matched the bare ``claude`` entry, whose
   ``supports_reasoning_effort=False`` suppressed the extended-thinking budget
   the model accepts — a capability loss no error ever reported.

4. **The list could not be maintained.** A new model needed a code change and a
   release. The operator's own local models matched nothing at all.

**Resolution order now** (see :meth:`ModelCapabilityRegistry.resolve`):

1. an exact, case-insensitive match against a profile *declared for this
   configuration* — normally the one derived from the active
   ``[providers.<name>]`` block;
2. the generic fallback, which asserts only what is safe for an unknown
   endpoint: system turn and temperature accepted, reasoning effort and vision
   not claimed.

Nothing is inferred from a model name, so there is no table to age.

**Two sources of declared profiles**, deliberately separated:

* :func:`registry_from_config` derives them from one ``UBTConfig`` — the
  per-job answer. This replaced a call to ``config.bootstrap_runtime_environment()``
  that *registered into the process-wide singleton*, which meant job A's
  capability override silently became job B's capability for the same model id
  and the registry grew without bound.
* :func:`get_default_registry` holds only what the process environment declares
  (``UBT_MODEL_PROFILES_JSON`` / ``UBT_MODEL_PROFILES_FILE``). It is a
  process-level escape hatch, so it belongs to the process rather than to a job.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import pydantic

from ubt.core.router.capabilities import (
    ExtractionStrategy,
    ModelProfile,
    PromptStrategy,
)

logger = logging.getLogger(__name__)

#: The pattern that marks a profile as the generic fallback rather than a
#: declared model. Kept as a constant because several call sites compare against
#: it to answer "did this model resolve to anything real?".
FALLBACK_PATTERN = "*"


def fallback_profile(model_name: str | None = None) -> ModelProfile:
    """The conservative profile for an endpoint that declared no capabilities.

    Every field is the value that cannot provoke a 400: a system turn and a
    temperature are accepted by essentially every OpenAI-compatible server, and
    nothing is claimed about reasoning or vision. Assuming *more* than the
    endpoint supports costs a failed call on every block; assuming less only
    leaves performance on the table, and the operator can declare the truth in
    the provider block.
    """
    return ModelProfile(
        model_pattern=FALLBACK_PATTERN,
        prompt_strategy=PromptStrategy.HYBRID,
        extraction_strategy=ExtractionStrategy.AUTO,
        supports_reasoning_effort=False,
        supports_system_prompt=True,
        supports_temperature=True,
        supports_vision=False,
        display_name=f"Undeclared ({model_name or 'unspecified'})",
    )


class ModelCapabilityRegistry:
    """The capability profiles one configuration declares, plus the fallback.

    Not a global model database: an instance holds exactly the profiles handed to
    it, so two jobs on the same model id cannot see each other's overrides.
    """

    def __init__(self, profiles: list[ModelProfile] | None = None) -> None:
        # Guards the read-modify-write in :meth:`register`. Reentrant because a
        # caller may register from inside an iteration over ``list_profiles()``.
        self._lock = threading.RLock()
        self._profiles: list[ModelProfile] = []
        for profile in profiles or []:
            self.register(profile)

    def register(self, profile: ModelProfile, override: bool = True) -> None:
        """Add a profile; with ``override`` it replaces a same-named one.

        A replacement keeps its position so that the list stays stable for a UI
        rendering it, and so a re-registration cannot reorder resolution.
        """
        with self._lock:
            existing_idx = next(
                (
                    idx
                    for idx, p in enumerate(self._profiles)
                    if p.model_pattern.lower() == profile.model_pattern.lower()
                ),
                None,
            )
            if existing_idx is not None:
                if not override:
                    raise ValueError(
                        f"Model profile for pattern {profile.model_pattern!r} already exists"
                    )
                self._profiles[existing_idx] = profile
            else:
                self._profiles.append(profile)

    def resolve(self, model_name: str | None = None) -> ModelProfile:
        """The profile for ``model_name``: an exact declared match, else the fallback.

        Case-insensitive and exact. There is deliberately no substring pass and
        no name-token heuristic — see the module docstring for what those cost.
        A model id that was never declared is not an error; it runs on the
        conservative fallback, which :attr:`ModelProfile.is_fallback` marks so a
        diagnostic screen can say so instead of implying the model is unknown to
        the engine.
        """
        name_lower = (model_name or "").strip().lower()
        if name_lower:
            with self._lock:
                for profile in self._profiles:
                    if profile.model_pattern.lower() == name_lower:
                        return profile
        return fallback_profile(model_name)

    def list_profiles(self) -> list[ModelProfile]:
        """Return a copy of the declared profiles (never the fallback)."""
        with self._lock:
            return list(self._profiles)

    def get(self, model_pattern: str) -> ModelProfile | None:
        """Find a declared profile by exact pattern name."""
        p_lower = model_pattern.lower()
        with self._lock:
            for p in self._profiles:
                if p.model_pattern.lower() == p_lower:
                    return p
        return None


def _parse_profile_item(item: Any, source: str) -> ModelProfile | None:
    """One env-declared profile, or ``None`` with a logged reason.

    Per item rather than per batch: the previous loader wrapped the whole loop in
    one ``try``, so a single invalid enum discarded every other profile in the
    list and left a truncated 200-character warning. An operator who declares
    eight models and mistypes one must keep the other seven.
    """
    if not isinstance(item, dict):
        logger.warning("Ignoring %s entry that is not an object: %.120r", source, item)
        return None
    try:
        return ModelProfile(**item)
    except (pydantic.ValidationError, TypeError, ValueError) as exc:
        # ``ModelProfile`` forbids unknown keys, so this also reports a
        # misspelled capability instead of letting it fall back to the default.
        logger.warning(
            "Ignoring invalid %s entry %r: %s",
            source,
            item.get("model_pattern", "<no model_pattern>"),
            exc,
        )
        return None


def _load_environment_profiles() -> list[ModelProfile]:
    """Profiles declared by the process environment, skipping malformed entries.

    Reads ``UBT_MODEL_PROFILES_JSON`` / ``UBT_MODEL_PROFILES_FILE`` directly
    instead of through ``UBTConfig``: this runs while the default registry is
    being built, and constructing a ``UBTConfig`` here would re-enter that
    bootstrap. The process environment is the config layer's only external
    source (no dotenv), so the direct read sees exactly what ``UBTConfig`` would.
    """
    loaded: list[ModelProfile] = []

    json_env = os.environ.get("UBT_MODEL_PROFILES_JSON", "").strip()
    if json_env:
        try:
            raw_list = json.loads(json_env)
        except ValueError as exc:
            logger.warning("Ignoring malformed UBT_MODEL_PROFILES_JSON: %s", exc)
            raw_list = None
        if isinstance(raw_list, list):
            for item in raw_list:
                profile = _parse_profile_item(item, "UBT_MODEL_PROFILES_JSON")
                if profile is not None:
                    loaded.append(profile)
        elif raw_list is not None:
            logger.warning("Ignoring UBT_MODEL_PROFILES_JSON that is not a JSON array")

    file_env = os.environ.get("UBT_MODEL_PROFILES_FILE", "").strip()
    if file_env:
        path = Path(file_env)
        if not path.is_file():
            logger.warning("UBT_MODEL_PROFILES_FILE does not exist: %s", path)
        else:
            try:
                with path.open(encoding="utf-8") as handle:
                    raw_list = json.load(handle)
            except (OSError, ValueError) as exc:
                logger.warning("Ignoring unreadable model profiles file %s: %s", path, exc)
                raw_list = None
            if isinstance(raw_list, list):
                for item in raw_list:
                    profile = _parse_profile_item(item, str(path))
                    if profile is not None:
                        loaded.append(profile)
            elif raw_list is not None:
                logger.warning("Ignoring model profiles file %s that is not a JSON array", path)

    return loaded


_default_registry: ModelCapabilityRegistry | None = None
_bootstrap_lock = threading.Lock()


def get_default_registry() -> ModelCapabilityRegistry:
    """The process-wide registry of *environment-declared* profiles.

    Locked: under a threaded server the check-then-assign raced, letting two
    callers build separate registries and losing whichever profile set landed
    second.

    Holds no built-in model table — only ``UBT_MODEL_PROFILES_*``. Per-job
    capability configuration must go through :func:`registry_from_config`;
    registering it here is what leaked one job's overrides into the next.
    """
    global _default_registry
    if _default_registry is None:
        with _bootstrap_lock:
            if _default_registry is None:
                _default_registry = ModelCapabilityRegistry(_load_environment_profiles())
    return _default_registry


def registry_from_config(config: Any) -> ModelCapabilityRegistry:
    """A registry holding the profiles one ``UBTConfig`` declares.

    Built fresh per call and never cached globally, so a job's capability
    overrides die with the job. Environment-declared profiles are layered in
    first (they are process-wide by definition) and the configuration's own
    profile last, so the active provider block wins.

    The returned registry always resolves — an undeclared model gets the
    fallback rather than an error.
    """
    registry = ModelCapabilityRegistry(get_default_registry().list_profiles())
    declared = getattr(config, "capability_profiles", None)
    if callable(declared):
        for profile in declared():
            registry.register(profile)
    return registry

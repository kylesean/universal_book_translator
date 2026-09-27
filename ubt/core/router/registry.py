"""Model capability registry managing built-in and enterprise custom profiles."""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.core.router.capabilities import (
    ExtractionStrategy,
    ModelProfile,
    PromptStrategy,
)

if TYPE_CHECKING:
    from ubt.core.config import UBTConfig

logger = logging.getLogger(__name__)


class ModelCapabilityRegistry:
    """Registry maintaining model capability profiles.

    Enables declarative model strategy dispatch without brittle hardcoded string branches.
    Supports enterprise custom overrides via code registration, config files, or environment variables.
    """

    def __init__(self) -> None:
        # Guards ``_profiles`` against the read-modify-write in :meth:`register`.
        # Reentrant because construction itself registers profiles, and because
        # the singleton is also mutated at runtime by ``POST /model-profiles``
        # while other threads resolve models against it.
        self._lock = threading.RLock()
        self._profiles: list[ModelProfile] = []
        self._load_builtin_profiles()
        self._load_environment_profiles()

    def _load_builtin_profiles(self) -> None:
        """Initialize built-in model capability profiles."""
        self._profiles = [
            ModelProfile(
                model_pattern="o1",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=False,
                supports_temperature=False,
                display_name="OpenAI o1 reasoning",
            ),
            ModelProfile(
                model_pattern="o3",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=False,
                display_name="OpenAI o3 reasoning",
            ),
            ModelProfile(
                model_pattern="o4",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=False,
                display_name="OpenAI o4 reasoning",
            ),
            ModelProfile(
                model_pattern="claude-3-7",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=True,
                display_name="Anthropic Claude 3.7 hybrid reasoning",
            ),
            ModelProfile(
                model_pattern="claude",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=False,
                supports_system_prompt=True,
                supports_temperature=True,
                display_name="Anthropic Claude series",
            ),
            ModelProfile(
                model_pattern="qwen",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=True,
                display_name="Qwen series",
            ),
            ModelProfile(
                model_pattern="muse-spark",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=True,
                display_name="Meta Muse Spark (Zen responses-only)",
            ),
            ModelProfile(
                model_pattern="deepseek-reasoner",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=False,
                display_name="DeepSeek Reasoner",
            ),
            ModelProfile(
                model_pattern="deepseek-r1",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=False,
                display_name="DeepSeek R1 series",
            ),
            ModelProfile(
                model_pattern="reasoning",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=False,
                display_name="Reasoning model",
            ),
            ModelProfile(
                model_pattern="reasoner",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=False,
                display_name="Reasoner model",
            ),
            ModelProfile(
                model_pattern="deepseek",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=True,
                display_name="DeepSeek series",
            ),
            ModelProfile(
                model_pattern="flash",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=True,
                display_name="Flash draft tier",
            ),
            ModelProfile(
                model_pattern="pro",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=True,
                display_name="Pro repair tier",
            ),
        ]

    def _load_environment_profiles(self, config: UBTConfig | None = None) -> None:
        """Load external profiles from UBTConfig (single env source).

        Reads ``model_profiles_json`` / ``model_profiles_file`` from the
        canonical config layer instead of ``os.getenv`` directly.
        """
        if config is None:
            from ubt.core.config import UBTConfig as _UBTConfig

            config = _UBTConfig()
        # 1. Direct JSON array string in UBT_MODEL_PROFILES_JSON
        json_env = config.model_profiles_json.strip()
        if json_env:
            try:
                raw_list = json.loads(json_env)
                if isinstance(raw_list, list):
                    for item in raw_list:
                        if isinstance(item, dict):
                            self.register(ModelProfile(**item), override=True)
            except Exception:
                # An operator typo must not vanish silently — a
                # malformed custom profile would degrade routing to the
                # built-ins with zero trace.
                logger.warning(
                    "Ignoring malformed UBT_MODEL_PROFILES_JSON (custom model "
                    "profiles not applied): %.200s",
                    json_env,
                )

        # 2. File path in UBT_MODEL_PROFILES_FILE
        file_env = config.model_profiles_file.strip()
        if file_env:
            p = Path(file_env)
            if p.is_file():
                try:
                    with p.open(encoding="utf-8") as f:
                        raw_list = json.load(f)
                        if isinstance(raw_list, list):
                            for item in raw_list:
                                if isinstance(item, dict):
                                    self.register(ModelProfile(**item), override=True)
                except Exception:
                    logger.warning(
                        "Ignoring malformed model profiles file %s (custom "
                        "model profiles not applied)",
                        p,
                    )

    def register(self, profile: ModelProfile, override: bool = True) -> None:
        """Register or override a model profile.

        Custom profiles are inserted at index 0 to guarantee higher resolution priority over built-ins.
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
                self._profiles.pop(existing_idx)
            self._profiles.insert(0, profile)

    def resolve(self, model_name: str | None = None) -> ModelProfile:
        """Resolve the capability profile for a given model name.

        Matching logic:
        1. Exact match (case-insensitive)
        2. Substring match (case-insensitive pattern in model_name)
        3. Fallback: conservative default profile for unknown models
        """
        name_lower = model_name.strip().lower() if model_name else ""

        if name_lower:
            # First pass: exact match
            for profile in self._profiles:
                if profile.model_pattern.lower() == name_lower:
                    return profile

            # Second pass: separator-bounded pattern match, uniformly for every
            # pattern length. Longest matching pattern wins. Ties keep
            # registration order, so custom profiles (inserted at index 0) beat
            # same-length built-ins. The previous split — bounded regex only
            # for patterns <= 4 chars, plain substring otherwise — let a long
            # pattern match inside an unrelated word and was easy to break.
            best: ModelProfile | None = None
            for profile in self._profiles:
                pat = profile.model_pattern.lower()
                if pat == "*":
                    continue
                cleaned_pat = pat.lstrip("-")
                if re.search(
                    rf"(?:^|[-_/:.]){re.escape(cleaned_pat)}(?:$|[-_/:.0-9])", name_lower
                ) and (best is None or len(pat) > len(best.model_pattern)):
                    best = profile
            if best is not None:
                return best

        # Heuristic inference for modern reasoning models (e.g. custom/unknown models with reasoning/r1 tokens)
        if name_lower and any(
            token in name_lower for token in ("reasoner", "reasoning", "thinking", "r1")
        ):
            return ModelProfile(
                model_pattern=name_lower,
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=True,
                supports_system_prompt=True,
                supports_temperature=False,
                display_name=f"Heuristic Reasoning ({model_name or 'unspecified'})",
            )

        # Unknown model safe fallback: rich prompt + auto extraction, no assumed reasoning_effort
        return ModelProfile(
            model_pattern="*",
            prompt_strategy=PromptStrategy.RICH,
            extraction_strategy=ExtractionStrategy.AUTO,
            supports_reasoning_effort=False,
            supports_system_prompt=True,
            supports_temperature=True,
            display_name=f"Unknown ({model_name or 'unspecified'})",
        )

    def list_profiles(self) -> list[ModelProfile]:
        """Return a copy of all registered profiles."""
        return list(self._profiles)

    def get(self, model_pattern: str) -> ModelProfile | None:
        """Find an exact registered profile by pattern name."""
        p_lower = model_pattern.lower()
        for p in self._profiles:
            if p.model_pattern.lower() == p_lower:
                return p
        return None


_default_registry: ModelCapabilityRegistry | None = None


_bootstrap_lock = threading.Lock()


def get_default_registry() -> ModelCapabilityRegistry:
    """Get the process-wide default ModelCapabilityRegistry singleton.

    Locked: under a threaded server the check-then-assign raced, letting two
    callers build separate registries and losing whichever profile set landed
    second — and with it any custom profiles registered against the loser.
    """
    global _default_registry
    if _default_registry is None:
        with _bootstrap_lock:
            if _default_registry is None:
                _default_registry = ModelCapabilityRegistry()
    return _default_registry

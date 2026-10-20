"""Declarative model capabilities and strategy models for translation routing.

A capability profile describes **what a model accepts on the wire** — whether it
takes a system turn, a temperature, a reasoning effort, an image — plus the
prompt-assembly strategy to use for it. The router reads these flags to decide
what to *send*; a wrong flag is not cosmetic, it either drops a supported
parameter or provokes a 400 on every call.

**Where the truth lives.** A profile is resolved from the model's own
``[providers.<name>]`` block, which already names the endpoint, the wire
protocol and the model id. Capability therefore travels with the configuration
that names it, and two providers pointing at the same model id can disagree —
which is correct, because the same weights behind two gateways can accept
different parameters.

Nothing here infers capability from a model *name*. An earlier design shipped a
fourteen-entry table of name patterns (``o1``, ``pro``, ``flash``, …) and
resolved by substring, which silently disabled reasoning on models whose names
did not contain the right token and removed the output-token cap from models
that merely ended in ``-pro``.
:meth:`ubt.core.router.registry.ModelCapabilityRegistry.resolve` documents the
resolution order that replaced it.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PromptStrategy(StrEnum):
    """Prompt assembly strategy."""

    RICH = "rich"  # Full LLM: static system prompt + multi-tier context + XML boundary schema
    MINIMAL = "minimal"  # Direct translation instruction without system prompt
    HYBRID = "hybrid"  # Intermediary: concise system role + structured user prompt


class ExtractionStrategy(StrEnum):
    """Output extraction strategy."""

    XML_TAG = "xml_tag"  # Extract strictly from <translation> or <final_translation>
    RAW = "raw"  # Direct strip(), no XML or prefix extraction
    CONVERSATIONAL_PREFIX = "cp"  # Strip markdown fences and conversational lead-in prefixes
    AUTO = "auto"  # Try XML first, fallback to code-fence and prefix cleaning


class ModelProfile(BaseModel):
    """Declarative capability profile for one model on one provider.

    Every field is a statement about the endpoint, so a profile is only ever
    attached to a specific ``[providers.<name>]`` block or to the generic
    fallback — never to a model *name* alone.
    """

    # ``forbid`` rather than ``ignore``: a misspelled capability
    # (``supports_temperatur``) used to fall back to the field default and
    # silently assert the opposite of what the operator wrote. An unknown key is
    # a configuration error and must be reported as one.
    model_config = ConfigDict(extra="forbid")

    # Model identification
    model_pattern: str = Field(
        ...,
        description=(
            "The model id this profile describes, exactly as it is sent on the "
            "wire. Resolution is an exact, case-insensitive match; there is no "
            "substring matching."
        ),
    )

    # Capability strategies
    prompt_strategy: PromptStrategy = Field(
        default=PromptStrategy.HYBRID,
        description="Prompt assembly strategy (RICH, MINIMAL, HYBRID).",
    )
    extraction_strategy: ExtractionStrategy = Field(
        default=ExtractionStrategy.AUTO,
        description="Output parsing strategy (XML_TAG, RAW, CONVERSATIONAL_PREFIX, AUTO).",
    )

    # Parameter compatibility
    supports_reasoning_effort: bool = Field(
        default=False,
        description=(
            "Whether the endpoint accepts a reasoning-effort control. True also "
            "means the output-token cap is left unset, because reasoning tokens "
            "bill against it and capping them corrupts the visible answer."
        ),
    )
    supports_system_prompt: bool = Field(
        default=True,
        description="Whether the model supports system turns; if False, merged into user prompt.",
    )
    supports_temperature: bool = Field(
        default=True,
        description=(
            "Whether the model accepts a temperature. False for the reasoning "
            "families that reject it with a 400."
        ),
    )
    supports_vision: bool = Field(
        default=False,
        description="Whether the model natively accepts image inputs (multimodal / VLM).",
    )

    # Metadata
    display_name: str = Field(
        default="",
        description="Human-readable model name for diagnostics and logging.",
    )
    deployment_backend: str = Field(
        default="",
        description=(
            "Backend identifier if specialized. The self-hosted families "
            "('self_hosted', 'ollama', 'llama.cpp', 'llama-swap', 'vllm', "
            "'lmstudio', ...) enable the local retry when a gateway drops the "
            "model; 'ollama' is kept as a legacy alias."
        ),
    )

    @property
    def is_fallback(self) -> bool:
        """True for the generic profile used when nothing is declared.

        The wildcard pattern is the only one that is not tied to a real model id,
        so callers that report "what did this resolve to" can distinguish a
        declared capability set from the conservative default.
        """
        return self.model_pattern == "*"

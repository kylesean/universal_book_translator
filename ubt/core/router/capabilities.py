"""Declarative model capabilities and strategy models for translation routing."""

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
    """Declarative capability profile for a translation model.

    Decouples model execution logic from brittle name matching, enabling
    enterprise self-hosted models to configure behavior declaratively.
    """

    model_config = ConfigDict(extra="ignore")

    # Model identification
    model_pattern: str = Field(
        ...,
        description="Exact model name or substring pattern used to resolve this profile.",
    )

    # Capability strategies
    prompt_strategy: PromptStrategy = Field(
        default=PromptStrategy.RICH,
        description="Prompt assembly strategy (RICH, MINIMAL, HYBRID).",
    )
    extraction_strategy: ExtractionStrategy = Field(
        default=ExtractionStrategy.AUTO,
        description="Output parsing strategy (XML_TAG, RAW, CONVERSATIONAL_PREFIX, AUTO).",
    )

    # Parameter compatibility
    supports_reasoning_effort: bool = Field(
        default=False,
        description="Whether the provider/model accepts the reasoning_effort parameter.",
    )
    supports_system_prompt: bool = Field(
        default=True,
        description="Whether the model supports system turns; if False, merged into user prompt.",
    )
    supports_temperature: bool = Field(
        default=True,
        description="Whether model supports explicit temperature control.",
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

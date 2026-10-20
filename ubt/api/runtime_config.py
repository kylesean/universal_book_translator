"""A read-only view of the model/provider configuration actually in force.

The console's "provider matrix" used to show ``GET /model-profiles``, which is
the engine's *capability preset table* — fourteen hard-coded name patterns
(``o1``, ``o3``, ``claude``, …) that decided prompt strategy and parameter
support when a model name matched. Nothing on that list was a configured model:
on the reference machine the operator runs ``local-gemma4`` / ``gemma4-e4b``,
which matched none of the fourteen and therefore fell through to the generic
default. The screen was titled "model routing and latency matrix" while showing
neither routing nor latency.

That table is gone. Capability is now declared per provider block, so this module
answers the question the operator actually has — *what is my engine going to
call, and where did each of those answers come from?* — without exposing any
credential.

**Credential safety.** ``load_provider_block`` returns the merged block as
written, and a block may legitimately contain ``api_key = "${OPENCODE_API_KEY}"``
or even a literal secret. Only fields on :data:`_EXPOSED_FIELDS` are ever read
out, so a future config key cannot leak by being added to a TOML file.
"""

from __future__ import annotations

from typing import Any

from ubt.core.config import UBTConfig
from ubt.core.providers import list_providers, load_provider_block
from ubt.core.router.registry import FALLBACK_PATTERN, registry_from_config

#: Provider-block fields the console may display. Everything else — most
#: importantly ``api_key``, ``extra_headers`` and any credential — is dropped
#: even when the block declares it. This is an allow-list on purpose: a deny-list
#: would leak the next credential field somebody adds to ``providers.toml``.
_EXPOSED_FIELDS: tuple[str, ...] = (
    "base_url",
    "api_mode",
    "draft_model",
    "repair_model",
    "is_free",
    "supports_temperature",
    "supports_reasoning_effort",
    "supports_system_prompt",
    "supports_vision",
    "extraction_strategy",
    "deployment_backend",
)


def _safe_block(name: str) -> dict[str, Any]:
    """The displayable subset of one provider block; never raises.

    A malformed block must not break the whole page — the operator needs to see
    the *other* providers and the rest of the diagnostics in order to work out
    what is wrong.
    """
    try:
        block = load_provider_block(name)
    except Exception:  # a broken block is reported as unavailable, not fatal
        return {"available": False}
    exposed = {
        key: block[key] for key in _EXPOSED_FIELDS if key in block and block[key] is not None
    }
    # ``is_free`` is what tells the operator a run costs nothing; absent means
    # "not declared", which is not the same as "costs money".
    exposed.setdefault("is_free", False)
    exposed["available"] = True
    return exposed


def describe_runtime_config(
    config: UBTConfig,
    *,
    can_write_credential: bool = False,
    frozen_fields_changed: list[str] | None = None,
) -> dict[str, Any]:
    """What the engine will call, which profile that resolves to, and the menu.

    ``active`` is the part the old screen never showed: the provider and models
    in force *right now*, plus the capability profile the router picks for the
    draft model. ``declared_capability`` says which of those flags the operator
    actually stated in the provider block versus which fell back to the generic
    default — the difference between "I told the engine this model rejects
    temperature" and "the engine assumed it".
    """
    # Per-configuration, matching what a job started from this config will use.
    registry = registry_from_config(config)
    draft_model = config.draft_model or ""
    resolved = registry.resolve(draft_model) if draft_model else None
    # ``resolve`` always returns a profile, so a declared match is only real when
    # the pattern is not the catch-all.
    matched = (
        None
        if resolved is None or resolved.model_pattern == FALLBACK_PATTERN
        else resolved.model_pattern
    )

    active = {
        "provider": config.provider or "",
        "draft_model": draft_model,
        "repair_model": config.repair_model or "",
        "base_url": config.base_url or "",
        "api_mode": config.api_mode or "",
        "matched_pattern": matched,
        "prompt_strategy": resolved.prompt_strategy.value if resolved else "",
        "display_name": resolved.display_name if resolved else "",
        # Which capability flags the provider block states outright. The rest
        # are the generic default, and a diagnostic screen should say so rather
        # than present an assumption as a measurement.
        "declared_capability": sorted(
            field
            for field in (
                "supports_temperature",
                "supports_reasoning_effort",
                "supports_system_prompt",
                "supports_vision",
            )
            if getattr(config, field, None) is not None
        ),
        "supports_temperature": resolved.supports_temperature if resolved else None,
        "supports_reasoning_effort": resolved.supports_reasoning_effort if resolved else None,
        "supports_system_prompt": resolved.supports_system_prompt if resolved else None,
        "supports_vision": resolved.supports_vision if resolved else None,
        "extraction_strategy": resolved.extraction_strategy.value if resolved else "",
    }

    # Split the menu into what the operator declared and what ships with the
    # engine. ``list_providers`` returns all three kinds at once — declared
    # blocks, shipped wire protocols (``openai-chat``) and bare aliases
    # (``chat``, ``claude``, ``responses``) — and rendering them as one list put
    # seven aliases above the operator's own six entries, which is how a
    # seventeen-row wall appears for a six-provider config.
    from ubt.core.providers import PROTOCOL_ALIASES, declared_provider_names

    declared = declared_provider_names()
    providers: list[dict[str, Any]] = []
    for name in list_providers():
        block = _safe_block(name)
        providers.append(
            {
                "name": name,
                "active": name == config.provider,
                "available": block.pop("available", False),
                "declared": name in declared,
                "is_alias": name in PROTOCOL_ALIASES,
                **block,
            }
        )

    return {
        "active": active,
        "providers": providers,
        # Provider settings are editable from the console, but credentials are
        # not: a job payload may never carry one, and this endpoint will not read
        # one back out. Saying where they live is more useful than a dead button.
        "provider_config_source": "ubt.toml [providers.<name>] or UBT_* environment variables",
        # Deliberately worded without the literal credential key name: the
        # console-contract guard asserts no response body contains it, and a
        # string that merely *names* the field would trip the same check. The
        # guard is worth more than the phrasing.
        "credential_source": (
            "the provider block's credential (${VAR} expansion) or UBT_LLM_API_KEY"
        ),
        # Whether the console may write a credential here. Passed in rather than
        # inferred: it depends on the request's bind address and the boot-time auth
        # posture, which only the caller knows. A UI that offered the field and
        # then failed on submit would be the same "control that lies" pattern this
        # panel exists to remove.
        "can_write_credential": bool(can_write_credential),
        "frozen_fields_changed": list(frozen_fields_changed or []),
    }

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

from ubt.api.provider_editor import default_provider_name
from ubt.core.config import UBTConfig, credential_required, has_real_credential
from ubt.core.providers import credential_resolution, list_providers, load_provider_block
from ubt.core.router.pricing import endpoint_is_local
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
    # Whether this block can authenticate, as a code rather than a sentence. The
    # list rendered six rows with no hint that `opencode` — whose block names an
    # unset ``${OPENCODE_API_KEY}`` — could not make a single call. A code lets
    # the console say which export is missing instead of "no credential".
    try:
        state = credential_resolution(name)
        # A loopback endpoint is not authenticated at all: llama.cpp, Ollama and
        # llama-swap accept an empty bearer, and this machine's `local-gemma4`
        # runs that way. Reporting "missing" for it would send the operator to
        # fix a configuration that is already correct — the same false alarm the
        # old `bool(raw)` check produced, just in the other direction.
        local_endpoint = endpoint_is_local(block.get("base_url"))
        exposed["credential_issue"] = (
            "unresolved_variable"
            if state["unresolved_vars"]
            else (None if state["authenticates"] or local_endpoint else "missing")
        )
        # The variable *name* is deliberately not echoed here. This payload
        # renders every provider at once, and a name like `OPENCODE_API_KEY` is
        # one step closer to the credential than a code is; the guard in
        # test_console_contract asserts exactly that. The actionable detail —
        # *which* export is missing — lives on the single-provider endpoint the
        # operator opens to fix it (`/system/providers/{name}`), where naming the
        # variable is the whole point.
    except Exception:  # a block that cannot be resolved is already "unavailable"
        pass
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
    # The same predicate ``/jobs/submit`` uses for its auto-rehearsal decision: a
    # model id is required, and either a real credential or a loopback endpoint.
    # Testing only the credential reported ``can_translate: false`` for a working
    # local model, so the wizard announced a rehearsal the submit endpoint would
    # not actually have run.
    can_translate = bool(draft_model) and (
        has_real_credential(config) or not credential_required(config)
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
        # Whether a submit can actually translate. The server downgrades a keyless
        # submit to a zero-token rehearsal, so the console must be able to say so
        # *before* the button is pressed. Derived here rather than in the client:
        # the rule is "no real credential", not "no declared provider" — an
        # operator using a built-in protocol with UBT_LLM_API_KEY set has no
        # declared block and still runs for real.
        "can_translate": can_translate,
    }

    # Split the menu into what the operator declared and what ships with the
    # engine. ``list_providers`` returns all three kinds at once — declared
    # blocks, shipped wire protocols (``openai-chat``) and bare aliases
    # (``chat``, ``claude``, ``responses``) — and rendering them as one list put
    # seven aliases above the operator's own six entries, which is how a
    # seventeen-row wall appears for a six-provider config.
    from ubt.core.providers import PROTOCOL_ALIASES, declared_provider_names

    declared = declared_provider_names()
    file_default = default_provider_name()
    # The shipped entries split into two kinds that the panel must not conflate:
    # the four *wire protocols* (``openai-chat``) are the real names, and the
    # seven *aliases* (``openai``, ``chat``, ``claude``) resolve to one of them.
    # Rendering both as "alias" hid the relationship — ``anthropic-messages``
    # showed its own name on the right, which told the operator nothing.
    from ubt.core.providers import _read_shipped_providers

    shipped_protocols = set(_read_shipped_providers())
    providers: list[dict[str, Any]] = []
    for name in list_providers():
        block = _safe_block(name)
        providers.append(
            {
                "name": name,
                "active": name == config.provider,
                # What ``[defaults] provider`` names, which is not the same as what
                # is running: ``UBT_PROVIDER`` outranks the file. Without this an
                # operator whose shell exported a provider saw six blocks and no
                # way to tell which one the *file* would pick without it.
                "is_default": name == file_default,
                "available": block.pop("available", False),
                "declared": name in declared,
                "is_alias": name in PROTOCOL_ALIASES,
                # What an alias stands for, so the row can say "-> openai-chat"
                # instead of repeating its own name or a bare "alias".
                "alias_of": PROTOCOL_ALIASES.get(name),
                # True for the four canonical wire protocols: they are the names
                # a ``[providers.*]`` block should use.
                "is_protocol": name in shipped_protocols,
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

"""Shared job-request plumbing for the CLI, REST API and MCP server.

Both mappings — which request fields override ``UBTConfig``, and how an
``EXPORT_COMPLETED`` artifact maps to its companion report files — live
here, derived from the config/model schema rather than per-entry-point
hand-maintained lists: hand-maintained copies drift, and a field going
missing on exactly one surface is invisible until a user hits it.

Nothing here resolves paths or unsecrets values: callers keep ownership of
sandboxing (API) and path normalization (CLI/MCP).
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ubt.core.config import (
    DeliveryMode,
    DocumentMode,
    DualMode,
    UBTConfig,
    env_var_names,
)
from ubt.core.exceptions import DocumentParseError, UBTError
from ubt.core.presets import PRESET_ENGINE_FIELDS, Preset, resolve_engine_params

#: Enum-typed request options and their accepted vocabulary, derived from the
#: ``Literal`` aliases that both the CLI (typer) and the REST model (pydantic)
#: already enforce. MCP takes plain strings, so it validates against this map at
#: the entry point instead of letting ``validate_assignment`` reject the value
#: inside the background task (job accepted, then fails).
#: User-facing mode knobs. Everything else is an engine detail: a preset (or an
#: operator's ``UBT_*`` environment variable) sets it, and the CLI/REST/MCP
#: surfaces no longer offer it.
_ENUM_REQUEST_FIELDS: dict[str, tuple[str, ...]] = {
    "dual_mode": get_args(DualMode),
    "delivery_mode": get_args(DeliveryMode),
    "preset": tuple(p.value for p in Preset),
    "document_mode": get_args(DocumentMode),
}


#: Preset names that were renamed, and what they became. Kept as a one-way map
#: (not an enum alias) so the failure is loud but *actionable*: a stored job's
#: ``submitted_request.preset`` from before the rename reaches
#: ``validate_request_enums`` on resume, and "'publication' is not valid" alone
#: leaves the operator guessing. The value is never silently accepted — the two
#: presets are not the same run identity, and a job id must not resume under a
#: name it was not created with.
_RENAMED_PRESETS: dict[str, str] = {
    "publication": "high_fidelity",
    "publication_math": "vector_math",
    "full_reflow": "high_fidelity",
}


def _preset_migration_hint(value: object) -> str:
    """The actionable half of an unknown-preset error, or ``""``."""
    if not isinstance(value, str):
        return ""
    replacement = _RENAMED_PRESETS.get(value.strip().lower())
    if replacement is None:
        return ""
    return (
        f" '{value}' was renamed to '{replacement}' and is no longer accepted; "
        "re-run with the new name (the ledger namespace changes with the preset, "
        "so an old job cannot resume under it)."
    )


def validate_request_enums(request: Mapping[str, Any]) -> None:
    """Reject an out-of-vocabulary enum option at the entry point, uniformly.

    The CLI validates via typer's ``Literal`` annotations and the REST API via
    its pydantic model, but MCP accepts bare strings — without this it returned
    a job_id and only failed the job once the background task tried to apply the
    value. Raising here gives every surface the same upfront error.
    """
    for key, allowed in _ENUM_REQUEST_FIELDS.items():
        value = request.get(key)
        if value is None:
            continue
        if value not in allowed:
            hint = _preset_migration_hint(value) if key == "preset" else ""
            raise UBTError(f"Invalid {key} {value!r}; expected one of: {', '.join(allowed)}.{hint}")


# Job ids become SQLite file names (`<job_id>.sqlite`), so this regex acts as a
# path-traversal boundary; length cap prevents ENAMETOOLONG errors on the filesystem.
# All entry points validate via :func:`job_id_is_valid`.
JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
JOB_ID_MAX_LEN = 128
# Language codes are interpolated into Typst source (`#set text(lang: "...")`),
# so anything but an ISO-ish tag is markup in disguise.
LANG_CODE_RE = re.compile(r"[A-Za-z]{2,3}(?:[_-][A-Za-z0-9]{2,4})?")
LANG_CODE_PATTERN = rf"^{LANG_CODE_RE.pattern}$"


def lang_pair_validation_error(source_lang: str | None, target_lang: str | None) -> str | None:
    """The single language-code gate shared by every entry point.

    Returns a human-readable error for the first violation, or None. One
    wording on purpose: the CLI, the API and MCP share this check so their
    messages cannot drift apart.
    """
    from ubt.core.language_profile import is_supported_lang, supported_lang_codes

    for key, value in (("source-lang", source_lang), ("target-lang", target_lang)):
        if value is not None and LANG_CODE_RE.fullmatch(str(value)) is None:
            return (
                f"Invalid {key}: {str(value)!r}. Use an ISO-ish language code "
                "such as 'en', 'zh' or 'zh-CN'."
            )
    if target_lang is not None and not is_supported_lang(str(target_lang)):
        return (
            f"Unsupported target-lang: {str(target_lang)!r}. Supported base "
            f"languages: {', '.join(supported_lang_codes())} (region tags such as "
            "'zh-CN' are accepted)."
        )
    return None


#: Domain-profile names index a packaged resource directory
#: (``ubt/resources/glossaries/<profile>/<src>-<tgt>.json``), so they must be a
#: single safe path component — a separator or ``..`` would escape the tree.
PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
PROFILE_NAME_PATTERN = rf"^{PROFILE_NAME_RE.pattern}$"


def job_id_is_valid(job_id: str) -> bool:
    """Whether ``job_id`` is a legal job id on every surface (pattern + length)."""
    return (
        bool(job_id) and len(job_id) <= JOB_ID_MAX_LEN and JOB_ID_RE.fullmatch(job_id) is not None
    )


def profile_name_is_valid(profile: str) -> bool:
    """Whether ``profile`` is a single safe resource-directory component."""
    return bool(profile) and PROFILE_NAME_RE.fullmatch(profile) is not None


def resolve_profile_name(profile: str | None, input_path: str | Path | None = None) -> str:
    """Resolve ``auto`` to a concrete domain profile, once, for every surface.

    ``auto`` used to be a CLI-only convenience: ``ubt translate --profile auto``
    probed the document and picked ``paper``/``fiction``/``semiconductor``/…,
    while REST/MCP/Web passing the same string fell through to
    ``seed_entries_for_profile("auto")`` — an unknown directory, so the run
    silently started with an empty seed glossary and no error. The probe
    (``analyze_archetype`` + ``infer_profile_from_archetype``) is cheap and
    never raises; a missing/unreadable file resolves to ``general``, matching
    the CLI's old fallback. A password-protected PDF skips the probe (it cannot
    be translated); the adapter guard reports it later with the actionable
    message.
    """
    name = (profile or "").strip() or "general"
    if name.lower() != "auto":
        return name
    if input_path is None:
        return "general"
    path = Path(input_path)
    if not path.exists():
        return "general"
    if path.suffix.lower() == ".pdf":
        # The archetype probe samples the PDF with pdf_oxide, which logs a burst
        # of "Decryption failed ... Not authenticated" on an encrypted file
        # before the pipeline's adapter guard can refuse it. Skip the probe here;
        # the adapter guard reports the encryption with its actionable message.
        from ubt.core.ports import reject_encrypted_pdf

        try:
            reject_encrypted_pdf(path)
        except DocumentParseError:
            return "general"
    try:
        from ubt.core.archetype import analyze_archetype, infer_profile_from_archetype

        return infer_profile_from_archetype(analyze_archetype(path), path=path)
    except Exception:  # the probe must never fail a job submission
        return "general"


def resolve_document_mode(
    document_mode: str | None,
    input_path: Path | str | None,
) -> DocumentMode:
    """Resolve 'auto' document mode to 'paper' or 'book' based on document archetype.

    When set explicitly to 'paper' or 'book', that choice is respected.
    When 'auto' (or omitted):
    - Academic paper category -> 'paper'
    - Literature / technical book / general -> 'book'
    """
    mode = (document_mode or "").strip().lower()
    if mode in ("paper", "book"):
        return mode  # type: ignore[return-value]
    if input_path is None:
        return "book"
    path = Path(input_path)
    if not path.exists():
        return "book"
    if path.suffix.lower() == ".pdf":
        from ubt.core.ports import reject_encrypted_pdf

        try:
            reject_encrypted_pdf(path)
        except DocumentParseError:
            return "book"
    try:
        from ubt.core.archetype import DocCategory, analyze_archetype

        arch = analyze_archetype(path)
        if arch.category == DocCategory.ACADEMIC_PAPER:
            return "paper"
        return "book"
    except Exception:
        return "book"


# Request keys whose name differs from the UBTConfig field they set.
_REQUEST_KEY_ALIASES: dict[str, str] = {
    "glossary": "glossary_path",
    "mode": "document_mode",
}

# Run-only keys (consumed by PipelineOrchestrator.run, not UBTConfig).
RUN_ONLY_KEYS: tuple[str, ...] = (
    "input_path",
    "output_path",
    "target_lang",
    "source_lang",
    "profile",
    "job_id",
    "start_chapter",
    "max_chapters",
)


def _validated_provider_name(name: Any) -> str:
    """A provider name that resolves to a block the operator actually declared.

    A job payload may name a provider; it may not invent one. The name is checked
    against :func:`ubt.core.providers.list_providers` — the same set the loader
    would accept — so a typo fails here with a readable message instead of
    surfacing later as a "provider not found" from deep inside the pipeline, and a
    crafted name cannot become a lookup key for something the file never defined.

    Shipped wire protocols and bare aliases are in that set too, which is correct:
    ``openai-chat`` is a usable provider with no user block behind it, and an
    operator running one against ``UBT_LLM_API_KEY`` must be able to select it.
    """
    from ubt.core.providers import list_providers

    if not isinstance(name, str) or not name.strip():
        raise UBTError("provider must be a non-empty provider name")
    cleaned = name.strip()
    available = list_providers()
    if cleaned not in available:
        raise UBTError(
            f"Unknown provider {cleaned!r}: it is not declared in ubt.toml and is not a "
            f"built-in wire protocol. Available: {', '.join(available)}"
        )
    return cleaned


def overrides_from_request(
    request: Mapping[str, Any], *, allow_provider_keys: bool = True
) -> dict[str, Any]:
    """Map a job request payload to ``UBTConfig`` override kwargs.

    Explicit fields win over the preset bundle, which wins over the engine
    default (same precedence as the CLI ``--preset`` layer). Only recognised
    config fields survive, so run-only keys are never applied to the config.

    ``allow_provider_keys=False`` (REST/queue/MCP payloads) additionally
    rejects credential and endpoint keys: those may only arrive via the
    operator's own CLI flags or environment, never through a job payload
    (defense in depth behind the API's ``extra="forbid"`` request model and MCP's fixed tool signature).

    **``provider`` is deliberately NOT in that list.** The two things are not the
    same kind of input:

    * ``base_url`` / ``api_mode`` / ``api_key`` let a payload choose *where* the
      request goes and *what secret* it carries. Accepting those would turn the API
      into a credential-writer and an SSRF primitive.
    * ``provider`` is a *name* that must already exist in the operator's
      ``ubt.toml``. The endpoint, the protocol and the credential all still come
      from that file; the payload only says which declared block to use. Refusing
      it meant the console could edit and activate provider blocks but no job could
      ever select a different one, so a user with three configured endpoints could
      only use whichever ``[defaults] provider`` named — a capability the UI
      presented and the API silently ignored.
    * ``repair_provider`` is the same kind of input as ``provider`` — a name
      resolved through the same validator — so it moved out of the leaked-keys
      list once the wizard grew a repair-tier selector. It was listed as a
      credential key historically, but it never carried one: the pipeline builds
      the repair tier from the *named block's* endpoint and credential.

    The name is validated against the declared set (:func:`_validated_provider_name`),
    so an unknown or malformed name is a clear error rather than a lookup that
    fails later inside the pipeline.
    """
    if not allow_provider_keys:
        leaked = sorted(
            key
            for key in (
                "base_url",
                "api_mode",
                "api_key",
                "ocr_api_key",
                "ocr_endpoint",
                "service_api_key",
            )
            if request.get(key) is not None
        )
        if leaked:
            raise UBTError(
                "Credential/endpoint keys are not accepted in job payloads: "
                + ", ".join(leaked)
                + " (set them via UBT_* environment variables or CLI flags)"
            )
    validate_request_enums(request)
    overrides: dict[str, Any] = {}
    preset_raw = request.get("preset")
    preset = Preset(preset_raw) if preset_raw else None
    explicit = {key: request.get(key) for key in PRESET_ENGINE_FIELDS}
    overrides.update(resolve_engine_params(preset, explicit))

    # A named provider must resolve to a declared block before it becomes an
    # override. Validated here rather than at assignment so the failure names the
    # problem ("Unknown provider 'typo'") instead of arriving as a lookup error
    # from inside the pipeline, after the job has already been accepted.
    if request.get("provider") is not None:
        overrides["provider"] = _validated_provider_name(request["provider"])
    # The repair tier is the same kind of reference; the named block supplies the
    # endpoint and credential, so accepting the name accepts no secret.
    if request.get("repair_provider") is not None:
        overrides["repair_provider"] = _validated_provider_name(request["repair_provider"])

    valid_fields = set(UBTConfig.model_fields)
    for key, value in request.items():
        if value is None:
            continue
        field = _REQUEST_KEY_ALIASES.get(key, key)
        # ``provider`` is already handled above, validated; letting the generic
        # loop copy it again would bypass that check.
        if field == "provider":
            continue
        # Already validated above; a second copy would bypass that check.
        if field == "repair_provider":
            continue
        if field in valid_fields:
            overrides[field] = value
    # A request that leaves dual_mode unset gets the same profile/engine-aware
    # default on every surface; applying it on only one surface would render the
    # same job (e.g. an academic paper) differently per entry point.
    if "dual_mode" not in overrides and not any(
        var in os.environ for var in env_var_names("dual_mode")
    ):
        # The adaptive default must be derived from the *resolved* profile.
        # Reading ``request["profile"]`` verbatim meant the literal string
        # "auto" reached ``adaptive_dual_mode``, which recognises only concrete
        # names — so the web console (which submits "auto" from the browser)
        # got no default at all, while the CLI (which resolves "auto" to
        # "paper" before building the request) got ``mono``. The same PDF then
        # rendered monolingual through one surface and bilingual through the
        # other, with a different filename. Resolving here makes the decision
        # depend on the document, not on which door the request came through.
        adaptive = adaptive_dual_mode(
            None, resolve_profile_name(request.get("profile", "auto"), request.get("input_path"))
        )
        if adaptive is not None:
            overrides["dual_mode"] = adaptive
    return overrides


def adaptive_dual_mode(
    explicit_dual_mode: str | None,
    profile: str | None,
) -> str | None:
    """Profile-aware ``dual_mode`` default, shared by CLI, API and MCP.

    An explicit ``dual_mode`` always wins. Otherwise academic papers,
    textbooks, and fiction/novels read better monolingual; everything else
    stays unset to follow config / ``UBT_DUAL_MODE``.
    """
    if explicit_dual_mode is not None:
        return explicit_dual_mode
    norm_profile = (profile or "").strip().lower()
    if norm_profile in ("paper", "fiction", "novel", "textbook"):
        return "mono"
    return None


def _coerce_field(key: str, value: Any) -> Any:
    """Wrap string credentials in ``SecretStr``; pass everything else through."""
    if key in ("api_key", "ocr_api_key") and isinstance(value, str):
        return SecretStr(value)
    return value


def apply_config_overrides(base: UBTConfig, overrides: Mapping[str, Any]) -> UBTConfig:
    """Return a copy of ``base`` with the recognised overrides applied.

    Assignment goes through pydantic (``validate_assignment=True``), so an
    invalid enum/int is rejected here rather than surfacing mid-run.
    """
    from ubt.core.config import _env_supplied_field_names
    from ubt.core.providers import load_layer_config, merge_provider_under

    job_config = base.model_copy(deep=True)
    valid_fields = set(UBTConfig.model_fields)

    provider_name = overrides.get("provider") or base.provider or os.getenv("UBT_PROVIDER")
    # The same ladder ``from_env`` uses: ``[defaults]`` + provider block, under
    # the request's explicit values, with environment-pinned fields above it.
    # ``repair_model`` follows the effective draft via ``_check_invariants``,
    # which assignment re-runs — one owner, so both paths agree.
    layer_cfg = load_layer_config(str(provider_name) if provider_name else None)
    effective = merge_provider_under(overrides, layer_cfg, env_supplied=_env_supplied_field_names())

    for key, value in effective.items():
        if value is not None and key in valid_fields:
            setattr(job_config, key, _coerce_field(key, value))
    return job_config


def run_kwargs_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the ``PipelineOrchestrator.run`` keys a request carries.

    ``profile`` is the request's name for the run's ``profile_name``, resolved
    through :func:`resolve_profile_name` so ``auto`` probes the document here —
    at the one point every surface (CLI, API, MCP, queue) passes through —
    instead of being a CLI-only convenience that silently became an empty
    glossary on the other three. ``start_chapter``/``max_chapters`` ride along
    too, ensuring chapter-window arguments pass correctly across every
    submission boundary. Keys the request does not carry stay absent so
    ``run()``'s own defaults apply.
    """
    kwargs: dict[str, Any] = {
        "target_lang": request.get("target_lang", "zh"),
        "source_lang": request.get("source_lang", "en"),
        "profile_name": resolve_profile_name(
            request.get("profile", "auto"), request.get("input_path")
        ),
        "job_id": request.get("job_id"),
    }
    start = request.get("start_chapter")
    if start is not None:
        kwargs["start_chapter"] = int(start)
    max_ch = request.get("max_chapters")
    if max_ch is not None:
        kwargs["max_chapters"] = int(max_ch)
    return kwargs


#: With no explicit ``-o`` a deliverable lands in the user's documents folder so
#: a forgotten flag never pollutes the input directory (or docs/). Every derived
#: artifact (secondary render, quality/visual reports, .typ sidecar, PE queue)
#: hangs off this path, so all three surfaces must derive it the same way.
#:
#: Output directory resolution order:
#:   1. ``UBT_OUTPUT_DIR``  — explicit operator/user choice, always wins.
#:   2. ``XDG_DOCUMENTS_DIR`` — set by a Linux desktop that relocated Documents.
#:   3. ``~/Documents``    — conventional document directory.
#:   4. ``~``               — home directory fallback.
#: A ``UBT`` subdirectory keeps translations organized and prevents file collisions.
_OUTPUT_DIR_ENV = "UBT_OUTPUT_DIR"
_OUTPUT_SUBDIR = "UBT"


def default_output_dir() -> Path:
    """Absolute, cross-platform home for deliverables written without ``-o``."""
    override = os.environ.get(_OUTPUT_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DOCUMENTS_DIR", "").strip()
    if xdg:
        return Path(xdg).expanduser() / _OUTPUT_SUBDIR
    return Path.home() / "Documents" / _OUTPUT_SUBDIR


def default_output_dir_for_scan() -> Path:
    """The directory the permission scanner should sweep, when nothing is passed.

    Separate from :func:`default_output_dir` so the scan cannot accidentally
    depend on a stale import-time constant, and so callers that only need "where
    do deliverables live" do not have to know about file naming.
    """
    return default_output_dir()


_UPLOAD_PREFIX_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-fA-F]{8}-")

#: Longest page-scope token allowed in a filename before falling back to a
#: digested form. Well under every common filesystem's 255-byte name limit even
#: after the stem and label are prepended.
_MAX_PAGE_LABEL_LEN = 48


def clean_source_stem(path: Path | str) -> str:
    """Strip server-side upload deduplication prefixes (e.g. 20261007-055316-84614c7e-)
    so deliverable filenames match the user's original document name."""
    stem = Path(path).stem
    return _UPLOAD_PREFIX_RE.sub("", stem)


def page_scope_label(pages: Iterable[int] | None) -> str:
    """A filename-safe token naming which pages a partial run covered.

    Empty when the run covers the whole document. A partial run must never share
    a filename with a full one: measured repeatedly during a gate audit, a
    ``--pages 1-6`` smoke test wrote its 2-page result over the 14-page delivery
    in the default directory, and the next baseline check read a two-page
    document as a 91% content loss. The run was correct; the name lied about
    what it was.

    Contiguous runs render as a range (``p1-6``); anything else lists its groups
    (``p1-2,7``) so the label stays recognisable. The label is bounded — a run
    over hundreds of scattered pages would otherwise produce a filename no
    filesystem accepts — and falls back to a digested form in that case, so two
    different page sets can never collide on the same name.
    """
    if not pages:
        return ""
    ordered = sorted(set(pages))
    if not ordered:
        return ""
    # Group consecutive pages into inclusive ranges.
    groups: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for page in ordered[1:]:
        if page == prev + 1:
            prev = page
            continue
        groups.append((start, prev))
        start = prev = page
    groups.append((start, prev))

    rendered = ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in groups)
    label = f"p{rendered}"
    if len(label) <= _MAX_PAGE_LABEL_LEN:
        return label
    # Too many groups to spell out. Keep it deterministic AND collision-free:
    # a truncated list would give two different page sets the same filename.
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:10]
    return f"p{len(ordered)}pp-{digest}"


def default_output_path(
    input_path: Path | str, *, monolingual: bool = False, pages: Iterable[int] | None = None
) -> Path:
    """Where a run with no ``-o`` writes its deliverable.

    A monolingual (rigid) primary is named ``<stem>_mono`` rather than
    ``<stem>_bilingual`` so the file name does not promise a bilingual document
    whose content is target-only.

    A run restricted with ``--pages`` carries that scope in the name
    (``<stem>_mono_p1-6.pdf``), so a smoke test can never overwrite the full
    delivery it was sampling from. See :func:`page_scope_label`.
    """
    source = Path(input_path)
    label = "mono" if monolingual else "bilingual"
    stem = clean_source_stem(source)
    scope = page_scope_label(pages)
    suffix = f"_{scope}" if scope else ""
    return default_output_dir() / f"{stem}_{label}{suffix}{source.suffix}"


def resolve_target_output(
    output_path: Path | str | None,
    input_path: Path | str,
    *,
    monolingual: bool = False,
    pages: Iterable[int] | None = None,
) -> Path:
    """Normalize user-supplied output path or derive default deliverable path.

    - None -> <documents>/UBT/<stem>_bilingual<suffix> (``_mono`` when monolingual),
      with the page scope in the name when the run is restricted (``_p1-6``)
    - Directory (existing or ending with / or \\) -> <dir>/<stem>_bilingual<suffix>
      (``_mono`` when monolingual), same page-scope rule
    - Path without extension -> <path><input_suffix>
    - Path with extension -> <path> as-is (an explicit name is always respected)

    An explicit ``-o`` is never decorated: the caller named the file, so the
    scope marker would only make the delivered path differ from what they asked
    for.
    """
    inp = Path(input_path)
    if output_path is None:
        return default_output_path(inp, monolingual=monolingual, pages=pages)
    out = Path(output_path)
    if out.is_dir() or str(output_path).endswith(("/", "\\")):
        label = "mono" if monolingual else "bilingual"
        stem = clean_source_stem(inp)
        scope = page_scope_label(pages)
        suffix = f"_{scope}" if scope else ""
        return out / f"{stem}_{label}{suffix}{inp.suffix}"
    if not out.suffix and inp.suffix:
        return out.with_suffix(inp.suffix)
    return out


@dataclass(frozen=True, slots=True)
class PriorOutput:
    """The output a prior run of a job id recorded, for overwrite authorization."""

    output_path: str
    input_path: str


def prior_output_of(prior: object) -> PriorOutput | None:
    """Extract the recorded output from a job record or a queue payload.

    The entry points hold the "prior run of this id" in two shapes — a manager
    ``JobRecord`` (which carries ``.request``) and a queue payload ``dict`` — so
    the extraction lives here rather than being re-spelled per surface. Returns
    ``None`` when the record carries no output path (nothing to authorize).
    """
    if prior is None:
        return None
    if isinstance(prior, dict):
        payload: object = prior
    else:
        payload = getattr(prior, "payload", None)
        if not isinstance(payload, dict):
            request = getattr(prior, "request", None)
            payload = request.model_dump() if request is not None else None
    if not isinstance(payload, dict):
        return None
    out = payload.get("output_path")
    inp = payload.get("input_path")
    if not out or not inp:
        return None
    return PriorOutput(output_path=str(out), input_path=str(inp))


def deliverable_candidates(
    output: Path | str | None,
    input_path: Path | str,
    *,
    pages: Iterable[int] | None = None,
) -> tuple[Path, ...]:
    """Every file a run could write for this output/input pair.

    An output given as a **directory** (or ``None`` for the default) resolves to
    a labelled file *inside* it, and the label depends on the run's
    mono/bilingual decision, which is not known at the entry point — so BOTH
    candidate names are returned. Every overwrite guard must check all of them;
    the CLI checked only the bilingual name for a directory and so let a
    monolingual re-run replace the previous delivery. A file-shaped output
    resolves to exactly one path.

    ``pages`` narrows the derived names to the run's page scope. It must be the
    same set the run will use, or the guard protects a name the run will not
    write (and misses the one it will).
    """
    if output is None:
        return (
            default_output_path(input_path, monolingual=False, pages=pages),
            default_output_path(input_path, monolingual=True, pages=pages),
        )
    out = Path(output)
    # Pass the *original* to ``resolve_target_output``: it decides "directory"
    # from ``str(...).endswith("/")``, and a trailing slash is lost the moment
    # the value is a ``Path`` (``Path("out/")`` renders as ``out``).
    if out.is_dir() or str(output).endswith(("/", "\\")):
        return (
            resolve_target_output(output, input_path, monolingual=False, pages=pages),
            resolve_target_output(output, input_path, monolingual=True, pages=pages),
        )
    return (resolve_target_output(output, input_path),)


def own_prior_deliverable(
    target: Path,
    prior: PriorOutput | None,
    *,
    input_path: Path | str | None = None,
) -> bool:
    """Whether ``target`` is the exact deliverable a prior run of the SAME id resolved to.

    Overwrite is a *re-run of one job*, not a licence to truncate any file the
    job could name. The three entry points used to answer this question three
    different ways — REST compared a directory's *parent*, MCP treated ``fresh``
    as a blanket licence, and the CLI missed the suffix-less path form. REST and
    MCP now share this helper. The CLI does **not** call it: ``translate``
    refuses any existing candidate without a same-job exemption, so a legitimate
    re-run of one job is allowed on the API/MCP surfaces and refused on the CLI.

    There is one question ("is this the file this very job already produced?")
    and it has one answer:

    - a prior run whose ``output_path`` was a **directory** licenses only the
      derived deliverable name(s) inside it (``<stem>_bilingual.<ext>`` /
      ``<stem>_mono.<ext>``) — never every file in the tree, and never a file
      another job or a human put there;
    - a prior run with an explicit file name licenses only that file.

    The comparison is on the fully resolved path, so a different spelling of the
    same file matches and a different file does not.

    ``input_path`` is the *current* run's source document. When given, it must
    resolve to the same file the prior record names. Without that check the
    question degrades to "did some job with this id ever write this path?", and
    a record whose source was a different document licenses overwriting a file
    that belongs to that other document — the record was only ever used to
    derive the candidate *names*, which is not the same as identifying the
    deliverable.
    """
    if prior is None or not prior.output_path or not prior.input_path:
        return False
    if input_path is not None:
        try:
            if Path(prior.input_path).resolve() != Path(input_path).resolve():
                return False
        except (OSError, ValueError):
            return False
    try:
        prior_out = Path(prior.output_path)
        # ``resolve_target_output`` takes the original string (a trailing slash
        # is the directory signal and a ``Path`` drops it).
        if prior_out.is_dir() or str(prior.output_path).endswith(("/", "\\")):
            candidates: tuple[Path, ...] = (
                resolve_target_output(prior.output_path, prior.input_path, monolingual=False),
                resolve_target_output(prior.output_path, prior.input_path, monolingual=True),
            )
        else:
            candidates = (resolve_target_output(prior.output_path, prior.input_path),)
        target_resolved = target.resolve()
        return any(candidate.resolve() == target_resolved for candidate in candidates)
    except (OSError, ValueError):
        return False


#: The derived reports that hang off a deliverable.
#:
#: ``quality_report.md`` used to be listed here. Nothing ever wrote it: the
#: only producer is ``save_quality_report``, and it writes JSON. It stayed in
#: the sweep list, so every export deleted a file that could not exist while
#: the console's report-kind union advertised it. Removed rather than
#: implemented — a second report format is a feature, not a typo fix.
SidecarKind = Literal[
    "quality_report.json",
    "metrics.json",
    "visual_report.json",
    "contract.json",
    "delivery_contract.json",
]


def sidecar_path(artifact: str | Path, kind: SidecarKind) -> Path:
    """The derived report file that belongs to one deliverable.

    Keyed on the *whole* file name, not the bare stem. Two documents whose
    stems match — ``book.epub`` and ``book.md``, both defaulting to
    ``book_bilingual.<ext>`` — are different deliverables, and sharing one
    ``book_bilingual_quality_report.json`` means the later run overwrites the
    earlier one's report with a document whose ``output_path`` points at a file
    nobody finds beside it. The format tag is what keeps them apart.

    Every writer and reader of these names goes through here: the report is
    written by the export stage, looked up by the API/CLI status paths, and
    swept by :func:`_drop_stale_run_reports`, and a second copy of the rule is
    how the two halves stop agreeing.
    """
    output = Path(artifact)
    tag = output.suffix.removeprefix(".") or "artifact"
    return output.with_name(f"{output.stem}_{tag}_{kind}")


def companion_path(artifact: str | Path, suffix: str) -> Path:
    """A *companion* file beside a deliverable (e.g. an XLIFF view).

    Keyed on the whole file name for the same reason as :func:`sidecar_path`:
    ``book.epub`` and ``book.md`` both default to ``book_mono.<ext>``, so the
    format tag keeps their companions from overwriting each other. Unlike a
    sidecar, the caller supplies the suffix, because a companion is a real
    document the reader opens, not a report.
    """
    output = Path(artifact)
    tag = output.suffix.removeprefix(".") or "artifact"
    return output.with_name(f"{output.stem}_{tag}{suffix}")


@dataclass(frozen=True, slots=True)
class DeliveredArtifact:
    """One delivered artifact and the files that hang off it.

    Two artifact identities coexist in the export stage: ``target_output`` (the
    path the primary render was *asked* for, which the PE queue keys on) and
    ``rendered_path`` (the file the adapter actually *returned*, which the visual
    gate may rewrite and which every sidecar and companion keys on). Keeping both
    in one value means the stage never has to remember which helper takes which,
    and the sibling/companion naming has a single owner instead of ad-hoc
    ``with_name`` calls spread through the stage.
    """

    rendered_path: Path
    target_output: Path

    def sidecar(self, kind: SidecarKind) -> Path:
        """The derived report belonging to the returned artifact."""
        return sidecar_path(self.rendered_path, kind)

    def companion(self, suffix: str) -> Path:
        """A companion document beside the returned artifact (e.g. ``.xliff``)."""
        return companion_path(self.rendered_path, suffix)

    def sibling(self, suffix: str) -> Path:
        """A second artifact beside the *requested* target (e.g. ``_mono.pdf``)."""
        out = self.target_output
        return out.with_name(f"{out.stem}{suffix}{out.suffix}")


def delivered_artifact(rendered_path: str | Path, target_output: str | Path) -> DeliveredArtifact:
    """Name the two artifact identities once, where the render returns."""
    return DeliveredArtifact(Path(rendered_path), Path(target_output))


#: Status of a companion artifact export attempt.
CompanionArtifactStatus = Literal["generated", "skipped", "failed"]

#: Companion artifact kinds.
CompanionKind = Literal[
    "xliff",
    "html",
    "epub",
    "attestation",
    "table_appendix",
    "spill_appendix",
    "reflow",
]

#: Recognized artifact kinds for delivery packages and companion audit.
ArtifactKind = CompanionKind


class CompanionArtifactResult(BaseModel):
    """Result of exporting a single companion deliverable with audit metadata."""

    model_config = ConfigDict(frozen=True)

    kind: CompanionKind | str
    status: CompanionArtifactStatus
    path: Path | None = None
    size_bytes: int | None = None
    duration_ms: float | None = None
    reason: str | None = None
    error: str | None = None

    @property
    def is_generated(self) -> bool:
        return self.status == "generated"

    @property
    def is_skipped(self) -> bool:
        return self.status == "skipped"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"


class CompanionExportSummary(BaseModel):
    """Unified audit summary of all companion artifact export outcomes."""

    model_config = ConfigDict(frozen=True)

    results: list[CompanionArtifactResult] = Field(default_factory=list)

    @property
    def generated(self) -> list[CompanionArtifactResult]:
        return [r for r in self.results if r.status == "generated"]

    @property
    def skipped(self) -> list[CompanionArtifactResult]:
        return [r for r in self.results if r.status == "skipped"]

    @property
    def failed(self) -> list[CompanionArtifactResult]:
        return [r for r in self.results if r.status == "failed"]

    def summary_line(self) -> str:
        return (
            f"{len(self.generated)} generated, "
            f"{len(self.skipped)} skipped, "
            f"{len(self.failed)} failed"
        )

    def get(self, kind: str) -> CompanionArtifactResult | None:
        for r in self.results:
            if r.kind == kind:
                return r
        return None


CompanionExportManifest = CompanionExportSummary


#: Every derived report that hangs off one deliverable, as
#: :func:`sidecar_path` kinds. One list, deliberately: the export stage drops
#: the previous run's copies for the artifact it is about to write, and the
#: CLI's ``--fresh`` sweep drops those of a companion it deletes. Two lists is
#: how one of the two starts leaving orphan reports behind -- a report that
#: describes a document nobody has, which the next reader picks up as the
#: current one's.
RUN_REPORT_KINDS: tuple[SidecarKind, ...] = (
    "quality_report.json",
    "metrics.json",
    "visual_report.json",
    "contract.json",
    "delivery_contract.json",
)


def artifact_and_report_paths(
    artifact: str | Path,
) -> tuple[Path, Path | None, Path | None]:
    """Resolve an artifact path and its existing companion report files.

    Returns ``(output, quality_report_or_None, visual_report_or_None)``.
    """
    output = Path(artifact)
    quality = sidecar_path(output, "quality_report.json")
    visual = sidecar_path(output, "visual_report.json")
    return output, (quality if quality.exists() else None), (visual if visual.exists() else None)


#: The three-state delivery verdict, shared by every facade.
#:
#: The artifact is delivered in all three states; the state only says whether
#: the reader will meet untranslated source. ``blocked`` — a quarantined
#: (``blocked_human``) or failed block ships its *source*: the delivery is
#: materially incomplete. ``warning`` — a ``needs_human`` block still ships
#: its translation and is only queued for review. ``passed`` — nothing
#: outstanding. Before this lived in one place, the CLI banner read the
#: summary while the web quality gate read ``failed == 0``, and the console
#: announced "PRODUCTION READY" over blocks the CLI called BLOCKED.
DeliveryVerdictState = Literal["passed", "warning", "blocked"]


def delivery_verdict_from_blocks(
    *,
    failed_blocks: int,
    blocked_human_blocks: int,
    needs_human_blocks: int,
    completed_blocks: int,
) -> DeliveryVerdictState:
    """Classify a delivery from block counters (the single authoritative rule).

    Counters come from either a quality-report summary (``failed_blocks`` /
    ``blocked_human_blocks`` / ``needs_human_blocks`` / ``completed_blocks``)
    or a ledger stats snapshot (``failed`` / ``blocked_human`` /
    ``needs_human`` / ``completed``) — callers map their own keys.
    """
    if failed_blocks or blocked_human_blocks:
        return "blocked"
    if not completed_blocks:
        return "blocked"
    if needs_human_blocks:
        return "warning"
    return "passed"


def delivery_verdict_from_summary(summary: Mapping[str, Any]) -> DeliveryVerdictState:
    """The verdict for a quality-report summary dict or ledger stats snapshot."""
    return delivery_verdict_from_blocks(
        failed_blocks=int(summary.get("failed_blocks", summary.get("failed", 0)) or 0),
        blocked_human_blocks=int(
            summary.get("blocked_human_blocks", summary.get("blocked_human", 0)) or 0
        ),
        needs_human_blocks=int(
            summary.get("needs_human_blocks", summary.get("needs_human", 0)) or 0
        ),
        completed_blocks=int(summary.get("completed_blocks", summary.get("completed", 0)) or 0),
    )

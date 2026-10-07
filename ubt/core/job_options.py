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

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, get_args

from ubt.core.config import (
    CoverMode,
    DualMode,
    ExecMode,
    FormulaEnrichment,
    FormulaMode,
    FormulaRender,
    MathBackend,
    OcrMode,
    PromptStrategyName,
    QeEngine,
    UBTConfig,
    env_var_names,
)
from ubt.core.exceptions import UBTError
from ubt.core.presets import PRESET_ENGINE_FIELDS, Preset, resolve_engine_params

#: Enum-typed request options and their accepted vocabulary, derived from the
#: ``Literal`` aliases that both the CLI (typer) and the REST model (pydantic)
#: already enforce. MCP takes plain strings, so it validates against this map at
#: the entry point instead of letting ``validate_assignment`` reject the value
#: inside the background task (job accepted, then fails).
_ENUM_REQUEST_FIELDS: dict[str, tuple[str, ...]] = {
    "dual_mode": get_args(DualMode),
    "exec_mode": get_args(ExecMode),
    "formula_mode": get_args(FormulaMode),
    "qe_engine": get_args(QeEngine),
    "prompt_strategy": get_args(PromptStrategyName),
    "cover_mode": get_args(CoverMode),
    "formula_enrichment": get_args(FormulaEnrichment),
    "formula_render": get_args(FormulaRender),
    "math_backend": get_args(MathBackend),
    "ocr_mode": get_args(OcrMode),
    "preset": tuple(p.value for p in Preset),
}


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
            raise UBTError(f"Invalid {key} {value!r}; expected one of: {', '.join(allowed)}")


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
    wording on purpose: the CLI, the API and MCP each used to word this check
    differently and the messages drifted apart.
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


# Request keys whose name differs from the UBTConfig field they set.
_REQUEST_KEY_ALIASES: dict[str, str] = {"glossary": "glossary_path"}

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
    """
    if not allow_provider_keys:
        leaked = sorted(
            key
            for key in (
                "provider",
                "repair_provider",
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
                "Provider selection/credential keys are not accepted in job payloads: "
                + ", ".join(leaked)
                + " (set them via UBT_* environment variables or CLI flags)"
            )
    validate_request_enums(request)
    overrides: dict[str, Any] = {}
    preset_raw = request.get("preset")
    preset = Preset(preset_raw) if preset_raw else None
    explicit = {key: request.get(key) for key in PRESET_ENGINE_FIELDS}
    overrides.update(resolve_engine_params(preset, explicit))

    valid_fields = set(UBTConfig.model_fields)
    for key, value in request.items():
        if value is None:
            continue
        field = _REQUEST_KEY_ALIASES.get(key, key)
        if field in valid_fields:
            overrides[field] = value
    # A request that leaves dual_mode unset gets the same profile/engine-aware
    # default on every surface. Previously only the CLI applied it, so the same
    # job (e.g. an academic paper) rendered bilingual inline through the API/MCP
    # and monolingual through the CLI.
    if "dual_mode" not in overrides and not any(
        var in os.environ for var in env_var_names("dual_mode")
    ):
        # The adaptive default is a *default*: writing it into the overrides
        # layer would outrank UBT_DUAL_MODE, which the CLI help documents as
        # "unset follows config / UBT_DUAL_MODE".
        adaptive = adaptive_dual_mode(None, request.get("profile"))
        if adaptive is not None:
            overrides["dual_mode"] = adaptive
    return overrides


def adaptive_dual_mode(
    explicit_dual_mode: str | None,
    profile: str | None,
) -> str | None:
    """Profile-aware ``dual_mode`` default, shared by CLI, API and MCP.

    An explicit ``dual_mode`` always wins. Otherwise academic papers and
    fiction/novels read better monolingual; everything else stays unset to
    follow config / ``UBT_DUAL_MODE``.
    """
    if explicit_dual_mode is not None:
        return explicit_dual_mode
    norm_profile = (profile or "").strip().lower()
    if norm_profile in ("paper", "fiction", "novel"):
        return "monolingual"
    return None


def _coerce_field(key: str, value: Any) -> Any:
    """Wrap string credentials in ``SecretStr``; pass everything else through."""
    if key in ("api_key", "ocr_api_key") and isinstance(value, str):
        from pydantic import SecretStr

        return SecretStr(value)
    return value


def apply_config_overrides(base: UBTConfig, overrides: Mapping[str, Any]) -> UBTConfig:
    """Return a copy of ``base`` with the recognised overrides applied.

    Assignment goes through pydantic (``validate_assignment=True``), so an
    invalid enum/int is rejected here rather than surfacing mid-run.
    """
    from ubt.core.config import _env_supplied_field_names
    from ubt.core.providers import load_layer, merge_provider_under

    job_config = base.model_copy(deep=True)
    valid_fields = set(UBTConfig.model_fields)

    provider_name = overrides.get("provider") or base.provider or os.getenv("UBT_PROVIDER")
    # The same ladder ``from_env`` uses: ``[defaults]`` + provider block, under
    # the request's explicit values, with environment-pinned fields above it.
    # ``repair_model`` follows the effective draft via ``_check_invariants``,
    # which assignment re-runs — one owner, so both paths agree.
    fields = load_layer(str(provider_name) if provider_name else None)
    effective = merge_provider_under(overrides, fields, env_supplied=_env_supplied_field_names())

    for key, value in effective.items():
        if value is not None and key in valid_fields:
            setattr(job_config, key, _coerce_field(key, value))
    return job_config


def run_kwargs_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the ``PipelineOrchestrator.run`` keys a request carries.

    ``profile`` is the request's name for the run's ``profile_name``.
    ``start_chapter``/``max_chapters`` ride along here too, ensuring chapter-window
    arguments pass correctly across CLI, API, MCP, and queue submission boundaries.
    Keys the request does not carry stay absent so ``run()``'s own defaults apply.
    """
    kwargs: dict[str, Any] = {
        "target_lang": request.get("target_lang", "zh"),
        "source_lang": request.get("source_lang", "en"),
        "profile_name": request.get("profile", "general"),
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


def default_output_path(input_path: Path | str, *, monolingual: bool = False) -> Path:
    """Where a run with no ``-o`` writes its deliverable.

    A monolingual (rigid) primary is named ``<stem>_mono`` rather than
    ``<stem>_bilingual`` so the file name does not promise a bilingual document
    whose content is target-only.
    """
    source = Path(input_path)
    label = "mono" if monolingual else "bilingual"
    return default_output_dir() / f"{source.stem}_{label}{source.suffix}"


def resolve_target_output(
    output_path: Path | str | None,
    input_path: Path | str,
    *,
    monolingual: bool = False,
) -> Path:
    """Normalize user-supplied output path or derive default deliverable path.

    - None -> <documents>/UBT/<stem>_bilingual<suffix> (``_mono`` when monolingual)
    - Directory (existing or ending with / or \\) -> <dir>/<stem>_bilingual<suffix>
      (``_mono`` when monolingual)
    - Path without extension -> <path><input_suffix>
    - Path with extension -> <path> as-is (an explicit name is always respected)
    """
    inp = Path(input_path)
    if output_path is None:
        return default_output_path(inp, monolingual=monolingual)
    out = Path(output_path)
    if out.is_dir() or str(output_path).endswith(("/", "\\")):
        label = "mono" if monolingual else "bilingual"
        return out / f"{inp.stem}_{label}{inp.suffix}"
    if not out.suffix and inp.suffix:
        return out.with_suffix(inp.suffix)
    return out


#: The derived reports that hang off a deliverable.
SidecarKind = Literal[
    "quality_report.json",
    "quality_report.md",
    "metrics.json",
    "visual_report.json",
    "contract.json",
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

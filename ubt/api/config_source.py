"""One owner for "what configuration is in force", re-read when the file changes.

**The problem this solves.** Configuration had five different lifetimes at once:

===========================================  ==================  =========================
consumer                                     when it read        symptom
===========================================  ==================  =========================
``create_app()`` → ``ApiScope.config``       once, at boot       the panel showed the
                                                                 pre-save value forever
``apply_config_overrides()``                 per job             a job saw the new value
                                                                 while the panel did not
``_env_supplied_field_names()``              per merge           ``UBT_*`` silently beat
                                                                 the file
``load_provider_block()``                    per call            the file was live here
``bootstrap_runtime_environment()``          once, at boot       global side effects
===========================================  ==================  =========================

A save through the console therefore produced a *successful* write that the
screen then contradicted, and in the same response: ``active.draft_model`` held
the old value while ``providers[name].draft_model`` held the new one.

**The fix, and its limit.** :class:`ConfigSource` re-reads the file when its
signature (path, mtime, size) changes, so every reader sees the same document and
a save is visible without a restart. It deliberately does **not** reload
everything: :data:`DEPLOYMENT_FIELDS` are bound at boot and stay bound, because
rebinding them at runtime would move the ledger out from under open connections
and re-open the auth gate on a running server. Those fields are reported as
frozen rather than silently ignored, so the console can say "this needs a
restart" instead of showing a value that will never take effect.
"""

from __future__ import annotations

import contextlib
import logging
import stat
import threading
import tomllib
from pathlib import Path
from typing import Any

import tomlkit

from ubt.core.atomic import atomic_write_text
from ubt.core.config import UBTConfig
from ubt.core.providers import find_config_file

logger = logging.getLogger(__name__)

#: Fields fixed at process start. Changing them in the file has no effect on the
#: running server, so the console must report them as "restart required" rather
#: than presenting the new value as though it were in force.
#:
#: Each one is here for a concrete reason, not by default:
#:
#: * ``db_dir`` / ``job_queue_path`` — the ledger, the job queue and the upload
#:   staging area are open on the old path; switching mid-process would orphan
#:   every running job's checkpoints.
#: * ``job_mode`` — the queue vs embedded decision builds or skips the worker
#:   pool at boot.
#: * ``service_api_key`` / ``strict_auth`` / ``env`` — the inbound auth gate was
#:   evaluated at boot. Reloading it would either open a gated server or lock out
#:   a session mid-request.
#: * ``trusted_hosts`` — the Host-header guard is a middleware bound to the app.
#: * ``allowed_dirs`` — used to resolve paths that are already open.
#:
#: ``host`` / ``port`` are not ``UBTConfig`` fields (they are CLI options), so they
#: are absent here on purpose: :meth:`ConfigSource.frozen_fields_changed` skips any
#: name the model does not declare, and listing them would imply a comparison that
#: never runs.
DEPLOYMENT_FIELDS: frozenset[str] = frozenset(
    {
        "db_dir",
        "job_queue_path",
        "job_mode",
        "service_api_key",
        "strict_auth",
        "env",
        "trusted_hosts",
        "allowed_dirs",
    }
)

#: The shared glossary is a first-class console asset. Its file lives under the
#: deployment-bound ``db_dir`` and is enabled by a ``[defaults]`` key, so the
#: console can create it without asking a product user to edit TOML by hand.
SHARED_GLOSSARY_FILENAME = "glossary.csv"


def shared_glossary_path(config: UBTConfig) -> Path:
    """The console-managed glossary file for this deployment."""
    return config.db_dir / SHARED_GLOSSARY_FILENAME


def ensure_shared_glossary_file(config: UBTConfig) -> Path:
    """Create the shared glossary and point ``[defaults]`` at it when unset.

    An explicit ``glossary_path`` (file or ``UBT_GLOSSARY_PATH``) is never
    replaced: the console enables the asset only when no glossary exists yet.
    The managed file is a two-column headerless CSV, which the engine's own
    loader already reads and every glossary editor can open.
    """
    configured = config.glossary_path
    if configured is not None:
        return configured

    path = shared_glossary_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        atomic_write_text(path, "")
    with contextlib.suppress(OSError):
        path.chmod(0o600)

    config_path = find_config_file() or (Path.cwd() / "ubt.toml")
    if config_path.is_file():
        try:
            document = tomlkit.parse(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(
                f"configuration file at {config_path} is not valid TOML: {exc}"
            ) from exc
    else:
        document = tomlkit.document()
    defaults = document.get("defaults")
    if defaults is None:
        defaults = tomlkit.table()
        document["defaults"] = defaults
    defaults["glossary_path"] = tomlkit.item(str(path.resolve()))
    atomic_write_text(config_path, tomlkit.dumps(document))
    try:
        if stat.S_IMODE(config_path.stat().st_mode) & (stat.S_IRWXG | stat.S_IRWXO):
            config_path.chmod(0o600)
    except OSError:
        pass
    return path


def _file_signature() -> tuple[str | None, int, int]:
    """Identity of the config file for cache invalidation: path, mtime, size.

    The same shape ``ubt.core.router.pricing`` uses for its price overlay. mtime
    alone is not enough — a file replaced by one of the same age (a checkout, a
    ``cp -p``) would keep serving the stale parse — so size rides along.
    """
    path = find_config_file()
    if path is None:
        return (None, 0, 0)
    try:
        stat = path.stat()
    except OSError:
        return (str(path), 0, 0)
    return (str(path), stat.st_mtime_ns, stat.st_size)


class ConfigSource:
    """The configuration in force, rebuilt whenever the file behind it changes.

    Thread-safe: a threaded server asks for the current config from many request
    handlers while a console save may be rebuilding it. The lock covers the
    check-and-rebuild, not the returned object — a ``UBTConfig`` is immutable
    enough for a job to hold one while a newer revision exists, and a job must
    keep the revision it started with.
    """

    def __init__(self, initial: UBTConfig, *, overrides: dict[str, Any] | None = None) -> None:
        self._lock = threading.RLock()
        #: Explicit construction overrides (``create_app(config=...)``, tests). A
        #: source built around a caller's config must not silently replace it with
        #: whatever the file says — the caller's values are the authority.
        self._overrides = dict(overrides or {})
        self._config = initial
        self._signature = _file_signature()
        #: Set when the file changed but rebuilding it failed. Reported rather
        #: than raised: one bad save must not take the whole console down, and the
        #: last good configuration keeps running.
        self._last_error: str | None = None

    @property
    def config(self) -> UBTConfig:
        """The current configuration, re-read if the file changed since last call."""
        with self._lock:
            signature = _file_signature()
            if signature == self._signature:
                return self._config
            self._signature = signature
            try:
                rebuilt = UBTConfig.from_env(**self._overrides)
            except Exception as exc:
                # Keep the last good config: a malformed file is a save the
                # operator can fix, not a reason to stop serving.
                self._last_error = str(exc)
                logger.warning("config file changed but could not be loaded: %s", exc)
                return self._config
            # Re-apply the boot values for deployment-bound fields. A plain
            # ``from_env()`` would happily move ``db_dir`` out from under the open
            # ledger, or re-open the auth gate, the moment an operator edited an
            # unrelated key in the same file. ``frozen_fields_changed`` then
            # reports the difference, which is how the console can say "restart
            # required" instead of showing a value that is not in force.
            for field in DEPLOYMENT_FIELDS:
                if field in UBTConfig.model_fields:
                    # A field whose own validator rejects the boot value cannot
                    # happen (the boot config passed the same validator), but a
                    # future assignment-guarded field must not break reloading.
                    with contextlib.suppress(ValueError, TypeError):
                        setattr(rebuilt, field, getattr(self._config, field))
            self._last_error = None
            self._config = rebuilt
            logger.info("configuration reloaded from the changed config file")
            return self._config

    def invalidate(self) -> None:
        """Force the next :attr:`config` read to rebuild.

        Called after a console save so the panel reflects it immediately rather
        than after the mtime granularity tick.
        """
        with self._lock:
            self._signature = (None, 0, 0)

    def set_overrides(self, updates: dict[str, Any]) -> UBTConfig:
        """Merge runtime overrides and return the configuration now in force.

        The console's page-egress toggle lands here rather than in ``ubt.toml``:
        the file schema is ``[defaults]``/``[providers]`` only and has no key
        for this field, so an honest toggle is an explicit runtime layer that
        dies with the process — the restart reverts to the environment, which
        is exactly the permanence the toggle's copy discloses. Merged into
        ``_overrides`` (not swapped) so later file-triggered rebuilds keep the
        operator's runtime choice.
        """
        with self._lock:
            self._overrides.update(updates)
            try:
                rebuilt = UBTConfig.from_env(**self._overrides)
            except Exception as exc:
                self._last_error = str(exc)
                raise
            self._config = rebuilt
            self._signature = _file_signature()
            self._last_error = None
            return self._config

    @property
    def last_error(self) -> str | None:
        """Why the most recent reload failed, or ``None`` when it succeeded."""
        return self._last_error

    def frozen_fields_changed(self) -> list[str]:
        """Deployment fields whose file value differs from the running one.

        A ``db_dir`` the operator just changed is not in force and will not be
        until a restart. Reporting that is the difference between a control that
        works and one that lies, and it is the same class of bug as the
        "latency matrix" heading that named a number no endpoint produced.
        """
        with self._lock:
            live = self._config
        try:
            on_disk = UBTConfig.from_env(**self._overrides)
        except Exception:
            return []
        changed: list[str] = []
        for field in sorted(DEPLOYMENT_FIELDS):
            if field not in UBTConfig.model_fields:
                continue
            if getattr(live, field, None) != getattr(on_disk, field, None):
                changed.append(field)
        return changed

import contextlib
import copy
import fcntl
import logging
import os
import pathlib
import tempfile
from collections.abc import Generator
from typing import Any, cast

import ruamel.yaml

from pivot import config_files, exceptions, project
from pivot import state_dir as state_dir_mod
from pivot.config import models

logger = logging.getLogger(__name__)

# Module-level cache for merged config to avoid repeated disk I/O
_merged_config_cache: models.PivotConfig | None = None

DEFAULT_CHECKOUT_MODE_ORDER: list[models.CheckoutMode] = list(models.CheckoutMode)


def get_local_config_path() -> pathlib.Path:
    """Get project-level config path (.pivot/config.yaml)."""
    return project.get_project_root() / ".pivot" / "config.yaml"


def get_config_path(scope: models.ConfigScope) -> pathlib.Path:
    """Get config path for the specified scope."""
    match scope:
        case models.ConfigScope.GLOBAL:
            return config_files.get_global_config_path()
        case models.ConfigScope.LOCAL:
            return get_local_config_path()


def _config_lock_path(path: pathlib.Path) -> pathlib.Path:
    """Lock file guarding a config file: ``config.yaml`` -> ``config.yaml.lock``."""
    return path.with_name(f"{path.name}.lock")


def _load_config_raw(path: pathlib.Path) -> dict[str, Any]:
    """Load YAML config as plain dict."""
    return config_files.load_yaml(path, as_dict=True)


def _load_config_preserving_structure(path: pathlib.Path) -> Any:
    """Load YAML config preserving ruamel structure for comments."""
    return config_files.load_yaml(path, as_dict=False)


def parse_config_yaml(raw: bytes, source: str) -> dict[str, Any]:
    """Parse config file contents exactly as ``load_config_file`` parses a file."""
    return config_files.parse_yaml(raw, source, as_dict=True)


def _save_config(path: pathlib.Path, data: dict[str, Any]) -> None:
    """Save YAML config atomically using temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            yaml = ruamel.yaml.YAML(typ="rt")
            yaml.default_flow_style = False
            with os.fdopen(fd, "w") as f:
                yaml.dump(data, f)
            os.rename(tmp_path, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
    except PermissionError:
        raise exceptions.ConfigError(f"Permission denied writing {path}") from None
    except OSError as e:
        raise exceptions.ConfigError(f"Error writing {path}: {e}") from e


@contextlib.contextmanager
def edit_config(
    scope: models.ConfigScope = models.ConfigScope.LOCAL,
) -> Generator[dict[str, Any]]:
    """Context manager for atomic config edits with file locking.

    Usage:
        with edit_config(ConfigScope.LOCAL) as config:
            config["cache"]["dir"] = "/new/path"
        # File is automatically saved on context exit
    """
    path = get_config_path(scope)
    lock_path = _config_lock_path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            raw_data = _load_config_preserving_structure(path)
            data: dict[str, Any] = raw_data if raw_data is not None else {}

            yield data

            _save_config(path, data)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_config_file(path: pathlib.Path) -> dict[str, Any]:
    """Load YAML config, returns empty dict if missing."""
    return _load_config_raw(path)


def set_nested(data: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set nested value, creating intermediate dicts as needed.

    Non-dict intermediate values are overwritten with empty dicts.
    """
    if not keys:
        raise ValueError("keys cannot be empty")

    current: dict[str, Any] = data
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def unset_nested(data: dict[str, Any], keys: list[str]) -> bool:
    """Remove nested key, cleaning up empty parent dicts. Returns True if removed."""
    if not keys:
        return False

    if len(keys) == 1:
        if keys[0] in data:
            del data[keys[0]]
            return True
        return False

    parent_keys = keys[:-1]
    final_key = keys[-1]

    parent = config_files.get_nested(data, parent_keys)
    if not isinstance(parent, dict) or final_key not in parent:
        return False

    del parent[final_key]

    if not parent:
        unset_nested(data, parent_keys)

    return True


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge override into base, recursively for nested dicts."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            nested_override = cast("dict[str, Any]", val)
            result[key] = deep_merge(result[key], nested_override)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _default_cache_dir() -> str:
    """Return the default cache display path without resolving or warning about project state."""
    root = project.get_project_root()
    cache_dir = state_dir_mod.state_dir_or_default(root, warn=False) / "cache"
    try:
        return cache_dir.relative_to(root).as_posix()
    except ValueError:
        return str(cache_dir)


def get_merged_config() -> models.PivotConfig:
    """Load and merge configs: defaults < global < local.

    A global ``core.state_dir`` is left out, so the merged config is the one in effect. Results
    are cached to avoid repeated disk I/O within a single command. Call clear_config_cache() to
    reset (e.g., in tests). The cache is process-wide, so it holds nothing derived from the state
    directory: an unset ``cache.dir`` keeps the model default and stays out of
    ``cache.model_fields_set``, and the cache in effect is then ``<state dir>/cache``, derived
    from the current state-directory record where it is used (``locate_cache_dir``).
    """
    global _merged_config_cache
    if _merged_config_cache is not None:
        return _merged_config_cache

    defaults = models.PivotConfig.get_default().model_dump()
    del defaults["cache"]["dir"]

    global_data = load_config_file(config_files.get_global_config_path())
    unset_nested(global_data, ["core", "state_dir"])
    merged = deep_merge(defaults, global_data)

    local_data = load_config_file(get_local_config_path())
    merged = deep_merge(merged, local_data)

    _merged_config_cache = models.PivotConfig.model_validate(merged)
    return _merged_config_cache


def _configured_cache_dir() -> str | None:
    """``cache.dir`` as a config file sets it, or None when unset (``get_merged_config``)."""
    cache = get_merged_config().cache
    return cache.dir if "dir" in cache.model_fields_set else None


def clear_config_cache() -> None:
    """Clear cached configuration and its project state directory."""
    global _merged_config_cache
    _merged_config_cache = None
    state_dir_mod.clear_state_dir_cache()


def get_config_value(key: str) -> tuple[Any, models.ConfigSource]:
    """Get value for dotted key, returns (value, source).

    A global ``core.state_dir`` is reported with source ``global, ignored``, and the ignored-value
    warning is issued.
    """
    keys = config_files.parse_dotted_key(key)
    if key == "core.state_dir":
        state_dir_mod.report_ignored_global_state_dir()

    local_data = load_config_file(get_local_config_path())
    local_value = config_files.get_nested(local_data, keys)
    if local_value is not config_files.NOT_FOUND:
        return local_value, models.ConfigSource.LOCAL

    global_data = load_config_file(config_files.get_global_config_path())
    global_value = config_files.get_nested(global_data, keys)
    if global_value is not config_files.NOT_FOUND:
        if key == "core.state_dir":
            return global_value, models.ConfigSource.GLOBAL_IGNORED
        return global_value, models.ConfigSource.GLOBAL

    if key == "cache.dir":
        return _default_cache_dir(), models.ConfigSource.DEFAULT
    if key in models.CONFIG_KEY_DESCRIPTIONS:
        return models.get_config_default(key), models.ConfigSource.DEFAULT

    return None, models.ConfigSource.UNKNOWN


def _changes_local_state_dir(key: str, scope: models.ConfigScope) -> bool:
    return key == "core.state_dir" and scope is models.ConfigScope.LOCAL


def set_config_value(
    key: str, value: Any, scope: models.ConfigScope = models.ConfigScope.LOCAL
) -> list[str]:
    """Set a configuration value and return any state-transition notices.

    A global ``core.state_dir`` is refused: it would have no effect.
    """
    if key == "core.state_dir" and scope is models.ConfigScope.GLOBAL:
        raise exceptions.ConfigValidationError(state_dir_mod.global_state_dir_refusal())
    changes_state_dir = _changes_local_state_dir(key, scope)
    old_state_dir: pathlib.Path | None = None
    if changes_state_dir:
        state_dir_mod.validate_state_dir_target(cast("str", value))
        old_state_dir = state_dir_mod.current_state_dir()
    keys = config_files.parse_dotted_key(key)
    with edit_config(scope) as data:
        set_nested(data, keys, value)
    clear_config_cache()
    if not changes_state_dir:
        return []
    return state_dir_mod.transition_local_state_dir(old_state_dir)


def unset_config_value(
    key: str, scope: models.ConfigScope = models.ConfigScope.LOCAL
) -> tuple[bool, list[str]]:
    """Remove config value from specified scope.

    Returns whether it was removed, and any state-transition notices.
    """
    keys = config_files.parse_dotted_key(key)
    path = get_config_path(scope)

    if not path.exists():
        return False, []

    changes_state_dir = _changes_local_state_dir(key, scope)
    old_state_dir: pathlib.Path | None = None
    if changes_state_dir:
        old_state_dir = state_dir_mod.current_state_dir()
    lock_path = _config_lock_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            raw_data = _load_config_preserving_structure(path)
            if raw_data is None:
                return False, []

            data: dict[str, Any] = raw_data
            removed = unset_nested(data, keys)
            if removed:
                _save_config(path, data)
                clear_config_cache()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    if not removed or not changes_state_dir:
        return removed, []
    return removed, state_dir_mod.transition_local_state_dir(old_state_dir)


def get_checkout_mode_order() -> list[models.CheckoutMode]:
    """Get checkout mode fallback order from merged config."""
    merged = get_merged_config()
    checkout_modes = list(merged.cache.checkout_mode)
    return checkout_modes if checkout_modes else DEFAULT_CHECKOUT_MODE_ORDER.copy()


def get_run_history_retention() -> int:
    """Get run history retention limit from merged config."""
    merged = get_merged_config()
    return merged.core.run_history_retention


def _effective_cache_dir() -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Resolve the effective cache without creating state or cache directories.

    ``PIVOT_CACHE_DIR``, then a configured ``cache.dir``, each as given. Unset, the cache is
    ``<state dir>/cache`` of the state directory this record resolved, derived here at each use
    and never stored, so a pipeline loaded after an edit of ``core.state_dir`` keeps its cache
    objects beside its locks.
    """
    root = project.get_project_root()
    configured = os.environ.get("PIVOT_CACHE_DIR", "").strip() or _configured_cache_dir()
    state_dir = state_dir_mod.state_dir_or_default(root)
    fallback_cache_dir = state_dir / "cache"
    if configured is None:
        cache_dir = fallback_cache_dir
    else:
        cache_dir = pathlib.Path(configured)
        if not cache_dir.is_absolute():
            cache_dir = root / cache_dir
    cache_dir = pathlib.Path(os.path.realpath(cache_dir))
    stages_dir = state_dir / "stages"
    if (
        cache_dir == stages_dir
        or cache_dir.is_relative_to(stages_dir)
        or stages_dir.is_relative_to(cache_dir)
    ):
        state_dir_mod.report_cache_stage_overlap(cache_dir, stages_dir, fallback_cache_dir)
        cache_dir = fallback_cache_dir
    return root, state_dir, cache_dir


def locate_cache_dir() -> pathlib.Path:
    """Return the effective cache location without creating project state."""
    _, _, cache_dir = _effective_cache_dir()
    return cache_dir


def get_cache_dir() -> pathlib.Path:
    """Return the effective writer-facing cache location, creating state only when required."""
    root, state_dir, cache_dir = _effective_cache_dir()
    if state_dir != root / ".pivot" and cache_dir.is_relative_to(state_dir):
        state_dir_mod.ensure_state_dir(root)
        if cache_dir != state_dir / "cache":
            state_dir_mod.make_nested_cache_dir(cache_dir)
    return cache_dir


def register_cache_use() -> None:
    """Register this project in a naming unit before a command links cache objects.

    `checkout`, `pull` and a pipeline run restore workspace links from records already on disk,
    outside the naming units that publish objects. They call this first, so a collection reads
    those records. Read-only commands never call it. It creates neither the cache nor the state
    directory: a cache that does not exist holds nothing to collect, and a state directory that
    cannot be written gives an entry without a project id, which a collection refuses.
    """
    from pivot.storage import gc

    try:
        root = project.get_project_root()
    except exceptions.ProjectNotInitializedError:
        return
    cache_dir = locate_cache_dir()
    if not cache_dir.is_dir():
        return
    with gc.naming_unit(cache_dir, project=(root, locate_state_dir())):
        pass  # the unit registers the project under the shared lock


def locate_state_dir() -> pathlib.Path:
    """Return the current project's configured state directory without creating it.

    The merged config is loaded first, so an invalid config fails here with its own message.
    A reader that opens the state database still creates the directory there (``StateDB``).
    """
    get_merged_config()
    return state_dir_mod.locate_state_dir(project.get_project_root())


def get_state_dir() -> pathlib.Path:
    """``locate_state_dir``, created when absent, for a command that writes state."""
    get_merged_config()
    return state_dir_mod.ensure_state_dir(project.get_project_root())


def project_state_dir(root: pathlib.Path) -> pathlib.Path:
    """Return ``root``'s state directory through the shared state-directory resolver.

    This compatibility helper is for a collector reading another project's records. It creates
    nothing, and resolves the project's own ``core.state_dir`` rather than this process's
    configuration.
    """
    return state_dir_mod.locate_state_dir(root)


def get_max_workers() -> int:
    """Get max workers from merged config."""
    merged = get_merged_config()
    return merged.core.max_workers


def get_resource_capacities() -> dict[str, int]:
    """Get configured scheduler resource capacities."""
    return dict(get_merged_config().resources)


def get_remote_jobs() -> int:
    """Get remote transfer parallel jobs from merged config."""
    merged = get_merged_config()
    return merged.remote.jobs


def get_remote_retries() -> int:
    """Get remote transfer retry count from merged config."""
    merged = get_merged_config()
    return merged.remote.retries


def get_remote_connect_timeout() -> int:
    """Get remote transfer connection timeout from merged config."""
    merged = get_merged_config()
    return merged.remote.connect_timeout


def get_watch_debounce() -> int:
    """Get watch mode debounce delay in milliseconds from merged config."""
    merged = get_merged_config()
    return merged.watch.debounce


def get_display_precision() -> int:
    """Get display precision for floating point numbers from merged config."""
    merged = get_merged_config()
    return merged.display.precision


def get_diff_max_rows() -> int:
    """Get max rows for data diff operations from merged config."""
    merged = get_merged_config()
    return merged.diff.max_rows

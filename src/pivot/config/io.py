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

from pivot import exceptions, project
from pivot.config import models

logger = logging.getLogger(__name__)

_NOT_FOUND: object = object()

# Module-level cache for merged config to avoid repeated disk I/O
_merged_config_cache: models.PivotConfig | None = None

DEFAULT_CHECKOUT_MODE_ORDER: list[models.CheckoutMode] = list(models.CheckoutMode)


def get_global_config_path() -> pathlib.Path:
    """Get user-level config path (~/.config/pivot/config.yaml)."""
    return pathlib.Path.home() / ".config" / "pivot" / "config.yaml"


def get_local_config_path() -> pathlib.Path:
    """Get project-level config path (.pivot/config.yaml)."""
    return project.get_project_root() / ".pivot" / "config.yaml"


def get_config_path(scope: models.ConfigScope) -> pathlib.Path:
    """Get config path for the specified scope."""
    match scope:
        case models.ConfigScope.GLOBAL:
            return get_global_config_path()
        case models.ConfigScope.LOCAL:
            return get_local_config_path()


def _load_yaml(path: pathlib.Path, *, as_dict: bool) -> Any:
    """Load YAML config with error handling."""
    if not path.exists():
        return {} if as_dict else None

    try:
        yaml = ruamel.yaml.YAML(typ="rt")
        with path.open() as f:
            data = yaml.load(f)
            if as_dict:
                return dict(data) if data is not None else {}
            return data
    except ruamel.yaml.YAMLError as e:
        raise exceptions.ConfigError(f"Invalid YAML in {path}: {e}") from e
    except PermissionError:
        raise exceptions.ConfigError(f"Permission denied reading {path}") from None
    except OSError as e:
        raise exceptions.ConfigError(f"Error reading {path}: {e}") from e


def _load_config_raw(path: pathlib.Path) -> dict[str, Any]:
    """Load YAML config as plain dict."""
    return _load_yaml(path, as_dict=True)


def _load_config_preserving_structure(path: pathlib.Path) -> Any:
    """Load YAML config preserving ruamel structure for comments."""
    return _load_yaml(path, as_dict=False)


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
    lock_path = path.with_suffix(".lock")

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock_file:
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


def parse_dotted_key(key: str) -> list[str]:
    """Parse 'cache.checkout_mode' into ['cache', 'checkout_mode']."""
    return key.split(".")


def get_nested(data: dict[str, Any], keys: list[str]) -> Any:
    """Traverse nested dict by key path, returns _NOT_FOUND sentinel if not found."""
    current: Any = data
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return _NOT_FOUND
        current_dict = cast("dict[str, Any]", current)
        current = current_dict[k]
    return current


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

    parent = get_nested(data, parent_keys)
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


def get_merged_config() -> models.PivotConfig:
    """Load and merge configs: defaults < global < local.

    Results are cached to avoid repeated disk I/O within a single command.
    Call clear_config_cache() to reset (e.g., in tests).
    """
    global _merged_config_cache
    if _merged_config_cache is not None:
        return _merged_config_cache

    defaults = models.PivotConfig.get_default().model_dump()

    global_data = load_config_file(get_global_config_path())
    merged = deep_merge(defaults, global_data)

    local_data = load_config_file(get_local_config_path())
    merged = deep_merge(merged, local_data)

    _merged_config_cache = models.PivotConfig.model_validate(merged)
    return _merged_config_cache


def clear_config_cache() -> None:
    """Clear the merged config cache. Call this when config files change."""
    global _merged_config_cache
    _merged_config_cache = None


def get_config_value(key: str) -> tuple[Any, models.ConfigSource]:
    """Get value for dotted key, returns (value, source)."""
    keys = parse_dotted_key(key)

    local_data = load_config_file(get_local_config_path())
    local_value = get_nested(local_data, keys)
    if local_value is not _NOT_FOUND:
        return local_value, models.ConfigSource.LOCAL

    global_data = load_config_file(get_global_config_path())
    global_value = get_nested(global_data, keys)
    if global_value is not _NOT_FOUND:
        return global_value, models.ConfigSource.GLOBAL

    default_value = models.get_config_default(key)
    if default_value is not None:
        return default_value, models.ConfigSource.DEFAULT

    return None, models.ConfigSource.UNKNOWN


def set_config_value(
    key: str, value: Any, scope: models.ConfigScope = models.ConfigScope.LOCAL
) -> None:
    """Set config value in specified scope with file locking."""
    keys = parse_dotted_key(key)
    with edit_config(scope) as data:
        set_nested(data, keys, value)


def unset_config_value(key: str, scope: models.ConfigScope = models.ConfigScope.LOCAL) -> bool:
    """Remove config value from specified scope. Returns True if removed."""
    keys = parse_dotted_key(key)
    path = get_config_path(scope)
    lock_path = path.with_suffix(".lock")

    if not path.exists():
        return False

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            raw_data = _load_config_preserving_structure(path)
            if raw_data is None:
                return False

            data: dict[str, Any] = raw_data
            removed = unset_nested(data, keys)

            if removed:
                _save_config(path, data)

            return removed
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def get_checkout_mode_order() -> list[models.CheckoutMode]:
    """Get checkout mode fallback order from merged config."""
    merged = get_merged_config()
    checkout_modes = list(merged.cache.checkout_mode)
    return checkout_modes if checkout_modes else DEFAULT_CHECKOUT_MODE_ORDER.copy()


def get_run_history_retention() -> int:
    """Get run history retention limit from merged config."""
    merged = get_merged_config()
    return merged.core.run_history_retention


def get_cache_dir() -> pathlib.Path:
    """Get cache directory from merged config, resolved to absolute path."""
    merged = get_merged_config()
    cache_dir = pathlib.Path(merged.cache.dir)
    if not cache_dir.is_absolute():
        cache_dir = project.get_project_root() / cache_dir
    return cache_dir


def get_state_dir() -> pathlib.Path:
    """Get state directory from merged config, resolved to absolute path."""
    merged = get_merged_config()
    state_dir = pathlib.Path(merged.core.state_dir)
    if not state_dir.is_absolute():
        state_dir = project.get_project_root() / state_dir
    return state_dir


def get_max_workers() -> int:
    """Get max workers from merged config."""
    merged = get_merged_config()
    return merged.core.max_workers


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

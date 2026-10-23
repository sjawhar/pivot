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


def _config_lock_path(path: pathlib.Path) -> pathlib.Path:
    """Lock file guarding a config file: ``config.yaml`` -> ``config.yaml.lock``."""
    return path.with_name(f"{path.name}.lock")


def _parse_yaml(stream: Any, source: object, *, as_dict: bool) -> Any:
    """Parse config YAML as ``config set`` writes it (YAML 1.2), with 0.5.2's error message."""
    try:
        data = ruamel.yaml.YAML(typ="rt").load(stream)
    except ruamel.yaml.YAMLError as e:
        raise exceptions.ConfigError(f"Invalid YAML in {source}: {e}") from e
    if as_dict:
        return dict(data) if data is not None else {}
    return data


def _load_yaml(path: pathlib.Path, *, as_dict: bool) -> Any:
    """Load YAML config with error handling."""
    if not path.exists():
        return {} if as_dict else None

    try:
        with path.open() as f:
            return _parse_yaml(f, path, as_dict=as_dict)
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


def parse_config_yaml(raw: bytes, source: str) -> dict[str, Any]:
    """Parse config file contents exactly as ``load_config_file`` parses a file."""
    return _parse_yaml(raw, source, as_dict=True)


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


def read_config_file_value(path: pathlib.Path, key: str) -> tuple[bool, Any]:
    """Read one dotted key from one config file: (present, value).

    The file is parsed as ``load_config_file`` parses it; a file that is not a mapping holds no
    key.
    """
    data = _load_config_preserving_structure(path)
    if not isinstance(data, dict):
        return False, None
    value = get_nested(cast("dict[str, Any]", data), parse_dotted_key(key))
    if value is _NOT_FOUND:
        return False, None
    return True, value


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


def _default_cache_dir() -> str:
    """Return the default cache display path without resolving or warning about project state."""
    root = project.get_project_root()
    cache_dir = project.state_dir_or_default(root, warn=False) / "cache"
    try:
        return cache_dir.relative_to(root).as_posix()
    except ValueError:
        return str(cache_dir)


def get_merged_config() -> models.PivotConfig:
    """Load and merge configs: defaults < global < local.

    A global ``core.state_dir`` is left out, so the merged config is the one in effect. An
    unset ``cache.dir`` is ``<state dir>/cache``. Results are cached to avoid repeated disk I/O
    within a single command. Call clear_config_cache() to reset (e.g., in tests).
    """
    global _merged_config_cache
    if _merged_config_cache is not None:
        return _merged_config_cache

    defaults = models.PivotConfig.get_default().model_dump()

    global_data = copy.deepcopy(load_config_file(get_global_config_path()))
    unset_nested(global_data, ["core", "state_dir"])
    merged = deep_merge(defaults, global_data)

    local_data = load_config_file(get_local_config_path())
    merged = deep_merge(merged, local_data)

    if all(get_nested(data, ["cache", "dir"]) is _NOT_FOUND for data in (global_data, local_data)):
        merged["cache"]["dir"] = _default_cache_dir()

    _merged_config_cache = models.PivotConfig.model_validate(merged)
    return _merged_config_cache


def clear_config_cache() -> None:
    """Clear cached configuration and its project state directory."""
    global _merged_config_cache
    _merged_config_cache = None
    project._state_dir_cache.clear()  # pyright: ignore[reportPrivateUsage] - shared resolver cache


def get_config_value(key: str) -> tuple[Any, models.ConfigSource]:
    """Get value for dotted key, returns (value, source).

    A global ``core.state_dir`` is reported with source ``global, ignored``, and the ignored-value
    warning is issued.
    """
    keys = parse_dotted_key(key)
    if key == "core.state_dir":
        project.report_ignored_global_state_dir()

    local_data = load_config_file(get_local_config_path())
    local_value = get_nested(local_data, keys)
    if local_value is not _NOT_FOUND:
        return local_value, models.ConfigSource.LOCAL

    global_data = load_config_file(get_global_config_path())
    global_value = get_nested(global_data, keys)
    if global_value is not _NOT_FOUND:
        if key == "core.state_dir":
            return global_value, models.ConfigSource.GLOBAL_IGNORED
        return global_value, models.ConfigSource.GLOBAL

    if key == "cache.dir":
        return _default_cache_dir(), models.ConfigSource.DEFAULT
    if key in models.CONFIG_KEY_DESCRIPTIONS:
        return models.get_config_default(key), models.ConfigSource.DEFAULT

    return None, models.ConfigSource.UNKNOWN


def _refuse_dangling_component(root: pathlib.Path, value: str) -> None:
    """Refuse a value whose path reaches a dangling symbolic link.

    Components are walked as the kernel walks them: ``..`` applies to the directory reached so
    far, through any link already followed, so ``link/../state`` checks the link target's
    sibling. A dangling link anywhere on that walk can never become a directory.
    """
    written = pathlib.Path(value)
    if written.is_absolute():
        current = pathlib.Path(written.anchor)
        parts = written.parts[1:]
    else:
        current = root
        parts = written.parts
    for index, part in enumerate(parts):
        if part == ".":
            continue
        if part == "..":
            current = current.parent
            continue
        current = current / part
        try:
            is_link = current.is_symlink()
        except OSError:
            # An unsearchable parent: the access check below refuses it.
            is_link = False
        if not is_link:
            # A real or missing directory: a later ``..`` steps back past it, as realpath does.
            continue
        if not current.exists():
            role = "core.state_dir" if index == len(parts) - 1 else "core.state_dir parent"
            raise exceptions.ConfigValidationError(f"{role} is a dangling symlink: {current}")
        current = pathlib.Path(os.path.realpath(current))


def _validate_state_dir_target(value: str) -> None:
    """Reject state-directory paths that cannot become a usable directory, before persistence.

    A dangling link on the path is refused. Then the path the resolver will use
    (``project.resolve_state_dir_value``) must pass ``project.state_dir_target_problem``.
    """
    root = project.get_project_root()
    _refuse_dangling_component(root, value)

    problem = project.state_dir_target_problem(project.resolve_state_dir_value(root, value))
    if problem is not None:
        raise exceptions.ConfigValidationError(problem)


def _has_runtime_state(state_dir: pathlib.Path) -> bool:
    """Whether a previous state directory holds more than the project's marker files."""
    marker_files = {"config.yaml", "config.yaml.lock", ".gitignore"}
    return any(path.name not in marker_files for path in state_dir.iterdir())


def _current_state_dir() -> pathlib.Path | None:
    """The project's state directory, or None when the configured value cannot be used."""
    try:
        return project.get_state_dir()
    except exceptions.ConfigError:
        return None


def _transition_local_state_dir(old_state_dir: pathlib.Path | None) -> list[str]:
    """Initialize the new directory without moving or discarding prior project state.

    ``.pivot/.gitignore`` is never touched: ``.pivot`` still holds the config and its lock. An
    old value that names no readable directory is reported as invalid, as 0.5.2 never read it.
    """
    root = project.get_project_root()
    new_state_dir = project.ensure_state_dir(root)
    invalid = ["The previous core.state_dir was invalid; nothing was moved."]
    if old_state_dir is None:
        return invalid
    if old_state_dir == new_state_dir:
        return []
    try:
        has_runtime_state = _has_runtime_state(old_state_dir)
    except FileNotFoundError:
        return []
    except OSError:
        return invalid
    if has_runtime_state:
        return [f"Existing state at {old_state_dir} is no longer used."]
    return []


def _global_state_dir_refusal() -> str:
    """The refusal of a global ``core.state_dir`` write, worded as the ignored-value warning."""
    try:
        root: pathlib.Path | None = project.get_project_root()
    except exceptions.ProjectNotInitializedError:
        root = None
    shadowed = root is not None and project.state_dir_is_configured(root)
    return project.ignored_global_state_dir_message(
        get_global_config_path(), root, shadowed=shadowed
    )


def _changes_local_state_dir(key: str, scope: models.ConfigScope) -> bool:
    return key == "core.state_dir" and scope is models.ConfigScope.LOCAL


def set_config_value(
    key: str, value: Any, scope: models.ConfigScope = models.ConfigScope.LOCAL
) -> list[str]:
    """Set a configuration value and return any state-transition notices.

    A global ``core.state_dir`` is refused: it would have no effect.
    """
    if key == "core.state_dir" and scope is models.ConfigScope.GLOBAL:
        raise exceptions.ConfigValidationError(_global_state_dir_refusal())
    changes_state_dir = _changes_local_state_dir(key, scope)
    old_state_dir: pathlib.Path | None = None
    if changes_state_dir:
        _validate_state_dir_target(cast("str", value))
        old_state_dir = _current_state_dir()
    keys = parse_dotted_key(key)
    with edit_config(scope) as data:
        set_nested(data, keys, value)
    clear_config_cache()
    if not changes_state_dir:
        return []
    return _transition_local_state_dir(old_state_dir)


def unset_config_value(
    key: str, scope: models.ConfigScope = models.ConfigScope.LOCAL
) -> tuple[bool, list[str]]:
    """Remove config value from specified scope.

    Returns whether it was removed, and any state-transition notices.
    """
    keys = parse_dotted_key(key)
    path = get_config_path(scope)

    if not path.exists():
        return False, []

    changes_state_dir = _changes_local_state_dir(key, scope)
    old_state_dir: pathlib.Path | None = None
    if changes_state_dir:
        old_state_dir = _current_state_dir()
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
    return removed, _transition_local_state_dir(old_state_dir)


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
    """Resolve the effective cache without creating state or cache directories."""
    root = project.get_project_root()
    env_cache = os.environ.get("PIVOT_CACHE_DIR", "").strip()
    configured = pathlib.Path(env_cache or get_merged_config().cache.dir)
    cache_dir = configured if configured.is_absolute() else root / configured
    cache_dir = pathlib.Path(os.path.realpath(cache_dir))
    state_dir = project.state_dir_or_default(root)
    stages_dir = state_dir / "stages"
    fallback_cache_dir = state_dir / "cache"
    if (
        cache_dir == stages_dir
        or cache_dir.is_relative_to(stages_dir)
        or stages_dir.is_relative_to(cache_dir)
    ):
        project.report_cache_stage_overlap(cache_dir, stages_dir, fallback_cache_dir)
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
        project.ensure_state_dir(root)
        if cache_dir != state_dir / "cache":
            project.make_nested_cache_dir(cache_dir)
    return cache_dir


def get_state_dir() -> pathlib.Path:
    """Get the current project's configured state directory, created when absent."""
    get_merged_config()
    root = project.get_project_root()
    try:
        return project.ensure_state_dir(root)
    except exceptions.ConfigError:
        return project.ensure_state_dir_or_default(root)


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

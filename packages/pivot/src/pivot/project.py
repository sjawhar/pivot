import logging
import os
import pathlib
import typing

logger = logging.getLogger(__name__)

_project_root_cache: pathlib.Path | None = None
_state_dir_cache = dict[pathlib.Path, pathlib.Path]()

STATE_GITIGNORE_CONTENT = """\
# Cache directory (stage outputs, file hashes)
/cache/

# State database (file hashes, generation counters)
/state.lmdb/

# Artifact and restore locks (flock files for concurrent execution)
/locks/

# Agent RPC socket (left behind by TUI and --serve sessions)
/agent.sock

# Generated plots gallery (pivot plots show)
/plots/

# Key for secret params' digests in lock files; never commit it
/secret.key
"""


def find_project_root() -> pathlib.Path:
    """Walk up from cwd to find the top-most .pivot directory.

    Raises:
        ProjectNotInitializedError: If no .pivot directory exists above cwd.
    """
    from pivot import exceptions

    current = pathlib.Path.cwd().resolve()
    topmost_pivot: pathlib.Path | None = None

    for parent in [current, *current.parents]:
        if (parent / ".pivot").is_dir():
            topmost_pivot = parent

    if topmost_pivot is None:
        msg = f"No .pivot directory found above '{current}'. Run 'pivot init' to initialize a Pivot project."
        raise exceptions.ProjectNotInitializedError(msg)

    logger.debug(f"Project root: {topmost_pivot}")
    return topmost_pivot


def get_project_root() -> pathlib.Path:
    """Get project root (cached after first call)."""
    global _project_root_cache
    if _project_root_cache is None:
        _project_root_cache = find_project_root()
    return _project_root_cache


def state_dir_for(root: pathlib.Path) -> pathlib.Path:
    """Return the state directory configured for ``root``.

    ``core.state_dir`` comes from ``<root>/.pivot/config.yaml``. Relative values are rooted at
    ``root``; absent configuration keeps state in ``<root>/.pivot``. The ``.pivot`` marker and
    its configuration always remain at the project root.
    """
    if (cached := _state_dir_cache.get(root)) is not None:
        return cached

    import yaml

    from pivot import exceptions

    config_path = root / ".pivot" / "config.yaml"
    try:
        data: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = None
    except yaml.YAMLError as e:
        raise exceptions.ConfigError(f"Error reading {config_path}: {e}") from e

    configured: object = None
    if isinstance(data, dict):
        config_data = typing.cast("dict[str, object]", data)
        core = config_data.get("core")
        if isinstance(core, dict):
            configured = typing.cast("dict[str, object]", core).get("state_dir")

    if configured is None:
        state_dir = root / ".pivot"
    elif isinstance(configured, str) and configured:
        state_dir = pathlib.Path(configured)
        state_dir = state_dir if state_dir.is_absolute() else root / state_dir
    else:
        raise exceptions.ConfigError(f"core.state_dir in {config_path} must be a path")

    _state_dir_cache[root] = state_dir
    return state_dir


def get_state_dir() -> pathlib.Path:
    """Return the state directory of the current project."""
    return state_dir_for(get_project_root())


def ensure_state_dir(root: pathlib.Path) -> pathlib.Path:
    """Create an initialized project's state directory and default ignore file when absent."""
    state_dir = state_dir_for(root)
    if not (root / ".pivot").is_dir():
        return state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    gitignore_path = state_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(STATE_GITIGNORE_CONTENT)
    return state_dir


def resolve_path(path: str | pathlib.Path) -> pathlib.Path:
    """Resolve relative path from project root; absolute paths unchanged."""
    p = pathlib.Path(os.fspath(path))
    if p.is_absolute():
        return p.resolve()
    return (get_project_root() / p).resolve()


def normalize_path(path: str | pathlib.Path, base: pathlib.Path | None = None) -> pathlib.Path:
    """Make path absolute from base (default: project root), preserving symlinks.

    Accepts both Unix (/) and Windows (\\) path separators.
    """
    from pathlib import PureWindowsPath

    if base is None:
        base = get_project_root()

    # Normalize Windows paths to POSIX (handles both \\ and /)
    p = pathlib.Path(PureWindowsPath(os.fspath(path)).as_posix())

    abs_path = p.absolute() if p.is_absolute() else (base / p).absolute()
    return pathlib.Path(os.path.normpath(abs_path))


def contains_symlink_in_path(path: pathlib.Path, base: pathlib.Path) -> bool:
    """Check if any component from base to path is a symlink.

    Example: If /project/data is a symlink, and path is /project/data/file.csv,
    returns True because 'data' component is a symlink.

    Args:
        path: Path to check for symlink components
        base: Base path to stop checking at

    Returns:
        True if any component in the path is a symlink
    """
    current = path.absolute()
    base_abs = base.absolute()

    while current != base_abs:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    return False


def resolve_path_for_comparison(path: str, context: str) -> pathlib.Path:
    """Resolve path for overlap comparison, falling back to normalized for missing stage outputs."""
    try:
        return resolve_path(path)
    except PermissionError as e:
        raise PermissionError(f"Permission denied for {context} '{path}'") from e
    except RuntimeError as e:
        raise RuntimeError(f"Circular symlink in {context} '{path}'") from e
    except FileNotFoundError:
        if "stage output" in context.lower():
            return normalize_path(path)
        raise
    except OSError as e:
        raise OSError(f"Filesystem error for {context} '{path}': {e}") from e


def to_relative_path(abs_path: str | pathlib.Path, base: pathlib.Path | None = None) -> str:
    """Convert absolute path to relative from base (default: project root).

    If path is already relative or outside base, returns as-is with warning for outside paths.
    """
    path = pathlib.Path(abs_path)
    if not path.is_absolute():
        return str(abs_path)

    if base is None:
        base = get_project_root()

    try:
        return str(path.relative_to(base))
    except ValueError:
        logger.warning(f"Path '{abs_path}' is outside base '{base}', keeping absolute")
        return str(abs_path)

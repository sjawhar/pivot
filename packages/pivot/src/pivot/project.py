import logging
import os
import pathlib
import typing

logger = logging.getLogger(__name__)

_project_root_cache: pathlib.Path | None = None
_state_dir_cache = dict[pathlib.Path, pathlib.Path]()

# The CLI turns this on; loky workers and shell completion never do, so they stay silent.
_ignored_global_state_dir_warning_enabled = False
# Global config files whose ignored core.state_dir was already reported in this process.
_reported_ignored_global_state_dirs = set[pathlib.Path]()

STATE_GITIGNORE_CONTENT = """\
# Cache directory (stage outputs, file hashes)
/cache/

# State database (file hashes, generation counters)
/state.lmdb/

# Artifact and restore locks (flock files for concurrent execution)
/locks/

# Agent RPC socket (left behind by TUI and --serve sessions)
/agent.sock

# Config lock (held while editing config.yaml)
/config.yaml.lock

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


def _read_core_state_dir(config_path: pathlib.Path) -> tuple[bool, object]:
    """Read ``core.state_dir`` from one config file: (present, value)."""
    import yaml

    from pivot import exceptions

    try:
        data: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, None
    except yaml.YAMLError as e:
        raise exceptions.ConfigError(f"Error reading {config_path}: {e}") from e
    if not isinstance(data, dict):
        return False, None
    core = typing.cast("dict[str, object]", data).get("core")
    if not isinstance(core, dict):
        return False, None
    core_data = typing.cast("dict[str, object]", core)
    return "state_dir" in core_data, core_data.get("state_dir")


def resolve_state_dir_value(root: pathlib.Path, value: str) -> pathlib.Path:
    """Return the directory a ``core.state_dir`` value names, as the kernel will open it.

    Existing components are resolved, symbolic links included, so ``link/../state`` names the
    link target's sibling; ``..`` after a missing component applies to the path so far. Every
    reader and writer uses this result, never a lexical spelling of the value.
    """
    return pathlib.Path(os.path.realpath(root / value))


def state_dir_for(root: pathlib.Path) -> pathlib.Path:
    """Return the state directory configured for ``root``.

    ``core.state_dir`` comes from ``<root>/.pivot/config.yaml`` only: a state directory belongs
    to one project, so a global value is ignored, and reported once per process when the CLI
    enabled that. A configured value is resolved with ``resolve_state_dir_value``; absent
    configuration keeps state in ``<root>/.pivot``. The ``.pivot`` marker and its configuration
    always remain at the project root.
    """
    if (cached := _state_dir_cache.get(root)) is not None:
        return cached

    from pivot import exceptions

    config_path = root / ".pivot" / "config.yaml"
    present, configured = _read_core_state_dir(config_path)
    warn_if_global_state_dir_ignored(root, shadowed=present)

    if configured is None:
        state_dir = root / ".pivot"
    elif is_valid_state_dir_value(configured):
        state_dir = resolve_state_dir_value(root, typing.cast("str", configured))
    else:
        raise exceptions.ConfigError(f"core.state_dir in {config_path} must be a path")

    _state_dir_cache[root] = state_dir
    return state_dir


def is_valid_state_dir_value(value: object) -> bool:
    """Whether a configured ``core.state_dir`` value names a path, as ``state_dir_for`` requires."""
    return isinstance(value, str) and bool(value)


def state_dir_is_configured(root: pathlib.Path) -> bool:
    """Return whether the local configuration explicitly contains core.state_dir."""
    return _read_core_state_dir(root / ".pivot" / "config.yaml")[0]


def get_state_dir() -> pathlib.Path:
    """Return the state directory of the current project."""
    return state_dir_for(get_project_root())


def enable_ignored_global_state_dir_warning() -> None:
    """Report an ignored global ``core.state_dir`` from now on in this process."""
    global _ignored_global_state_dir_warning_enabled
    _ignored_global_state_dir_warning_enabled = True


def ignored_global_state_dir_message(
    global_path: pathlib.Path, root: pathlib.Path | None, *, shadowed: bool
) -> str:
    """The one-line notice that a global ``core.state_dir`` has no effect."""
    local_path = (
        "the project's .pivot/config.yaml" if root is None else root / ".pivot" / "config.yaml"
    )
    if shadowed:
        return f"core.state_dir in {global_path} is shadowed by {local_path} and ignored"
    return f"core.state_dir in {global_path} is ignored; set it in {local_path}"


def warn_if_global_state_dir_ignored(root: pathlib.Path | None, *, shadowed: bool) -> None:
    """Log, once per process, that the global config's ``core.state_dir`` is ignored.

    Silent until ``enable_ignored_global_state_dir_warning`` runs. ``shadowed`` says the project
    sets its own value.
    """
    if not _ignored_global_state_dir_warning_enabled:
        return
    from pivot.config import io as config_io

    global_path = config_io.get_global_config_path()
    if global_path in _reported_ignored_global_state_dirs:
        return
    if not _read_core_state_dir(global_path)[0]:
        return
    _reported_ignored_global_state_dirs.add(global_path)
    logger.warning(ignored_global_state_dir_message(global_path, root, shadowed=shadowed))


def report_ignored_global_state_dir() -> None:
    """Report the ignored global value for the current project, as ``state_dir_for`` does."""
    root = get_project_root()
    warn_if_global_state_dir_ignored(root, shadowed=state_dir_is_configured(root))


def create_gitignore_if_absent(path: pathlib.Path) -> bool:
    """Write the default ignore template to ``path`` only if nothing exists there.

    ``O_EXCL`` fails on any existing entry, including a dangling symlink, and never follows one.
    Returns whether the file was created.
    """
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w") as gitignore:
        gitignore.write(STATE_GITIGNORE_CONTENT)
    return True


def ensure_state_dir(root: pathlib.Path) -> pathlib.Path:
    """Create an initialized project's state directory and default ignore file when absent."""
    state_dir = state_dir_for(root)
    if not (root / ".pivot").is_dir():
        return state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    create_gitignore_if_absent(state_dir / ".gitignore")
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

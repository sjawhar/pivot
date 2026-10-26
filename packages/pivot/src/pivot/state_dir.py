"""The project's state directory: where ``core.state_dir`` points, and Pivot's single creator of it.

Resolution reads ``<root>/.pivot/config.yaml`` through ``pivot.config_files`` only, so this module
never loads the ``pivot.config`` package. ``config.io`` imports this module for its
state-directory transitions and effective cache; nothing here imports ``pivot.config``.
"""

import contextlib
import contextvars
import logging
import os
import pathlib
import typing
from collections.abc import Iterator

from pivot import config_files, exceptions, project

logger = logging.getLogger(__name__)


class StateDirWarnings(typing.TypedDict):
    """What one invocation has already reported about the state directory."""

    # Global config files whose ignored core.state_dir was reported.
    ignored_global: set[pathlib.Path]
    # Project config files whose unusable state directory was reported.
    unusable: set[pathlib.Path]
    # Configured state directories whose 0.5.2 lock layout was reported.
    legacy: set[pathlib.Path]
    # Effective cache directories reported as overlapping the stage-lock directory.
    cache_stage_overlap: set[tuple[pathlib.Path, pathlib.Path]]


class StateDirInvocation(typing.TypedDict):
    """One invocation's state-directory record: what it resolved, and what it reported."""

    # The state directory of each project root, read from its config once per invocation.
    resolved: dict[pathlib.Path, pathlib.Path]
    # None under --quiet and for a worker's stage: these warnings stay silent.
    reported: StateDirWarnings | None


# Set by the CLI callback for one CLI invocation, by a worker for one stage, and by an engine
# run that no invocation covers (``run_invocation``). Outside one (shell completion, a library
# caller between runs), every lookup reads the config and nothing is reported.
_invocation = contextvars.ContextVar[StateDirInvocation | None](
    "state_dir_invocation", default=None
)

# What a lookup outside an invocation last resolved for a project root. The key is the project
# root, and an entry answers only while the exact bytes of `<root>/.pivot/config.yaml` equal the
# bytes that answer was parsed from. Each such lookup reads the file again, so an edit by this
# process or by another one is seen on the next lookup; no timestamp, size or inode is trusted.
# `clear_state_dir_cache` empties it, which `config.io.clear_config_cache` does at the start of
# each CLI invocation and after `config set` or `config unset`. A lookup inside an invocation or
# an engine run never reads or writes it: the record answers there.
_resolved_outside_invocation = dict[pathlib.Path, tuple[bytes, pathlib.Path]]()


def _reported() -> StateDirWarnings | None:
    invocation = _invocation.get()
    return None if invocation is None else invocation["reported"]


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


def resolve_state_dir_value(root: pathlib.Path, value: str) -> pathlib.Path:
    """Return the directory a ``core.state_dir`` value names, as the kernel will open it.

    Existing components are resolved, symbolic links included, so ``link/../state`` names the
    link target's sibling; ``..`` after a missing component applies to the path so far. Every
    reader and writer uses this result, never a lexical spelling of the value.
    """
    return pathlib.Path(os.path.realpath(root / value))


def _config_file_content(config_path: pathlib.Path) -> bytes | None:
    """The config file's bytes, or None when this call cannot read them.

    Only the resolution cache uses them. A file this call cannot read is left to the parser,
    which reports the problem as it always did.
    """
    try:
        return config_path.read_bytes()
    except OSError:
        return None


def state_dir_for(root: pathlib.Path) -> pathlib.Path:
    """Return the state directory configured for ``root``.

    ``core.state_dir`` comes from ``<root>/.pivot/config.yaml`` only: a state directory belongs
    to one project, so a global value is ignored, and reported once per invocation when the CLI
    enabled that. A configured value is resolved with ``resolve_state_dir_value``; absent
    configuration keeps state in ``<root>/.pivot``. The ``.pivot`` marker and its configuration
    always remain at the project root. Within one invocation or engine run, each root's config
    is read once and the record answers. Outside one, every call reads the file's bytes; it
    reuses an earlier answer only for the same root and identical bytes, and otherwise parses
    the file. An unreadable file is always passed to the parser, which reports it.
    """
    invocation = _invocation.get()
    if invocation is not None and (cached := invocation["resolved"].get(root)) is not None:
        return cached

    config_path = root / ".pivot" / "config.yaml"
    content = None if invocation is not None else _config_file_content(config_path)
    if content is not None:
        remembered = _resolved_outside_invocation.get(root)
        if remembered is not None and remembered[0] == content:
            return remembered[1]

    present, configured = config_files.read_config_file_value(config_path, "core.state_dir")
    if configured is None:
        state_dir = root / ".pivot"
    elif is_valid_state_dir_value(configured):
        state_dir = resolve_state_dir_value(root, typing.cast("str", configured))
    elif configured == "":
        raise exceptions.EmptyStateDirError(f"core.state_dir in {config_path} must be a path")
    else:
        raise exceptions.ConfigError(f"core.state_dir in {config_path} must be a path")

    warn_if_global_state_dir_ignored(root, shadowed=present)
    if invocation is not None:
        invocation["resolved"][root] = state_dir
    elif content is not None:
        _resolved_outside_invocation[root] = (content, state_dir)
    return state_dir


def state_dir_or_default(root: pathlib.Path, *, warn: bool = True) -> pathlib.Path:
    """The state directory of a command that 0.5.2 ran without one; nothing is created here.

    The configured directory when ``<root>/.pivot/config.yaml`` can be read, its
    ``core.state_dir`` is a path, and that path is or can become a directory this process
    writes (``state_dir_target_problem``). Otherwise the default ``<root>/.pivot``. Callers
    that merely display stored configuration pass ``warn=False``; a command that reads or writes
    state uses the warning-enabled resolver and then fails where 0.5.2 failed.
    """
    config_path = root / ".pivot" / "config.yaml"
    reason = "The state directory is unknown"
    try:
        state_dir = state_dir_for(root)
    except exceptions.ConfigError as e:
        problem = str(e)
    except UnicodeDecodeError as e:
        problem = f"{config_path} is not valid UTF-8: {e}"
    else:
        if state_dir == root / ".pivot":
            return state_dir
        target_problem = state_dir_target_problem(state_dir)
        if target_problem is None:
            return state_dir
        problem = f"{target_problem} (core.state_dir in {config_path})"
        reason = "The state directory cannot be used"
    reported = _reported()
    if warn and reported is not None and config_path not in reported["unusable"]:
        reported["unusable"].add(config_path)
        logger.warning(f"{problem}. {reason}; Pivot uses the default .pivot.")
    return root / ".pivot"


def ensure_state_dir_or_default(root: pathlib.Path) -> pathlib.Path:
    """``state_dir_or_default``, created as ``ensure_state_dir`` creates it.

    For a command that 0.5.2 ran and that writes where 0.5.2 wrote to ``.pivot`` (the plot
    gallery, the secret key).
    """
    state_dir = state_dir_or_default(root)
    if state_dir != root / ".pivot":
        ensure_state_dir(root)
    return state_dir


def state_dir_target_problem(target: pathlib.Path) -> str | None:
    """Why ``target`` cannot become a state directory, or None when it is or can be one.

    Its nearest existing entry must be a directory this process can create entries in and
    search (``W_OK | X_OK``).
    """
    for candidate in (target, *target.parents):
        if not os.path.lexists(candidate):
            continue
        role = "core.state_dir" if candidate == target else "core.state_dir parent"
        if not os.path.isdir(candidate):
            return f"{role} must be a directory: {candidate}"
        if not os.access(candidate, os.W_OK | os.X_OK):
            return f"{role} is not a writable, searchable directory: {candidate}"
        return None
    return None


def is_valid_state_dir_value(value: object) -> bool:
    """Whether a configured ``core.state_dir`` value names a path, as ``state_dir_for`` requires."""
    return isinstance(value, str) and bool(value)


def state_dir_is_configured(root: pathlib.Path) -> bool:
    """Return whether the local configuration explicitly contains core.state_dir."""
    config_path = root / ".pivot" / "config.yaml"
    return config_files.read_config_file_value(config_path, "core.state_dir")[0]


def get_state_dir() -> pathlib.Path:
    """Return the configured state directory of the current project."""
    return state_dir_for(project.get_project_root())


def legacy_state_dir_for(state_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the 0.5.2 state directory paired with ``state_dir``, if this project has one."""
    try:
        root = project.get_project_root()
        configured = state_dir_for(root)
    except (exceptions.ConfigError, exceptions.ProjectNotInitializedError, UnicodeDecodeError):
        return None
    legacy = root / ".pivot"
    if configured != state_dir or configured == legacy:
        return None
    return legacy


def legacy_stages_dir_for(stages_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the old stages directory corresponding to a configured stages directory."""
    legacy = legacy_state_dir_for(stages_dir.parent)
    return None if legacy is None else legacy / "stages"


def legacy_cache_files_dir_for(cache_files_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the old cache object directory corresponding to a default configured cache."""
    legacy = legacy_state_dir_for(cache_files_dir.parent.parent)
    if legacy is None:
        return None
    return legacy / "cache" / "files"


def report_legacy_state_dir(state_dir: pathlib.Path) -> None:
    """Warn once per invocation that reads fall back to the 0.5.2 state layout for ``state_dir``."""
    reported = _reported()
    if reported is None or state_dir in reported["legacy"]:
        return
    legacy = legacy_state_dir_for(state_dir)
    if legacy is None:
        return
    reported["legacy"].add(state_dir)
    root = project.get_project_root()
    try:
        target = (state_dir / "stages").relative_to(root)
    except ValueError:
        move = f"mv {legacy / 'stages'} {state_dir / 'stages'}"
    else:
        move = f"git mv .pivot/stages {target}"
    logger.warning(
        "core.state_dir is %s, but Pivot 0.5.2 stored locks and cache in %s. Pivot reads missing locks and cache entries there; move the locks with `%s`.",
        state_dir,
        legacy,
        move,
    )


def report_cache_stage_overlap(
    cache_dir: pathlib.Path, stages_dir: pathlib.Path, fallback_cache_dir: pathlib.Path
) -> None:
    """Warn once per invocation that an effective cache directory overlaps durable stage locks."""
    reported = _reported()
    if reported is None:
        return
    key = (cache_dir, stages_dir)
    if key in reported["cache_stage_overlap"]:
        return
    reported["cache_stage_overlap"].add(key)
    logger.warning(
        "cache directory %s overlaps the stage-lock directory %s; Pivot uses %s.",
        cache_dir,
        stages_dir,
        fallback_cache_dir,
    )


def begin_invocation(
    *, report: bool, state_dirs: dict[pathlib.Path, pathlib.Path] | None = None
) -> contextvars.Token[StateDirInvocation | None]:
    """Start one invocation's record, used until the returned token is reset.

    Its resolution cache starts with ``state_dirs`` and reads the config of any other project
    afresh. A worker's stage passes the state directory its Pivot process decided for the
    command, so a ``core.state_dir`` edited during a ``--watch`` session never reaches it. With
    ``report``, each state-directory problem is warned once; without it, none is.
    """
    reported = (
        StateDirWarnings(
            ignored_global=set[pathlib.Path](),
            unusable=set[pathlib.Path](),
            legacy=set[pathlib.Path](),
            cache_stage_overlap=set[tuple[pathlib.Path, pathlib.Path]](),
        )
        if report
        else None
    )
    return _invocation.set(
        StateDirInvocation(
            resolved=dict[pathlib.Path, pathlib.Path](state_dirs or {}), reported=reported
        )
    )


def end_invocation(token: contextvars.Token[StateDirInvocation | None]) -> None:
    """End an invocation, restoring whatever record ``begin_invocation`` replaced."""
    _invocation.reset(token)


@contextlib.contextmanager
def run_invocation() -> Iterator[None]:
    """Hold one engine run's record, when no invocation already holds one.

    A run decides its state directory before its first stage and writes its locks, its run
    manifest and its stages' cache entries there. The engine fingerprints a stage when it
    dispatches it, so without a record a library caller's later stages would read a
    ``core.state_dir`` that an earlier stage of the same run edited, and file their entries
    under it. Within the run the record answers every lookup, in the threads that fingerprint
    at dispatch as well, because they copy this context. A CLI invocation already holds the
    value for the whole command, and this leaves that record alone. The run reports nothing: a
    library caller's warnings are silent outside a run too.
    """
    if _invocation.get() is not None:
        yield
        return
    token = begin_invocation(report=False)
    try:
        yield
    finally:
        end_invocation(token)


def clear_state_dir_cache() -> None:
    """Forget every resolved state directory, so the next lookup parses the config again."""
    _resolved_outside_invocation.clear()
    invocation = _invocation.get()
    if invocation is not None:
        invocation["resolved"].clear()


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


def warn_if_global_state_dir_ignored(root: pathlib.Path, *, shadowed: bool) -> None:
    """Log, once per invocation, that the global config's ``core.state_dir`` is ignored.

    Silent outside an invocation that reports (``begin_invocation``). ``shadowed`` says the
    project sets its own value. The global file is read only to decide whether to warn: when it
    cannot be read, this says nothing, and reporting that stays with the normal config load.
    """
    reported = _reported()
    if reported is None:
        return
    global_path = config_files.get_global_config_path()
    if global_path in reported["ignored_global"]:
        return
    try:
        present, _ = config_files.read_config_file_value(global_path, "core.state_dir")
    except (exceptions.ConfigError, UnicodeDecodeError):
        return
    if not present:
        return
    reported["ignored_global"].add(global_path)
    logger.warning(ignored_global_state_dir_message(global_path, root, shadowed=shadowed))


def report_ignored_global_state_dir() -> bool:
    """Report an ignored global value from a config command; return whether a project ignores it.

    Outside a project nothing ignores the value, so nothing is reported. Neither config file has
    to load here: when the local one cannot be read, the warning is left out.
    """
    try:
        root = project.get_project_root()
    except exceptions.ProjectNotInitializedError:
        return False
    try:
        shadowed = state_dir_is_configured(root)
    except (exceptions.ConfigError, UnicodeDecodeError):
        return True
    warn_if_global_state_dir_ignored(root, shadowed=shadowed)
    return True


def create_gitignore_if_absent(path: pathlib.Path) -> bool:
    """Write the default ignore template to ``path`` only if nothing exists there.

    ``O_EXCL`` fails on any existing entry, including a dangling symlink, and never follows one.
    Returns whether the file was created.
    """
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w") as gitignore:
        gitignore.write(STATE_GITIGNORE_CONTENT)
    return True


def make_state_dir(state_dir: pathlib.Path) -> pathlib.Path:
    """Create ``state_dir`` and its default ignore file when this call makes the directory."""
    try:
        state_dir.mkdir(parents=True)
    except FileExistsError:
        return state_dir
    create_gitignore_if_absent(state_dir / ".gitignore")
    return state_dir


def make_dirs(path: pathlib.Path) -> None:
    """``path.mkdir(parents=True, exist_ok=True)`` for every directory Pivot creates for state.

    When ``path`` lies inside the current project's configured state directory, that directory
    is first created through ``ensure_state_dir``, so it gets its ``.gitignore`` before anything
    is written inside it. Outside a project, or outside its state directory, this only creates
    ``path``.
    """
    if not path.is_dir():
        try:
            root = project.get_project_root()
            state_dir = state_dir_or_default(root)
        except (exceptions.ProjectNotInitializedError, OSError):
            pass
        else:
            if state_dir != root / ".pivot" and path.absolute().is_relative_to(state_dir):
                ensure_state_dir(root)
    path.mkdir(parents=True, exist_ok=True)


def make_nested_cache_dir(cache_dir: pathlib.Path) -> None:
    """Create a custom cache directory inside the state directory, ignoring all of it.

    The state directory's template ignores only ``/cache/``, so a cache at another path inside
    it gets its own ``*`` ignore file when this call creates the directory. An existing
    directory is left as it is.
    """
    try:
        cache_dir.mkdir(parents=True)
    except FileExistsError:
        return
    descriptor = os.open(cache_dir / ".gitignore", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    with os.fdopen(descriptor, "w") as gitignore:
        gitignore.write("# Pivot cache; never commit it\n*\n")


def locate_state_dir(root: pathlib.Path) -> pathlib.Path:
    """Return an initialized project's state directory without creating anything.

    An empty hand-edited ``core.state_dir`` falls back to ``<root>/.pivot`` through
    ``state_dir_or_default``. Other invalid config values still raise ``ConfigError``.
    """
    try:
        return state_dir_for(root)
    except exceptions.EmptyStateDirError:
        return state_dir_or_default(root)


def ensure_state_dir(root: pathlib.Path) -> pathlib.Path:
    """``locate_state_dir``, created when absent.

    A directory this call creates gets the default ``.gitignore``. One that already exists is
    left as it is, so a deleted ``.pivot/.gitignore`` stays deleted until ``init`` writes it. A
    directory that cannot be created (``state_dir_target_problem``) is returned as it is, so the
    command fails where it opens it, as 0.5.2 did.
    """
    state_dir = locate_state_dir(root)
    if (
        os.path.isdir(state_dir)
        or not (root / ".pivot").is_dir()
        or state_dir_target_problem(state_dir) is not None
    ):
        return state_dir
    return make_state_dir(state_dir)


# --- `config set` / `config unset` of core.state_dir ---


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


def validate_state_dir_target(value: str) -> None:
    """Reject state-directory paths that cannot become a usable directory, before persistence.

    A dangling link on the path is refused. Then the path the resolver will use
    (``resolve_state_dir_value``) must pass ``state_dir_target_problem``.
    """
    root = project.get_project_root()
    _refuse_dangling_component(root, value)

    problem = state_dir_target_problem(resolve_state_dir_value(root, value))
    if problem is not None:
        raise exceptions.ConfigValidationError(problem)


def _has_runtime_state(state_dir: pathlib.Path) -> bool:
    """Whether a previous state directory holds more than the project's marker files."""
    marker_files = {"config.yaml", "config.yaml.lock", ".gitignore"}
    return any(path.name not in marker_files for path in state_dir.iterdir())


def current_state_dir() -> pathlib.Path | None:
    """The project's state directory, or None when the configured value cannot be used."""
    try:
        return get_state_dir()
    except exceptions.ConfigError:
        return None


def transition_local_state_dir(old_state_dir: pathlib.Path | None) -> list[str]:
    """Initialize the new directory without moving or discarding prior project state.

    ``.pivot/.gitignore`` is never touched: ``.pivot`` still holds the config and its lock. An
    old value that names no readable directory is reported as invalid, as 0.5.2 never read it.
    """
    root = project.get_project_root()
    new_state_dir = ensure_state_dir(root)
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


def global_state_dir_refusal() -> str:
    """The refusal of a global ``core.state_dir`` write, worded as the ignored-value warning."""
    try:
        root: pathlib.Path | None = project.get_project_root()
    except exceptions.ProjectNotInitializedError:
        root = None
    shadowed = root is not None and state_dir_is_configured(root)
    return ignored_global_state_dir_message(
        config_files.get_global_config_path(), root, shadowed=shadowed
    )

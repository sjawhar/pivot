"""The project's state directory: where ``core.state_dir`` points, and Pivot's single creator of it.

Resolution reads ``<root>/.pivot/config.yaml`` through ``pivot.config_files`` only, so this module
never loads the ``pivot.config`` package. ``config.io`` imports this module for its
state-directory transitions and effective cache; nothing here imports ``pivot.config``.
"""

import contextlib
import contextvars
import copy
import errno
import logging
import os
import pathlib
import shlex
import typing
from collections.abc import Callable, Iterator, Mapping

from pivot import config_files, exceptions, project

if typing.TYPE_CHECKING:
    from pivot.storage.lock import DirectoryRecord
    from pivot.storage.state import StateDB

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
    # Configured state directories whose stale ``.pivot`` locks were reported.
    stale: set[pathlib.Path]


# One project root's ``core.state_dir`` as a record resolved it: the directory the value names
# (``<root>/.pivot`` when it is unset), or the error that says why it cannot be used. A value
# that names a path which cannot become a directory this process writes is kept as
# ``StateDirTargetError``, with the outcome decided with the value: the stages keep their state
# in ``.pivot``, and a command that needs the configured directory falls back to ``.pivot`` too
# or fails on the path (``StateDirTargetError.fallback``).
type StateDirResolution = pathlib.Path | exceptions.ConfigError | UnicodeDecodeError


class StageRunMention(typing.TypedDict):
    """One run's record of a stage in one state directory's database."""

    run_id: str
    input_hash: str
    # Whether that run ran the stage (and wrote its lock), rather than keeping or restoring it.
    ran: bool
    # The state directory whose database recorded the run.
    state_dir: str


class OrderingView(typing.TypedDict):
    """A project's run records in both of its state databases, as one record read them."""

    # The state directories whose database exists, in the order their entries are looked up.
    dirs: tuple[str, ...]
    # Each database's newest run ID when its runs were listed.
    newest: tuple[str | None, ...]
    # Each stage's runs that ran or kept it, newest first across both databases.
    mentions: dict[str, list[StageRunMention]]
    # Each database's last committed write when the view was read or last checked.
    markers: tuple[int, ...]
    # Per stage, the output hashes the run-cache entry of its newest run records, from the first
    # database that has one (None when none has). None as a whole after a write that recorded no
    # run, as during a run: the entries are then looked up per stage.
    first_outputs: dict[str, list[str] | None] | None


class LockOrdering(typing.TypedDict):
    """What one record keeps to tell which of a stage's two locks is current.

    A lock in both of a project's state directories is ordered by the run records of both
    databases (``lock.resolve_stage_lock_path``). The record keeps what that reads, so a command
    opens each database at most once, lists its runs once, and, while nothing writes, orders
    every stage without a database transaction.
    """

    # Each state directory's database, opened read-only at most once per record by the process
    # that holds the entry (its ID, then the data file's inode, then the database), and closed
    # when the record ends. A replaced data file adds an entry; the last one is current.
    databases: "dict[str, list[tuple[int, int, StateDB]]]"
    # Per pair of state directories, both databases' run records (``OrderingView``). Kept while
    # neither database commits a write; listed again when either records a new run, as during a
    # ``--watch`` session.
    views: dict[tuple[str, ...], OrderingView]
    # Per lock in the configured directory: what its answer was read from (both lock files'
    # identities and both databases' last committed writes) and whether the other directory's
    # lock is the current one. A changed file or any write reads the answer again.
    decisions: dict[str, tuple[object, bool]]
    # Configured state directories with a stage whose ``.pivot`` lock is not the current one; the
    # notice naming every such lock is printed once, when the record ends.
    stale: set[pathlib.Path]


class StateDirInvocation(typing.TypedDict):
    """One invocation's state-directory record: what it resolved, and what it reported."""

    # Each project root's resolution, read from its config once per invocation, the check that
    # the path it names can be used included. Everything derived from a state directory is
    # derived from this where it is used, the default cache ``<state dir>/cache`` included
    # (``config.io.locate_cache_dir``).
    resolved: dict[pathlib.Path, StateDirResolution]
    # The directory ``locate_state_dir`` answers for a root whose ``core.state_dir`` is ``''``:
    # the fallback ``state_dir_or_default`` decided for it, once per invocation.
    fallback: dict[pathlib.Path, pathlib.Path]
    # None under --quiet and for a worker's stage: these warnings stay silent.
    reported: StateDirWarnings | None
    # A library entry point's record (``entry_invocation``): it reports only a path that cannot
    # become a directory, once per pipeline load (``report_unusable_path_to_library``).
    library: bool
    # What ordering a stage's two locks needs, kept for the whole record
    # (``lock.resolve_stage_lock_path``).
    lock_ordering: LockOrdering
    # Per lock file decoded for the directory outputs it records (``lock.recorded_directories``):
    # the file's identity when it was decoded (device, inode, size, change time) and what it
    # records. A file whose identity changed is decoded again, as during a ``--watch`` session.
    recorded_locks: "dict[str, tuple[tuple[int, int, int, int], tuple[DirectoryRecord, ...]]]"


# Set by the CLI callback for one CLI invocation, by a worker for one stage, and by a library
# entry point that no invocation covers (``entry_invocation``): an engine run, a pipeline's load
# and each of its fingerprints, discovery, and each ``fingerprint.get_stage_fingerprint_cached``
# call. Outside one (shell completion, a low-level call such as
# ``fingerprint.get_stage_fingerprint`` by hand), every lookup reads and resolves the config
# again and nothing is reported. Nothing else in the process remembers a resolution or anything
# derived from one.
_invocation = contextvars.ContextVar[StateDirInvocation | None](
    "state_dir_invocation", default=None
)


def _reported() -> StateDirWarnings | None:
    invocation = _invocation.get()
    return None if invocation is None else invocation["reported"]


def lock_ordering() -> LockOrdering | None:
    """The current record's lock-ordering state; None outside a record."""
    invocation = _invocation.get()
    return None if invocation is None else invocation["lock_ordering"]


def recorded_locks() -> (
    "dict[str, tuple[tuple[int, int, int, int], tuple[DirectoryRecord, ...]]] | None"
):
    """The current record's lock files read for their recorded directory outputs; None outside
    a record."""
    invocation = _invocation.get()
    return None if invocation is None else invocation["recorded_locks"]


# Called with a record as it ends (``end_invocation``), before the record it replaced returns.
# ``storage.lock`` reports the stale locks the record found and closes the databases it held.
_record_end_hooks = list[Callable[["StateDirInvocation"], None]]()


def on_record_end(hook: Callable[["StateDirInvocation"], None]) -> None:
    """Call ``hook`` with each record as it ends."""
    _record_end_hooks.append(hook)


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
    reader and writer uses this result, never a lexical spelling of the value. A value holding a
    NUL byte names nothing the kernel can open, so it is kept as spelled
    (``state_dir_target_problem`` reports it).
    """
    if "\0" in value:
        return root / value
    return pathlib.Path(os.path.realpath(root / value))


def state_dir_for(root: pathlib.Path) -> pathlib.Path:
    """Return the state directory configured for ``root``.

    ``core.state_dir`` comes from ``<root>/.pivot/config.yaml`` only: a state directory belongs
    to one project, so a global value is ignored, and reported once per invocation when the CLI
    enabled that. A configured value is resolved with ``resolve_state_dir_value``; absent
    configuration keeps state in ``<root>/.pivot``. The ``.pivot`` marker and its configuration
    always remain at the project root. A value that cannot be used raises: ``ConfigError``
    (``EmptyStateDirError`` for ``''``), or ``UnicodeDecodeError`` for a config that is not
    UTF-8. A path that cannot become a directory is returned as the value names it, unreported:
    it is the directory configured, which ``config set`` and ``config unset`` read to tell what
    the old value leaves behind. A command that uses the state directory looks it up with
    ``locate_state_dir``, which answers and reports as the record decided. Within one
    invocation (``begin_invocation``, ``entry_invocation``), each root's config is read and
    resolved once and the record keeps the outcome, an error included, so every later lookup
    answers or raises from it; outside one, every call reads and resolves it again.
    """
    resolution = resolution_for(root)
    if isinstance(resolution, pathlib.Path):
        return resolution
    if isinstance(resolution, exceptions.StateDirTargetError):
        return resolution.path
    # A fresh copy each time, so lookups never share one exception's traceback or context.
    raise copy.copy(resolution)


def resolution_for(root: pathlib.Path) -> StateDirResolution:
    """``root``'s resolution: the record's, or read and resolved now (and kept in the record).

    A worker's record starts with its Pivot process's resolution, so it takes every answer,
    the outcome for a path that cannot be used included, from the same decision.
    """
    invocation = _invocation.get()
    resolution = None if invocation is None else invocation["resolved"].get(root)
    if resolution is None:
        resolution = _resolve_state_dir(root)
        if invocation is not None:
            invocation["resolved"][root] = resolution
    return resolution


def current_resolution(root: pathlib.Path) -> StateDirResolution:
    """``root``'s resolution read from its config now, whatever the current record holds: for a
    check that a resolution kept from earlier (a loaded pipeline's) still holds."""
    return _resolve_state_dir(root)


def _resolve_state_dir(root: pathlib.Path) -> StateDirResolution:
    """Read and resolve ``root``'s ``core.state_dir``: the directory, or why it cannot be used."""
    config_path = root / ".pivot" / "config.yaml"
    try:
        present, configured = config_files.read_config_file_value(config_path, "core.state_dir")
    except (exceptions.ConfigError, UnicodeDecodeError) as e:
        return e.with_traceback(None)
    if configured is None:
        state_dir = root / ".pivot"
    elif is_valid_state_dir_value(configured):
        state_dir = resolve_state_dir_value(root, typing.cast("str", configured))
    elif configured == "":
        return exceptions.EmptyStateDirError(f"core.state_dir in {config_path} must be a path")
    else:
        return exceptions.ConfigError(f"core.state_dir in {config_path} must be a path")

    if state_dir != root / ".pivot":
        target_problem = state_dir_target_problem(state_dir)
        if target_problem is not None:
            # An unusable local state directory owns the one warning, as an unusable value does.
            return exceptions.StateDirTargetError(
                f"{target_problem} (core.state_dir in {config_path})",
                state_dir,
                _ran_in_052(state_dir),
            )
    warn_if_global_state_dir_ignored(root, shadowed=present)
    return state_dir


# What Pivot 0.5.2 opened under the state directory at the start of a state-using command: its
# database, and the directory of its artifact locks. 0.5.2 created each when it was missing.
_052_STATE_ENTRIES = ("state.lmdb", "locks")


def _ran_in_052(state_dir: pathlib.Path) -> bool:
    """Whether Pivot 0.5.2 ran a state-using command with ``state_dir``, which cannot be written.

    0.5.2 opened ``<state_dir>/state.lmdb`` and ``<state_dir>/locks`` there, and created each
    when it was missing, so it ran only where both are already directories it can write. Then
    its state is there and the command's work goes to ``.pivot``: the fallback. Where 0.5.2
    would have had to create one of them, it failed, and so does the command: it gets the path
    and fails where it opens it.
    """
    return all(
        os.path.isdir(state_dir / entry) and os.access(state_dir / entry, os.W_OK | os.X_OK)
        for entry in _052_STATE_ENTRIES
    )


def resolved_state_dir(root: pathlib.Path, resolution: StateDirResolution) -> pathlib.Path:
    """The directory one resolution keeps ``root``'s state in, as ``state_dir_or_default`` does.

    The directory the value names, or the default ``<root>/.pivot`` when the value, or the path
    it names, cannot be used: the directory a pipeline loaded with this resolution records on
    its stages.
    """
    return resolution if isinstance(resolution, pathlib.Path) else root / ".pivot"


def kept_by_pipeline(resolution: StateDirResolution) -> StateDirResolution:
    """The resolution a loaded pipeline keeps for its fingerprints and its runs.

    Its stages record ``.pivot`` for a path that cannot become a directory, so the pipeline keeps
    that path's fallback outcome: every reader in its fingerprints and runs uses ``.pivot`` too,
    whether or not the path can be used by then. Any other resolution is kept as it is.
    """
    if isinstance(resolution, exceptions.StateDirTargetError) and not resolution.fallback:
        return exceptions.StateDirTargetError(str(resolution), resolution.path, True)
    return resolution


def state_dir_or_default(root: pathlib.Path, *, warn: bool = True) -> pathlib.Path:
    """The state directory of a command that 0.5.2 ran without one; nothing is created here.

    The configured directory when ``<root>/.pivot/config.yaml`` can be read, its
    ``core.state_dir`` is a path, and that path is or can become a directory this process
    writes (``state_dir_target_problem``), as the record found them when it resolved the value.
    Otherwise the default ``<root>/.pivot``. Callers that merely display stored configuration
    pass ``warn=False``; a command that reads or writes state uses the warning-enabled resolver
    and then fails where 0.5.2 failed.
    """
    return _state_dir_or_default(root, resolution_for(root), warn=warn)


def _state_dir_or_default(
    root: pathlib.Path, resolution: StateDirResolution, *, warn: bool
) -> pathlib.Path:
    if isinstance(resolution, pathlib.Path):
        return resolution
    config_path = root / ".pivot" / "config.yaml"
    reported = _reported()
    if warn and reported is not None and config_path not in reported["unusable"]:
        reported["unusable"].add(config_path)
        logger.warning(_unusable_message(root, resolution))
    return root / ".pivot"


def _unusable_message(
    root: pathlib.Path, resolution: exceptions.ConfigError | UnicodeDecodeError
) -> str:
    """The one warning about a value, or the path it names, that cannot be used."""
    reason = "The state directory is unknown"
    if isinstance(resolution, UnicodeDecodeError):
        problem = f"{root / '.pivot' / 'config.yaml'} is not valid UTF-8: {resolution}"
    else:
        problem = str(resolution)
        if isinstance(resolution, exceptions.StateDirTargetError):
            reason = "The state directory cannot be used"
    return f"{problem}. {reason}; Pivot uses the default .pivot."


def report_unusable_path_to_library(root: pathlib.Path, resolution: StateDirResolution) -> None:
    """Warn a library caller, at each pipeline load, that the configured path cannot be used.

    The pipeline then keeps ``.pivot`` for its fingerprints (``kept_by_pipeline``), and a run of
    it refuses once the path can be used (``discovery.require_current_state_dirs``), so the
    caller gets the warning a command prints, once per load. Silent under a CLI invocation,
    which reports it once per command, and in a worker.
    """
    invocation = _invocation.get()
    if (
        invocation is not None
        and invocation["library"]
        and isinstance(resolution, exceptions.StateDirTargetError)
    ):
        logger.warning(_unusable_message(root, resolution))


def _unusable_target(
    root: pathlib.Path, resolution: exceptions.StateDirTargetError
) -> pathlib.Path:
    """The state directory of a command that needs one, for a path the record found unusable.

    The record decided it with the value, and nothing checks the path again. ``.pivot``, where
    the stages keep their state, when the path falls back (it holds a database this process can
    write, so 0.5.2 ran the command); otherwise the path, which ``ensure_state_dir`` leaves
    uncreated, so the command fails where it opens it, as 0.5.2 did. Either way the command
    reports the path once, as every command that uses the state directory does.
    """
    default = _state_dir_or_default(root, resolution, warn=True)
    return default if resolution.fallback else resolution.path


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
    search (``W_OK | X_OK``). A path holding a NUL byte cannot be opened at all, as 0.5.2 found
    where it created it.
    """
    if "\0" in os.fspath(target):
        return f"core.state_dir holds a NUL byte, which no path can: {os.fspath(target)!r}"
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


def other_state_dir_for(state_dir: pathlib.Path) -> pathlib.Path | None:
    """The project's other state directory, where state it recorded earlier can still be.

    One resolver answers "where does this project's recorded state live for reading", in both
    directions. A reader that misses an entry under ``state_dir`` looks there; a writer never
    does. ``.pivot`` for the configured directory: Pivot 0.5.2 kept its state there, and so
    does a run made before ``core.state_dir`` was set or while it cannot be used. The
    configured path for ``.pivot``, when the record fell back to ``.pivot`` because that path
    cannot be written although it still holds this project's state
    (``StateDirTargetError.fallback``): reading it needs no write permission, so a run
    recorded there stays readable. Returns None when this project has no other directory.
    """
    try:
        root = project.get_project_root()
    except exceptions.ProjectNotInitializedError:
        return None
    resolution = resolution_for(root)
    default = root / ".pivot"
    if isinstance(resolution, exceptions.StateDirTargetError):
        if resolution.fallback and state_dir == default:
            return resolution.path
        configured = resolution.path
    elif isinstance(resolution, pathlib.Path):
        configured = resolution
    else:
        return None
    if configured != state_dir or configured == default:
        return None
    return default


def other_stages_dir_for(stages_dir: pathlib.Path) -> pathlib.Path | None:
    """The other stages directory of the project whose locks are in ``stages_dir``."""
    other = other_state_dir_for(stages_dir.parent)
    return None if other is None else other / "stages"


def other_cache_files_dir_for(cache_files_dir: pathlib.Path) -> pathlib.Path | None:
    """The other cache-object directory paired with a default configured cache."""
    other = other_state_dir_for(cache_files_dir.parent.parent)
    if other is None:
        return None
    return other / "cache" / "files"


def report_legacy_state_dir(state_dir: pathlib.Path) -> None:
    """Warn once per invocation that reads fall back to ``.pivot`` for ``state_dir``.

    ``.pivot`` holds stage locks or cache objects while ``core.state_dir`` names another
    directory. Earlier Pivot releases kept them there, and so does a run while
    ``core.state_dir`` is unset or cannot be used (``''``, or a path that cannot become a
    directory). The files cannot tell these apart, so the notice names them all. When
    ``.pivot/stages`` exists, it says how to move the locks: with ``mv`` and absolute paths,
    which works whether or not git tracks them and from any directory, while ``state_dir`` is a
    directory without ``stages``; otherwise ``mv`` would put the directory inside the existing
    one, or fail, so it says to move the lock files into ``<state_dir>/stages``. Reading the
    other way round, from ``.pivot`` into a configured path that cannot be written, says
    nothing: the command already reported that path once.
    """
    reported = _reported()
    if reported is None or state_dir in reported["legacy"]:
        return
    legacy = other_state_dir_for(state_dir)
    if legacy is None or legacy != project.get_project_root() / ".pivot":
        return
    reported["legacy"].add(state_dir)
    source, destination = legacy / "stages", state_dir / "stages"
    delete = ", or delete them if a run that could not use core.state_dir made them"
    if not os.path.isdir(source):
        advice = ""
    elif os.path.isdir(state_dir) and not os.path.lexists(destination):
        command = f"mv {shlex.quote(str(source))} {shlex.quote(str(destination))}"
        advice = f"; move the locks with `{command}`{delete}"
    else:
        advice = f"; move the lock files under {source} into {destination}{delete}"
    logger.warning(
        "core.state_dir is %s, but %s holds stage locks or cache objects of this project: earlier Pivot releases kept them there, and so does a run while core.state_dir is unset or cannot be used (an empty value, or a path that is not a writable directory). Pivot reads missing locks and cache entries there%s.",
        state_dir,
        legacy,
        advice,
    )


def report_stale_locks(
    state_dir: pathlib.Path,
    find_stale: Callable[[pathlib.Path, pathlib.Path], tuple[list[pathlib.Path], bool] | None],
) -> None:
    """Warn once per invocation that ``.pivot/stages`` holds locks the configured ones replaced.

    ``find_stale(stages_dir, legacy_stages_dir)`` names each lock in ``<root>/.pivot/stages``
    whose stage also has a lock in ``<state_dir>/stages`` and whose run records make that one
    current, and says whether those are every lock there; None when it cannot tell. A run made
    before ``core.state_dir`` was set, or while it could not be used, left such a lock. Pivot
    reads neither outputs nor parameters from it, but reads both directories' run records to
    find which lock is current, on every command that reads the stage. So the notice gives the
    exact command that removes them: ``rm -r`` of ``.pivot/stages`` when every lock there is
    stale, or ``rm`` of each stale lock, which has to come before moving any other lock out of
    ``.pivot/stages`` (``report_legacy_state_dir``) so that none replaces a current one. It is
    looked for only once per invocation, and only in this direction: a configured lock older
    than its ``.pivot`` lock is the notice about locks in ``.pivot``.
    """
    reported = _reported()
    if reported is None or state_dir in reported["stale"]:
        return
    legacy = other_state_dir_for(state_dir)
    if legacy is None or legacy != project.get_project_root() / ".pivot":
        return
    reported["stale"].add(state_dir)
    source = legacy / "stages"
    found = find_stale(state_dir / "stages", source)
    if found is None or not found[0]:
        return
    stale, every_lock = found
    if every_lock:
        command = f"rm -r {shlex.quote(str(source))}"
        order = ""
    else:
        command = "rm " + " ".join(shlex.quote(str(path)) for path in stale)
        order = f", before moving any other lock out of {source}"
    stages = "stage" if len(stale) == 1 else "stages"
    logger.warning(
        "core.state_dir is %s, and %s also holds older locks of %d %s whose current lock is in %s: a run made before core.state_dir was set, or while it could not be used, left them. Pivot uses none of them, but reads both directories' run records to tell which lock is current on every command that reads these stages; remove them with `%s`%s.",
        state_dir,
        source,
        len(stale),
        stages,
        state_dir / "stages",
        command,
        order,
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
    *,
    report: bool,
    state_dirs: Mapping[pathlib.Path, StateDirResolution] | None = None,
    library: bool = False,
) -> contextvars.Token[StateDirInvocation | None]:
    """Start one invocation's record, used until the returned token is reset.

    Its resolutions start with ``state_dirs`` and it reads the config of any other project
    afresh. A worker's stage passes the resolution its Pivot process decided for the command,
    the outcome for a path that cannot be used included, so the worker reads every state file
    where its Pivot process does, and a ``core.state_dir`` edited during a ``--watch`` session
    never reaches it. With ``report``, each state-directory problem is warned once; without it,
    none is, except that a ``library`` record reports an unusable path at each pipeline load.
    """
    reported = (
        StateDirWarnings(
            ignored_global=set[pathlib.Path](),
            unusable=set[pathlib.Path](),
            legacy=set[pathlib.Path](),
            cache_stage_overlap=set[tuple[pathlib.Path, pathlib.Path]](),
            stale=set[pathlib.Path](),
        )
        if report
        else None
    )
    return _invocation.set(
        StateDirInvocation(
            resolved=dict[pathlib.Path, StateDirResolution](state_dirs or {}),
            fallback=dict[pathlib.Path, pathlib.Path](),
            reported=reported,
            library=library,
            lock_ordering=LockOrdering(
                databases=dict[str, list[tuple[int, int, "StateDB"]]](),
                views=dict[tuple[str, ...], OrderingView](),
                decisions=dict[str, tuple[object, bool]](),
                stale=set[pathlib.Path](),
            ),
            recorded_locks=dict[
                str, tuple[tuple[int, int, int, int], tuple["DirectoryRecord", ...]]
            ](),
        )
    )


def end_invocation(token: contextvars.Token[StateDirInvocation | None]) -> None:
    """End an invocation, restoring whatever record ``begin_invocation`` replaced.

    The record's end hooks (``on_record_end``) run first, while it is still the current record.
    """
    invocation = _invocation.get()
    try:
        if invocation is not None:
            for hook in _record_end_hooks:
                hook(invocation)
    finally:
        _invocation.reset(token)


@contextlib.contextmanager
def entry_invocation(
    state_dirs: Mapping[pathlib.Path, StateDirResolution] | None = None,
) -> Iterator[Mapping[pathlib.Path, StateDirResolution]]:
    """Hold one library entry point's record for its whole call, when no invocation holds one.

    A library caller has no CLI invocation. Each public entry point that produces cache entries
    or writes state (an engine run, discovery, a pipeline's load, ``Pipeline.ensure_fingerprint``,
    ``fingerprint.get_stage_fingerprint_cached``) holds this record, so every lookup in the
    call, in the threads that copy this context as well, reads the record: each root's
    ``core.state_dir`` is read and resolved once per call, as the kernel opens it then, the
    check that the path can be a directory included, and an edit of the value during the call
    moves nothing. The record starts with ``state_dirs``; a pipeline passes the resolutions it
    made when it was loaded, a value it could not use included and a path it could not use as
    its ``.pivot`` fallback (``kept_by_pipeline``), so its fingerprints and its runs use the
    directory its stages' locks are in and read no config. A CLI invocation, or an entry point
    already in progress, holds the record for the whole call, and this leaves it alone. The
    record reports one thing: each pipeline loaded under it warns once when the path
    ``core.state_dir`` names cannot be used (``report_unusable_path_to_library``), as a command
    would; a library caller's other state-directory warnings stay silent.

    Yields the record's resolutions by project root, so the caller can keep them.
    """
    invocation = _invocation.get()
    if invocation is not None:
        yield invocation["resolved"]
        return
    token = begin_invocation(report=False, state_dirs=state_dirs, library=True)
    try:
        yield typing.cast("StateDirInvocation", _invocation.get())["resolved"]
    finally:
        end_invocation(token)


def clear_state_dir_cache() -> None:
    """Forget this invocation's resolved state directories, so the next lookup reads the config."""
    invocation = _invocation.get()
    if invocation is not None:
        invocation["resolved"].clear()
        invocation["fallback"].clear()


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
    """Create ``state_dir`` and its default ignore file when this call makes the directory.

    A path holding a NUL byte cannot become a directory, so creating it fails with an
    ``OSError``, as for any path that cannot (``state_dir_target_problem``), not with the
    ``ValueError`` the kernel interface raises: a best-effort writer then skips it as it skips
    every other unusable state directory.
    """
    if "\0" in os.fspath(state_dir):
        raise OSError(errno.EINVAL, "core.state_dir holds a NUL byte", os.fspath(state_dir))
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
    is written inside it. The decision is made on the path the kernel will open
    (``os.path.realpath``), which is how the configured value itself is resolved
    (``resolve_state_dir_value``), so one destination gets one answer however it is spelled: a
    ``..`` segment, a relative path from a subdirectory of the project, or a path through a
    symbolic link to the project. Outside a project, or outside its state directory, this only
    creates ``path``.
    """
    if not path.is_dir():
        try:
            root = project.get_project_root()
            state_dir = state_dir_or_default(root)
            resolved = pathlib.Path(os.path.realpath(path))
        except (exceptions.ProjectNotInitializedError, OSError):
            pass
        else:
            if state_dir != root / ".pivot" and resolved.is_relative_to(state_dir):
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
    ``state_dir_or_default``, decided once per invocation and kept in its record. A path that
    cannot become a directory answers as the record decided it (``_unusable_target``). Other
    invalid config values still raise ``ConfigError``.
    """
    resolution = resolution_for(root)
    if isinstance(resolution, pathlib.Path):
        return resolution
    if isinstance(resolution, exceptions.StateDirTargetError):
        return _unusable_target(root, resolution)
    if not isinstance(resolution, exceptions.EmptyStateDirError):
        raise copy.copy(resolution)
    invocation = _invocation.get()
    if invocation is None:
        return _state_dir_or_default(root, resolution, warn=True)
    fallback = invocation["fallback"].get(root)
    if fallback is None:
        fallback = _state_dir_or_default(root, resolution, warn=True)
        invocation["fallback"][root] = fallback
    return fallback


def ensure_state_dir(root: pathlib.Path) -> pathlib.Path:
    """``locate_state_dir``, created when absent.

    A directory this call creates gets the default ``.gitignore``. One that already exists is
    left as it is, so a deleted ``.pivot/.gitignore`` stays deleted until ``init`` writes it. A
    path the record found unable to become a directory is left uncreated: the command gets
    ``.pivot`` or the path, as the record decided, and fails where it opens the path, as 0.5.2
    did.
    """
    state_dir = locate_state_dir(root)
    if (
        os.path.isdir(state_dir)
        or not (root / ".pivot").is_dir()
        or isinstance(resolution_for(root), exceptions.StateDirTargetError)
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

"""Per-stage lock files for tracking pipeline state.

StageLock provides persistent lock files (.lock) for change detection,
storing fingerprints, params, and hashes to detect when re-runs are needed.
A lock records a directory by its listing's hash, its total size and its file count
(``listing``, ``size``, ``nfiles``), and then keeps all its entries under ``records``
(``_RECORDS_KEY``); the directory's listing is committed beside the lock
(``storage/listing.py``), and ``StageLock.write`` writes both. A stage that keeps an
incremental output commits a restart proof beside its lock (``restart_proof_path``), so an
older pivot's rewrite of that lock is caught (``restart_proofs``). A lock an older pivot wrote
is read as it is (``StageLock.read``).

For runtime mutual exclusion during concurrent execution, see
``pivot.storage.artifact_lock`` which uses flock-based artifact locks.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import os
import pathlib
import posixpath
import re
import shlex
import stat
import threading
from typing import TYPE_CHECKING, NamedTuple, cast, final, override

import yaml

from pivot import exceptions, path_policy, yaml_config
from pivot import state_dir as state_dir_mod
from pivot.storage import cache, listing
from pivot.types import (
    ArtifactIdentity,
    ArtifactRef,
    ArtifactTag,
    DepEntry,
    DirHash,
    FileHash,
    HashInfo,
    LockData,
    OutEntry,
    StageStatus,
    StorageLockData,
    identity_from_key,
    identity_key,
    is_dir_hash,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator, Mapping

    from pivot.storage import state

logger = logging.getLogger(__name__)

_VALID_STAGE_NAME = re.compile(
    r"[a-zA-Z0-9_@./-]+"
)  # Allow / for pipeline-prefixed names, . for DVC matrix keys
_VALID_VARIANT_NAME = re.compile(r"[a-zA-Z0-9_.-]+")
_MAX_STAGE_NAME_LEN = 200  # Leave room for ".lock" suffix within filesystem NAME_MAX (255)
_REQUIRED_LOCK_KEYS = frozenset({"code_manifest", "params", "deps", "outs"})
_LOCK_SUFFIX = ".lock"
_DIR_HASH_KEY = "listing"
"""The key a lock entry records a directory's hash under; a file's is ``hash``. Every released
pivot reads an entry that has a ``hash`` and no inline ``manifest`` as one file, and would
restore a directory's listing in the directory's place, so no directory record has a ``hash``."""
_RECORDS_KEY = "records"
"""Where a lock that records a directory keeps its ``deps`` and ``outs``, leaving the top-level
``deps`` and ``outs`` empty. Every released pivot (0.5.2 … 0.5.11 read alike) validates those
empty lists as a lock that records nothing: it reruns an ordinary stage from its inputs, and for
a stage with an incremental output it stops with "Cache missing for incremental output" instead
of restarting that output from empty, which it would do for a lock it read as missing. A lock
that records only files keeps them at the top level, where those pivots read them as before."""
_READS_KEY = "reads"
"""Where a lock lists the objects its run read by hash (``pivot.read_by_hash``): a top-level list
of content hashes, sorted, each once, written only when the run read one. Every released pivot
allows an unknown top-level key (``lock_data_error``), so it reads such a lock as it reads any
other, and its own records stay where it looks for them."""
_RESTART_PROOF_SUFFIX = ".incremental"
"""The suffix of the file this pivot writes beside the lock of a stage that keeps an incremental
output and commits with it (``restart_proof_path``). No released pivot writes one, so this file
beside a lock an older pivot wrote (``older_format``) is proof that this pivot wrote that stage's
lock first and the older pivot replaced it — removing the lock and restarting the output from
empty, or rerunning the stage — so the lock it left is not the stage's history
(``restart_proofs``)."""
_RESTART_PROOF_TEXT = """\
# Pivot wrote this file beside the lock of a stage that keeps an incremental output, and reads
# it to tell its own lock from one an older pivot wrote in that lock's place, which may have
# restarted the output from empty. Commit it with the lock; it is the same in every project.
"""

SCHEMA_VERSION = 3


def stages_rel_path(state_dir: pathlib.Path, root: pathlib.Path) -> str:
    """Return the git-relative directory holding ``state_dir``'s lock files."""
    return (state_dir / "stages").resolve().relative_to(root.resolve()).as_posix()


def resolve_stage_lock_path(stages_dir: pathlib.Path, stage_name: str) -> pathlib.Path:
    """Return the stage's current lock in either of the project's state directories.

    A lock in only one directory is current. When both exist, run records decide which was
    written last: a fallback run in legacy ``.pivot`` must not be hidden by an older configured
    lock when the configured directory becomes writable again.
    """
    return _resolve_stage_lock(stages_dir, stage_name)[0]


def _resolve_stage_lock(
    stages_dir: pathlib.Path, stage_name: str
) -> tuple[pathlib.Path, pathlib.Path, object | None]:
    """Return the current lock path, its directory, and parsed YAML when ordering read it, with
    the notices that choice calls for (``_choose_stage_lock``)."""
    choice = _choose_stage_lock(
        stages_dir, stage_name, state_dir_mod.other_stages_dir_for(stages_dir)
    )
    if choice.stale:
        ordering = state_dir_mod.lock_ordering()
        if ordering is not None:
            ordering["stale"].add(stages_dir.parent)
    if choice.legacy:
        state_dir_mod.report_legacy_state_dir(stages_dir.parent)
    ordered = None if choice.ordered is None else choice.ordered.valid_document()
    return choice.path, choice.stages_dir, ordered


class _LockChoice(NamedTuple):
    """A stage's current lock in either of the project's state directories."""

    path: pathlib.Path
    stages_dir: pathlib.Path
    ordered: _LockFile | None
    """The chosen file, when the stage has a lock in each directory and run records ordered
    them; None when it has one lock."""
    legacy: bool
    """The current lock is in the other directory (``state_dir.report_legacy_state_dir``)."""
    stale: bool
    """The other directory holds a lock that is not the current one (the stale notice)."""


def _choose_stage_lock(
    stages_dir: pathlib.Path, stage_name: str, other_stages_dir: pathlib.Path | None
) -> _LockChoice:
    """Which of a stage's locks is current, in ``stages_dir`` or ``other_stages_dir`` (the
    project's other stages directory, ``state_dir.other_stages_dir_for``), saying nothing."""
    primary_path = stages_dir / f"{stage_name}{_LOCK_SUFFIX}"
    if other_stages_dir is None:
        return _LockChoice(primary_path, stages_dir, None, legacy=False, stale=False)
    other_path = other_stages_dir / f"{stage_name}{_LOCK_SUFFIX}"
    other_identity = _lock_identity(other_path)
    if other_identity is None:
        return _LockChoice(primary_path, stages_dir, None, legacy=False, stale=False)
    primary_identity = _lock_identity(primary_path)
    if primary_identity is None:
        return _LockChoice(other_path, other_stages_dir, None, legacy=True, stale=False)
    primary = _LockFile(primary_path, primary_identity)
    other = _LockFile(other_path, other_identity)
    if _other_lock_is_current(
        stage_name, primary, other, stages_dir.parent, other_stages_dir.parent
    ):
        return _LockChoice(other_path, other_stages_dir, other, legacy=True, stale=False)
    return _LockChoice(primary_path, stages_dir, primary, legacy=False, stale=True)


# A lock file's device, inode, size and change time.
type _LockIdentity = tuple[int, int, int, int]


def _lock_identity(path: pathlib.Path) -> _LockIdentity | None:
    """Return a regular lock's identity, or None when the path is not a regular file."""
    try:
        info = os.stat(path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return info.st_dev, info.st_ino, info.st_size, info.st_ctime_ns


@final
class _LockFile:
    """A lock file read without a word (for ordering, and for other stages' recorded directories),
    decoded at most once by ``parse_committed``."""

    path: pathlib.Path
    identity: _LockIdentity
    _document: object | None
    _committed: CommittedLock | None
    _output_hashes: list[str] | None
    _read: bool

    def __init__(self, path: pathlib.Path, identity: _LockIdentity) -> None:
        self.path = path
        self.identity = identity
        self._document = None
        self._committed = None
        self._output_hashes = None
        self._read = False

    def committed(self) -> CommittedLock | None:
        """The decoded lock, or None when any error naming, reading or decoding it occurs."""
        if not self._read:
            self._read = True
            try:
                with open(self.path) as f:
                    document: object = yaml.load(f, Loader=yaml_config.Loader)
                parsed = parse_committed(document)
            except (OSError, UnicodeError, yaml.YAMLError, ValueError, exceptions.PivotError):
                return None
            if isinstance(parsed, CommittedLock):
                self._document = document
                self._committed = parsed
                self._output_hashes = sorted(out["hash"] for out in parsed.data["outs"])
        return self._committed

    def output_hashes(self) -> list[str] | None:
        """Return the lock's output hashes, or None when it cannot be read as a lock."""
        _ = self.committed()
        return self._output_hashes

    def valid_document(self) -> object | None:
        """Return the parsed YAML if it was a valid lock, otherwise None."""
        _ = self.committed()
        return self._document


# ``StateDB.list_runs`` keeps at most this many runs (`prune_runs`).
_ALL_RUNS = 1 << 31

# Guards databases held by one state-directory invocation.
_held_databases_lock = threading.Lock()


def _other_lock_is_current(
    stage_name: str,
    primary: _LockFile,
    other: _LockFile,
    primary_state_dir: pathlib.Path,
    other_state_dir: pathlib.Path,
) -> bool:
    """Whether the lock in the other state directory is current."""
    import lmdb

    from pivot.storage import state

    try:
        return _order_locks(stage_name, primary, other, primary_state_dir, other_state_dir)
    except (exceptions.PivotError, state.StateDBError, lmdb.Error, OSError, ValueError):
        logger.debug("Cannot order the locks of %s; reading %s", stage_name, primary.path)
        return False


def _order_locks(
    stage_name: str,
    primary: _LockFile,
    other: _LockFile,
    primary_state_dir: pathlib.Path,
    other_state_dir: pathlib.Path,
) -> bool:
    with contextlib.ExitStack() as stack:
        databases = _ordering_databases(stack, (primary_state_dir, other_state_dir))
        view = _ordering_view(databases)
        return _order_with(stage_name, primary, other, databases, view, other_state_dir)


def _order_with(
    stage_name: str,
    primary: _LockFile,
    other: _LockFile,
    databases: list[tuple[pathlib.Path, state.StateDB]],
    view: state_dir_mod.OrderingView,
    other_state_dir: pathlib.Path,
) -> bool:
    """Order two lock files, retaining an answer while their inputs stay unchanged."""
    ordering = state_dir_mod.lock_ordering()
    primary_key = os.fspath(primary.path)
    key: object = None
    if ordering is not None:
        key = (view["dirs"], view["markers"], primary.identity, other.identity)
        kept = ordering["decisions"].get(primary_key)
        if kept is not None and kept[0] == key:
            return kept[1]
    answer = _answer(stage_name, primary, other, databases, view, os.fspath(other_state_dir))
    if ordering is not None:
        ordering["decisions"][primary_key] = (key, answer)
    return answer


def _answer(
    stage_name: str,
    primary: _LockFile,
    other: _LockFile,
    databases: list[tuple[pathlib.Path, state.StateDB]],
    view: state_dir_mod.OrderingView,
    other_state_dir: str,
) -> bool:
    """Order two locks by the stage's newest reusable runs."""
    mentions = view["mentions"].get(stage_name)
    if not mentions:
        return False
    first_outputs = view["first_outputs"]
    for index, mention in enumerate(mentions):
        if index == 0 and first_outputs is not None:
            outputs = first_outputs[stage_name]
        else:
            outputs = _run_cache_outputs(stage_name, mention["input_hash"], databases)
        if outputs is None:
            break
        if outputs == primary.output_hashes():
            return False
        if outputs == other.output_hashes():
            return True
    newest = mentions[0]
    return newest["ran"] and newest["state_dir"] == other_state_dir


def _ordering_databases(
    stack: contextlib.ExitStack, state_dirs: tuple[pathlib.Path, pathlib.Path]
) -> list[tuple[pathlib.Path, state.StateDB]]:
    """Open each existing state database once for the current record."""
    from pivot.storage import state

    ordering = state_dir_mod.lock_ordering()
    databases = list[tuple[pathlib.Path, state.StateDB]]()
    for state_dir in state_dirs:
        if ordering is not None:
            held = _held_database(ordering, state_dir)
            if held is not None:
                databases.append((state_dir, held))
        elif (state_dir / "state.lmdb").is_dir():
            databases.append(
                (state_dir, stack.enter_context(state.StateDB(state_dir, readonly=True)))
            )
    return databases


def _held_database(
    ordering: state_dir_mod.LockOrdering, state_dir: pathlib.Path
) -> state.StateDB | None:
    """Return this invocation's database for ``state_dir``, opening it if needed."""
    from pivot.storage import state

    pid = os.getpid()
    key = os.fspath(state_dir)
    data_file = os.path.join(key, "state.lmdb", "data.mdb")
    current = _current_held(ordering, key, pid, data_file)
    if current is not None:
        return current
    with _held_databases_lock:
        current = _current_held(ordering, key, pid, data_file)
        if current is not None:
            return current
        if not os.path.isdir(os.path.join(key, "state.lmdb")):
            return None
        database = state.StateDB(state_dir, readonly=True)
        ordering["databases"].setdefault(key, []).append((pid, os.stat(data_file).st_ino, database))
        return database


def _current_held(
    ordering: state_dir_mod.LockOrdering, key: str, pid: int, data_file: str
) -> state.StateDB | None:
    """Return the current process's still-current held database for ``key``."""
    held = ordering["databases"].get(key)
    if not held:
        return None
    opener, inode, database = held[-1]
    if opener != pid:
        return None
    try:
        return database if os.stat(data_file).st_ino == inode else None
    except FileNotFoundError:
        return None


def _ordering_view(
    databases: list[tuple[pathlib.Path, state.StateDB]],
) -> state_dir_mod.OrderingView:
    """Return the record's current view of runs in both state databases."""
    dirs = tuple(os.fspath(state_dir) for state_dir, _database in databases)
    markers = tuple(database.last_change() for _state_dir, database in databases)
    ordering = state_dir_mod.lock_ordering()
    view = None if ordering is None else ordering["views"].get(dirs)
    if view is not None and view["markers"] == markers:
        return view
    newest = tuple(database.newest_run_id() for _state_dir, database in databases)
    if view is not None and view["newest"] == newest:
        view = state_dir_mod.OrderingView(
            dirs=dirs,
            newest=newest,
            mentions=view["mentions"],
            markers=markers,
            first_outputs=None,
        )
    else:
        view = _list_runs(databases, dirs, newest, markers, prefetch=ordering is not None)
    if ordering is not None:
        ordering["views"][dirs] = view
    return view


def _list_runs(
    databases: list[tuple[pathlib.Path, state.StateDB]],
    dirs: tuple[str, ...],
    newest: tuple[str | None, ...],
    markers: tuple[int, ...],
    *,
    prefetch: bool,
) -> state_dir_mod.OrderingView:
    """List runs from both databases and, for a record, prefetch their newest entries."""
    mentions = dict[str, list[state_dir_mod.StageRunMention]]()
    for (_state_dir, database), state_dir in zip(databases, dirs, strict=True):
        for manifest in database.list_runs(limit=_ALL_RUNS):
            for name, record in manifest["stages"].items():
                input_hash = record["input_hash"]
                status = record["status"]
                if input_hash is None or status not in (StageStatus.RAN, StageStatus.CACHED):
                    continue
                mentions.setdefault(name, []).append(
                    state_dir_mod.StageRunMention(
                        run_id=manifest["run_id"],
                        input_hash=input_hash,
                        ran=status == StageStatus.RAN,
                        state_dir=state_dir,
                    )
                )
    for runs in mentions.values():
        runs.sort(key=lambda mention: mention["run_id"], reverse=True)
    if not prefetch:
        return state_dir_mod.OrderingView(
            dirs=dirs, newest=newest, mentions=mentions, markers=markers, first_outputs=None
        )
    first_outputs = dict[str, list[str] | None]()
    pending = dict[str, str]()
    for name, runs in mentions.items():
        first_outputs[name] = None
        pending[name] = runs[0]["input_hash"]
    for _state_dir, database in databases:
        if not pending:
            break
        for (name, _input_hash), entry in database.lookup_run_cache_many(pending.items()).items():
            first_outputs[name] = sorted(output["hash"] for output in entry["output_hashes"])
            del pending[name]
    return state_dir_mod.OrderingView(
        dirs=dirs,
        newest=newest,
        mentions=mentions,
        markers=markers,
        first_outputs=first_outputs,
    )


def _run_cache_outputs(
    stage_name: str, input_hash: str, databases: list[tuple[pathlib.Path, state.StateDB]]
) -> list[str] | None:
    """Return one run-cache entry's output hashes from either state database."""
    for _state_dir, database in databases:
        entry = database.lookup_run_cache(stage_name, input_hash)
        if entry is not None:
            return sorted(output["hash"] for output in entry["output_hashes"])
    return None


def _end_lock_ordering(invocation: state_dir_mod.StateDirInvocation) -> None:
    """Report the stale legacy locks a record found and close its held databases."""
    ordering = invocation["lock_ordering"]
    try:
        if invocation["reported"] is not None:
            for state_dir in sorted(ordering["stale"]):
                state_dir_mod.report_stale_locks(state_dir, _find_stale_locks)
    finally:
        with _held_databases_lock:
            held = [entry for entries in ordering["databases"].values() for entry in entries]
            ordering["databases"].clear()
        pid = os.getpid()
        for opener, _inode, database in held:
            if opener == pid:
                database.close()


def _find_stale_locks(
    stages_dir: pathlib.Path, other_stages_dir: pathlib.Path
) -> tuple[list[pathlib.Path], bool] | None:
    """Return legacy locks the configured locks replaced, or None if they cannot be ordered."""
    import lmdb

    from pivot.storage import state

    state_dir, other_state_dir = stages_dir.parent, other_stages_dir.parent
    try:
        names = sorted(
            posixpath.join(os.path.relpath(directory, other_stages_dir), name)
            .removeprefix("./")
            .removesuffix(_LOCK_SUFFIX)
            for directory, _subdirectories, files in os.walk(other_stages_dir)
            for name in files
            if name.endswith(_LOCK_SUFFIX)
        )
        stale = list[pathlib.Path]()
        with contextlib.ExitStack() as stack:
            databases = _ordering_databases(stack, (state_dir, other_state_dir))
            view = _ordering_view(databases)
            for stage_name in names:
                if stage_name_error(stage_name) is not None:
                    continue
                primary_path = stages_dir / f"{stage_name}{_LOCK_SUFFIX}"
                other_path = other_stages_dir / f"{stage_name}{_LOCK_SUFFIX}"
                primary_identity = _lock_identity(primary_path)
                other_identity = _lock_identity(other_path)
                if primary_identity is None or other_identity is None:
                    continue
                primary = _LockFile(primary_path, primary_identity)
                other = _LockFile(other_path, other_identity)
                if not _order_with(stage_name, primary, other, databases, view, other_state_dir):
                    stale.append(other_path)
    except (exceptions.PivotError, state.StateDBError, lmdb.Error, OSError, ValueError):
        logger.debug("Cannot order the locks under %s; not reporting stale locks", other_stages_dir)
        return None
    return stale, len(stale) == len(names)


state_dir_mod.on_record_end(_end_lock_ordering)


def get_stages_dir(state_dir: pathlib.Path) -> pathlib.Path:
    """Return the state directory's git-tracked lock file directory."""
    return state_dir / "stages"


def restart_proof_path(stages_dir: pathlib.Path, stage_name: str) -> pathlib.Path:
    """Where pivot records that it wrote ``stage_name``'s lock, beside that lock, for a stage
    that keeps an incremental output (``_RESTART_PROOF_SUFFIX``). A stage name is short enough
    for a lock file name, so it is short enough for this one."""
    return stages_dir / f"{stage_name}{_RESTART_PROOF_SUFFIX}"


def _write_restart_proof(path: pathlib.Path) -> None:
    """Write the restart proof at ``path`` atomically, leaving one that holds its bytes
    untouched: the file belongs to the project's history, and every run writes the same bytes."""
    data = _RESTART_PROOF_TEXT.encode()
    try:
        if path.read_bytes() == data:
            return
    except OSError:
        pass

    def write_bytes(fd: int) -> None:
        with os.fdopen(fd, "wb") as f:
            _ = f.write(data)

    cache.atomic_write_file(path, write_bytes)


def stage_name_error(name: str) -> str | None:
    """Return why ``name`` cannot name a lock file, or None if it can.

    This is the single rule for stage, pipeline and variant names. A name is a relative path
    of non-empty segments under the state directory's ``stages`` directory: no leading or doubled
    ``/``, and no ``.`` or ``..`` segment, so the lock file can never land outside that directory.
    """
    if not name or not _VALID_STAGE_NAME.fullmatch(name):
        return "use only letters, digits and the characters _ @ . / -"
    segments = name.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        return "'/'-separated parts must be non-empty and not '.' or '..'"
    if len(name) > _MAX_STAGE_NAME_LEN:
        return f"too long ({len(name)} > {_MAX_STAGE_NAME_LEN} characters)"
    return None


def variant_name_error(name: str) -> str | None:
    """Return why ``name`` cannot be a variant name, or None if it can.

    A variant is one part of a stage name, so it follows the stage-name rule without ``/``
    or ``@``: those separate a stage name's pipeline and variant parts, and a variant
    containing them could name the same lock file and outputs as another stage.
    """
    if not _VALID_VARIANT_NAME.fullmatch(name):
        return "use only letters, digits and the characters _ . -"
    return stage_name_error(name)


@dataclasses.dataclass(frozen=True)
class UnsafeLockPath:
    """A recorded path that is not project-relative POSIX form."""

    field: str
    value: str

    @override
    def __str__(self) -> str:
        return f"{self.field}: {self.value!r} is not a project-relative POSIX path"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_count(value: object) -> bool:
    return _is_int(value) and cast("int", value) >= 0


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _path_error(value: object, field: str) -> str | UnsafeLockPath | None:
    if not isinstance(value, str) or not value:
        return field
    if path_policy.recorded_path_error(value) is not None:
        return UnsafeLockPath(field, value)
    return None


def _is_str_mapping(value: object, *, str_values: bool) -> bool:
    if not isinstance(value, dict):
        return False
    mapping = cast("dict[object, object]", value)
    if not all(isinstance(key, str) for key in mapping):
        return False
    return not str_values or all(isinstance(item, str) for item in mapping.values())


def _is_hash_list(value: object) -> bool:
    return isinstance(value, list) and all(
        cache.is_valid_hash(item) for item in cast("list[object]", value)
    )


def _entry_error(
    raw: object, field: str, list_key: str, *, listed: bool
) -> str | UnsafeLockPath | None:
    """Check one ``deps`` or ``outs`` entry as a lock file holds it: a file's hash under
    ``hash``, a directory's under ``listing`` (``_DIR_HASH_KEY``) beside its size and file
    count. A directory entry stands only under ``records`` (``listed``, ``_RECORDS_KEY``)."""
    if not isinstance(raw, dict):
        return field
    entry = cast("dict[str, object]", raw)
    # Every recorded hash is a cache key (cache.get_cache_path). Null and empty hashes come
    # from stages never executed locally (e.g. pulled with incomplete state); a hand-edited
    # short or non-hex one would otherwise fail every restore instead of meaning stale.
    if _DIR_HASH_KEY in entry:
        if not listed:
            return f"{field}.{_DIR_HASH_KEY}"
        if "hash" in entry:
            return f"{field}.hash"
        if not cache.is_valid_hash(entry[_DIR_HASH_KEY]):
            return f"{field}.{_DIR_HASH_KEY}"
        if not _is_count(entry.get("nfiles")):
            return f"{field}.nfiles"
        if not _is_count(entry.get("size")):
            return f"{field}.size"
    else:
        if not cache.is_valid_hash(entry.get("hash")):
            return f"{field}.hash"
        if "nfiles" in entry:
            return f"{field}.nfiles"
        if "size" in entry and not _is_count(entry["size"]):
            return f"{field}.size"
    if list_key == "deps" and not _is_nonempty_str(entry.get("producer")):
        return f"{field}.producer"
    if "key" not in entry or not (entry["key"] is None or isinstance(entry["key"], str)):
        return f"{field}.key"
    if (
        list_key == "outs"
        and "path" in entry
        and (error := _path_error(entry["path"], f"{field}.path"))
    ):
        return error
    return None


def lock_data_error(data: object) -> str | UnsafeLockPath | None:
    """Return the first reason a parsed lock file cannot safely be consumed, or None.

    The reason is the first field that breaks this format, or an ``UnsafeLockPath`` for a
    recorded path that is not project-relative POSIX form. A lock that fails is treated as
    missing by every caller ("no lock = needs re-run"), so a hand-edited or partially written
    lock makes the stage stale instead of crashing a reader or letting status and repro
    disagree. Extra top-level keys are allowed for forward compatibility.
    """
    if not isinstance(data, dict):
        return "<document>"
    document = cast("dict[object, object]", data)
    if document.get("schema_version") != SCHEMA_VERSION:
        return "schema_version"
    for key in sorted(_REQUIRED_LOCK_KEYS):
        if key not in document:
            return key
    if not _is_str_mapping(document["code_manifest"], str_values=True):
        return "code_manifest"
    if not _is_str_mapping(document["params"], str_values=False):
        return "params"
    if _READS_KEY in document and not _is_hash_list(document[_READS_KEY]):
        return _READS_KEY
    records: dict[object, object] = document
    prefix = ""
    if _RECORDS_KEY in document:
        for list_key in ("deps", "outs"):
            if document[list_key] != []:
                return list_key
        raw_records = document[_RECORDS_KEY]
        if not isinstance(raw_records, dict):
            return _RECORDS_KEY
        records = cast("dict[object, object]", raw_records)
        if records.keys() != {"deps", "outs"}:
            return _RECORDS_KEY
        prefix = f"{_RECORDS_KEY}."
    for list_key in ("deps", "outs"):
        entries = records[list_key]
        if not isinstance(entries, list):
            return f"{prefix}{list_key}"
        for index, raw in enumerate(cast("list[object]", entries)):
            error = _entry_error(raw, f"{prefix}{list_key}[{index}]", list_key, listed=bool(prefix))
            if error is not None:
                return error
    return None


def _from_document(document: object) -> StorageLockData:
    """A lock file's data, ``lock_data_error`` having passed it, with its entries where
    ``StorageLockData`` holds them (``deps`` and ``outs``) and each directory's hash under
    ``hash``."""
    data = dict(cast("dict[str, object]", document))
    records = cast("dict[str, object]", data.pop(_RECORDS_KEY, None) or data)
    for list_key in ("deps", "outs"):
        data[list_key] = [
            {("hash" if key == _DIR_HASH_KEY else key): value for key, value in entry.items()}
            for entry in cast("list[dict[str, object]]", records[list_key])
        ]
    return cast("StorageLockData", cast("object", data))


def _shaped(document: dict[str, object]) -> dict[str, object]:
    """``document``, whose entries stand in its ``deps`` and ``outs``, as its lock file holds
    it: under ``records`` when one of them records a directory (``_RECORDS_KEY``)."""
    entries = [*cast("list[object]", document["deps"]), *cast("list[object]", document["outs"])]
    if not any(isinstance(entry, dict) and _DIR_HASH_KEY in entry for entry in entries):
        return document
    return {
        **document,
        "deps": [],
        "outs": [],
        _RECORDS_KEY: {"deps": document["deps"], "outs": document["outs"]},
    }


def _to_document(data: StorageLockData) -> dict[str, object]:
    """``data`` as its lock file holds it: each directory's hash under ``listing``, and every
    entry under ``records`` when one records a directory (``_shaped``)."""

    def entry(item: DepEntry | OutEntry) -> dict[str, object]:
        if "nfiles" not in item:
            return dict(item)
        return {(_DIR_HASH_KEY if key == "hash" else key): value for key, value in item.items()}

    return _shaped(
        {
            **data,
            "deps": [entry(item) for item in data["deps"]],
            "outs": [entry(item) for item in data["outs"]],
        }
    )


def _schema_version(data: object) -> int | None:
    """The ``schema_version`` of parsed YAML that has every lock key: 0 when it names none,
    None when it is not such a mapping or the version is not an integer."""
    if not isinstance(data, dict):
        return None
    document = cast("dict[object, object]", data)
    if not document.keys() >= _REQUIRED_LOCK_KEYS:
        return None
    version = document.get("schema_version", 0)
    return cast("int", version) if _is_int(version) else None


def _is_old_format(data: object) -> bool:
    """Whether parsed YAML is a lock an older pivot wrote, listing each directory's files inside
    it: a mapping with the lock keys whose ``schema_version`` is below this pivot's."""
    version = _schema_version(data)
    return version is not None and version < SCHEMA_VERSION


@functools.cache
def _warn_format_once(path: str, mtime_ns: int, older: bool) -> None:
    """Warn once per process for each lock in another pivot's format, as it is on disk
    (``mtime_ns``)."""
    del mtime_ns
    if older:
        logger.warning(
            "%s was written by an older pivot; pivot reads it as it is, and writes it in this "
            + "pivot's format, with a listing file beside it for each directory it records, the "
            + "next time its stage's lock is written",
            path,
        )
    else:
        logger.warning(
            "%s was written by a newer pivot; this pivot cannot read it and reads it as missing, "
            + "so its stage is decided afresh",
            path,
        )


class CommittedLock(NamedTuple):
    """A lock read from git history or from another repository, and the listings of the
    directory outputs an older pivot recorded inside it (by output key)."""

    data: StorageLockData
    listings: dict[str | None, listing.DirListing]


class _ConvertedLock(NamedTuple):
    """An older pivot's lock as this format's document, with the listings it held inline."""

    document: dict[str, object]
    output_listings: dict[str | None, listing.DirListing]  # by output key
    listing_bytes: dict[str, bytes]  # every inline listing's bytes, by directory hash
    inline: dict[str, listing.DirListing]  # every inline listing, by this format's hash
    legacy_hashes: set[str]  # the directory hashes that lock recorded, this format's apart


def _convert_old(data: object) -> _ConvertedLock | None:
    """An older pivot's lock (``_is_old_format``) as this format's document: each directory
    entry, which lists its files inline, becomes the record of its listing (the listing's hash,
    its total size and its file count). None when an inline listing is malformed or names a
    path outside its directory; the document still has to pass ``lock_data_error``."""
    old = cast("dict[str, object]", data)
    document: dict[str, object] = {**old, "schema_version": SCHEMA_VERSION}
    output_listings = dict[str | None, listing.DirListing]()
    listing_bytes = dict[str, bytes]()
    inline = dict[str, listing.DirListing]()
    legacy_hashes = set[str]()
    for list_key in ("deps", "outs"):
        entries = old[list_key]
        if not isinstance(entries, list):
            return None
        converted = list[object]()
        for raw in cast("list[object]", entries):
            if not isinstance(raw, dict):
                converted.append(raw)
                continue
            entry = cast("dict[str, object]", raw)
            if "manifest" not in entry:
                converted.append(entry)
                continue
            members = listing.from_inline(entry["manifest"], entry.get("empty_dirs", []))
            if members is None:
                return None
            data_bytes = listing.encode(members)
            record = listing.record_of(members, data_bytes)
            new_entry = {
                key: value
                for key, value in entry.items()
                if key not in ("hash", "manifest", "empty_dirs", "size")
            }
            new_entry.update(
                {_DIR_HASH_KEY: record["hash"], "size": record["size"], "nfiles": record["nfiles"]}
            )
            converted.append(new_entry)
            listing_bytes[record["hash"]] = data_bytes
            inline[record["hash"]] = members
            if cache.is_valid_hash(entry.get("hash")):
                legacy_hashes.add(cast("str", entry["hash"]))
            key = entry.get("key")
            if list_key == "outs" and (key is None or isinstance(key, str)):
                output_listings[key] = members
        document[list_key] = converted
    return _ConvertedLock(_shaped(document), output_listings, listing_bytes, inline, legacy_hashes)


def parse_committed(document: object) -> CommittedLock | UnsafeLockPath | None:
    """A lock as git history or another repository holds it: this format, or an older pivot's
    format, read whole so a past revision stays readable.

    An older pivot's directory entry lists its files inline; it becomes this format's record
    (the hash of its listing) and its listing is returned beside the data. None when the
    document is not a lock; an ``UnsafeLockPath`` when it records a path outside its project.
    """
    listings = dict[str | None, listing.DirListing]()
    if _is_old_format(document):
        converted = _convert_old(document)
        if converted is None:
            return None
        document = converted.document
        listings = converted.output_listings
    error = lock_data_error(document)
    if isinstance(error, UnsafeLockPath):
        return error
    if error is not None:
        return None
    return CommittedLock(_from_document(document), listings)


class RecordedHashes(NamedTuple):
    """Every cache object a lock names, for a reader that must keep all of them."""

    keys: set[str]
    """Each hash the lock holds itself: a file's, a directory's listing hash, and, for a lock an
    older pivot wrote, that lock's own directory hash (its whole-tree object) and every file its
    inline listing names."""
    directories: list[DirHash]
    """The directories whose members live in a listing the reader must read. A lock an older
    pivot wrote lists them inline, so it yields none."""


class UninterpretableLock(NamedTuple):
    """A lock this pivot cannot fully interpret, so it may name cache objects this pivot cannot
    see: one that is not a lock, lacks a key, has a schema or a key this pivot does not know, or
    holds a field it cannot read."""

    reason: str


# Every key a lock this pivot or an older one writes, at the top level, under ``records`` and in
# a ``deps`` or ``outs`` entry. ``dep_generations`` is an old pivot's (generation counts, no
# hashes; read as nothing). ``reads`` lists the objects the run read by hash, and is a root. A
# key outside these may be a newer pivot's list of hashes, so a reader that must keep every
# object a lock names refuses it; a key that names objects joins this list and the roots in
# ``recorded_hashes`` in the same change.
_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "code_manifest",
        "params",
        "deps",
        "outs",
        _RECORDS_KEY,
        "dep_generations",
        _READS_KEY,
    }
)
_ENTRY_KEYS = frozenset(
    {"producer", "key", "hash", _DIR_HASH_KEY, "size", "nfiles", "path", "manifest", "empty_dirs"}
)


def _uninterpretable(document: object) -> str | None:
    """Why a parsed lock file is not one this pivot can fully interpret before its entries are
    read, or None: not a mapping, a lock key missing, a schema version that is not a number or
    is newer than this pivot's, or a key this pivot does not know."""
    if not isinstance(document, dict):
        return f"no lock: its YAML is {type(document).__name__}, not a mapping"
    mapping = cast("dict[object, object]", document)
    missing = sorted(_REQUIRED_LOCK_KEYS - mapping.keys())
    if missing:
        return f"no {', '.join(missing)} key, which every lock holds"
    version = mapping.get("schema_version", 0)
    if not _is_int(version):
        return f"schema_version {version!r}, which is not a whole number"
    if cast("int", version) > SCHEMA_VERSION:
        return f"schema_version {version}, newer than this pivot's {SCHEMA_VERSION}"
    unknown = sorted(str(key) for key in mapping if key not in _LOCK_KEYS)
    if unknown:
        return f"the top-level key {', '.join(unknown)}, which this pivot does not know"
    records = mapping.get(_RECORDS_KEY, mapping)
    lists = cast("dict[object, object]", records) if isinstance(records, dict) else {}
    if records is not mapping and set(lists) - {"deps", "outs"}:
        extra = ", ".join(sorted(str(key) for key in set(lists) - {"deps", "outs"}))
        return f"the key {extra} under {_RECORDS_KEY}, which this pivot does not know"
    for list_key in ("deps", "outs"):
        entries = lists.get(list_key)
        for raw in cast("list[object]", entries) if isinstance(entries, list) else ():
            if isinstance(raw, dict):
                extra = set(cast("dict[object, object]", raw)) - _ENTRY_KEYS
                if extra:
                    names = ", ".join(sorted(str(key) for key in extra))
                    return f"the key {names} in a {list_key} entry, which this pivot does not know"
    return None


def recorded_hashes(document: object) -> RecordedHashes | UninterpretableLock:
    """What a parsed lock file names, for `pivot gc`: every cache key the lock itself holds (each
    ``deps`` and ``outs`` hash, and each object its run read by hash, ``reads``) and the
    directories whose members a reader must read from a listing.

    Any lock this pivot cannot fully interpret is ``UninterpretableLock``, with the reason: not a
    mapping, a lock key missing, a schema version that is not a whole number or is newer than
    this pivot's, a key this pivot does not know (at the top level, under ``records`` or in an
    entry), an inline listing it cannot read, or a field ``lock_data_error`` refuses. A newer
    pivot may record objects in any of those, so a collector refuses rather than read the rest.

    A lock an older pivot wrote lists each directory's files inline (``_convert_old``). Its
    members, the directory hash it recorded and the hash of the listing this pivot derives from
    it are all in ``keys``, so a collection keeps the whole-tree object of that format as well
    as the listing object any read of that lock stored.
    """
    reason = _uninterpretable(document)
    if reason is not None:
        return UninterpretableLock(reason)
    keys = set[str]()
    directories = list[DirHash]()
    listed = True
    if _is_old_format(document):
        converted = _convert_old(document)
        if converted is None:
            return UninterpretableLock("a directory listed inline that this pivot cannot read")
        keys |= converted.legacy_hashes
        for directory_hash, members in converted.inline.items():
            keys.add(directory_hash)
            keys.update(entry["hash"] for entry in members["files"])
        document = converted.document
        listed = False
    error = lock_data_error(document)
    if error is not None:
        return UninterpretableLock(
            str(error) if isinstance(error, UnsafeLockPath) else f"a field {error} it cannot read"
        )
    data = _from_document(document)
    # `lock_data_error` refused a `reads` that is not a list of content hashes, and an older
    # pivot's lock keeps its top-level keys through `_convert_old`, so this is every one.
    if _READS_KEY in data:
        keys.update(data[_READS_KEY])
    for entry in (*data["deps"], *data["outs"]):
        hash_info = entry_hash_info(entry)
        keys.add(hash_info["hash"])
        if listed and is_dir_hash(hash_info):
            directories.append(hash_info)
    return RecordedHashes(keys, directories)


def _record_dir_fields(entry: DepEntry | OutEntry, hash_info: HashInfo) -> None:
    """Copy a directory's size and file count onto a lock entry."""
    if is_dir_hash(hash_info):
        entry["size"] = hash_info["size"]
        entry["nfiles"] = hash_info["nfiles"]


def entry_hash_info(entry: DepEntry | OutEntry) -> HashInfo:
    """Return the hash a lock ``deps`` or ``outs`` entry records."""
    if "nfiles" not in entry:
        return FileHash(hash=entry["hash"])
    return DirHash(
        hash=entry["hash"], size=entry["size"] if "size" in entry else 0, nfiles=entry["nfiles"]
    )


def _convert_to_storage_format(data: LockData) -> StorageLockData:
    """Convert internal LockData to storage format (list-based, identity keys, sorted)."""
    normalized_deps = _ensure_identity_keys(data["dep_hashes"])
    normalized_outs = _ensure_identity_keys(data["output_hashes"])

    deps_list = list[DepEntry]()
    for identity, hash_info in normalized_deps.items():
        entry = DepEntry(producer=identity.producer, key=identity.key, hash=hash_info["hash"])
        _record_dir_fields(entry, hash_info)
        deps_list.append(entry)
    deps_list.sort(key=lambda e: (e["producer"], e["key"] or ""))

    outs_list = list[OutEntry]()
    for identity, hash_info in normalized_outs.items():
        entry = OutEntry(key=identity.key, hash=hash_info["hash"])
        if "output_paths" in data and identity in data["output_paths"]:
            entry["path"] = data["output_paths"][identity]
        _record_dir_fields(entry, hash_info)
        outs_list.append(entry)
    outs_list.sort(key=lambda e: e["key"] or "")

    sorted_code_manifest = dict(sorted(data["code_manifest"].items()))

    storage = StorageLockData(
        schema_version=SCHEMA_VERSION,
        code_manifest=sorted_code_manifest,
        params=data["params"],
        deps=deps_list,
        outs=outs_list,
    )
    if "reads" in data and data["reads"]:
        storage["reads"] = sorted(set(data["reads"]))
    return storage


def _convert_from_storage_format(data: StorageLockData, *, stage_name: str) -> LockData:
    """Convert storage format (list-based) to internal LockData (identity-keyed)."""
    dep_hashes = dict[ArtifactIdentity, HashInfo]()
    for entry in data["deps"]:
        dep_hashes[ArtifactIdentity(entry["producer"], entry["key"])] = entry_hash_info(entry)

    output_hashes = dict[ArtifactIdentity, HashInfo]()
    output_paths = dict[ArtifactIdentity, str]()
    for entry in data["outs"]:
        identity = ArtifactIdentity(stage_name, entry["key"])
        output_hashes[identity] = entry_hash_info(entry)
        if "path" in entry:
            output_paths[identity] = entry["path"]

    lock_data = LockData(
        code_manifest=data["code_manifest"],
        params=data["params"],
        dep_hashes=dep_hashes,
        output_hashes=output_hashes,
    )
    if output_paths:
        lock_data["output_paths"] = output_paths
    if "reads" in data and data["reads"]:
        lock_data["reads"] = data["reads"]
    return lock_data


def _ensure_identity_keys(
    hashes: dict[ArtifactIdentity, HashInfo] | dict[str, HashInfo],
) -> dict[ArtifactIdentity, HashInfo]:
    if not hashes:
        return {}
    first_key = next(iter(hashes))
    if isinstance(first_key, ArtifactIdentity):
        return cast("dict[ArtifactIdentity, HashInfo]", hashes)
    return {identity_from_key(k): v for k, v in cast("dict[str, HashInfo]", hashes).items()}


def _hashed_directory(
    record: DirHash,
    hashed_dirs: Mapping[str, listing.HashedDirectory],
    committed: pathlib.Path,
    cache_dir: pathlib.Path | None,
) -> tuple[bytes, listing.DirListing]:
    """The listing bytes and members of directory ``record``: the caller's, else the committed
    file or the cache's object (``listing.read_bytes``), checked against ``record``. A run-cache
    hit or an output already on disk records a directory this run did not list."""
    if record["hash"] in hashed_dirs:
        hashed = hashed_dirs[record["hash"]]
        return hashed["data"], hashed["listing"]
    data = listing.read_bytes(committed, record, cache_dir)
    return data, listing.parse(data, record, committed)


def _recorded_paths(
    outs: list[OutEntry], written: list[_ListingWrite]
) -> Iterator[tuple[str, str | None, str | None]]:
    """Yield each path a lock and its listings record with its output path or dependency name.

    A yielded output path names the directory that owns a member. A yielded dependency names
    the directory dependency that owns its member. The caller formats either only when it must
    report a validation refusal or control-character warning.
    """
    for out in outs:
        if "path" in out:
            yield out["path"], None, None
    for item in written:
        for entry in item.members["files"]:
            yield entry["relpath"], item.out_path, item.dependency
        for relpath in item.members["empty_dirs"]:
            yield relpath, item.out_path, item.dependency


class _ListingWrite(NamedTuple):
    """A listing file ``StageLock.write`` writes beside its lock."""

    target: pathlib.Path
    record: DirHash
    data: bytes
    members: listing.DirListing
    out_path: str | None  # the directory output's path, for messages
    dependency: str | None  # the directory dependency's name, for messages


class StageLock:
    """Manages lock file for a single pipeline stage."""

    stage_name: str
    stages_dir: pathlib.Path
    path: pathlib.Path
    _read_stages_dir: pathlib.Path
    _read_path: pathlib.Path

    def __init__(self, stage_name: str, stages_dir: pathlib.Path) -> None:
        """Initialize a stage lock for the given stage in stages_dir."""
        reason = stage_name_error(stage_name)
        if reason is not None:
            raise ValueError(f"Invalid stage name {stage_name!r}: {reason}")
        self.stage_name = stage_name
        self.stages_dir = stages_dir
        self.path = stages_dir / f"{stage_name}{_LOCK_SUFFIX}"
        self._read_stages_dir = stages_dir
        self._read_path = self.path

    def _load(self) -> object | None:
        """The current lock's parsed YAML, or None when it is absent or unparseable."""
        read_path, self._read_stages_dir, ordered = _resolve_stage_lock(
            self.stages_dir, self.stage_name
        )
        self._read_path = read_path
        if ordered is not None:
            return ordered
        try:
            with open(read_path) as f:
                return yaml.load(f, Loader=yaml_config.Loader)
        except FileNotFoundError:
            return None  # Normal case - lock doesn't exist yet
        except (UnicodeDecodeError, yaml.YAMLError) as e:
            logger.warning("Failed to parse lock file %s: %s", read_path, e)
            return None

    def read(
        self, *, cache_dir: pathlib.Path | None = None, warn_old_format: bool = True
    ) -> LockData | None:
        """Read lock file, converting storage format to internal format.

        A lock an older pivot wrote is read as it is (``_convert_old``): each directory it
        lists inline becomes the record of its listing, and that listing is put in the cache
        at ``cache_dir`` (``listing.store_object``), where every listing read finds it, since no
        listing file is committed beside such a lock. Nothing reruns or restores differently
        because of the format alone; the stage's next lock write writes this format. A lock a
        newer pivot wrote is read as missing. ``warn_old_format`` logs either, once per process
        for each such lock; a worker reads without it, since the process that dispatched the
        stage has already warned.
        """
        data = self._load()
        if data is None:
            return None
        version = _schema_version(data)
        converted = None
        if version is not None and version < SCHEMA_VERSION:
            converted = _convert_old(data)
            if converted is not None:
                data = converted.document
        error = lock_data_error(data)
        if error is not None:
            if version is not None and version > SCHEMA_VERSION:
                if warn_old_format:
                    self._warn_format(older=False)
            elif isinstance(error, UnsafeLockPath):
                logger.warning(
                    "Lock file %s failed validation: %s; treating it as missing",
                    self.path,
                    error,
                )
            else:
                logger.debug(
                    "Lock file %s failed validation at %s; treating it as missing",
                    self.path,
                    error,
                )
            return None
        if converted is not None:
            if warn_old_format:
                self._warn_format(older=True)
            if cache_dir is not None:
                for directory_hash, listing_data in converted.listing_bytes.items():
                    try:
                        listing.store_object(cache_dir, directory_hash, listing_data)
                    except OSError as e:
                        logger.debug("Could not cache a listing %s holds: %s", self.path, e)
        result = _convert_from_storage_format(_from_document(data), stage_name=self.stage_name)
        if converted is not None:
            result["older_format"] = True
        return result

    def _warn_format(self, *, older: bool) -> None:
        try:
            mtime_ns = self._read_path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        _warn_format_once(str(self._read_path), mtime_ns, older)

    def output_listing_path(self, identity: ArtifactIdentity) -> pathlib.Path:
        """Where the listing of this stage's directory output is committed."""
        return listing.output_listing_path(self._read_stages_dir, identity)

    @property
    def restart_proof_path(self) -> pathlib.Path:
        """Where the active lock's restart proof belongs."""
        return restart_proof_path(self._read_stages_dir, self.stage_name)

    def write(
        self,
        data: LockData,
        *,
        hashed_dirs: Mapping[str, listing.HashedDirectory] | None = None,
        input_paths: Mapping[ArtifactIdentity, str] | None = None,
        cache_dir: pathlib.Path | None = None,
        outs: Iterable[ArtifactRef] = (),
    ) -> list[pathlib.Path]:
        """Write the lock file, the listing of each directory it records and, for a stage that
        ``outs`` says keeps an incremental output, the restart proof, atomically, and return the
        paths of the files it wrote or left unchanged.

        A directory output's listing goes to ``listing.output_listing_path``; a directory
        dependency named in ``input_paths`` (a pipeline input, by its project-relative path)
        has its listing at ``listing.input_listing_path``. A listing comes from ``hashed_dirs``
        (by directory hash), else from the cache object in ``cache_dir``. Each listing, and the
        restart proof, is written before the lock, so no lock names a listing not yet written
        and none stands without its proof; a file already holding the bytes is left as it is.

        ``outs`` are the stage's registered outputs. A stage with an incremental one commits the
        restart proof beside its lock (``_RESTART_PROOF_SUFFIX``), so that an older pivot's
        rewrite of that lock is caught (``restart_proofs``).

        Every recorded path must pass ``path_policy.recorded_path_error``, the rule ``read``
        applies to a lock's paths, so this never writes a lock that ``read`` refuses; a path that
        fails raises InvalidPathError naming it. A recorded name that holds a control character
        is written, and one warning names every such name.
        """
        self._read_stages_dir = self.stages_dir
        incremental = any(ref.tag is ArtifactTag.INCREMENTAL for ref in outs)
        hashed_dirs = hashed_dirs or {}
        input_paths = input_paths or {}
        normalized_outs = _ensure_identity_keys(data["output_hashes"])
        to_write = list[_ListingWrite]()
        for identity, hash_info in normalized_outs.items():
            if not is_dir_hash(hash_info):
                continue
            target = self.output_listing_path(identity)
            data_bytes, members = _hashed_directory(hash_info, hashed_dirs, target, cache_dir)
            to_write.append(
                _ListingWrite(
                    target=target,
                    record=hash_info,
                    data=data_bytes,
                    members=members,
                    out_path=(
                        data["output_paths"][identity]
                        if "output_paths" in data and identity in data["output_paths"]
                        else identity_key(identity)
                    ),
                    dependency=None,
                )
            )
        for identity, hash_info in _ensure_identity_keys(data["dep_hashes"]).items():
            if not is_dir_hash(hash_info) or identity not in input_paths:
                continue
            target = listing.input_listing_path(self.stages_dir, input_paths[identity])
            data_bytes, members = _hashed_directory(hash_info, hashed_dirs, target, cache_dir)
            to_write.append(
                _ListingWrite(
                    target=target,
                    record=hash_info,
                    data=data_bytes,
                    members=members,
                    out_path=None,
                    dependency=identity_key(identity),
                )
            )

        # Refuse before writing anything when two overlong names alias to the same short target.
        targets = dict[pathlib.Path, str]()
        for item in to_write:
            prior = targets.get(item.target)
            if prior is not None:
                this = item.out_path or item.dependency or "?"
                raise exceptions.InvalidPathError(
                    f"Stage '{self.stage_name}' cannot write two listings to the same file "
                    + f"'{item.target.name}': '{prior}' and '{this}' both map to it because "
                    + "their XXH64 short names collide; rename one of the outputs"
                )
            targets[item.target] = (
                item.out_path
                or item.dependency
                or identity_key(ArtifactIdentity(self.stage_name, None))
            )

        storage_data = _convert_to_storage_format(data)
        with_control = list[str]()
        for recorded, out_path, dependency in _recorded_paths(storage_data["outs"], to_write):
            reason = path_policy.recorded_path_error(recorded)
            has_control = reason is None and path_policy.has_control_character(recorded)
            if reason is None and not has_control:
                continue
            if dependency is not None:
                shown = f"{recorded!r} in dependency {dependency!r}"
            elif out_path is not None:
                shown = repr(posixpath.join(out_path, recorded))
            else:
                shown = repr(recorded)
            if reason is not None:
                raise exceptions.InvalidPathError(
                    f"Stage '{self.stage_name}' cannot record {shown}: {reason}; rename it"
                )
            with_control.append(shown)
        if with_control:
            one = len(with_control) == 1
            logger.warning(
                "Stage '%s' records %s, whose %s a control character; a later release will "
                + "refuse such names, so rename %s",
                self.stage_name,
                ", ".join(with_control),
                "name holds" if one else "names hold",
                "it" if one else "them",
            )

        # Serialize before changing a stable listing path.  If YAML cannot represent the lock,
        # the existing lock and the listings it names must remain a matching set.
        serialized = yaml.dump(
            _to_document(storage_data), Dumper=yaml_config.Dumper, sort_keys=False
        )

        written = list[pathlib.Path]()
        for item in to_write:
            if cache_dir is not None:
                listing.store_object(cache_dir, item.record["hash"], item.data)
            listing.write(item.target, item.data)
            written.append(item.target)
        if incremental:
            _write_restart_proof(self.restart_proof_path)
            written.append(self.restart_proof_path)

        def write_yaml(fd: int) -> None:
            with os.fdopen(fd, "w") as f:
                _ = f.write(serialized)

        cache.atomic_write_file(self.path, write_yaml)
        written.append(self.path)
        return written


def committed_listing(lock_data: LockData, path: pathlib.Path) -> pathlib.Path | None:
    """``path``, the committed listing file of a directory ``lock_data`` records, for a reader
    to pass to ``listing.read``; None for a lock an older pivot wrote (``older_format``). Such a
    lock listed each directory inline, and ``StageLock.read`` put that listing in the cache; a
    listing file beside it is another lock's (a newer pivot's, before an older one reran the
    stage), and would be refused or warned about though nothing is wrong."""
    return None if "older_format" in lock_data else path


def restart_proofs(
    stage_lock: StageLock, lock_data: LockData | None, outs: Iterable[ArtifactRef]
) -> list[pathlib.Path]:
    """The files this pivot wrote that still stand beside ``lock_data``, a lock an older pivot
    rewrote (``older_format``), for a stage with an incremental output; empty for any other
    stage or lock.

    Each proves this pivot wrote the stage's lock before the older pivot rewrote it: the
    restart proof, which every such lock is written with, and the listing of each directory
    output, which an older pivot writes for none. A released pivot stops at the stage instead
    of restarting its incremental output from this pivot's lock (``_RECORDS_KEY``), so it
    rewrites that lock only after a person removed it, which starts the output from empty: the
    lock it leaves is not to be taken as the stage's history (``restarted_reason``).
    """
    if lock_data is None or "older_format" not in lock_data:
        return []
    refs = list(outs)
    if not any(ref.tag is ArtifactTag.INCREMENTAL for ref in refs):
        return []
    candidates = [stage_lock.restart_proof_path]
    candidates.extend(stage_lock.output_listing_path(ref.identity) for ref in refs)
    return [path for path in candidates if path.is_file()]


def restarted_reason(stage_lock: StageLock, proofs: list[pathlib.Path]) -> str:
    """Why pivot refuses a stage ``restart_proofs`` names, and the two ways to settle it."""
    lock_shown = shlex.quote(os.path.relpath(stage_lock.path))
    proofs_shown = " ".join(shlex.quote(os.path.relpath(path)) for path in proofs)
    stage = shlex.quote(stage_lock.stage_name)
    one = len(proofs) == 1
    return (
        f"an older pivot rewrote {lock_shown} after this pivot wrote it ({proofs_shown}, "
        + f"which this pivot wrote beside it, {'is' if one else 'are'} still there), and the "
        + "stage keeps an incremental output, so that pivot may have restarted the output from "
        + "empty. Restore the output from the commit before the older pivot's (`git checkout "
        + f"<commit> -- {lock_shown} {proofs_shown}`, then `pivot checkout --force {stage}`), "
        + "or rerun the stage from what that lock records, on purpose, with "
        + f"`pivot repro --force {stage}`. "
        + "Everyone who works on a project upgrades pivot together"
    )


def stage_lock_files(stages_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Each lock file under ``stages_dir`` (``<stage name>.lock`, a regular file or a symlink
    to one) whose name can name a stage, by its stage name; none in a directory that cannot be
    listed (absent, unreadable, or a configured path no directory can have). Symlinked
    directories are not entered, and an entry whose type cannot be read (a symlink that loops
    or dangles into a directory it cannot search) or whose name is invalid is no lock, as
    ``Path.is_file`` and the stage-name rule read it. Each entry's type comes from the directory
    listing (``os.scandir``), so only a symlink costs a ``stat``.
    """
    found = dict[str, pathlib.Path]()
    pending = [(os.fspath(stages_dir), "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except (OSError, ValueError):  # ValueError: a path holding a NUL byte
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append((entry.path, f"{prefix}{entry.name}/"))
                elif entry.name.endswith(_LOCK_SUFFIX) and entry.is_file():
                    name = f"{prefix}{entry.name[: -len(_LOCK_SUFFIX)]}"
                    if stage_name_error(name) is None:
                        found[name] = pathlib.Path(entry.path)
            except OSError:
                continue
    return found


def instance_files(
    stages_dirs: Iterable[pathlib.Path], template: str
) -> dict[str, list[pathlib.Path]]:
    """Each instance of mapped stage ``template`` (``Pipeline.map``) that keeps a file in one of
    ``stages_dirs``, by instance name: its locks, the listing of each directory output
    (``listing.output_listing_path``) and its restart proofs (``restart_proof_path``), every lock
    last. An instance is ``<template>@<key>``, its key one the variant-name rule allows and the
    whole name one the stage-name rule allows (no lock is written under any other), so a key
    holds no ``/`` and every such file is in the directory of ``template``'s own lock; an
    instance of a mapped variant (``<template>@<variant>@<key>``) is that variant's, not
    ``template``'s. A file that is not a regular one (a symlink counts as its target) is none,
    and so is every file of a directory that cannot be listed."""
    parent, _, base = template.rpartition("/")
    prefix = f"{base}@"
    found = dict[str, list[pathlib.Path]]()
    for stages_dir in stages_dirs:
        try:
            entries = list(os.scandir(stages_dir / parent if parent else stages_dir))
        except (OSError, ValueError):  # ValueError: a path holding a NUL byte
            continue
        for entry in entries:
            name = entry.name
            if not name.startswith(prefix):
                continue
            if name.endswith((_LOCK_SUFFIX, _RESTART_PROOF_SUFFIX)):
                key = name[len(prefix) :].rsplit(".", 1)[0]
            elif name.endswith(listing.SUFFIX) and "=" in name:
                key = name[len(prefix) :].split("=", 1)[0]
            else:
                continue
            instance = f"{template}@{key}"
            try:
                if (
                    variant_name_error(key) is not None
                    or stage_name_error(instance) is not None
                    or not entry.is_file()
                ):
                    continue
            except OSError:
                continue
            found.setdefault(instance, []).append(pathlib.Path(entry.path))
    for files in found.values():
        files.sort(key=lambda path: path.name.endswith(_LOCK_SUFFIX))
    return found


def remove_stage_files(files: Iterable[pathlib.Path]) -> None:
    """Remove the lock, listing and restart-proof files of a stage that no longer exists, in the
    order given (``instance_files`` puts the lock last, so a removal cut short leaves the lock,
    which names the stage to the next removal). A file already gone is skipped."""
    for path in files:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


class DirectoryRecord(NamedTuple):
    """A directory output one lock file records at a path (``recorded_directories``)."""

    path: str
    """As the lock records it, relative to the project root."""
    identity: ArtifactIdentity
    output_hash: DirHash
    inline: listing.DirListing | None
    """The directory's listing, which an older pivot's lock holds inline; None for a lock of
    this format, whose listing is committed beside it."""


class RecordedDirectory(NamedTuple):
    """A directory output a stage lock records at a path (``recorded_directories``)."""

    stage_name: str
    record: DirectoryRecord
    lock_files: tuple[pathlib.Path, ...]
    """Every lock file of the stage: one per state directory of the 0.5.2 layout holding one."""
    stages_dir: pathlib.Path
    """The stages directory of the lock the record was read from."""

    @property
    def state_dir(self) -> pathlib.Path:
        """The state directory of that lock (its stage's ``nocommit:`` record is there)."""
        return self.stages_dir.parent

    def read_listing(self, cache_dir: pathlib.Path | None) -> listing.DirListing:
        """The directory's listing: the older lock's inline one, else the one committed beside
        the lock or the cache's copy, read without a word (``listing.OnStale.QUIET``): the stage
        is not the command's."""
        if self.record.inline is not None:
            return self.record.inline
        return listing.read(
            listing.output_listing_path(self.stages_dir, self.record.identity),
            self.record.output_hash,
            cache_dir,
            on_stale=listing.OnStale.QUIET,
        )


def recorded_directories(
    stages_dir: pathlib.Path, exclude: Collection[str]
) -> dict[str, RecordedDirectory]:
    """Every directory output that a lock in ``stages_dir``, or in the project's other stages
    directory (``state_dir.other_stages_dir_for``), records at a path, by that path as the lock
    records it (relative to the project root), for each stage not in ``exclude``.

    This reads what runs left behind, for stages the command does not run, so it says nothing
    and never fails: the stage's current lock is chosen as ``StageLock`` chooses it, without its
    notices (``_choose_stage_lock``), and decoded as ``parse_committed`` decodes a lock, writing
    nothing; a lock that cannot be read or decoded (unreadable, damaged, a newer pivot's, a
    scalar the safe loader cannot build) records nothing here. Within one state-directory record
    (``state_dir.recorded_locks``) a lock file is decoded again only when it changed, so a
    ``--watch`` run sees every lock written since the last.
    """
    other_dir = state_dir_mod.other_stages_dir_for(stages_dir)
    listed = [stage_lock_files(stages_dir)]
    if other_dir is not None:
        listed.append(stage_lock_files(other_dir))
    kept = state_dir_mod.recorded_locks()
    found = dict[str, RecordedDirectory]()
    for stage_name in {name for files in listed for name in files} - set(exclude):
        try:
            choice = _choose_stage_lock(stages_dir, stage_name, other_dir)
            lock_files = tuple(files[stage_name] for files in listed if stage_name in files)
            records = _directory_records(stage_name, choice, kept)
        except (OSError, UnicodeError, ValueError, exceptions.PivotError):
            continue
        for record in records:
            found[record.path] = RecordedDirectory(
                stage_name, record, lock_files, choice.stages_dir
            )
    return found


def _directory_records(
    stage_name: str,
    choice: _LockChoice,
    kept: dict[str, tuple[_LockIdentity, tuple[DirectoryRecord, ...]]] | None,
) -> tuple[DirectoryRecord, ...]:
    """The directory outputs the chosen lock file records at a path; decoded again only when
    the file's identity changed since ``kept`` recorded it."""
    lock_file = choice.ordered
    identity = _lock_identity(choice.path) if lock_file is None else lock_file.identity
    if identity is None:
        return ()
    key = os.fspath(choice.path)
    if kept is not None and key in kept and kept[key][0] == identity:
        return kept[key][1]
    try:
        committed = (lock_file or _LockFile(choice.path, identity)).committed()
        records = _decoded_directories(stage_name, committed)
    except (ValueError, exceptions.PivotError) as e:
        logger.debug("Reading no recorded directory from %s: %s", choice.path, e)
        records = ()
    if kept is not None:
        kept[key] = (identity, records)
    return records


def _decoded_directories(
    stage_name: str, committed: CommittedLock | None
) -> tuple[DirectoryRecord, ...]:
    if committed is None:
        return ()
    lock_data = _convert_from_storage_format(committed.data, stage_name=stage_name)
    if "output_paths" not in lock_data:
        return ()
    return tuple(
        DirectoryRecord(
            path=lock_data["output_paths"][identity],
            identity=identity,
            output_hash=output_hash,
            inline=committed.listings.get(identity.key),
        )
        for identity, output_hash in lock_data["output_hashes"].items()
        if is_dir_hash(output_hash) and identity in lock_data["output_paths"]
    )


def find_orphaned_locks(
    stages_dir: pathlib.Path,
    registered_stages: set[str],
    registered_listings: Iterable[pathlib.Path] = (),
) -> list[str]:
    """Find lock, listing and restart-proof files in stages_dir that belong to no registered
    stage or input.

    When a pipeline is restructured (stages renamed/removed), old lock files
    persist and old output files remain on disk. Detecting orphaned locks helps
    users identify stale artifacts from previous pipeline configurations.
    A listing file is orphaned when ``registered_listings`` (the listing paths of the
    registered stages' outputs and inputs) does not hold it; a restart proof
    (``restart_proof_path``) is orphaned when no registered stage has its name.

    Returns:
        Sorted list of the stage names that have lock files but aren't registered, and of the
        paths, relative to stages_dir, of the orphaned listing and restart-proof files.
    """
    if not stages_dir.is_dir():
        return []

    orphaned = [name for name in stage_lock_files(stages_dir) if name not in registered_stages]

    for proof_file in stages_dir.rglob(f"*{_RESTART_PROOF_SUFFIX}"):
        rel = str(proof_file.relative_to(stages_dir))
        if proof_file.is_file() and rel[: -len(_RESTART_PROOF_SUFFIX)] not in registered_stages:
            orphaned.append(rel)

    known = set(registered_listings)
    for listing_file in stages_dir.rglob(f"*{listing.SUFFIX}"):
        if listing_file.is_file() and listing_file not in known:
            orphaned.append(str(listing_file.relative_to(stages_dir)))

    orphaned.sort()
    return orphaned

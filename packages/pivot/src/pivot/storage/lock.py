"""Per-stage lock files for tracking pipeline state.

StageLock provides persistent lock files (.lock) for change detection,
storing fingerprints, params, and hashes to detect when re-runs are needed.

For runtime mutual exclusion during concurrent execution, see
``pivot.storage.artifact_lock`` which uses flock-based artifact locks.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import pathlib
import posixpath
import re
import stat
import threading
from typing import TYPE_CHECKING, TypeGuard, cast, final, override

import yaml

from pivot import exceptions, path_policy, yaml_config
from pivot import state_dir as state_dir_mod
from pivot.storage import cache
from pivot.types import (
    ArtifactIdentity,
    DepEntry,
    DirHash,
    DirManifestEntry,
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
    from collections.abc import Iterator

    from pivot.storage import state

logger = logging.getLogger(__name__)

_VALID_STAGE_NAME = re.compile(
    r"[a-zA-Z0-9_@./-]+"
)  # Allow / for pipeline-prefixed names, . for DVC matrix keys
_VALID_VARIANT_NAME = re.compile(r"[a-zA-Z0-9_.-]+")
_MAX_STAGE_NAME_LEN = 200  # Leave room for ".lock" suffix within filesystem NAME_MAX (255)
_REQUIRED_LOCK_KEYS = frozenset({"code_manifest", "params", "deps", "outs"})


def stages_rel_path(state_dir: pathlib.Path, root: pathlib.Path) -> str:
    """Return the git-relative directory holding lock files."""
    return (state_dir / "stages").resolve().relative_to(root.resolve()).as_posix()


def get_stages_dir(state_dir: pathlib.Path) -> pathlib.Path:
    """Return the state directory's git-tracked lock file directory."""
    return state_dir / "stages"


def resolve_stage_lock_path(stages_dir: pathlib.Path, stage_name: str) -> pathlib.Path:
    """Return the stage's current lock, in ``stages_dir`` or the project's other stages directory.

    A lock in only one of them is that one: 0.5.2's layout, a run made before ``core.state_dir``
    was set, or one made while it could not be written. When both hold one, the current lock is
    the one the project's run records name (``_other_lock_is_current``), never the first
    directory's: a run made while the configured directory could not be written leaves its lock
    in ``.pivot`` beside an older one in the configured directory, and the older one would
    restore stale outputs once that directory can be written again.
    """
    return _resolve_stage_lock(stages_dir, stage_name)[0]


def _resolve_stage_lock(
    stages_dir: pathlib.Path, stage_name: str
) -> tuple[pathlib.Path, StorageLockData | None]:
    """``resolve_stage_lock_path``, with the returned lock's data when ordering read it.

    The data is the lock's validated content when ordering the two locks read that file, and
    None otherwise, so ``StageLock.read`` parses and checks each lock file at most once. A
    configured lock that is current while ``.pivot`` also holds the stage's lock is noted for
    the notice ``report_stale_locks`` prints when the record ends.
    """
    primary_path = stages_dir / f"{stage_name}.lock"
    other_stages_dir = state_dir_mod.other_stages_dir_for(stages_dir)
    if other_stages_dir is None:
        return primary_path, None
    other_path = other_stages_dir / f"{stage_name}.lock"
    other_identity = _lock_identity(other_path)
    if other_identity is None:
        return primary_path, None
    primary_identity = _lock_identity(primary_path)
    if primary_identity is not None:
        primary = _LockFile(primary_path, primary_identity)
        other = _LockFile(other_path, other_identity)
        if not _other_lock_is_current(
            stage_name, primary, other, stages_dir.parent, other_stages_dir.parent
        ):
            ordering = state_dir_mod.lock_ordering()
            if ordering is not None:
                ordering["stale"].add(stages_dir.parent)
            return primary_path, primary.valid_data()
        state_dir_mod.report_legacy_state_dir(stages_dir.parent)
        return other_path, other.valid_data()
    state_dir_mod.report_legacy_state_dir(stages_dir.parent)
    return other_path, None


# A lock file's device, inode, size and change time.
type _Identity = tuple[int, int, int, int]


def _lock_identity(path: pathlib.Path) -> _Identity | None:
    """A lock file's identity, or None when ``path`` is not a regular file.

    One ``stat`` tells whether the lock exists and what notices that a kept answer may be out
    of date (``_order_with``); it never orders two locks.
    """
    try:
        info = os.stat(path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return info.st_dev, info.st_ino, info.st_size, info.st_ctime_ns


@final
class _LockFile:
    """One of a stage's two lock files, read and checked at most once while they are ordered."""

    path: pathlib.Path
    identity: _Identity
    _data: StorageLockData | None
    _read: bool

    def __init__(self, path: pathlib.Path, identity: _Identity) -> None:
        self.path = path
        self.identity = identity
        self._data = None
        self._read = False

    def output_hashes(self) -> list[str] | None:
        """The output hashes the lock records; None when it cannot be read or is not a lock."""
        if not self._read:
            self._read = True
            try:
                with open(self.path) as f:
                    data: object = yaml.load(f, Loader=yaml_config.Loader)
            except (OSError, UnicodeDecodeError, yaml.YAMLError):
                data = None
            self._data = data if is_lock_data(data) else None
        if self._data is None:
            return None
        return sorted(out["hash"] for out in self._data["outs"])

    def valid_data(self) -> StorageLockData | None:
        """The lock's content if it was read and is a valid lock, else None."""
        return self._data


# list_runs reads at most this many runs: every run a database keeps (`prune_runs`).
_ALL_RUNS = 1 << 31

# Guards the databases a record holds, which the threads that copy its context share.
_held_databases_lock = threading.Lock()


def _other_lock_is_current(
    stage_name: str,
    primary: _LockFile,
    other: _LockFile,
    primary_state_dir: pathlib.Path,
    other_state_dir: pathlib.Path,
) -> bool:
    """Whether, of a stage's locks in both state directories, the other directory's is current.

    The run records of both directories' databases answer (``_order_locks``). Any failure to
    read keeps the first lock.
    """
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
    """``_other_lock_is_current``, raising what it cannot read.

    The stage's runs that ran or kept it, newest first, ordered by their timestamp-prefixed IDs
    across the two databases, answer; a lock file's modification time is never read. The first
    of them whose run-cache entry holds one lock's outputs names that lock: the first lock is
    read first, and the other only when the first does not match. A run whose outputs are in
    neither lock (a ``--no-commit`` run) defers to an older one. When a run has no run-cache
    entry (a result that is not reusable, or a pruned one), the newest run decides: the other
    lock is current when that run ran the stage there, since a run writes its lock where it
    records itself.

    Within a record, each database is opened once (``_held_database``), the runs are read once
    (``_ordering_view``), and each answer is kept until either lock file changes or either
    database commits a write.
    """
    with contextlib.ExitStack() as stack:
        dbs = _ordering_databases(stack, (primary_state_dir, other_state_dir))
        view = _ordering_view(dbs)
        return _order_with(stage_name, primary, other, dbs, view, other_state_dir)


def _order_with(
    stage_name: str,
    primary: _LockFile,
    other: _LockFile,
    dbs: list[tuple[pathlib.Path, state.StateDB]],
    view: state_dir_mod.OrderingView,
    other_state_dir: pathlib.Path,
) -> bool:
    """``_order_locks`` with both databases open and their runs read."""
    ordering = state_dir_mod.lock_ordering()
    primary_key = os.fspath(primary.path)
    key: object = None
    if ordering is not None:
        key = (view["dirs"], view["markers"], primary.identity, other.identity)
        kept = ordering["decisions"].get(primary_key)
        if kept is not None and kept[0] == key:
            return kept[1]
    answer = _answer(stage_name, primary, other, dbs, view, os.fspath(other_state_dir))
    if ordering is not None:
        ordering["decisions"][primary_key] = (key, answer)
    return answer


def _answer(
    stage_name: str,
    primary: _LockFile,
    other: _LockFile,
    dbs: list[tuple[pathlib.Path, state.StateDB]],
    view: state_dir_mod.OrderingView,
    other_state_dir: str,
) -> bool:
    """Whether the other lock is current, from the stage's runs newest first (``_order_locks``)."""
    mentions = view["mentions"].get(stage_name)
    if not mentions:
        return False
    first_outputs = view["first_outputs"]
    for index, mention in enumerate(mentions):
        if index == 0 and first_outputs is not None:
            outputs = first_outputs[stage_name]
        else:
            outputs = _run_cache_outputs(stage_name, mention["input_hash"], dbs)
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
    """Each of ``state_dirs`` whose database exists, with that database open read-only.

    Within a record the database is the one the record holds (``_held_database``); outside one
    it is opened for this call and closed by ``stack``.
    """
    from pivot.storage import state

    ordering = state_dir_mod.lock_ordering()
    dbs = list[tuple[pathlib.Path, state.StateDB]]()
    for state_dir in state_dirs:
        if ordering is not None:
            held = _held_database(ordering, state_dir)
            if held is not None:
                dbs.append((state_dir, held))
        elif (state_dir / "state.lmdb").is_dir():
            dbs.append((state_dir, stack.enter_context(state.StateDB(state_dir, readonly=True))))
    return dbs


def _held_database(
    ordering: state_dir_mod.LockOrdering, state_dir: pathlib.Path
) -> state.StateDB | None:
    """The record's read-only database of ``state_dir``; None when it has none.

    Opened on first use and kept until the record ends (``_end_lock_ordering``). It is opened
    again by a process forked from the one that opened it, since a database cannot be used
    across a fork, and when its data file was replaced, as a long ``--watch`` session can see;
    the replaced one stays open, for any thread still reading it, until the record ends.
    """
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
        db = state.StateDB(state_dir, readonly=True)
        ordering["databases"].setdefault(key, []).append((pid, os.stat(data_file).st_ino, db))
        return db


def _current_held(
    ordering: state_dir_mod.LockOrdering, key: str, pid: int, data_file: str
) -> state.StateDB | None:
    """The record's database under ``key`` if this process opened it on the current data file;
    None otherwise. Reading the list needs no lock: entries are only appended."""
    held = ordering["databases"].get(key)
    if not held:
        return None
    opener, inode, db = held[-1]
    if opener != pid:
        return None
    try:
        return db if os.stat(data_file).st_ino == inode else None
    except FileNotFoundError:
        return None


def _ordering_view(dbs: list[tuple[pathlib.Path, state.StateDB]]) -> state_dir_mod.OrderingView:
    """Both databases' run records, as the record keeps them (``state_dir.OrderingView``).

    While neither database has committed a write since the view was read
    (``StateDB.last_change``), it is used as it is, with no transaction. After a write that
    recorded no new run, as each stage of a run makes, the runs are kept and their run-cache
    entries are looked up per stage. After a new run, the runs are listed again. Outside a
    record they are listed for each call.
    """
    dirs = tuple(os.fspath(state_dir) for state_dir, _db in dbs)
    markers = tuple(db.last_change() for _state_dir, db in dbs)
    ordering = state_dir_mod.lock_ordering()
    view = None if ordering is None else ordering["views"].get(dirs)
    if view is not None and view["markers"] == markers:
        return view
    newest = tuple(db.newest_run_id() for _state_dir, db in dbs)
    if view is not None and view["newest"] == newest:
        view = state_dir_mod.OrderingView(
            dirs=dirs, newest=newest, mentions=view["mentions"], markers=markers, first_outputs=None
        )
    else:
        view = _list_runs(dbs, dirs, newest, markers, prefetch=ordering is not None)
    if ordering is not None:
        ordering["views"][dirs] = view
    return view


def _list_runs(
    dbs: list[tuple[pathlib.Path, state.StateDB]],
    dirs: tuple[str, ...],
    newest: tuple[str | None, ...],
    markers: tuple[int, ...],
    *,
    prefetch: bool,
) -> state_dir_mod.OrderingView:
    """List both databases' runs, and, with ``prefetch``, the entry of each stage's newest run.

    Each stage's runs that ran or kept it are ordered newest first across the databases, in the
    order the databases are consulted for runs with equal IDs. Each newest run's run-cache entry
    is looked up in one transaction per database, in the order ``_run_cache_outputs`` looks them
    up. A record prefetches, as it keeps the view for many stages; a call outside one does not.
    """
    mentions = dict[str, list[state_dir_mod.StageRunMention]]()
    for (_state_dir, db), state_dir in zip(dbs, dirs, strict=True):
        for manifest in db.list_runs(limit=_ALL_RUNS):
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
    for _state_dir, db in dbs:
        if not pending:
            break
        for (name, _input_hash), entry in db.lookup_run_cache_many(pending.items()).items():
            first_outputs[name] = sorted(output["hash"] for output in entry["output_hashes"])
            del pending[name]
    return state_dir_mod.OrderingView(
        dirs=dirs, newest=newest, mentions=mentions, markers=markers, first_outputs=first_outputs
    )


def _run_cache_outputs(
    stage_name: str, input_hash: str, dbs: list[tuple[pathlib.Path, state.StateDB]]
) -> list[str] | None:
    """The output hashes a run-cache entry records for ``input_hash``, from either database."""
    for _state_dir, db in dbs:
        entry = db.lookup_run_cache(stage_name, input_hash)
        if entry is not None:
            return sorted(output["hash"] for output in entry["output_hashes"])
    return None


def _end_lock_ordering(invocation: state_dir_mod.StateDirInvocation) -> None:
    """As a record ends: report its stale ``.pivot`` locks, then close the databases it held."""
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
        for opener, _inode, db in held:
            if opener == pid:
                db.close()


def _find_stale_locks(
    stages_dir: pathlib.Path, other_stages_dir: pathlib.Path
) -> tuple[list[pathlib.Path], bool] | None:
    """Every lock in ``other_stages_dir`` that its stage's lock in ``stages_dir`` replaced, and
    whether that is every lock there; None when an ordering cannot be read.

    Each stage with a lock in both directories is ordered as ``resolve_stage_lock_path`` orders
    it, with the answers the record already kept, so a command that read every stage parses no
    lock again. When any ordering cannot be read, it answers None: the notice suggests removing
    files, so it names only locks it knows Pivot no longer uses.
    """
    import lmdb

    from pivot.storage import state

    state_dir, other_state_dir = stages_dir.parent, other_stages_dir.parent
    try:
        names = sorted(
            posixpath.join(os.path.relpath(directory, other_stages_dir), name)
            .removeprefix("./")
            .removesuffix(".lock")
            for directory, _subdirectories, files in os.walk(other_stages_dir)
            for name in files
            if name.endswith(".lock")
        )
        stale = list[pathlib.Path]()
        with contextlib.ExitStack() as stack:
            dbs = _ordering_databases(stack, (state_dir, other_state_dir))
            view = _ordering_view(dbs)
            for stage_name in names:
                primary_path = stages_dir / f"{stage_name}.lock"
                other_path = other_stages_dir / f"{stage_name}.lock"
                if stage_name_error(stage_name) is not None:
                    continue
                primary_identity = _lock_identity(primary_path)
                other_identity = _lock_identity(other_path)
                if primary_identity is None or other_identity is None:
                    continue
                primary = _LockFile(primary_path, primary_identity)
                other = _LockFile(other_path, other_identity)
                if not _order_with(stage_name, primary, other, dbs, view, other_state_dir):
                    stale.append(other_path)
    except (exceptions.PivotError, state.StateDBError, lmdb.Error, OSError, ValueError):
        logger.debug("Cannot order the locks under %s; not reporting stale locks", other_stages_dir)
        return None
    return stale, len(stale) == len(names)


state_dir_mod.on_record_end(_end_lock_ordering)


def stage_name_error(name: str) -> str | None:
    """Return why ``name`` cannot name a lock file, or None if it can.

    This is the single rule for stage, pipeline and variant names. A name is a relative path
    of non-empty segments under the state directory's ``stages`` directory: no leading or
    doubled ``/`, and no ``.`` or ``..`` segment, so the lock file can never land outside that
    directory.
    """
    if not name or not _VALID_STAGE_NAME.fullmatch(name):
        return "use only letters, digits and the characters _ @ . / -"
    if any(segment in {"", ".", ".."} for segment in name.split("/")):
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


def is_lock_data(data: object) -> TypeGuard[StorageLockData]:
    """Validate that parsed YAML has the storage format's shape, field by field.

    A lock that fails is treated as missing by every caller ("no lock = needs re-run"), so a
    hand-edited or partially written lock makes the stage stale instead of crashing a reader
    or letting status and repro disagree. The first failing field is logged at debug.
    """
    error = lock_data_error(data)
    if error is not None:
        logger.debug("Lock data rejected: %s", error)
        return False
    return True


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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


def _manifest_error(value: object, field: str) -> str | UnsafeLockPath | None:
    if not isinstance(value, list):
        return field
    for index, raw in enumerate(cast("list[object]", value)):
        entry_field = f"{field}[{index}]"
        if not isinstance(raw, dict):
            return entry_field
        entry = cast("dict[str, object]", raw)
        if error := _path_error(entry.get("relpath"), f"{entry_field}.relpath"):
            return error
        if not cache.is_valid_hash(entry.get("hash")):
            return f"{entry_field}.hash"
        if not _is_int(entry.get("size")):
            return f"{entry_field}.size"
        if not isinstance(entry.get("isexec"), bool):
            return f"{entry_field}.isexec"
    return None


def _empty_dirs_error(value: object, field: str) -> str | UnsafeLockPath | None:
    if not isinstance(value, list):
        return field
    for index, relpath in enumerate(cast("list[object]", value)):
        if error := _path_error(relpath, f"{field}[{index}]"):
            return error
    return None


def _entry_error(raw: object, field: str, list_key: str) -> str | UnsafeLockPath | None:
    """Check one ``deps`` or ``outs`` entry against DepEntry / OutEntry."""
    if not isinstance(raw, dict):
        return field
    entry = cast("dict[str, object]", raw)
    # Every recorded hash is a cache key (cache.get_cache_path). Null and empty hashes come
    # from stages never executed locally (e.g. pulled with incomplete state); a hand-edited
    # short or non-hex one would otherwise fail every restore instead of meaning stale.
    if not cache.is_valid_hash(entry.get("hash")):
        return f"{field}.hash"
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
    if "size" in entry and not _is_int(entry["size"]):
        return f"{field}.size"
    if "manifest" in entry and (error := _manifest_error(entry["manifest"], f"{field}.manifest")):
        return error
    if "empty_dirs" in entry:
        if "manifest" not in entry:
            return f"{field}.empty_dirs"
        return _empty_dirs_error(entry["empty_dirs"], f"{field}.empty_dirs")
    return None


def lock_data_error(data: object) -> str | UnsafeLockPath | None:
    """Return the first reason parsed data cannot safely be consumed as a lock, or None.

    The reason is the first field that breaks StorageLockData, or an ``UnsafeLockPath`` for a
    recorded path that is not project-relative POSIX form. Extra top-level keys are allowed for
    forward compatibility.
    """
    if not isinstance(data, dict):
        return "<document>"
    document = cast("dict[object, object]", data)
    for key in sorted(_REQUIRED_LOCK_KEYS):
        if key not in document:
            return key
    if not _is_str_mapping(document["code_manifest"], str_values=True):
        return "code_manifest"
    if not _is_str_mapping(document["params"], str_values=False):
        return "params"
    for list_key in ("deps", "outs"):
        entries = document[list_key]
        if not isinstance(entries, list):
            return list_key
        for index, raw in enumerate(cast("list[object]", entries)):
            error = _entry_error(raw, f"{list_key}[{index}]", list_key)
            if error is not None:
                return error
    if "schema_version" in document and not _is_int(document["schema_version"]):
        return "schema_version"
    return None


def _record_dir_fields(entry: DepEntry | OutEntry, hash_info: HashInfo) -> None:
    """Copy a directory hash's manifest and empty directories onto a lock entry.

    Each manifest entry is rebuilt in ``DirManifestEntry``'s field order: a hash read back from
    the run cache carries its entries' keys sorted, and a lock written from it would otherwise
    differ in bytes from the lock the same run writes.
    """
    if is_dir_hash(hash_info):
        entry["manifest"] = [
            DirManifestEntry(
                relpath=item["relpath"], hash=item["hash"], size=item["size"], isexec=item["isexec"]
            )
            for item in hash_info["manifest"]
        ]
        if "empty_dirs" in hash_info:
            entry["empty_dirs"] = hash_info["empty_dirs"]


def entry_hash_info(entry: DepEntry | OutEntry) -> HashInfo:
    """Return the hash a lock ``deps`` or ``outs`` entry records."""
    if "manifest" not in entry:
        return FileHash(hash=entry["hash"])
    dir_hash = DirHash(hash=entry["hash"], manifest=entry["manifest"])
    if "empty_dirs" in entry and entry["empty_dirs"]:
        dir_hash["empty_dirs"] = entry["empty_dirs"]
    return dir_hash


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

    return StorageLockData(
        schema_version=2,
        code_manifest=sorted_code_manifest,
        params=data["params"],
        deps=deps_list,
        outs=outs_list,
    )


def _member_relpaths(entry: DepEntry | OutEntry) -> Iterator[str]:
    """Yield the manifest relpaths and empty directories a directory entry records."""
    if "manifest" in entry:
        for item in entry["manifest"]:
            yield item["relpath"]
    if "empty_dirs" in entry:
        yield from entry["empty_dirs"]


def _recorded_paths(data: StorageLockData) -> Iterator[tuple[str, str | None, str | None]]:
    """Yield each path a lock records with its output path or dependency name.

    A yielded output path names the directory that owns a member. A yielded dependency names
    the directory dependency that owns its member. The caller formats either only when it must
    report a validation refusal or control-character warning.
    """
    for dep in data["deps"]:
        dependency = identity_key(ArtifactIdentity(dep["producer"], dep["key"]))
        for relpath in _member_relpaths(dep):
            yield relpath, None, dependency
    for out in data["outs"]:
        out_path = out["path"] if "path" in out else None
        if out_path is not None:
            yield out_path, None, None
        for relpath in _member_relpaths(out):
            yield relpath, out_path, None


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


class StageLock:
    """Manages lock file for a single pipeline stage."""

    stage_name: str
    path: pathlib.Path
    stages_dir: pathlib.Path

    def __init__(self, stage_name: str, stages_dir: pathlib.Path) -> None:
        """Initialize a stage lock for the given stage in stages_dir."""
        reason = stage_name_error(stage_name)
        if reason is not None:
            raise ValueError(f"Invalid stage name {stage_name!r}: {reason}")
        self.stage_name = stage_name
        self.stages_dir = stages_dir
        self.path = stages_dir / f"{stage_name}.lock"

    def read(self) -> LockData | None:
        """Read the stage's current lock, in either state directory (resolve_stage_lock_path)."""
        path, valid = _resolve_stage_lock(self.stages_dir, self.stage_name)
        if valid is not None:
            return _convert_from_storage_format(valid, stage_name=self.stage_name)
        try:
            with open(path) as f:
                data: object = yaml.load(f, Loader=yaml_config.Loader)
            if not is_lock_data(data):
                error = lock_data_error(data)
                assert error is not None
                if isinstance(error, UnsafeLockPath):
                    logger.warning(
                        "Lock file %s failed validation: %s; treating it as missing",
                        path,
                        error,
                    )
                else:
                    logger.debug("Lock file %s failed validation; treating it as missing", path)
                return None
            return _convert_from_storage_format(data, stage_name=self.stage_name)
        except FileNotFoundError:
            return None  # Normal case - lock doesn't exist yet
        except (UnicodeDecodeError, yaml.YAMLError) as e:
            logger.warning("Failed to parse lock file %s: %s", path, e)
            return None

    def write(self, data: LockData) -> None:
        """Write lock file atomically, converting to storage format.

        Every recorded path must pass ``path_policy.recorded_path_error``, the rule ``read``
        applies, so this never writes a lock that ``read`` refuses; a path that fails raises
        InvalidPathError naming it. A recorded name that holds a control character is written,
        and one warning names every such name.
        """
        storage_data = _convert_to_storage_format(data)
        with_control = list[str]()
        for recorded, out_path, dependency in _recorded_paths(storage_data):
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

        def write_yaml(fd: int) -> None:
            with os.fdopen(fd, "w") as f:
                yaml.dump(storage_data, f, Dumper=yaml_config.Dumper, sort_keys=False)

        cache.atomic_write_file(self.path, write_yaml)


def find_orphaned_locks(
    stages_dir: pathlib.Path,
    registered_stages: set[str],
) -> list[str]:
    """Find lock files in stages_dir that don't correspond to any registered stage.

    When a pipeline is restructured (stages renamed/removed), old lock files
    persist and old output files remain on disk. Detecting orphaned locks helps
    users identify stale artifacts from previous pipeline configurations.

    Returns:
        Sorted list of stage names that have lock files but aren't registered.
    """
    if not stages_dir.is_dir():
        return []

    orphaned = list[str]()
    suffix = ".lock"

    for lock_file in stages_dir.rglob(f"*{suffix}"):
        # Derive stage name from relative path (e.g., "base/stage_name.lock" -> "base/stage_name")
        rel = lock_file.relative_to(stages_dir)
        stage_name = str(rel)[: -len(suffix)]
        if stage_name not in registered_stages:
            orphaned.append(stage_name)

    orphaned.sort()
    return orphaned

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
import posixpath
import re
from typing import TYPE_CHECKING, TypeGuard, cast, override

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
    import pathlib
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
    primary_path = stages_dir / f"{stage_name}.lock"
    other_stages_dir = state_dir_mod.other_stages_dir_for(stages_dir)
    if other_stages_dir is None:
        return primary_path
    other_path = other_stages_dir / f"{stage_name}.lock"
    if not other_path.exists():
        return primary_path
    if primary_path.exists() and not _other_lock_is_current(
        stage_name, primary_path, other_path, stages_dir.parent, other_stages_dir.parent
    ):
        return primary_path
    state_dir_mod.report_legacy_state_dir(stages_dir.parent)
    return other_path


# list_runs reads at most this many runs: every run a database keeps (`prune_runs`).
_ALL_RUNS = 1 << 31


def _other_lock_is_current(
    stage_name: str,
    primary_path: pathlib.Path,
    other_path: pathlib.Path,
    primary_state_dir: pathlib.Path,
    other_state_dir: pathlib.Path,
) -> bool:
    """Whether, of a stage's locks in both state directories, the other directory's is current.

    The run records of both directories' databases answer: the stage's runs that ran or kept it,
    newest first, ordered by their timestamp-prefixed IDs across the two databases; a lock file's
    modification time is never read. The first of them whose run-cache entry holds one lock's
    outputs names that lock. A run whose outputs are in neither lock (a ``--no-commit`` run) defers
    to an older one. When a run has no run-cache entry (a result that is not reusable, or a pruned
    one), the newest run decides: the other lock is current when that run ran the stage there,
    since a run writes its lock where it records itself. Any failure to read keeps the first lock.
    """
    import lmdb

    from pivot.storage import state

    try:
        with contextlib.ExitStack() as stack:
            dbs = [
                (state_dir, stack.enter_context(state.StateDB(state_dir, readonly=True)))
                for state_dir in (primary_state_dir, other_state_dir)
                if (state_dir / "state.lmdb").is_dir()
            ]
            mentions = sorted(
                (
                    (mention, state_dir)
                    for state_dir, db in dbs
                    for mention in _stage_runs(state_dir, db).get(stage_name, ())
                ),
                key=lambda item: item[0]["run_id"],
                reverse=True,
            )
            if not mentions:
                return False
            primary_outputs = _lock_output_hashes(primary_path)
            other_outputs = _lock_output_hashes(other_path)
            for mention, _state_dir in mentions:
                outputs = _run_cache_outputs(stage_name, mention["input_hash"], dbs)
                if outputs is None:
                    break
                if outputs == primary_outputs:
                    return False
                if outputs == other_outputs:
                    return True
            newest, newest_dir = mentions[0]
            return newest["ran"] and newest_dir == other_state_dir
    except (exceptions.PivotError, state.StateDBError, lmdb.Error, OSError, ValueError):
        logger.debug("Cannot order the locks of %s; reading %s", stage_name, primary_path)
        return False


def _stage_runs(
    state_dir: pathlib.Path, db: state.StateDB
) -> dict[str, list[state_dir_mod.StageRunMention]]:
    """Each stage's runs in ``db`` that ran or kept it, newest first.

    Read once per state-directory record for each newest run the database holds, so a command
    lists the runs once, and a ``--watch`` session lists them again after it records one.
    """
    newest = db.newest_run_id()
    if newest is None:
        return {}
    cache = state_dir_mod.stage_run_cache()
    if cache is not None and state_dir in cache and cache[state_dir][0] == newest:
        return cache[state_dir][1]
    runs = dict[str, list[state_dir_mod.StageRunMention]]()
    for manifest in db.list_runs(limit=_ALL_RUNS):
        for name, record in manifest["stages"].items():
            input_hash = record["input_hash"]
            if input_hash is None or record["status"] not in (StageStatus.RAN, StageStatus.CACHED):
                continue
            runs.setdefault(name, []).append(
                state_dir_mod.StageRunMention(
                    run_id=manifest["run_id"],
                    input_hash=input_hash,
                    ran=record["status"] == StageStatus.RAN,
                )
            )
    if cache is not None:
        cache[state_dir] = (newest, runs)
    return runs


def _run_cache_outputs(
    stage_name: str, input_hash: str, dbs: list[tuple[pathlib.Path, state.StateDB]]
) -> list[str] | None:
    """The output hashes a run-cache entry records for ``input_hash``, from either database."""
    for _state_dir, db in dbs:
        entry = db.lookup_run_cache(stage_name, input_hash)
        if entry is not None:
            return sorted(output["hash"] for output in entry["output_hashes"])
    return None


def _lock_output_hashes(path: pathlib.Path) -> list[str] | None:
    """The output hashes the lock at ``path`` records; None when it cannot be read."""
    try:
        with open(path) as f:
            data: object = yaml.load(f, Loader=yaml_config.Loader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    if not is_lock_data(data):
        return None
    return sorted(out["hash"] for out in data["outs"])


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
        path = resolve_stage_lock_path(self.stages_dir, self.stage_name)
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

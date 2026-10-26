"""Per-stage lock files for tracking pipeline state.

StageLock provides persistent lock files (.lock) for change detection,
storing fingerprints, params, and hashes to detect when re-runs are needed.
A lock records a directory by ``DirHash`` (hash, size, file count); the directory's listing is
committed beside the lock (``storage/listing.py``), and ``StageLock.write`` writes both.

For runtime mutual exclusion during concurrent execution, see
``pivot.storage.artifact_lock`` which uses flock-based artifact locks.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import pathlib
import posixpath
import re
from typing import TYPE_CHECKING, NamedTuple, TypeGuard, cast, override

import yaml

from pivot import exceptions, path_policy, yaml_config
from pivot.storage import cache, listing
from pivot.types import (
    ArtifactIdentity,
    DepEntry,
    DirHash,
    FileHash,
    HashInfo,
    LockData,
    OutEntry,
    StorageLockData,
    identity_from_key,
    identity_key,
    is_dir_hash,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

logger = logging.getLogger(__name__)

_VALID_STAGE_NAME = re.compile(
    r"[a-zA-Z0-9_@./-]+"
)  # Allow / for pipeline-prefixed names, . for DVC matrix keys
_VALID_VARIANT_NAME = re.compile(r"[a-zA-Z0-9_.-]+")
_MAX_STAGE_NAME_LEN = 200  # Leave room for ".lock" suffix within filesystem NAME_MAX (255)
_REQUIRED_LOCK_KEYS = frozenset({"code_manifest", "params", "deps", "outs"})
_LOCK_SUFFIX = ".lock"

SCHEMA_VERSION = 3
STAGES_REL_PATH = ".pivot/stages"


def get_stages_dir(state_dir: pathlib.Path) -> pathlib.Path:
    """Return the stages directory for lock files.

    Lock files are stored in .pivot/stages/ (git-tracked) rather than
    .pivot/cache/stages/ so they can be versioned for reproducibility.
    """
    return state_dir / "stages"


def stage_name_error(name: str) -> str | None:
    """Return why ``name`` cannot name a lock file, or None if it can.

    This is the single rule for stage, pipeline and variant names. A name is a relative path
    of non-empty segments under ``.pivot/stages``: no leading or doubled ``/``, and no ``.``
    or ``..`` segment, so the lock file can never land outside that directory.
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
    if "nfiles" in entry:
        if not _is_count(entry["nfiles"]):
            return f"{field}.nfiles"
        if not _is_count(entry.get("size")):
            return f"{field}.size"
    elif "size" in entry and not _is_count(entry["size"]):
        return f"{field}.size"
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
    if document.get("schema_version") != SCHEMA_VERSION:
        return "schema_version"
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
    return None


def _is_old_format(data: object) -> bool:
    """Whether parsed YAML is a lock an older pivot wrote: a mapping with the lock keys whose
    ``schema_version`` is not this pivot's."""
    if not isinstance(data, dict):
        return False
    document = cast("dict[object, object]", data)
    return (
        document.keys() >= _REQUIRED_LOCK_KEYS and document.get("schema_version") != SCHEMA_VERSION
    )


@functools.cache
def _warn_old_format(path: str, mtime_ns: int) -> None:
    """Warn once per process for each old lock, as it is on disk (``mtime_ns``)."""
    del mtime_ns
    logger.warning(
        "%s was written by an older pivot; pivot reads it as missing, so its stage is decided "
        + "afresh (it runs, or its run-cache entry restores it) and the lock is rewritten. An "
        + "incremental stage keeps the output this lock records",
        path,
    )


class CommittedLock(NamedTuple):
    """A lock read from git history or from another repository, and the listings of the
    directory outputs an older pivot recorded inside it (by output key)."""

    data: StorageLockData
    listings: dict[str | None, listing.DirListing]


def _convert_old_entry(raw: object) -> tuple[object, listing.DirListing | None] | None:
    """An older pivot's lock entry as this format's, and its inline listing if it is a
    directory; None when the inline listing is malformed."""
    if not isinstance(raw, dict):
        return raw, None
    entry = cast("dict[str, object]", raw)
    if "manifest" not in entry:
        return entry, None
    members = listing.from_inline(entry["manifest"], entry.get("empty_dirs", []))
    if members is None:
        return None
    record = listing.record_of(members, listing.encode(members))
    converted = {k: v for k, v in entry.items() if k not in ("manifest", "empty_dirs")}
    converted.update(hash=record["hash"], size=record["size"], nfiles=record["nfiles"])
    return converted, members


def parse_committed(document: object) -> CommittedLock | UnsafeLockPath | None:
    """A lock as git history or another repository holds it: this format, or an older pivot's
    format, read whole so a past revision stays readable.

    An older pivot's directory entry lists its files inline; it becomes this format's record
    (the hash of its listing) and its listing is returned beside the data. None when the
    document is not a lock; an ``UnsafeLockPath`` when it records a path outside its project.
    """
    if not _is_old_format(document):
        error = lock_data_error(document)
        if isinstance(error, UnsafeLockPath):
            return error
        if error is not None:
            return None
        return CommittedLock(cast("StorageLockData", document), {})
    old = cast("dict[str, object]", document)
    listings = dict[str | None, listing.DirListing]()
    converted: dict[str, object] = {**old, "schema_version": SCHEMA_VERSION}
    for list_key in ("deps", "outs"):
        entries = old[list_key]
        if not isinstance(entries, list):
            return None
        new_entries = list[object]()
        for raw in cast("list[object]", entries):
            result = _convert_old_entry(raw)
            if result is None:
                return None
            new_entry, members = result
            new_entries.append(new_entry)
            if list_key == "outs" and members is not None:
                key = cast("dict[str, object]", new_entry).get("key")
                if key is None or isinstance(key, str):
                    listings[key] = members
        converted[list_key] = new_entries
    document_now: object = converted
    error = lock_data_error(document_now)
    if isinstance(error, UnsafeLockPath):
        return error
    if error is not None or not is_lock_data(document_now):
        return None
    return CommittedLock(document_now, listings)


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

    return StorageLockData(
        schema_version=SCHEMA_VERSION,
        code_manifest=sorted_code_manifest,
        params=data["params"],
        deps=deps_list,
        outs=outs_list,
    )


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


@dataclasses.dataclass(frozen=True)
class _ListingWrite:
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

    def __init__(self, stage_name: str, stages_dir: pathlib.Path) -> None:
        """Initialize a stage lock for the given stage in stages_dir."""
        reason = stage_name_error(stage_name)
        if reason is not None:
            raise ValueError(f"Invalid stage name {stage_name!r}: {reason}")
        self.stage_name = stage_name
        self.stages_dir = stages_dir
        self.path = stages_dir / f"{stage_name}{_LOCK_SUFFIX}"

    def _load(self) -> object | None:
        """The lock file's parsed YAML, or None when it is absent or unparseable."""
        try:
            with open(self.path) as f:
                return yaml.load(f, Loader=yaml_config.Loader)
        except FileNotFoundError:
            return None  # Normal case - lock doesn't exist yet
        except (UnicodeDecodeError, yaml.YAMLError) as e:
            logger.warning("Failed to parse lock file %s: %s", self.path, e)
            return None

    def read(self, *, warn_old_format: bool = True) -> LockData | None:
        """Read lock file, converting storage format to internal format.

        A lock an older pivot wrote is read as missing, so its stage runs once and the lock is
        rewritten; ``warn_old_format`` logs that, once per process for each such lock. A worker
        reads without it: the process that dispatched the stage has already warned.
        """
        data = self._load()
        if data is None:
            return None
        if _is_old_format(data):
            if warn_old_format:
                try:
                    mtime_ns = self.path.stat().st_mtime_ns
                except FileNotFoundError:
                    return None
                _warn_old_format(str(self.path), mtime_ns)
            return None
        if not is_lock_data(data):
            error = lock_data_error(data)
            assert error is not None
            if isinstance(error, UnsafeLockPath):
                logger.warning(
                    "Lock file %s failed validation: %s; treating it as missing",
                    self.path,
                    error,
                )
            else:
                logger.debug("Lock file %s failed validation; treating it as missing", self.path)
            return None
        return _convert_from_storage_format(data, stage_name=self.stage_name)

    def read_old_format_file_outputs(self) -> dict[ArtifactIdentity, FileHash]:
        """The file outputs a lock an older pivot wrote records, by identity; empty for any
        other lock.

        An incremental stage reads its prior output from these, once, when its lock is in an
        old format: running it from nothing instead would drop what it accumulated.
        """
        data = self._load()
        if not _is_old_format(data):
            return {}
        outs = cast("dict[object, object]", data)["outs"]
        found = dict[ArtifactIdentity, FileHash]()
        if not isinstance(outs, list):
            return found
        for raw in cast("list[object]", outs):
            if not isinstance(raw, dict):
                continue
            entry = cast("dict[str, object]", raw)
            key = entry.get("key")
            file_hash = entry.get("hash")
            if "manifest" in entry or not (key is None or isinstance(key, str)):
                continue
            if not cache.is_valid_hash(file_hash):
                continue
            found[ArtifactIdentity(self.stage_name, key)] = FileHash(hash=cast("str", file_hash))
        return found

    def output_listing_path(self, identity: ArtifactIdentity) -> pathlib.Path:
        """Where the listing of this stage's directory output ``identity`` is committed."""
        return listing.output_listing_path(self.stages_dir, identity)

    def write(
        self,
        data: LockData,
        *,
        hashed_dirs: Mapping[str, listing.HashedDirectory] | None = None,
        input_paths: Mapping[ArtifactIdentity, str] | None = None,
        cache_dir: pathlib.Path | None = None,
    ) -> list[pathlib.Path]:
        """Write the lock file and the listing of each directory it records, atomically, and
        return the paths of the files it wrote or left unchanged.

        A directory output's listing goes to ``listing.output_listing_path``; a directory
        dependency named in ``input_paths`` (a pipeline input, by its project-relative path)
        has its listing at ``listing.input_listing_path``. A listing comes from ``hashed_dirs``
        (by directory hash), else from the cache object in ``cache_dir``. Each listing is
        written before the lock, so no lock names a listing not yet written; a file already
        holding the bytes is left as it is.

        Every recorded path must pass ``path_policy.recorded_path_error``, the rule ``read``
        applies to a lock's paths, so this never writes a lock that ``read`` refuses; a path that
        fails raises InvalidPathError naming it. A recorded name that holds a control character
        is written, and one warning names every such name.
        """
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
        serialized = yaml.dump(storage_data, Dumper=yaml_config.Dumper, sort_keys=False)

        written = list[pathlib.Path]()
        for item in to_write:
            if cache_dir is not None:
                listing.store_object(cache_dir, item.record["hash"], item.data)
            listing.write(item.target, item.data)
            written.append(item.target)

        def write_yaml(fd: int) -> None:
            with os.fdopen(fd, "w") as f:
                _ = f.write(serialized)

        cache.atomic_write_file(self.path, write_yaml)
        written.append(self.path)
        return written


def find_orphaned_locks(
    stages_dir: pathlib.Path,
    registered_stages: set[str],
    registered_listings: Iterable[pathlib.Path] = (),
) -> list[str]:
    """Find lock and listing files in stages_dir that belong to no registered stage or input.

    When a pipeline is restructured (stages renamed/removed), old lock files
    persist and old output files remain on disk. Detecting orphaned locks helps
    users identify stale artifacts from previous pipeline configurations.
    A listing file is orphaned when ``registered_listings`` (the listing paths of the
    registered stages' outputs and inputs) does not hold it.

    Returns:
        Sorted list of the stage names that have lock files but aren't registered, and of the
        paths, relative to stages_dir, of the orphaned listing files.
    """
    if not stages_dir.is_dir():
        return []

    orphaned = list[str]()
    for lock_file in stages_dir.rglob(f"*{_LOCK_SUFFIX}"):
        if not lock_file.is_file():
            continue
        # Derive stage name from relative path (e.g., "base/stage_name.lock" -> "base/stage_name")
        rel = lock_file.relative_to(stages_dir)
        stage_name = str(rel)[: -len(_LOCK_SUFFIX)]
        if stage_name not in registered_stages:
            orphaned.append(stage_name)

    known = set(registered_listings)
    for listing_file in stages_dir.rglob(f"*{listing.SUFFIX}"):
        if listing_file.is_file() and listing_file not in known:
            orphaned.append(str(listing_file.relative_to(stages_dir)))

    orphaned.sort()
    return orphaned

"""Per-stage lock files for tracking pipeline state.

StageLock provides persistent lock files (.lock) for change detection,
storing fingerprints, params, and hashes to detect when re-runs are needed.
A lock records a directory by its listing's hash, its total size and its file count
(``listing``, ``size``, ``nfiles``); the directory's listing is committed beside the lock
(``storage/listing.py``), and ``StageLock.write`` writes both. A lock an older pivot wrote is
read as it is (``StageLock.read``).

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
from typing import TYPE_CHECKING, NamedTuple, cast, override

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
_DIR_HASH_KEY = "listing"
"""The key a lock entry records a directory's hash under; a file's is ``hash``. Every released
pivot reads an entry that has a ``hash`` and no inline ``manifest`` as one file, and would
restore a directory's listing in the directory's place; finding no ``hash``, it refuses the
lock instead, and decides the stage afresh."""

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
    """Check one ``deps`` or ``outs`` entry as a lock file holds it: a file's hash under
    ``hash``, a directory's under ``listing`` (``_DIR_HASH_KEY``) beside its size and file
    count."""
    if not isinstance(raw, dict):
        return field
    entry = cast("dict[str, object]", raw)
    # Every recorded hash is a cache key (cache.get_cache_path). Null and empty hashes come
    # from stages never executed locally (e.g. pulled with incomplete state); a hand-edited
    # short or non-hex one would otherwise fail every restore instead of meaning stale.
    if _DIR_HASH_KEY in entry:
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
    for list_key in ("deps", "outs"):
        entries = document[list_key]
        if not isinstance(entries, list):
            return list_key
        for index, raw in enumerate(cast("list[object]", entries)):
            error = _entry_error(raw, f"{list_key}[{index}]", list_key)
            if error is not None:
                return error
    return None


def _from_document(document: object) -> StorageLockData:
    """A lock file's data, ``lock_data_error`` having passed it, with each directory's hash
    under ``hash`` as ``StorageLockData`` holds it."""
    data = dict(cast("dict[str, object]", document))
    for list_key in ("deps", "outs"):
        data[list_key] = [
            {("hash" if key == _DIR_HASH_KEY else key): value for key, value in entry.items()}
            for entry in cast("list[dict[str, object]]", data[list_key])
        ]
    return cast("StorageLockData", cast("object", data))


def _to_document(data: StorageLockData) -> dict[str, object]:
    """``data`` as its lock file holds it: each directory's hash under ``listing``."""

    def entry(item: DepEntry | OutEntry) -> dict[str, object]:
        if "nfiles" not in item:
            return dict(item)
        return {(_DIR_HASH_KEY if key == "hash" else key): value for key, value in item.items()}

    return {
        **data,
        "deps": [entry(item) for item in data["deps"]],
        "outs": [entry(item) for item in data["outs"]],
    }


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


def _convert_old(data: object) -> _ConvertedLock | None:
    """An older pivot's lock (``_is_old_format``) as this format's document: each directory
    entry, which lists its files inline, becomes the record of its listing (the listing's hash,
    its total size and its file count). None when an inline listing is malformed or names a
    path outside its directory; the document still has to pass ``lock_data_error``."""
    old = cast("dict[str, object]", data)
    document: dict[str, object] = {**old, "schema_version": SCHEMA_VERSION}
    output_listings = dict[str | None, listing.DirListing]()
    listing_bytes = dict[str, bytes]()
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
            key = entry.get("key")
            if list_key == "outs" and (key is None or isinstance(key, str)):
                output_listings[key] = members
        document[list_key] = converted
    return _ConvertedLock(document, output_listings, listing_bytes)


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
            mtime_ns = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        _warn_format_once(str(self.path), mtime_ns, older)

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
        serialized = yaml.dump(
            _to_document(storage_data), Dumper=yaml_config.Dumper, sort_keys=False
        )

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


def committed_listing(lock_data: LockData, path: pathlib.Path) -> pathlib.Path | None:
    """``path``, the committed listing file of a directory ``lock_data`` records, for a reader
    to pass to ``listing.read``; None for a lock an older pivot wrote (``older_format``). Such a
    lock listed each directory inline, and ``StageLock.read`` put that listing in the cache; a
    listing file beside it is another lock's (a newer pivot's, before an older one reran the
    stage), and would be refused or warned about though nothing is wrong."""
    return None if "older_format" in lock_data else path


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

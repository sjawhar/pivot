"""Directory listings: the file a directory's hash is computed from, committed beside the lock
or ``.pvt`` that records the directory.

A lock or ``.pvt`` records a directory by ``DirHash`` (its hash, total size and file count).
The directory's members live in its listing, one line per file and one per deepest empty
directory, sorted by the raw bytes of the path. The directory's hash is the xxhash64 of the
listing's bytes. The listing is written in two places: a committed file a person can diff
(``output_listing_path``, ``input_listing_path``, ``tracked_listing_path``), and a cache object
at the directory's hash, so a restore, push or pull finds it even where the committed file is
absent. ``read`` takes either, and refuses bytes that do not hash to the recorded hash.

Line format::

    <hash> <size> <x|-> <path>\\n      a file (``x``: its owner may execute it)
    - - - <path>/\\n                   a deepest empty directory

A path is written as is when it is plain: valid UTF-8, no byte below 0x20, no 0x7f, no ``\\``,
not starting with ``"``, and not starting or ending with a space. Any other path is written
``"`` + escaped + ``"``, with ``\\\\``, ``\\"``, ``\\t``, ``\\n``, ``\\r``, and a three-digit
octal ``\\ooo`` for every other control byte, 0x7f, and every byte that is not part of valid
UTF-8; an empty directory's ``/`` goes inside the quotes. The listing is always valid UTF-8.
"""

from __future__ import annotations

import functools
import logging
import operator
import os
import re
from typing import TYPE_CHECKING, NewType, TypedDict, cast

import xxhash

from pivot import exceptions
from pivot.storage import cache
from pivot.types import ArtifactIdentity, DirHash, DirManifestEntry

if TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)

SUFFIX = ".listing"
INPUTS_DIR = "=inputs"
_NAME_MAX = 255
_HASH = re.compile(r"[0-9a-f]{16}")
_SIZE = re.compile(r"0|[1-9][0-9]*")
_ENC_KEEP = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.@-")
_NAMED_ESCAPES = {"\\": "\\\\", '"': '\\"', "\t": "\\t", "\n": "\\n", "\r": "\\r"}
_UNESCAPES = {"\\": 0x5C, '"': 0x22, "t": 0x09, "n": 0x0A, "r": 0x0D}
# A path field written as is: no control byte, DEL or backslash, no leading quote or space, no
# trailing space. Checked by the regex engine, not byte by byte in Python, since hashing a
# directory encodes every member's path.
_PLAIN = re.compile(rb'(?![" ])[^\x00-\x1f\x7f\\]*(?<! )')


class DirListing(TypedDict):
    """A directory's members: its files and its deepest empty directories."""

    files: list[DirManifestEntry]
    empty_dirs: list[str]  # relpaths, no trailing "/"


class HashedDirectory(TypedDict):
    """A directory as hashed: the record a lock keeps, its listing, and the listing's bytes."""

    record: DirHash
    listing: DirListing
    data: bytes  # encode(listing); digest(data) == record["hash"]


# =============================================================================
# Bytes
# =============================================================================


def digest(data: bytes) -> str:
    """The directory hash a listing's bytes give."""
    return xxhash.xxh64(data).hexdigest()


def record_of(listing: DirListing, data: bytes) -> DirHash:
    """The ``DirHash`` a lock records for ``listing``, whose bytes are ``data``."""
    return DirHash(
        hash=digest(data),
        size=sum(entry["size"] for entry in listing["files"]),
        nfiles=len(listing["files"]),
    )


def _field(raw: bytes) -> str:
    """The path field for a path whose bytes are ``raw``."""
    if _PLAIN.fullmatch(raw):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            pass
    out = ['"']
    for char in raw.decode("utf-8", "surrogateescape"):
        code = ord(char)
        if 0xDC80 <= code <= 0xDCFF:
            out.append(f"\\{code - 0xDC00:03o}")
        elif char in _NAMED_ESCAPES:
            out.append(_NAMED_ESCAPES[char])
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\{code:03o}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


# In the NUL-joined paths of a listing: a byte that forces a path to be quoted, a leading quote
# or space, or a trailing space. NUL never occurs in a path.
_NEEDS_QUOTING = re.compile(r'[\x01-\x1f\x7f\\]|(?:\A|\x00)[" ]| (?:\x00|\Z)')


def _all_plain(paths: list[str]) -> bool:
    """Whether every path is valid UTF-8 needing no quotes, checked in two C passes over all
    of them: for these, sorting by ``str`` is sorting by UTF-8 bytes."""
    joined = "\0".join(paths)
    try:
        joined.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return _NEEDS_QUOTING.search(joined) is None


def encode(listing: DirListing) -> bytes:
    """The listing's bytes: one line per file and per empty directory, sorted by path bytes."""
    files = listing["files"]
    empty_dirs = listing["empty_dirs"]
    keys = [entry["relpath"] for entry in files]
    keys.extend(f"{relpath}/" for relpath in empty_dirs)
    if _all_plain(keys):
        plain = [
            (key, f"{entry['hash']} {entry['size']} {'x' if entry['isexec'] else '-'} {key}\n")
            for key, entry in zip(keys, files, strict=False)
        ]
        plain.extend((key, f"- - - {key}\n") for key in keys[len(files) :])
        plain.sort(key=operator.itemgetter(0))
        return "".join(line for _, line in plain).encode("utf-8")
    lines = list[tuple[bytes, str]]()
    for entry in files:
        raw = os.fsencode(entry["relpath"])
        flag = "x" if entry["isexec"] else "-"
        lines.append((raw, f"{entry['hash']} {entry['size']} {flag} {_field(raw)}\n"))
    for relpath in empty_dirs:
        raw = os.fsencode(relpath) + b"/"
        lines.append((raw, f"- - - {_field(raw)}\n"))
    lines.sort(key=operator.itemgetter(0))
    return "".join(line for _, line in lines).encode("utf-8")


def _unquote(field: str) -> bytes:
    """The path bytes a quoted field names."""
    if len(field) < 2 or not field.endswith('"'):
        raise exceptions.CorruptListingError(f"unterminated quoted path {field!r}")
    body = field[1:-1]
    out = bytearray()
    index = 0
    while index < len(body):
        char = body[index]
        if char == '"':
            raise exceptions.CorruptListingError(f"unescaped quote in path {field!r}")
        if char != "\\":
            out += char.encode("utf-8")
            index += 1
            continue
        escape = body[index + 1 : index + 2]
        if escape in _UNESCAPES:
            out.append(_UNESCAPES[escape])
            index += 2
        elif re.fullmatch(r"[0-3][0-7][0-7]", body[index + 1 : index + 4]):
            out.append(int(body[index + 1 : index + 4], 8))
            index += 4
        else:
            raise exceptions.CorruptListingError(f"bad escape in path {field!r}")
    return bytes(out)


def _relpath_error(raw: bytes) -> str | None:
    if not raw:
        return "an empty path"
    if b"\0" in raw:
        return "a NUL byte"
    if raw.startswith(b"/"):
        return "an absolute path"
    if any(segment in (b"", b".", b"..") for segment in raw.split(b"/")):
        return "an empty, '.' or '..' segment"
    return None


def decode(data: bytes) -> DirListing:
    """Parse a listing, refusing any that ``encode`` could not have written: so one set of
    members has one listing, and so one hash. A path is written as ``encode`` writes it, and no
    member lies inside a file or an empty directory the listing records."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise exceptions.CorruptListingError(f"not valid UTF-8 ({e.reason})") from e
    if text and not text.endswith("\n"):
        raise exceptions.CorruptListingError("the last line has no final newline")
    files = list[DirManifestEntry]()
    empty_dirs = list[str]()
    previous: bytes | None = None
    seen = set[bytes]()  # every member path so far; sorting puts an ancestor first
    for number, line in enumerate(text.split("\n")[:-1], start=1):
        parts = line.split(" ", 3)
        if len(parts) != 4:
            raise exceptions.CorruptListingError(f"line {number}: expected 4 fields")
        file_hash, size, flag, field = parts
        raw = _unquote(field) if field.startswith('"') else field.encode("utf-8")
        if _field(raw) != field:
            raise exceptions.CorruptListingError(
                f"line {number}: path {field!r} is not written as pivot writes it"
            )
        is_dir = file_hash == "-"
        if is_dir:
            if size != "-" or flag != "-" or not raw.endswith(b"/"):
                raise exceptions.CorruptListingError(
                    f"line {number}: an empty directory is '- - - <path>/'"
                )
            member = raw[:-1]
        else:
            if not _HASH.fullmatch(file_hash):
                raise exceptions.CorruptListingError(f"line {number}: bad hash {file_hash!r}")
            if not _SIZE.fullmatch(size):
                raise exceptions.CorruptListingError(f"line {number}: bad size {size!r}")
            if flag not in ("x", "-"):
                raise exceptions.CorruptListingError(f"line {number}: bad flag {flag!r}")
            if raw.endswith(b"/"):
                raise exceptions.CorruptListingError(f"line {number}: a file path ends in '/'")
            member = raw
        if error := _relpath_error(member):
            raise exceptions.CorruptListingError(f"line {number}: {error}")
        if previous is not None and raw <= previous:
            raise exceptions.CorruptListingError(f"line {number}: paths are not sorted and unique")
        previous = raw
        segments = member.split(b"/")
        for depth in range(1, len(segments) + 1):
            if (inner := b"/".join(segments[:depth])) in seen:
                raise exceptions.CorruptListingError(
                    f"line {number}: {os.fsdecode(member)!r} is inside, or is, "
                    + f"{os.fsdecode(inner)!r}, which the listing records as a file or an "
                    + "empty directory"
                )
        seen.add(member)
        relpath = os.fsdecode(member)
        if is_dir:
            empty_dirs.append(relpath)
        else:
            files.append(
                DirManifestEntry(
                    relpath=relpath, hash=file_hash, size=int(size), isexec=flag == "x"
                )
            )
    return DirListing(files=files, empty_dirs=empty_dirs)


def from_inline(raw_files: object, raw_dirs: object) -> DirListing | None:
    """The listing an older pivot recorded inline (a lock entry's or a ``.pvt``'s ``manifest``
    and ``empty_dirs``), or None when it is malformed or names a path outside its directory."""
    if not isinstance(raw_files, list) or not isinstance(raw_dirs, list):
        return None
    files = list[DirManifestEntry]()
    for raw in cast("list[object]", raw_files):
        if not isinstance(raw, dict):
            return None
        entry = cast("dict[str, object]", raw)
        relpath, file_hash = entry.get("relpath"), entry.get("hash")
        size, isexec = entry.get("size"), entry.get("isexec", False)
        if not (
            isinstance(relpath, str)
            and isinstance(file_hash, str)
            and _HASH.fullmatch(file_hash)
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size >= 0
            and isinstance(isexec, bool)
        ):
            return None
        files.append(DirManifestEntry(relpath=relpath, hash=file_hash, size=size, isexec=isexec))
    empty_dirs = cast("list[object]", raw_dirs)
    if not all(isinstance(relpath, str) for relpath in empty_dirs):
        return None
    members = DirListing(files=files, empty_dirs=cast("list[str]", empty_dirs))
    try:
        return decode(encode(members))  # refuses traversal, absolute paths and duplicates
    except exceptions.CorruptListingError:
        return None


# =============================================================================
# Paths
# =============================================================================


def _enc(value: str) -> str:
    """``value``'s UTF-8 bytes with ``[A-Za-z0-9_.@-]`` kept and every other byte ``%XX``."""
    return "".join(
        chr(byte) if byte in _ENC_KEEP else f"%{byte:02X}"
        for byte in value.encode("utf-8", "surrogateescape")
    )


def _short(value: str) -> str:
    """A deterministic file-name-safe abbreviation for a listing name that would exceed
    NAME_MAX. The original value remains in the lock or `.pvt`; only its sidecar's name changes.
    """
    return f"~{xxhash.xxh64(os.fsencode(value)).hexdigest()}"


def _checked(path: pathlib.Path, fallback: pathlib.Path) -> pathlib.Path:
    """``path`` if its final name fits NAME_MAX, else its deterministic ``fallback``."""
    return path if len(os.fsencode(path.name)) <= _NAME_MAX else fallback


def output_listing_path(stages_dir: pathlib.Path, identity: ArtifactIdentity) -> pathlib.Path:
    """``<stages_dir>/<stage_name>=<enc(key)>.listing``, beside the writing stage's lock.

    A key whose encoded bytes would exceed NAME_MAX uses `~` plus its xxhash64 instead; the
    original key remains in the lock, so this changes no artifact identity.
    """
    key = identity.key or ""
    return _checked(
        stages_dir / f"{identity.producer}={_enc(key)}{SUFFIX}",
        stages_dir / f"{identity.producer}={_short(key)}{SUFFIX}",
    )


def input_listing_path(stages_dir: pathlib.Path, input_path: str) -> pathlib.Path:
    """``<stages_dir>/=inputs/<enc(project-relative path)>.listing``, one per project input.

    An encoded path beyond NAME_MAX uses `~` plus its xxhash64.
    """
    inputs = stages_dir / INPUTS_DIR
    return _checked(
        inputs / f"{_enc(input_path)}{SUFFIX}",
        inputs / f"{_short(input_path)}{SUFFIX}",
    )


def tracked_listing_path(pvt_path: pathlib.Path) -> pathlib.Path:
    """``<path>.pvt.listing``, beside a tracked directory's ``.pvt``; a name that would exceed
    NAME_MAX uses `.~<xxhash64>.pvt.listing` in the same directory."""
    direct = pvt_path.with_name(pvt_path.name + SUFFIX)
    return _checked(
        direct,
        pvt_path.with_name(f".{_short(pvt_path.name)}.pvt{SUFFIX}"),
    )


# =============================================================================
# Reading and writing
# =============================================================================


CheckedBytes = NewType("CheckedBytes", bytes)
"""Listing bytes known to hash to the ``DirHash`` they were read for (``check``,
``read_bytes``); only these reach ``parse``."""


def check(data: bytes, record: DirHash, source: pathlib.Path | str) -> CheckedBytes:
    """``data`` as listing bytes for ``record``, refused, naming ``source``, unless they hash to
    ``record["hash"]``."""
    actual = digest(data)
    if actual != record["hash"]:
        raise exceptions.CorruptListingError(
            f"{source}: its bytes hash to {actual}, not to the directory hash {record['hash']} "
            + "its lock or .pvt records"
        )
    return CheckedBytes(data)


@functools.cache
def _warn_stale(committed: str, record_hash: str) -> None:
    """Warn once per process for each committed listing a read had to pass over."""
    logger.warning(
        "%s does not hash to the directory hash %s its lock or .pvt records; using the cache's "
        + "copy of that listing. A stage run or `pivot track --force` rewrites the file; "
        + "`git checkout` restores the committed one",
        committed,
        record_hash,
    )


def _read_located(
    committed: pathlib.Path | None,
    record: DirHash,
    cache_dir: pathlib.Path | None,
    *,
    strict: bool = False,
) -> tuple[CheckedBytes, pathlib.Path | str]:
    """The checked listing bytes and the name ``parse`` gives where they came from.

    The committed file is taken when it hashes to ``record``. When it does not, the cache
    object at ``record``'s hash, which is checked and never changes, is taken instead with one
    warning naming the committed file: a shared input listing that a later run of another stage
    moved on, a lock reverted without its listing, or a run interrupted between writing the
    listing and the lock. Only when the cache lacks that object too is the committed file
    refused. ``strict`` refuses it at once, for ``pivot verify``: a fresh clone has no cache
    to fall back on.
    """
    mismatch: exceptions.CorruptListingError | None = None
    if committed is not None:
        try:
            return check(committed.read_bytes(), record, committed), committed
        except FileNotFoundError:
            pass
        except exceptions.CorruptListingError as e:
            if strict:
                raise
            mismatch = e
    cached = None if cache_dir is None else cache.get_cache_path(cache_dir, record["hash"])
    if cached is not None:
        try:
            data = check(cached.read_bytes(), record, cached)
        except FileNotFoundError:
            pass
        else:
            if mismatch is not None:
                _warn_stale(str(committed), record["hash"])
            return data, f"cache object {record['hash']}"
    if mismatch is not None:
        raise mismatch
    where = list[str]() if committed is None else [str(committed)]
    if cached is not None:
        where.append(f"the cache object {cached}")
    raise exceptions.ListingMissingError(
        f"No listing for directory {record['hash']}: neither {' nor '.join(where)} exists"
        if where
        else f"No listing for directory {record['hash']}: no listing file and no cache"
    )


def read_bytes(
    committed: pathlib.Path | None,
    record: DirHash,
    cache_dir: pathlib.Path | None,
    *,
    strict: bool = False,
) -> CheckedBytes:
    """The bytes of the listing ``record`` names, checked against its hash (``_read_located``)."""
    return _read_located(committed, record, cache_dir, strict=strict)[0]


def parse(data: CheckedBytes, record: DirHash, source: pathlib.Path | str) -> DirListing:
    """Decode listing bytes checked against ``record``, naming ``source`` on error."""
    try:
        parsed = decode(data)
    except exceptions.CorruptListingError as e:
        raise exceptions.CorruptListingError(f"{source}: {e}") from e
    sizes = sum(entry["size"] for entry in parsed["files"])
    if len(parsed["files"]) != record["nfiles"] or sizes != record["size"]:
        raise exceptions.CorruptListingError(
            f"{source}: lists {len(parsed['files'])} files of {sizes} bytes, but its lock or "
            + f".pvt records {record['nfiles']} files of {record['size']} bytes"
        )
    return parsed


def read(
    committed: pathlib.Path | None,
    record: DirHash,
    cache_dir: pathlib.Path | None,
    *,
    strict: bool = False,
) -> DirListing:
    """The listing ``record`` names (see ``_read_located``), parsed."""
    data, source = _read_located(committed, record, cache_dir, strict=strict)
    return parse(data, record, source)


def write(path: pathlib.Path, data: bytes) -> None:
    """Write a committed listing file atomically, leaving one that holds ``data`` untouched."""
    try:
        if path.read_bytes() == data:
            return
    except FileNotFoundError:
        pass

    def write_bytes(fd: int) -> None:
        with os.fdopen(fd, "wb") as f:
            f.write(data)

    cache.atomic_write_file(path, write_bytes)


def store_object(cache_dir: pathlib.Path, directory_hash: str, data: bytes) -> None:
    """Put a listing in the cache as the object at its directory's hash, read-only. Bytes that
    do not hash to ``directory_hash`` are refused: a cache object must be what its name says."""
    cache_path = cache.get_cache_path(cache_dir, directory_hash)
    if cache_path.exists():
        return
    if (actual := digest(data)) != directory_hash:
        raise ValueError(
            f"Listing bytes hash to {actual}, not to directory hash {directory_hash}; not cached"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache.staged_cache_file(cache_path) as staged:
        staged.write_bytes(data)
        os.chmod(staged, 0o444)
        cache.publish_cache_object(staged, cache_path)

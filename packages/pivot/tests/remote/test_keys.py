from __future__ import annotations

import errno
import os
from typing import TYPE_CHECKING

import pytest
import xxhash

from pivot import exceptions
from pivot.remote import keys
from pivot.storage import cache

if TYPE_CHECKING:
    import pathlib

# -----------------------------------------------------------------------------
# Hash to Key Conversion Tests
# -----------------------------------------------------------------------------


def test_hash_to_key_with_prefix() -> None:
    """Hash converts to key with prefix."""
    key = keys.hash_to_key("cache/", "abcdef1234567890")
    assert key == "cache/files/ab/cdef1234567890"


def test_hash_to_key_no_prefix() -> None:
    """Hash converts to key without prefix."""
    key = keys.hash_to_key("", "abcdef1234567890")
    assert key == "files/ab/cdef1234567890"


def test_key_to_hash_with_prefix() -> None:
    """Key converts back to hash."""
    hash_ = keys.key_to_hash("cache/", "cache/files/ab/cdef1234567890")
    assert hash_ == "abcdef1234567890"


def test_key_to_hash_no_prefix() -> None:
    """Key without prefix converts to hash."""
    hash_ = keys.key_to_hash("", "files/ab/cdef1234567890")
    assert hash_ == "abcdef1234567890"


def test_key_to_hash_wrong_prefix() -> None:
    """Key with wrong prefix returns None."""
    hash_ = keys.key_to_hash("cache/", "other/files/ab/cdef1234567890")
    assert hash_ is None


def test_key_to_hash_not_cache_file() -> None:
    """Non-cache key returns None."""
    hash_ = keys.key_to_hash("cache/", "cache/stages/my_stage.lock")
    assert hash_ is None


def test_key_to_hash_malformed() -> None:
    """Malformed key returns None."""
    hash_ = keys.key_to_hash("", "files/abcdef1234567890")  # Missing split
    assert hash_ is None


def test_key_to_hash_empty_parts() -> None:
    """Key with empty parts returns None."""
    assert keys.key_to_hash("", "files//abcdef") is None
    assert keys.key_to_hash("", "files/ab/") is None


def test_key_to_hash_rejects_invalid_hex() -> None:
    """Key producing non-hex or wrong-length hash returns None."""
    assert keys.key_to_hash("", "files/AB/CDEF1234567890") is None  # Uppercase
    assert keys.key_to_hash("", "files/ab/c") is None  # Too short
    assert keys.key_to_hash("", "files/ab/cdef12345678901234") is None  # Too long


def test_validate_hash_short_raises() -> None:
    """Hash with wrong length raises RemoteError."""
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        keys.validate_hash("")
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        keys.validate_hash("ab")
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        keys.validate_hash("abc")  # Was valid before, now too short
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        keys.validate_hash("abcdef123456789")  # 15 chars


def test_validate_hash_non_hex_raises() -> None:
    """Non-lowercase-hex hash raises RemoteError."""
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        keys.validate_hash("ghijklmnopqrstuv")  # Non-hex chars
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        keys.validate_hash("ABCDEF1234567890")  # Uppercase
    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        keys.validate_hash("abc/def123456789")  # Path separator


def test_validate_hash_valid() -> None:
    """Valid 16-char lowercase hex hash passes validation."""
    keys.validate_hash("abcdef1234567890")
    keys.validate_hash("0123456789abcdef")
    keys.validate_hash("0000000000000000")
    keys.validate_hash("ffffffffffffffff")


# -----------------------------------------------------------------------------
# Streaming Integrity Digest Tests
# -----------------------------------------------------------------------------


async def test_stream_to_fd_writes_every_chunk_and_returns_its_digest(
    tmp_path: pathlib.Path,
) -> None:
    """stream_to_fd writes chunks until an empty one and returns the xxh64 of all of them."""
    chunks = [b"first ", b"second ", b"third", b""]

    async def read_chunk() -> bytes:
        return chunks.pop(0)

    dest = tmp_path / "dest"
    dest.touch()
    fd = os.open(dest, os.O_WRONLY)
    try:
        digest = await keys.stream_to_fd(read_chunk, fd)
    finally:
        os.close(fd)

    assert dest.read_bytes() == b"first second third"
    assert digest == xxhash.xxh64(b"first second third").hexdigest()


def test_check_digest_accepts_match() -> None:
    """check_digest returns when the digest is the expected hash."""
    keys.check_digest("abcdef1234567890", "abcdef1234567890", "bucket/files/ab/cdef1234567890")


def test_check_digest_mismatch_names_location_and_recovery() -> None:
    """check_digest raises the hash-mismatch error naming the object and how to repair it."""
    with pytest.raises(exceptions.RemoteError) as exc_info:
        keys.check_digest("abcdef1234567890", "0000000000000000", "gs://b/files/ab/cdef1234567890")

    message = str(exc_info.value)
    assert message.startswith("Hash mismatch: expected abcdef1234567890, got 0000000000000000")
    assert "Delete gs://b/files/ab/cdef1234567890 from the remote, then run `pivot push`" in message
    assert "delete `.pivot/state.lmdb/`" in message


def test_check_digest_names_the_configured_state_database(tmp_path: pathlib.Path) -> None:
    """The repair instructions name the project state directory selected by configuration."""
    state_dir = tmp_path / "alternate-state"

    with pytest.raises(exceptions.RemoteError) as exc_info:
        keys.check_digest(
            "abcdef1234567890",
            "0000000000000000",
            "gs://b/files/ab/cdef1234567890",
            state_dir=state_dir,
        )

    assert f"delete `{state_dir}/state.lmdb/`" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Atomic Download Tests (every backend's download_file and download_batch)
# -----------------------------------------------------------------------------


def _writer(content: bytes):  # noqa: ANN202 - a download_verified stand-in
    async def download_verified(dest: pathlib.Path) -> None:
        dest.write_bytes(content)

    return download_verified


async def _failing(dest: pathlib.Path) -> None:
    dest.write_bytes(b"partial")
    raise exceptions.RemoteError("Hash mismatch: expected …")


@pytest.mark.parametrize("overwrite", [True, False], ids=["download_file", "download_batch"])
async def test_atomic_download_readonly_sets_0o444(tmp_path: pathlib.Path, overwrite: bool) -> None:
    dest = tmp_path / "dest"

    await keys.atomic_download(dest, _writer(b"content"), overwrite=overwrite, readonly=True)

    assert dest.read_bytes() == b"content"
    assert dest.stat().st_mode & 0o777 == 0o444


async def test_atomic_download_default_permissions_are_writable(tmp_path: pathlib.Path) -> None:
    dest = tmp_path / "dest"

    await keys.atomic_download(dest, _writer(b"content"), overwrite=True, readonly=False)

    assert dest.stat().st_mode & 0o200, oct(dest.stat().st_mode)


@pytest.mark.parametrize("overwrite", [True, False], ids=["download_file", "download_batch"])
async def test_atomic_download_failure_leaves_nothing(
    tmp_path: pathlib.Path, overwrite: bool
) -> None:
    dest = tmp_path / "dest"

    with pytest.raises(exceptions.RemoteError, match="Hash mismatch"):
        await keys.atomic_download(dest, _failing, overwrite=overwrite, readonly=True)

    assert [p.name for p in tmp_path.iterdir() if p.name != cache._PREFIX_LOCK] == []


async def test_atomic_download_overwrite_replaces_an_existing_file(tmp_path: pathlib.Path) -> None:
    dest = tmp_path / "dest"
    dest.write_bytes(b"old")

    await keys.atomic_download(dest, _writer(b"new"), overwrite=True, readonly=False)

    assert dest.read_bytes() == b"new"
    assert os.listdir(tmp_path) == ["dest"]


async def test_atomic_download_keeps_an_existing_cache_entry(tmp_path: pathlib.Path) -> None:
    """A cache entry another process may be hardlinking is never replaced (#461)."""
    dest = tmp_path / "entry"
    dest.write_bytes(b"content")
    inode = dest.stat().st_ino

    await keys.atomic_download(dest, _writer(b"content"), overwrite=False, readonly=True)

    assert dest.stat().st_ino == inode, "the existing cache entry must not be replaced"
    assert [p.name for p in tmp_path.iterdir() if p.name != cache._PREFIX_LOCK] == ["entry"]


@pytest.mark.parametrize("link_errno", [errno.EPERM, errno.ENOTSUP])
@pytest.mark.parametrize("exists", [False, True])
async def test_atomic_download_without_hard_links(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, link_errno: int, exists: bool
) -> None:
    """On a filesystem that refuses link() (vfat, many SMB and FUSE mounts), the verified file is
    still promoted, and an existing entry is still kept."""
    dest = tmp_path / "entry"
    inode = None
    if exists:
        dest.write_bytes(b"content")
        inode = dest.stat().st_ino

    def _refuse_link(src: object, dst: object) -> None:
        raise OSError(link_errno, os.strerror(link_errno))

    monkeypatch.setattr(os, "link", _refuse_link)

    await keys.atomic_download(dest, _writer(b"content"), overwrite=False, readonly=True)

    assert dest.read_bytes() == b"content"
    if inode is not None:
        assert dest.stat().st_ino == inode, "the existing cache entry must not be replaced"
    assert [p.name for p in tmp_path.iterdir() if p.name != cache._PREFIX_LOCK] == ["entry"]

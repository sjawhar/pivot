from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import xxhash

from pivot import exceptions
from pivot.remote import keys

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

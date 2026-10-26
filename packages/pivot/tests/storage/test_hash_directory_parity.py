from __future__ import annotations

import os
import pathlib
import time

import pytest
import xxhash

from pivot.storage import cache, state
from pivot.types import DirHash

# Digests `cache.hash_directory` gave for `_helper_write_parity_tree` at 5f8e39c8 (release
# 0.3.10): with the cache root passed, and with no cache (the cache-object link left out).
_PINNED_HASH_WITH_CACHE = "57f87df41ea09bcc"
_PINNED_HASH_NO_CACHE = "b0b0ad43a83da3c9"
_PINNED_EMPTY_DIRS = ["empty/deep/deeper", "empty2", "links"]
_CACHED_CONTENT = b"bytes held in the cache\n"

# Names that JSON escapes, that sort differently by whole path than directory by directory, or
# that are not UTF-8 (a str with surrogate escapes, as os.scandir returns them).
_PARITY_FILES: dict[str, bytes] = {
    "plain.txt": b"plain\n",
    'with "double" quotes.txt': b"double\n",
    "it's.txt": b"single\n",
    "back\\slash.txt": b"backslash\n",
    "caf\u00e9.txt": b"nfc\n",
    "cafe\u0301.txt": b"nfd\n",
    "\u65e5\u672c\u8a9e.txt": b"cjk\n",
    "emoji-\U0001f642.txt": b"emoji\n",
    "tab\tname.txt": b"tab\n",
    "new\nline.txt": b"newline\n",
    " leading.txt": b"leading\n",
    "trailing.txt ": b"trailing\n",
    os.fsdecode(b"\xff\xfe.bin"): b"not utf-8\n",
    "a/b.txt": b"nested\n",
    "a.b.txt": b"dot\n",
    "a-b.txt": b"dash\n",
    "a0.txt": b"zero\n",
    "A.txt": b"upper\n",
    "zero.bin": b"",
    "big.bin": bytes(range(256)) * 8,
}
_PARITY_MODES = {"run.sh": 0o755, "user-exec": 0o744, "group-exec-only": 0o654}


def _helper_write_parity_tree(root: pathlib.Path, cache_dir: pathlib.Path) -> None:
    """A directory exercising every rule `hash_directory` applies to names, links and modes."""
    root.mkdir()
    for relpath, content in _PARITY_FILES.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for name, mode in _PARITY_MODES.items():
        (root / name).write_bytes(name.encode())
        os.chmod(root / name, mode)
    (root / "empty" / "deep" / "deeper").mkdir(parents=True)
    (root / "empty2").mkdir()
    (root / "links").mkdir()
    (root / "links" / "only").symlink_to(root / "plain.txt")
    (root / "user-link.txt").symlink_to(root / "plain.txt")
    (root / "dirlink").symlink_to(root / "a", target_is_directory=True)
    cached_hash = xxhash.xxh64(_CACHED_CONTENT).hexdigest()
    cache_object = cache.get_cache_path(cache_dir, cached_hash)
    cache_object.parent.mkdir(parents=True)
    cache_object.write_bytes(_CACHED_CONTENT)
    (root / "from-cache.bin").symlink_to(cache_object)
    outside = root.parent / "elsewhere" / cached_hash[:2] / cached_hash[2:]
    outside.parent.mkdir(parents=True)
    outside.write_bytes(_CACHED_CONTENT)
    (root / "hash-named-outside.bin").symlink_to(outside)


def _helper_parity_tree(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    cache_dir = tmp_path / "cache" / "files"
    root = tmp_path / "tree"
    _helper_write_parity_tree(root, cache_dir)
    return root, cache_dir


@pytest.mark.parametrize("mmap_threshold", [cache.MMAP_THRESHOLD, 1024], ids=["read", "mmap"])
@pytest.mark.parametrize(
    ("with_cache", "expected_hash"),
    [
        pytest.param(True, _PINNED_HASH_WITH_CACHE, id="cache-root"),
        pytest.param(False, _PINNED_HASH_NO_CACHE, id="no-cache"),
    ],
)
def test_hash_directory_digest_matches_0_3_10(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mmap_threshold: int,
    with_cache: bool,
    expected_hash: str,
) -> None:
    """A directory hashes to the digest 0.3.10 recorded for it, so no lock entry changes."""
    monkeypatch.setattr(cache, "MMAP_THRESHOLD", mmap_threshold)
    root, cache_dir = _helper_parity_tree(tmp_path)

    result = cache.hash_directory(root, cache_dir=cache_dir if with_cache else None)

    assert result["hash"] == expected_hash
    assert "empty_dirs" in result
    assert result["empty_dirs"] == _PINNED_EMPTY_DIRS
    relpaths = [entry["relpath"] for entry in result["manifest"]]
    assert ("from-cache.bin" in relpaths) is with_cache
    assert relpaths == sorted(relpaths), "The manifest is sorted by whole relative path"


def test_hash_directory_gives_the_same_digest_through_the_state_db(tmp_path: pathlib.Path) -> None:
    """Hashing through a StateDB, on its first pass and from its cached hashes, and through a
    read-only one, gives the digest hashing without one gives."""
    root, cache_dir = _helper_parity_tree(tmp_path)
    expected = cache.hash_directory(root, cache_dir=cache_dir)

    with state.StateDB(tmp_path / "state") as db:
        first = cache.hash_directory(root, db, cache_dir=cache_dir)
        second = cache.hash_directory(root, db, cache_dir=cache_dir)
    with state.StateDB(tmp_path / "state", readonly=True) as readonly_db:
        readonly = cache.hash_directory(root, readonly_db, cache_dir=cache_dir)

    assert first == expected
    assert second == expected
    assert readonly == expected


def test_hash_directory_skips_hash_caching_for_a_path_too_long_for_the_state_db(
    tmp_path: pathlib.Path,
) -> None:
    """A file whose StateDB key exceeds LMDB's limit is hashed and recorded, not cached."""
    root = tmp_path / "deep"
    nested = root
    for level in range(10):
        nested = nested / (f"level{level}" + "d" * 50)
    nested.mkdir(parents=True)
    (nested / "leaf.txt").write_text("leaf")
    expected = cache.hash_directory(root)

    with state.StateDB(tmp_path / "state") as db:
        result = cache.hash_directory(root, db)

    assert result == expected


def _helper_floor_seconds(files: list[pathlib.Path]) -> float:
    """Reading and hashing every file, with no walk, manifest or path work: the floor."""
    start = time.perf_counter()
    for path in files:
        with open(path, "rb", buffering=0) as f:
            xxhash.xxh64(f.read()).hexdigest()
    return time.perf_counter() - start


def test_hash_directory_costs_little_more_than_reading_its_files(tmp_path: pathlib.Path) -> None:
    """Warm, a directory of 2,000 small files hashes within 3 times the cost of reading and
    hashing each file: per-file path resolution and repeated stats are what made it slower."""
    root = tmp_path / "many"
    files = list[pathlib.Path]()
    for index in range(2_000):
        path = root / f"d{index // 100:02d}" / f"f{index:05d}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(index.to_bytes(4, "big") * 256)
        files.append(path)
    cache.hash_directory(root)
    _helper_floor_seconds(files)

    floor = min(_helper_floor_seconds(files) for _ in range(3))
    elapsed = list[float]()
    for _ in range(3):
        start = time.perf_counter()
        cache.hash_directory(root)
        elapsed.append(time.perf_counter() - start)

    assert min(elapsed) < 3 * floor, (
        f"hash_directory took {min(elapsed):.4f}s for 2,000 files against {floor:.4f}s to read "
        + "and hash them"
    )


def test_hash_directory_hashes_a_name_that_is_not_utf8_through_the_state_db(
    tmp_path: pathlib.Path,
) -> None:
    """A file name that is not UTF-8 (os.scandir returns it with surrogate escapes) hashes
    through a StateDB as it does without one."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / os.fsdecode(b"\xff.bin")).write_bytes(b"x")
    expected = cache.hash_directory(root)

    with state.StateDB(tmp_path / "state") as db:
        result = cache.hash_directory(root, db)

    assert result == DirHash(hash=expected["hash"], manifest=expected["manifest"])


def test_hash_directory_reads_nothing_through_a_symlinked_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk never follows a symlink, so no file outside the directory is read or listed:
    this is why the per-file ``resolve()`` containment check is not needed. A link to a
    directory outside, and a base path that is itself a link, hash as the tree without it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "kept.txt").write_text("kept")
    expected = cache.hash_directory(root)
    (root / "sub" / "escape").symlink_to(outside, target_is_directory=True)
    base_link = tmp_path / "base-link"
    base_link.symlink_to(root, target_is_directory=True)
    read = list[str]()
    original_hash_contents = cache._hash_contents

    def recording_hash_contents(path: str | os.PathLike[str], size: int) -> str:
        read.append(os.fspath(path))
        return original_hash_contents(path, size)

    monkeypatch.setattr(cache, "_hash_contents", recording_hash_contents)

    with state.StateDB(tmp_path / "state") as db:
        through_root = cache.hash_directory(root, db)
        through_link = cache.hash_directory(base_link, db)

    assert through_root == expected, "A symlinked directory adds nothing to the manifest"
    assert through_link == expected, "A base path that is a link hashes as its target"
    assert not [path for path in read if "secret" in path], "No file outside the tree is read"

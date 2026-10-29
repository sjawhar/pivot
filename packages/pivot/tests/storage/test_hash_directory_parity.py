from __future__ import annotations

import contextlib
import os
import pathlib
import statistics
import time
from typing import TYPE_CHECKING

import pytest
import xxhash

from pivot.storage import cache, state
from pivot.types import DirHash

# The listing bytes and their xxh64 values for `_helper_write_parity_tree`. A directory's
# identity is the xxh64 of these bytes, not the pre-0.3.11 recursive manifest digest.
_EXPECTED_HASH_WITH_CACHE = "8dfdfb86c4134c7b"
_EXPECTED_LISTING_WITH_CACHE = (
    b'6d05f87c11460591 8 - " leading.txt"\n'
    b"c59f4514175f2079 6 - A.txt\n"
    b"021fa084edda38d7 5 - a-b.txt\n"
    b"ffebe8ef91f98782 4 - a.b.txt\n"
    b"48a9425c39957584 7 - a/b.txt\n"
    b"f6f209378be35c10 5 - a0.txt\n"
    b'364082a8cb2e4e95 10 - "back\\\\slash.txt"\n'
    b"68534a48b7bf5f4d 2048 - big.bin\n"
    b"6e7eefe6528f4ff6 4 - cafe\xcc\x81.txt\n"
    b"e7390d0e4d278775 4 - caf\xc3\xa9.txt\n"
    b"90420ee47cbc226b 6 - emoji-\xf0\x9f\x99\x82.txt\n"
    b"- - - empty/deep/deeper/\n"
    b"- - - empty2/\n"
    b"9aab28278308abe2 24 - from-cache.bin\n"
    b"39dc54237ecf0c8c 15 - group-exec-only\n"
    b"38f9f4673d2638f1 7 - it's.txt\n"
    b"- - - links/\n"
    b'9da1861a5eecb30f 8 - "new\\nline.txt"\n'
    b"760e5d8ccd4ca44e 6 - plain.txt\n"
    b"3bd7f94a57a5dbde 6 x run.sh\n"
    b'197b33186f96a274 4 - "tab\\tname.txt"\n'
    b'0fe05254166a9936 9 - "trailing.txt "\n'
    b"52b56604f19cba92 9 x user-exec\n"
    b'a8313c064255ffca 7 - with "double" quotes.txt\n'
    b"ef46db3751d8e999 0 - zero.bin\n"
    b"8b9b330095727779 4 - \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e.txt\n"
    b'b9ade7de3b62f0d8 10 - "\\377\\376.bin"\n'
)
_EXPECTED_HASH_NO_CACHE = "7d079989ceba3e25"
_EXPECTED_LISTING_NO_CACHE = (
    b'6d05f87c11460591 8 - " leading.txt"\n'
    b"c59f4514175f2079 6 - A.txt\n"
    b"021fa084edda38d7 5 - a-b.txt\n"
    b"ffebe8ef91f98782 4 - a.b.txt\n"
    b"48a9425c39957584 7 - a/b.txt\n"
    b"f6f209378be35c10 5 - a0.txt\n"
    b'364082a8cb2e4e95 10 - "back\\\\slash.txt"\n'
    b"68534a48b7bf5f4d 2048 - big.bin\n"
    b"6e7eefe6528f4ff6 4 - cafe\xcc\x81.txt\n"
    b"e7390d0e4d278775 4 - caf\xc3\xa9.txt\n"
    b"90420ee47cbc226b 6 - emoji-\xf0\x9f\x99\x82.txt\n"
    b"- - - empty/deep/deeper/\n"
    b"- - - empty2/\n"
    b"39dc54237ecf0c8c 15 - group-exec-only\n"
    b"38f9f4673d2638f1 7 - it's.txt\n"
    b"- - - links/\n"
    b'9da1861a5eecb30f 8 - "new\\nline.txt"\n'
    b"760e5d8ccd4ca44e 6 - plain.txt\n"
    b"3bd7f94a57a5dbde 6 x run.sh\n"
    b'197b33186f96a274 4 - "tab\\tname.txt"\n'
    b'0fe05254166a9936 9 - "trailing.txt "\n'
    b"52b56604f19cba92 9 x user-exec\n"
    b'a8313c064255ffca 7 - with "double" quotes.txt\n'
    b"ef46db3751d8e999 0 - zero.bin\n"
    b"8b9b330095727779 4 - \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e.txt\n"
    b'b9ade7de3b62f0d8 10 - "\\377\\376.bin"\n'
)
_EXPECTED_EMPTY_DIRS = ["empty/deep/deeper", "empty2", "links"]
if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

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
    ("with_cache", "expected_listing", "expected_hash", "expected_size", "expected_nfiles"),
    [
        pytest.param(
            True, _EXPECTED_LISTING_WITH_CACHE, _EXPECTED_HASH_WITH_CACHE, 2216, 24, id="cache-root"
        ),
        pytest.param(
            False, _EXPECTED_LISTING_NO_CACHE, _EXPECTED_HASH_NO_CACHE, 2192, 23, id="no-cache"
        ),
    ],
)
def test_hash_directory_record_matches_listing_bytes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mmap_threshold: int,
    with_cache: bool,
    expected_listing: bytes,
    expected_hash: str,
    expected_size: int,
    expected_nfiles: int,
) -> None:
    """A directory record is the xxh64 and metadata of its canonical listing bytes."""
    monkeypatch.setattr(cache, "MMAP_THRESHOLD", mmap_threshold)
    root, cache_dir = _helper_parity_tree(tmp_path)

    result = cache.hash_directory(root, cache_dir=cache_dir if with_cache else None)

    assert result["data"] == expected_listing
    assert xxhash.xxh64(expected_listing).hexdigest() == expected_hash
    assert result["record"] == DirHash(
        hash=expected_hash, size=expected_size, nfiles=expected_nfiles
    )
    assert result["listing"]["empty_dirs"] == _EXPECTED_EMPTY_DIRS
    relpaths = [entry["relpath"] for entry in result["listing"]["files"]]
    assert ("from-cache.bin" in relpaths) is with_cache
    assert relpaths == sorted(relpaths), "The listing is sorted by whole relative path"


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


def _helper_write_files(root: pathlib.Path, size: int) -> list[pathlib.Path]:
    """2,000 files of ``size`` bytes in 20 directories, under the same names in every root."""
    files = list[pathlib.Path]()
    for index in range(2_000):
        path = root / f"d{index // 100:02d}" / f"f{index:05d}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(index.to_bytes(4, "big") * (size // 4))
        files.append(path)
    return files


def _helper_floor_cpu_ns(files: list[pathlib.Path]) -> int:
    """CPU time to read and hash every file, with no walk, manifest or path work: the floor."""
    start = time.process_time_ns()
    for path in files:
        with open(path, "rb", buffering=0) as f:
            xxhash.xxh64(f.read()).hexdigest()
    return time.process_time_ns() - start


def _helper_hash_directory_cpu_ns(root: pathlib.Path) -> int:
    start = time.process_time_ns()
    cache.hash_directory(root)
    return time.process_time_ns() - start


def test_hash_directory_costs_little_more_than_reading_its_files(tmp_path: pathlib.Path) -> None:
    """Warm, a directory of 2,000 small files hashes within 4 times the CPU cost of reading and
    hashing each file, and reads the bytes of each file once.

    The direct loop does no walk, stat or canonical listing. How much more hash_directory costs
    varies with the CPU and with coverage tracing (1.3 to 2.6 times), so reading every file twice
    (2.2 to 3.7 times) needs its own check: two trees with the same names, of 1 KiB and 64 KiB
    files, cancel the per-file work. 63 KiB more in each file costs hash_directory 1.0 to 1.25
    times what it costs the loop, and 1.55 to 1.85 times when it reads every file twice.

    The test counts this process's CPU time, so time that other processes take from it on a busy
    runner is left out. It compares the four measurements within each of 10 interleaved rounds
    and takes the median round: in CPU time a round runs up to 1.8 times faster while the other
    hyperthread of its core is idle, and the minimum of each measurement would mix fast and slow
    rounds."""
    small_root = tmp_path / "small"
    large_root = tmp_path / "large"
    small = _helper_write_files(small_root, 1024)
    large = _helper_write_files(large_root, 64 * 1024)
    for root, files in ((small_root, small), (large_root, large)):
        cache.hash_directory(root)
        _helper_floor_cpu_ns(files)

    ratios = list[float]()
    slopes = list[float]()
    for _ in range(10):
        floor_small = _helper_floor_cpu_ns(small)
        cost_small = _helper_hash_directory_cpu_ns(small_root)
        floor_large = _helper_floor_cpu_ns(large)
        cost_large = _helper_hash_directory_cpu_ns(large_root)
        ratios.append(cost_small / floor_small)
        slopes.append((cost_large - cost_small) / (floor_large - floor_small))

    ratio, slope = statistics.median(ratios), statistics.median(slopes)
    assert ratio < 4, (
        f"hash_directory of 2,000 files took {ratio:.2f} times the CPU of reading and hashing "
        + f"them (median; by round: {', '.join(f'{value:.2f}' for value in ratios)})"
    )
    assert slope < 1.4, (
        f"63 KiB more in each of 2,000 files took hash_directory {slope:.2f} times the extra CPU "
        + "of reading and hashing them (median; by round: "
        + f"{', '.join(f'{value:.2f}' for value in slopes)}): it does more work per byte than one "
        + "read and hash (reading every file twice gives about 1.6 times)"
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

    assert result == expected


def test_hash_directory_reads_nothing_through_a_symlink(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk never follows a symlink, so no file outside the directory is read or listed:
    this is why the per-file ``resolve()`` containment check is not needed. A link to a
    directory outside, a link to a file outside (never read, then left out: only a link into
    the cache counts as content), and a base path that is itself a link, hash as the tree
    without them."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "kept.txt").write_text("kept")
    cache_dir = tmp_path / "cache" / "files"
    cache_dir.mkdir(parents=True)
    expected = cache.hash_directory(root, cache_dir=cache_dir)
    (root / "sub" / "escape").symlink_to(outside, target_is_directory=True)
    (root / "sub" / "secret-link.txt").symlink_to(outside / "secret.txt")
    base_link = tmp_path / "base-link"
    base_link.symlink_to(root, target_is_directory=True)
    read = list[str]()
    original_hash_contents = cache._hash_contents

    def recording_hash_contents(path: str | os.PathLike[str], size: int) -> str:
        read.append(os.fspath(path))
        return original_hash_contents(path, size)

    monkeypatch.setattr(cache, "_hash_contents", recording_hash_contents)

    with state.StateDB(tmp_path / "state") as db:
        through_root = cache.hash_directory(root, db, cache_dir=cache_dir)
        through_link = cache.hash_directory(base_link, db, cache_dir=cache_dir)

    assert through_root == expected, "A symlink to anything outside adds nothing to the manifest"
    assert through_link == expected, "A base path that is a link hashes as its target"
    assert not [path for path in read if "secret" in path], "No file outside the tree is read"


def _helper_plain_tree(root: pathlib.Path) -> None:
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "kept.txt").write_text("kept")
    (root / "top.txt").write_text("top")


def test_hash_directory_lists_each_name_of_a_hard_link(tmp_path: pathlib.Path) -> None:
    """Two names of one inode are two files of the artifact, each with its content."""
    root = tmp_path / "tree"
    _helper_plain_tree(root)
    os.link(root / "top.txt", root / "sub" / "twin.txt")
    with_copy = tmp_path / "copied"
    _helper_plain_tree(with_copy)
    (with_copy / "sub" / "twin.txt").write_text("top")

    with state.StateDB(tmp_path / "state") as db:
        result = cache.hash_directory(root, db)

    assert result == cache.hash_directory(with_copy)


def test_hash_directory_leaves_out_a_fifo(tmp_path: pathlib.Path) -> None:
    """A FIFO is neither a file nor a directory of the artifact: never opened, never listed."""
    root = tmp_path / "tree"
    _helper_plain_tree(root)
    expected = cache.hash_directory(root)
    os.mkfifo(root / "sub" / "pipe")

    with state.StateDB(tmp_path / "state") as db:
        result = cache.hash_directory(root, db)

    assert result == expected


def test_hash_directory_leaves_out_a_dangling_cache_object_link(tmp_path: pathlib.Path) -> None:
    """A link into the cache whose object is gone holds no content, so it is not stored."""
    root = tmp_path / "tree"
    _helper_plain_tree(root)
    cache_dir = tmp_path / "cache" / "files"
    cache_dir.mkdir(parents=True)
    expected = cache.hash_directory(root, cache_dir=cache_dir)
    gone = cache.get_cache_path(cache_dir, xxhash.xxh64(b"gone").hexdigest())
    (root / "sub" / "gone.txt").symlink_to(gone)

    with state.StateDB(tmp_path / "state") as db:
        result = cache.hash_directory(root, db, cache_dir=cache_dir)

    assert result == expected


def test_hash_directory_skips_a_file_removed_after_the_walk_lists_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file deleted between the walk and its stat is left out, with no error."""
    root = tmp_path / "tree"
    _helper_plain_tree(root)
    expected = cache.hash_directory(root)
    (root / "sub" / "doomed.txt").write_text("doomed")
    real_scandir = os.scandir

    @contextlib.contextmanager
    def scandir_then_remove(path: str) -> Generator[Iterator[os.DirEntry[str]]]:
        with real_scandir(path) as entries:
            listed = list(entries)
        for entry in listed:
            if entry.name.startswith("doomed"):
                os.unlink(entry.path)
        yield iter(listed)

    monkeypatch.setattr(os, "scandir", scandir_then_remove)

    with state.StateDB(tmp_path / "state") as db:
        result = cache.hash_directory(root, db)

    assert result == expected
    assert not (root / "sub" / "doomed.txt").exists()

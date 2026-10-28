"""End-to-end forged-listing tests with real XXH64 collisions, no monkeypatching.

Each test builds listing bytes whose XXH64 matches the target directory's hash, but whose
content differs from the real listing — then verifies that the byte-equality check refuses them.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import cast

import pytest
import xxhash

from pivot.storage import cache, listing
from pivot.types import DirHash, DirManifestEntry

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")


def _helper_project(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, DirHash]:
    """A project with one tracked directory, reproduced and committed."""
    root.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run([_PIVOT, "init"], cwd=root, env=env, capture_output=True, text=True, check=True)
    corpus = root / "data" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "a.txt").write_text("alpha")
    (corpus / "b.txt").write_text("beta")
    subprocess.run(
        [_PIVOT, "track", "data/corpus"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    pvt_listing = root / "data" / "corpus.pvt.listing"
    assert pvt_listing.is_file()
    real_bytes = pvt_listing.read_bytes()
    directory_hash = xxhash.xxh64(real_bytes).hexdigest()
    cache_dir = root / ".pivot" / "cache" / "files"
    return (
        root,
        cache_dir,
        DirHash(
            hash=directory_hash,
            size=sum(entry["size"] for entry in listing.decode(real_bytes)["files"]),
            nfiles=len(listing.decode(real_bytes)["files"]),
        ),
    )


def _forge_listing(target_hex: str) -> bytes:
    """Build listing bytes whose XXH64 equals ``target_hex`` but whose content differs."""
    from xxh64_forge import steer_with_relpath

    base_entries: list[dict[str, object]] = [
        {"relpath": "a.txt", "hash": "0123456789abcdef", "size": 5, "isexec": False},
    ]
    extra: dict[str, object] = {"hash": "fedcba9876543210", "size": 4, "isexec": False}
    forged_entries = steer_with_relpath(base_entries, extra, target_hex)
    members = listing.DirListing(
        files=cast("list[DirManifestEntry]", forged_entries),
        empty_dirs=[],
    )
    return listing.encode(members)


def test_committed_listing_collision_is_refused(tmp_path: pathlib.Path) -> None:
    """A committed listing with the right XXH64 but wrong bytes is refused by listing._resolve."""
    from pivot import exceptions

    root, cache_dir, record = _helper_project(tmp_path / "proj")
    pvt_listing = root / "data" / "corpus.pvt.listing"
    real_bytes = pvt_listing.read_bytes()

    # Forge bytes that hash to the same XXH64
    forged = _forge_listing(record["hash"])
    assert xxhash.xxh64(forged).hexdigest() == record["hash"]
    assert forged != real_bytes

    # Store the real bytes in the cache so the collision check fires
    listing.store_object(cache_dir, record["hash"], real_bytes)

    # Write the forged bytes as the committed listing
    pvt_listing.write_bytes(forged)

    # The read refuses the collision
    with pytest.raises(exceptions.CorruptListingError, match="XXH64 collision"):
        listing.read(pvt_listing, record, cache_dir, strict=True)


def test_cached_listing_collision_is_refused(tmp_path: pathlib.Path) -> None:
    """A cache object with the right XXH64 but wrong bytes is refused by listing.store_object."""
    root, cache_dir, record = _helper_project(tmp_path / "proj")
    pvt_listing = root / "data" / "corpus.pvt.listing"
    real_bytes = pvt_listing.read_bytes()

    # Store the real listing in the cache first
    listing.store_object(cache_dir, record["hash"], real_bytes)

    # Forge bytes with the same XXH64
    forged = _forge_listing(record["hash"])
    assert xxhash.xxh64(forged).hexdigest() == record["hash"]
    assert forged != real_bytes

    # Storing the forged bytes at the same hash is refused
    from pivot import exceptions

    with pytest.raises(exceptions.CorruptListingError, match="XXH64 collision"):
        listing.store_object(cache_dir, record["hash"], forged)


def test_tree_marker_collision_is_refused(tmp_path: pathlib.Path) -> None:
    """A tree marked with different canonical bytes refuses a listing that shares its XXH64."""
    root, cache_dir, record = _helper_project(tmp_path / "proj")
    pvt_listing = root / "data" / "corpus.pvt.listing"
    real_bytes = pvt_listing.read_bytes()
    real_members = listing.decode(real_bytes)

    # Forge bytes with the same XXH64
    forged = _forge_listing(record["hash"])
    assert xxhash.xxh64(forged).hexdigest() == record["hash"]

    # Create a tree and mark it with the real canonical bytes
    tree = cache.tree_path(cache_dir, record["hash"])
    tree.mkdir(parents=True)
    for entry in real_members["files"]:
        (tree / entry["relpath"]).write_text("x")
    cache._mark_tree(tree, real_bytes)
    assert cache._tree_marker_matches(tree, real_bytes)

    # The forged bytes do not match the marker, so the tree is not trusted
    assert not cache._tree_marker_matches(tree, forged)


def test_track_force_refuses_a_forged_listing_that_shares_the_directory_xxh64(
    tmp_path: pathlib.Path,
) -> None:
    """A listing file whose bytes hash to the directory's XXH64 but differ from the cached
    copy is treated as foreign by track --force."""
    root, cache_dir, record = _helper_project(tmp_path / "proj")
    pvt_listing = root / "data" / "corpus.pvt.listing"
    real_bytes = pvt_listing.read_bytes()

    # Store the real listing in the cache
    listing.store_object(cache_dir, record["hash"], real_bytes)

    # Forge bytes with the same XXH64 and put them at the listing path
    forged = _forge_listing(record["hash"])
    assert xxhash.xxh64(forged).hexdigest() == record["hash"]
    assert forged != real_bytes
    pvt_listing.write_bytes(forged)

    # track --force should refuse because the forged bytes differ from the cached copy
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    result = subprocess.run(
        [_PIVOT, "track", "--force", "data/corpus"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout + result.stderr
    assert result.returncode != 0, f"track --force should refuse a forged listing:\n{out}"
    assert "in the way" in out, out
    assert pvt_listing.read_bytes() == forged, "the forged listing is not overwritten"

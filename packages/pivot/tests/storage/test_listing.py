"""The committed directory listing: its bytes, the hash it gives a directory, and its path."""

from __future__ import annotations

import logging
import os
import pathlib

import hypothesis
import hypothesis.strategies as st
import pytest
import xxhash

from pivot import exceptions, types
from pivot.storage import cache, listing, lock

_A = xxhash.xxh64(b"a").hexdigest()
_RUN = xxhash.xxh64(b"#!/bin/sh\n").hexdigest()


def _helper_files(*rows: tuple[str, str, int, bool]) -> list[types.DirManifestEntry]:
    return [
        types.DirManifestEntry(relpath=relpath, hash=digest, size=size, isexec=isexec)
        for relpath, digest, size, isexec in rows
    ]


def _helper_record(data: bytes, size: int, nfiles: int) -> types.DirHash:
    return types.DirHash(hash=xxhash.xxh64(data).hexdigest(), size=size, nfiles=nfiles)


# =============================================================================
# Bytes
# =============================================================================


def test_encode_writes_a_line_per_file_and_per_empty_directory_sorted_by_path_bytes() -> None:
    data = listing.encode(
        listing.DirListing(
            files=_helper_files(
                ("bin/run.sh", _RUN, 10, True), ("a.txt", _A, 1, False), ("bin-z", _A, 1, False)
            ),
            empty_dirs=["empty", "bin/sub"],
        )
    )

    assert (
        data
        == (
            f"{_A} 1 - a.txt\n"
            + f"{_A} 1 - bin-z\n"
            + f"{_RUN} 10 x bin/run.sh\n"
            + "- - - bin/sub/\n"
            + "- - - empty/\n"
        ).encode()
    )


def test_an_empty_directory_listing_is_empty() -> None:
    assert listing.encode(listing.DirListing(files=[], empty_dirs=[])) == b""


@pytest.mark.parametrize(
    ("name", "field"),
    [
        pytest.param("plain name.txt", b"plain name.txt", id="inner-space"),
        pytest.param("caf\u00e9.txt", "caf\u00e9.txt".encode(), id="utf-8"),
        pytest.param('mid"quote', b'mid"quote', id="inner-quote"),
        pytest.param("new\nline", b'"new\\nline"', id="newline"),
        pytest.param("tab\tname", b'"tab\\tname"', id="tab"),
        pytest.param("cr\rname", b'"cr\\rname"', id="carriage-return"),
        pytest.param(" lead", b'" lead"', id="leading-space"),
        pytest.param("trail ", b'"trail "', id="trailing-space"),
        pytest.param("a\\b", b'"a\\\\b"', id="backslash"),
        pytest.param('"q', b'"\\"q"', id="leading-quote"),
        pytest.param("bell\x07", b'"bell\\007"', id="control"),
        pytest.param("del\x7f", b'"del\\177"', id="delete"),
        pytest.param(os.fsdecode(b"\xff.bin"), b'"\\377.bin"', id="not-utf-8"),
    ],
)
def test_a_name_round_trips_and_the_listing_stays_utf8(name: str, field: bytes) -> None:
    files = _helper_files((name, _A, 1, False))
    empty = [f"{name}-dir"]

    data = listing.encode(listing.DirListing(files=files, empty_dirs=empty))

    assert f"{_A} 1 - ".encode() + field + b"\n" in data
    data.decode("utf-8")
    assert listing.decode(data) == listing.DirListing(files=files, empty_dirs=empty)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"0123456789abcde 1 - a\n", id="short-hash"),
        pytest.param(b"0123456789ABCDEF 1 - a\n", id="upper-case-hash"),
        pytest.param(f"{_A} 1 - a".encode(), id="no-final-newline"),
        pytest.param(f"{_A} 1 - b\n{_A} 1 - a\n".encode(), id="unsorted"),
        pytest.param(f"{_A} 1 - a\n{_A} 1 - a\n".encode(), id="duplicate"),
        pytest.param(f"{_A} 1 - ../a\n".encode(), id="traversal"),
        pytest.param(f"{_A} 1 - /a\n".encode(), id="absolute"),
        pytest.param(f"{_A} one - a\n".encode(), id="bad-size"),
        pytest.param(f"{_A} -1 - a\n".encode(), id="negative-size"),
        pytest.param(f"{_A} 1 y a\n".encode(), id="bad-exec-flag"),
        pytest.param(b"- - - a\n", id="empty-dir-without-slash"),
        pytest.param(f'{_A} 1 - "a\n'.encode(), id="unterminated-quote"),
        pytest.param(f"{_A} 1 - a/\n".encode(), id="file-with-slash"),
        pytest.param(b"\xff\n", id="not-utf-8"),
        pytest.param(f'{_A} 1 - "a"\n'.encode(), id="quoted-plain-path"),
        pytest.param(f'{_A} 1 - "a\\011b"\n'.encode(), id="octal-for-named-escape"),
        pytest.param(f"{_A} 1 - a\tb\n".encode(), id="unquoted-tab"),
        pytest.param(f"{_A} 1 - a \n".encode(), id="unquoted-trailing-space"),
        pytest.param(f"{_A} 1 - a\\b\n".encode(), id="unquoted-backslash"),
        pytest.param(f"{_A} 1 - a\n- - - a/\n".encode(), id="file-and-empty-dir-same-path"),
        pytest.param(f"{_A} 1 - a\n{_A} 1 - a/b\n".encode(), id="file-inside-a-file"),
        pytest.param(f"- - - a/\n{_A} 1 - a/b\n".encode(), id="file-inside-an-empty-dir"),
        pytest.param(b"- - - a/\n- - - a/b/\n", id="empty-dir-inside-an-empty-dir"),
    ],
)
def test_decode_refuses_a_malformed_listing(data: bytes) -> None:
    with pytest.raises(exceptions.CorruptListingError):
        listing.decode(data)


def test_a_directory_hash_is_the_hash_of_its_listing(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "x"
    (root / "bin").mkdir(parents=True)
    (root / "empty").mkdir()
    (root / "a.txt").write_bytes(b"a")
    (root / "bin" / "run.sh").write_bytes(b"#!/bin/sh\n")
    (root / "bin" / "run.sh").chmod(0o755)

    hashed = cache.hash_directory(root)

    expected = f"{_A} 1 - a.txt\n{_RUN} 10 x bin/run.sh\n- - - empty/\n".encode()
    assert hashed["data"] == expected
    assert hashed["record"] == _helper_record(expected, size=11, nfiles=2)
    assert listing.decode(hashed["data"]) == hashed["listing"]


# =============================================================================
# Reading a listing: the committed file, else the cache object, checked against the record
# =============================================================================


def test_read_takes_the_committed_file_and_refuses_one_that_does_not_hash_to_the_record(
    tmp_path: pathlib.Path,
) -> None:
    data = f"{_A} 1 - a.txt\n".encode()
    record = _helper_record(data, size=1, nfiles=1)
    committed = tmp_path / "s=k.listing"
    committed.write_bytes(data)

    assert listing.read(committed, record, tmp_path / "cache") == listing.decode(data)

    committed.write_bytes(data.replace(b" 1 - ", b" 2 - "))
    with pytest.raises(exceptions.CorruptListingError, match=r"s=k\.listing") as info:
        listing.read(committed, record, tmp_path / "cache")
    assert record["hash"] in str(info.value)


def test_read_falls_back_to_the_cache_object_at_the_directory_hash(
    tmp_path: pathlib.Path,
) -> None:
    data = f"{_A} 1 - a.txt\n".encode()
    record = _helper_record(data, size=1, nfiles=1)
    cache_dir = tmp_path / "files"
    listing.store_object(cache_dir, record["hash"], data)

    assert listing.read(tmp_path / "absent.listing", record, cache_dir) == listing.decode(data)
    assert listing.read(None, record, cache_dir) == listing.decode(data)


def test_a_listing_that_does_not_match_its_record_falls_back_to_the_cache_object_unless_strict(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A shared input listing a later run moved on, or a lock reverted without its listing,
    reads the immutable cache object at the record's hash, warning once; `pivot verify` reads
    strictly, since a fresh clone has no cache to fall back on."""
    old_data = f"{_A} 1 - a.txt\n".encode()
    record = _helper_record(old_data, size=1, nfiles=1)
    cache_dir = tmp_path / "files"
    committed = tmp_path / "=inputs" / "data.listing"
    committed.parent.mkdir()
    committed.write_bytes(f"{_A} 1 - b.txt\n".encode())

    with pytest.raises(exceptions.CorruptListingError, match=r"data\.listing"):
        listing.read(committed, record, cache_dir)

    listing.store_object(cache_dir, record["hash"], old_data)
    with caplog.at_level(logging.WARNING):
        first = listing.read(committed, record, cache_dir)
        second = listing.read(committed, record, cache_dir)
    with pytest.raises(exceptions.CorruptListingError, match=r"data\.listing"):
        listing.read(committed, record, cache_dir, strict=True)

    assert first == second == listing.decode(old_data)
    warnings = [r.getMessage() for r in caplog.records if "data.listing" in r.getMessage()]
    assert len(warnings) == 1, warnings


def test_store_object_refuses_bytes_that_do_not_hash_to_the_directory_hash(
    tmp_path: pathlib.Path,
) -> None:
    data = f"{_A} 1 - a.txt\n".encode()
    record = _helper_record(data, size=1, nfiles=1)
    cache_dir = tmp_path / "files"

    with pytest.raises(ValueError, match=record["hash"]):
        listing.store_object(cache_dir, record["hash"], data.replace(b"a.txt", b"b.txt"))

    assert not cache.get_cache_path(cache_dir, record["hash"]).exists(), (
        "No cache object should be published under a hash its bytes do not have"
    )


def test_read_names_both_places_when_neither_holds_the_listing(tmp_path: pathlib.Path) -> None:
    record = _helper_record(b"", size=0, nfiles=0)

    with pytest.raises(exceptions.ListingMissingError, match=r"absent\.listing"):
        listing.read(tmp_path / "absent.listing", record, tmp_path / "files")


def test_write_replaces_a_listing_atomically_and_leaves_identical_bytes_alone(
    tmp_path: pathlib.Path,
) -> None:
    target = tmp_path / "stages" / "p" / "a=k.listing"
    data = f"{_A} 1 - a.txt\n".encode()

    listing.write(target, data)
    first = target.stat()
    listing.write(target, data)

    assert target.read_bytes() == data
    assert target.stat().st_mtime_ns == first.st_mtime_ns, "unchanged bytes must not be rewritten"


# =============================================================================
# Paths
# =============================================================================


@pytest.mark.parametrize(
    ("stage", "key", "expected"),
    [
        pytest.param("p/a", None, "p/a=.listing", id="return-value"),
        pytest.param("p/a", "parts", "p/a=parts.listing", id="typed-dict-field"),
        pytest.param("p/a", "b.c", "p/a=b.c.listing", id="dotted-key"),
        pytest.param("p/a.b", "c", "p/a.b=c.listing", id="dotted-stage"),
        pytest.param(
            "cg-regions/package@r97@production",
            "parts/train.csv",
            "cg-regions/package@r97@production=parts%2Ftrain.csv.listing",
            id="variant-and-slash",
        ),
        pytest.param("p/a", "50% off", "p/a=50%25%20off.listing", id="percent-and-space"),
    ],
)
def test_an_output_listing_sits_beside_its_writers_lock(
    stage: str, key: str | None, expected: str
) -> None:
    stages = pathlib.Path("/project/.pivot/stages")

    path = listing.output_listing_path(stages, types.ArtifactIdentity(stage, key))

    assert path == stages / expected
    assert path.parent == lock.StageLock(stage, stages).path.parent


def _helper_valid_key(key: str) -> bool:
    try:
        types.validate_artifact_identity(types.ArtifactIdentity("p", key))
    except ValueError:
        return False
    return True


_STAGE_NAMES = st.from_regex(r"[a-zA-Z0-9_@.\-]{1,6}(/[a-zA-Z0-9_@.\-]{1,6}){0,2}", fullmatch=True)
_OUTPUTS = st.tuples(
    _STAGE_NAMES.filter(lambda name: lock.stage_name_error(name) is None),
    st.none() | st.text(min_size=1, max_size=10).filter(_helper_valid_key),
)


@hypothesis.given(first=_OUTPUTS, second=_OUTPUTS)
def test_two_outputs_never_share_a_listing_path_and_no_listing_meets_a_lock(
    first: tuple[str, str | None], second: tuple[str, str | None]
) -> None:
    stages = pathlib.Path("/project/.pivot/stages")
    first_path = listing.output_listing_path(stages, types.ArtifactIdentity(*first))
    second_path = listing.output_listing_path(stages, types.ArtifactIdentity(*second))

    assert (first_path == second_path) == (first == second)
    for stage_name in (first[0], second[0]):
        lock_path = lock.StageLock(stage_name, stages).path
        assert lock_path != first_path
        assert not lock_path.is_relative_to(first_path)
        assert not first_path.is_relative_to(lock_path)


@pytest.mark.parametrize(
    ("input_path", "expected"),
    [
        pytest.param("data/raw/x", "=inputs/data%2Fraw%2Fx.listing", id="default-input"),
        pytest.param(
            "data/raw/x.listing/y", "=inputs/data%2Fraw%2Fx.listing%2Fy.listing", id="nested"
        ),
        pytest.param(
            "data/external/box/bundles/F19",
            "=inputs/data%2Fexternal%2Fbox%2Fbundles%2FF19.listing",
            id="bet-bundle",
        ),
    ],
)
def test_an_input_directory_listing_is_named_by_its_path_once_per_project(
    input_path: str, expected: str
) -> None:
    stages = pathlib.Path("/project/.pivot/stages")

    assert listing.input_listing_path(stages, input_path) == stages / expected


@hypothesis.given(first=st.text(min_size=1, max_size=20), second=st.text(min_size=1, max_size=20))
def test_two_input_directories_never_share_a_listing_path(first: str, second: str) -> None:
    stages = pathlib.Path("/project/.pivot/stages")

    same = listing.input_listing_path(stages, first) == listing.input_listing_path(stages, second)

    assert same == (first == second)


def test_a_listing_name_longer_than_a_file_name_uses_a_deterministic_short_name() -> None:
    key = "k" * 250
    assert (
        listing.output_listing_path(pathlib.Path("/s"), types.ArtifactIdentity("p/a", key))
        == pathlib.Path("/s") / f"p/a=~{xxhash.xxh64(key.encode()).hexdigest()}.listing"
    )


def test_a_tracked_directory_listing_sits_beside_its_pvt_file() -> None:
    assert listing.tracked_listing_path(pathlib.Path("data/raw/corpus.pvt")) == pathlib.Path(
        "data/raw/corpus.pvt.listing"
    )

"""The batch orchestration every remote backend shares, tested once against fake operations."""

from __future__ import annotations

import contextlib
import errno
import os
from typing import TYPE_CHECKING

import pytest

from pivot import exceptions, metrics
from pivot.remote import batch, keys

if TYPE_CHECKING:
    import pathlib
    from collections.abc import AsyncGenerator, AsyncIterator

    from pytest_mock import MockerFixture


class _Client:
    """A fake backend client: which objects exist, and a log of what was asked of it."""

    present: set[str]
    opened: int
    head_calls: list[str]
    listed: list[str]
    closed_listings: int
    exhausted_listings: int

    def __init__(self, present: set[str]) -> None:
        self.present = present
        self.opened = 0
        self.head_calls = []
        self.listed = []
        self.closed_listings = 0
        self.exhausted_listings = 0

    @contextlib.asynccontextmanager
    async def open(self) -> AsyncIterator[_Client]:
        self.opened += 1
        yield self


async def _exists(client: _Client, hash_: str) -> bool:
    client.head_calls.append(hash_)
    return hash_ in client.present


async def _names(client: _Client, key_prefix: str) -> AsyncGenerator[str]:
    client.listed.append(key_prefix)
    try:
        for hash_ in sorted(client.present):
            key = keys.hash_to_key("p/", hash_)
            if key.startswith(key_prefix):
                yield key
        yield f"{key_prefix}not-a-cache-object"
        client.exhausted_listings += 1
    finally:
        client.closed_listings += 1


async def _bulk(client: _Client, hashes: list[str], **overrides: object) -> dict[str, bool]:
    options: dict[str, object] = {
        "open_client": client.open,
        "exists_one": _exists,
        "list_names": _names,
        "prefix": "p/",
        "label": "Fake",
        "metric": "fake.bulk_exists",
    }
    options.update(overrides)
    return await batch.bulk_exists(hashes, 4, **options)  # pyright: ignore[reportArgumentType]


# -----------------------------------------------------------------------------
# bulk_exists
# -----------------------------------------------------------------------------


async def test_bulk_exists_below_the_threshold_asks_per_hash() -> None:
    client = _Client({"a000000000000001"})
    hashes = ["a000000000000001", "a000000000000002"]

    result = await _bulk(client, hashes)

    assert result == {"a000000000000001": True, "a000000000000002": False}
    assert sorted(client.head_calls) == hashes
    assert client.listed == []


async def test_bulk_exists_at_the_threshold_lists_each_two_char_prefix_once() -> None:
    hashes = [f"{i % 3:02x}{i:014x}" for i in range(batch.BULK_EXISTS_LIST_THRESHOLD)]
    present = set(hashes[::2])
    client = _Client(present)

    result = await _bulk(client, hashes)

    assert result == {h: h in present for h in hashes}
    assert client.head_calls == []
    assert sorted(client.listed) == ["p/files/00/", "p/files/01/", "p/files/02/"]
    assert (client.closed_listings, client.exhausted_listings) == (3, 3)


async def test_bulk_exists_listing_stops_once_everything_wanted_is_found() -> None:
    """Early exit closes the listing, so a backend's paginator stops fetching pages."""
    hashes = [f"00{i:014x}" for i in range(batch.BULK_EXISTS_LIST_THRESHOLD)]
    client = _Client(set(hashes) | {"00ffffffffffffff"})

    result = await _bulk(client, hashes)

    assert all(result.values())
    assert (client.closed_listings, client.exhausted_listings) == (1, 0)


async def test_bulk_exists_empty_opens_no_client() -> None:
    client = _Client(set())

    assert await _bulk(client, []) == {}
    assert client.opened == 0


async def test_bulk_exists_validates_every_hash_before_opening_a_client() -> None:
    client = _Client(set())

    with pytest.raises(exceptions.RemoteError, match="16 lowercase hex"):
        await _bulk(client, ["a000000000000001", "NOT-A-HASH"])

    assert client.opened == 0


async def test_bulk_exists_head_failure_is_one_error_named_by_label() -> None:
    async def refuse(client: _Client, hash_: str) -> bool:
        raise exceptions.RemoteConnectionError(f"denied {hash_}")

    with pytest.raises(exceptions.RemoteConnectionError) as exc_info:
        await _bulk(_Client(set()), ["a000000000000001"], exists_one=refuse)

    assert str(exc_info.value) == "Fake bulk_exists failed: denied a000000000000001"


async def test_bulk_exists_list_failure_is_one_error_named_by_label() -> None:
    async def refuse(client: _Client, key_prefix: str) -> AsyncGenerator[str]:
        if key_prefix:
            raise exceptions.RemoteConnectionError(f"cannot list {key_prefix}")
        yield key_prefix

    hashes = [f"00{i:014x}" for i in range(batch.BULK_EXISTS_LIST_THRESHOLD)]
    with pytest.raises(exceptions.RemoteConnectionError) as exc_info:
        await _bulk(_Client(set()), hashes, list_names=refuse)

    assert str(exc_info.value) == "Fake bulk_exists LIST failed: cannot list p/files/00/"


async def test_bulk_exists_records_its_metric_when_it_fails(mocker: MockerFixture) -> None:
    async def refuse(client: _Client, hash_: str) -> bool:
        raise exceptions.RemoteConnectionError("denied")

    spy_end = mocker.spy(metrics, "end")

    with pytest.raises(exceptions.RemoteConnectionError):
        await _bulk(_Client(set()), ["a000000000000001"], exists_one=refuse)

    spy_end.assert_any_call("fake.bulk_exists", mocker.ANY)


# -----------------------------------------------------------------------------
# cache_hashes / collect_hashes
# -----------------------------------------------------------------------------


async def test_cache_hashes_skips_names_that_are_not_cache_objects() -> None:
    async def names() -> AsyncGenerator[str]:
        for name in [
            "p/files/ab/cdef1234567890",  # A cache object
            "p/files/AB/CDEF1234567890",  # Uppercase
            "p/files/ab/c",  # Too short
            "p/files/stages/my_stage.lock",  # Not a cache file
            "q/files/12/3456789abcdef0",  # Outside the prefix
        ]:
            yield name

    assert [h async for h in batch.cache_hashes(names(), "p/")] == ["abcdef1234567890"]


async def test_collect_hashes_gathers_every_hash_and_records_its_metric(
    mocker: MockerFixture,
) -> None:
    async def hashes() -> AsyncGenerator[str]:
        yield "abcdef1234567890"
        yield "0123456789abcdef"

    spy_end = mocker.spy(metrics, "end")

    assert await batch.collect_hashes(hashes(), "fake.list_hashes") == {
        "abcdef1234567890",
        "0123456789abcdef",
    }
    spy_end.assert_any_call("fake.list_hashes", mocker.ANY)


# -----------------------------------------------------------------------------
# transfer_batch
# -----------------------------------------------------------------------------


async def _copy(client: _Client, hash_: str, local_path: pathlib.Path) -> None:
    if not local_path.name.startswith("ok"):
        raise OSError(f"cannot read {local_path.name}")


async def test_transfer_batch_reports_each_item_and_never_raises(tmp_path: pathlib.Path) -> None:
    client = _Client(set())
    items = [("a000000000000001", tmp_path / "ok-1"), ("a000000000000002", tmp_path / "bad")]

    results = await batch.transfer_batch(
        items, 4, None, open_client=client.open, transfer_one=_copy, metric="fake.transfer"
    )

    assert results == [
        {"hash": "a000000000000001", "success": True},
        {"hash": "a000000000000002", "success": False, "error": "cannot read bad"},
    ]
    assert client.opened == 1, "one client for the whole batch"


async def test_transfer_batch_words_an_os_error_as_its_message_and_path(
    tmp_path: pathlib.Path,
) -> None:
    """A local file the transfer cannot read is reported as the OS's message and its path, which
    push and pull print, not as Python's `[Errno 13] Permission denied: '<path>'`."""
    local_path = tmp_path / "unreadable"

    async def read(client: _Client, hash_: str, path: pathlib.Path) -> None:
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))

    results = await batch.transfer_batch(
        [("a000000000000001", local_path)],
        4,
        None,
        open_client=_Client(set()).open,
        transfer_one=read,
        metric="fake.transfer",
    )

    assert results == [
        {
            "hash": "a000000000000001",
            "success": False,
            "error": f"{os.strerror(errno.EACCES)}: {local_path}",
        }
    ]


async def _copy_or_fail_on_the_remote(
    client: _Client, hash_: str, local_path: pathlib.Path
) -> None:
    if local_path.name == "unreachable":
        raise exceptions.RemoteUnreachableError("cannot reach the remote")
    if local_path.name == "refused":
        raise exceptions.RemoteConnectionError("permission denied to read this object")
    await _copy(client, hash_, local_path)


async def test_transfer_batch_marks_only_a_transfer_that_could_not_reach_the_remote(
    tmp_path: pathlib.Path,
) -> None:
    """A transfer that raised RemoteUnreachableError says so, so a caller can stop using the
    remote; a transfer the remote refused (a RemoteConnectionError for one object) or that failed
    for another reason does not."""
    items = [
        ("a000000000000001", tmp_path / "unreachable"),
        ("a000000000000002", tmp_path / "refused"),
        ("a000000000000003", tmp_path / "bad"),
    ]

    results = await batch.transfer_batch(
        items,
        4,
        None,
        open_client=_Client(set()).open,
        transfer_one=_copy_or_fail_on_the_remote,
        metric="fake.transfer",
    )

    assert results == [
        {
            "hash": "a000000000000001",
            "success": False,
            "error": "cannot reach the remote",
            "connection_failed": True,
        },
        {
            "hash": "a000000000000002",
            "success": False,
            "error": "permission denied to read this object",
        },
        {"hash": "a000000000000003", "success": False, "error": "cannot read bad"},
    ]


async def test_transfer_batch_calls_back_after_each_success(tmp_path: pathlib.Path) -> None:
    calls = list[tuple[int, int, str]]()
    items = [(f"a00000000000000{i}", tmp_path / f"ok-{i}") for i in range(3)]
    items.append(("a000000000000009", tmp_path / "bad"))

    await batch.transfer_batch(
        items,
        1,
        lambda done, total, name: calls.append((done, total, name)),
        open_client=_Client(set()).open,
        transfer_one=_copy,
        metric="fake.transfer",
    )

    assert calls == [(1, 4, "ok-0"), (2, 4, "ok-1"), (3, 4, "ok-2")]


async def test_transfer_batch_empty_opens_no_client() -> None:
    client = _Client(set())

    results = await batch.transfer_batch(
        [], 4, None, open_client=client.open, transfer_one=_copy, metric="fake.transfer"
    )

    assert results == []
    assert client.opened == 0


async def test_transfer_batch_failure_to_open_the_client_raises_before_any_transfer(
    tmp_path: pathlib.Path,
) -> None:
    class NoCredentials:
        async def __aenter__(self) -> _Client:
            raise exceptions.RemoteConnectionError("no credentials")

        async def __aexit__(self, *exc_info: object) -> None:
            return None

    started = list[str]()

    async def transfer(client: _Client, hash_: str, local_path: pathlib.Path) -> None:
        started.append(hash_)

    with pytest.raises(exceptions.RemoteConnectionError, match="no credentials"):
        await batch.transfer_batch(
            [("a000000000000001", tmp_path / "ok")],
            4,
            None,
            open_client=NoCredentials,
            transfer_one=transfer,
            metric="fake.transfer",
        )

    assert started == []


async def test_transfer_batch_records_its_metric_when_items_fail(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    spy_end = mocker.spy(metrics, "end")

    await batch.transfer_batch(
        [("a000000000000001", tmp_path / "bad")],
        4,
        None,
        open_client=_Client(set()).open,
        transfer_one=_copy,
        metric="fake.transfer",
    )

    spy_end.assert_any_call("fake.transfer", mocker.ANY)

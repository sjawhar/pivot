"""Batch orchestration shared by every remote backend.

A backend supplies its client context and its per-object operations; these functions own the
rules that must not differ between backends: which existence strategy a batch uses, which keys
are cache objects, and that a failed transfer is a ``TransferResult`` rather than an exception.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from typing import TYPE_CHECKING

from pivot import exceptions, metrics
from pivot.remote import keys
from pivot.types import TransferResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

DEFAULT_CONCURRENCY = 20
# Threshold for listing by 2-char prefix instead of one existence request per hash in
# bulk_exists. Per-hash requests are O(n) but lower latency; listing is O(prefixes) requests
# (at most 256 for 2-char hex), each returning up to 1000 keys. At 50 hashes, listing wins
# unless the hashes are spread across 50+ prefixes (unlikely).
BULK_EXISTS_LIST_THRESHOLD = 50


async def cache_hashes(names: AsyncGenerator[str], prefix: str) -> AsyncGenerator[str]:
    """The cache hashes among object ``names`` under the remote's ``prefix``; others are skipped."""
    async with contextlib.aclosing(names):
        async for name in names:
            hash_ = keys.key_to_hash(prefix, name)
            if hash_ is not None:
                yield hash_


async def collect_hashes(hashes: AsyncIterator[str], metric: str) -> set[str]:
    """Every hash ``hashes`` yields, timed as ``metric``."""
    _t = metrics.start()
    collected = set[str]()
    async for hash_ in hashes:
        collected.add(hash_)
    metrics.end(metric, _t)
    return collected


async def bulk_exists[C](
    hashes: list[str],
    concurrency: int,
    *,
    open_client: Callable[[], AbstractAsyncContextManager[C]],
    exists_one: Callable[[C, str], Awaitable[bool]],
    list_names: Callable[[C, str], AsyncGenerator[str]],
    prefix: str,
    label: str,
    metric: str,
) -> dict[str, bool]:
    """Check which ``hashes`` exist on a remote whose cache objects live under ``prefix``.

    Below BULK_EXISTS_LIST_THRESHOLD hashes, asks ``exists_one`` for each in parallel; from the
    threshold up, lists each 2-char key prefix with ``list_names`` instead.

    Raises:
        RemoteError: A hash is not a valid cache hash (before any request).
        Exception: Whatever ``open_client`` raises, unchanged (``RemoteConnectionError`` when a
            backend cannot set up access: GCS unusable credentials, an S3 client that cannot
            start).
        RemoteConnectionError: An existence check or listing failed; the message starts with
            ``label`` (the backend's name).
    """
    _t = metrics.start()
    if not hashes:
        metrics.end(metric, _t)
        return {}

    for h in hashes:
        keys.validate_hash(h)

    try:
        # One client for every request in the batch reuses its connection pool
        async with open_client() as client:
            if len(hashes) >= BULK_EXISTS_LIST_THRESHOLD:
                return await _exists_via_list(
                    client, hashes, concurrency, list_names, prefix, label
                )
            return await _exists_via_head(client, hashes, concurrency, exists_one, label)
    finally:
        metrics.end(metric, _t)


async def _exists_via_head[C](
    client: C,
    hashes: list[str],
    concurrency: int,
    exists_one: Callable[[C, str], Awaitable[bool]],
    label: str,
) -> dict[str, bool]:
    semaphore = asyncio.Semaphore(concurrency)

    async def check_one(hash_: str) -> tuple[str, bool]:
        async with semaphore:
            return hash_, await exists_one(client, hash_)

    results = await asyncio.gather(*[check_one(h) for h in hashes], return_exceptions=True)

    output = dict[str, bool]()
    for result in results:
        if isinstance(result, BaseException):
            raise exceptions.RemoteConnectionError(
                f"{label} bulk_exists failed: {result}"
            ) from result
        hash_, exists = result
        output[hash_] = exists
    return output


async def _exists_via_list[C](
    client: C,
    hashes: list[str],
    concurrency: int,
    list_names: Callable[[C, str], AsyncGenerator[str]],
    prefix: str,
    label: str,
) -> dict[str, bool]:
    """List each 2-char key prefix once: P requests for N hashes spread over P prefixes."""
    by_prefix = defaultdict[str, set[str]](set)
    for h in hashes:
        by_prefix[h[:2]].add(h)

    semaphore = asyncio.Semaphore(concurrency)

    async def list_prefix(hash_prefix: str, wanted: set[str]) -> dict[str, bool]:
        found = set[str]()
        names = list_names(client, f"{prefix}files/{hash_prefix}/")
        async with semaphore, contextlib.aclosing(cache_hashes(names, prefix)) as listed:
            async for hash_ in listed:
                if hash_ in wanted:
                    found.add(hash_)
                    if found == wanted:  # Everything wanted is present: stop listing
                        break
        return {h: h in found for h in wanted}

    tasks = [list_prefix(hash_prefix, wanted) for hash_prefix, wanted in by_prefix.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = dict[str, bool]()
    for result in results:
        if isinstance(result, BaseException):
            raise exceptions.RemoteConnectionError(
                f"{label} bulk_exists LIST failed: {result}"
            ) from result
        output.update(result)
    return output


async def transfer_batch[C](
    items: list[tuple[str, Path]],
    concurrency: int,
    callback: Callable[[int, int, str], None] | None,
    *,
    open_client: Callable[[], AbstractAsyncContextManager[C]],
    transfer_one: Callable[[C, str, Path], Awaitable[None]],
    metric: str,
) -> list[TransferResult]:
    """Run ``transfer_one`` for every ``(hash, local_path)`` in parallel on one client.

    A transfer that raises becomes ``TransferResult(success=False, error=str(exc))``, so one
    failed file never stops the rest; a ``RemoteUnreachableError`` (no connection to the remote
    could be made) also sets ``connection_failed``.
    ``callback(completed, total, file_name)`` runs after each success.

    Raises:
        RemoteError: A hash is not a valid cache hash (before any transfer).
        Exception: Whatever ``open_client`` raises, unchanged, before any transfer starts.
    """
    _t = metrics.start()
    if not items:
        metrics.end(metric, _t)
        return []

    for h, _ in items:
        keys.validate_hash(h)

    try:
        semaphore = asyncio.Semaphore(concurrency)
        completed = 0

        async with open_client() as client:

            async def transfer(hash_: str, local_path: Path) -> TransferResult:
                nonlocal completed
                async with semaphore:
                    try:
                        await transfer_one(client, hash_, local_path)
                        completed += 1
                        if callback:
                            callback(completed, len(items), local_path.name)
                        return TransferResult(hash=hash_, success=True)
                    except exceptions.RemoteUnreachableError as e:
                        return TransferResult(
                            hash=hash_, success=False, error=str(e), connection_failed=True
                        )
                    except Exception as e:
                        return TransferResult(hash=hash_, success=False, error=str(e))

            return await asyncio.gather(*[transfer(h, p) for h, p in items])
    finally:
        metrics.end(metric, _t)

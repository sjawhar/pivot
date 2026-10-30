"""Red tests from the acceptance of sjawhar/pivot#544 (pivot.read_by_hash) at 52169fd1.

Each test pins one defect found on the real surface (`pivot repro` / `pivot status` in a scratch
project on pivot-slices-3); see /mnt/nvme/sami/tmp/BET408/reports/accept544.md. Place this file at
packages/pivot/tests/test_reads_accept544_red.py; it uses the suite's `set_project_root` and
`moto_services` fixtures.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import uuid
from typing import TYPE_CHECKING, Protocol, cast

import boto3
import pytest
import xxhash
import yaml

import pivot
from pivot import exceptions, reads
from pivot.remote import keys
from pivot.storage import cache

if TYPE_CHECKING:
    import pathlib


class _MotoS3Client(Protocol):
    def create_bucket(self, **kwargs: object) -> object: ...

    def put_object(self, **kwargs: object) -> object: ...

    def delete_object(self, **kwargs: object) -> object: ...


def _helper_hash(data: bytes) -> str:
    return xxhash.xxh64(data).hexdigest()


def _helper_put(files_dir: pathlib.Path, data: bytes) -> str:
    content_hash = _helper_hash(data)
    path = cache.get_cache_path(files_dir, content_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o444)
    return content_hash


def _helper_remote(
    root: pathlib.Path, endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[_MotoS3Client, str]:
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    bucket = f"reads-{uuid.uuid4().hex[:8]}"
    s3 = cast("_MotoS3Client", boto3.client("s3", endpoint_url=endpoint, region_name="us-east-1"))
    s3.create_bucket(Bucket=bucket)
    (root / ".pivot" / "config.yaml").write_text(
        yaml.safe_dump({"remotes": {"origin": f"s3://{bucket}/cache"}})
    )
    return s3, bucket


@pytest.fixture
def files_dir(set_project_root: pathlib.Path) -> pathlib.Path:
    return set_project_root / ".pivot" / "cache" / "files"


def test_a_read_from_a_coroutine_fetches_the_object_from_the_remote(
    files_dir: pathlib.Path,
    set_project_root: pathlib.Path,
    moto_services: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defect 1. A stage whose helper code runs an event loop calls `read_by_hash` (a plain
    function) from a coroutine. With the object only on the remote, the read raises
    `RuntimeError: asyncio.run() cannot be called from a running event loop` (observed:
    `pivot repro` of edge/asyncer), not the object. The docstring promises any thread of the
    stage's process may call it, and the event loop's thread is one."""
    s3, bucket = _helper_remote(set_project_root, moto_services["s3"], monkeypatch)
    content_hash = _helper_hash(b"snapshot\n")
    s3.put_object(Bucket=bucket, Key=keys.hash_to_key("cache/", content_hash), Body=b"snapshot\n")

    async def stage_helper() -> bytes:
        return pivot.read_by_hash(content_hash).read_bytes()

    with reads.recording(files_dir) as recorded:
        data = asyncio.run(stage_helper())

    assert data == b"snapshot\n"
    assert recorded.hashes() == [content_hash]


def _forked_read(content_hash: str, queue: multiprocessing.Queue[str]) -> None:
    try:
        _ = pivot.read_by_hash(content_hash)
        queue.put("read")
    except exceptions.ObjectReadError as error:
        queue.put(f"refused: {error}")


def test_a_read_in_a_forked_child_is_refused(files_dir: pathlib.Path) -> None:
    """Defect 2. A stage that forks a child process which calls `read_by_hash` got the object,
    but the child's copy of the recording is lost, so the stage's lock recorded no `reads`
    (observed: edge/forked's lock had no `reads`, and `pivot status` said cached after the object
    left the cache and the remote). Reads are recorded only in the stage's own process and its
    threads, so the child's read is refused, naming that rule (decision D2 of the acceptance)."""
    content_hash = _helper_put(files_dir, b"one\n")
    ctx = multiprocessing.get_context("fork")
    queue: multiprocessing.Queue[str] = ctx.Queue()

    with reads.recording(files_dir) as recorded:
        child = ctx.Process(target=_forked_read, args=(content_hash, queue))
        child.start()
        outcome = queue.get(timeout=60)
        child.join(timeout=60)

    assert outcome.startswith("refused: "), outcome
    assert "only in the stage's own process and its threads" in outcome, outcome
    assert recorded.hashes() == []

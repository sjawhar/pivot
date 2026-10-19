from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import boto3
import xxhash
from moto.server import ThreadedMotoServer

from pivot.remote import keys, storage, sync
from pivot.storage import cache, state

if TYPE_CHECKING:
    import pathlib

    import pytest


class _MotoS3Client(Protocol):
    """Methods used to set up a moto S3 object for this integration test."""

    def create_bucket(self, **kwargs: object) -> object: ...

    def put_object(self, **kwargs: object) -> object: ...


def test_pull_rejects_downloaded_bytes_that_do_not_match_cache_hash(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pull reports corrupt remote bytes and never promotes them into the local cache."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    expected_hash = xxhash.xxh64(b"expected content").hexdigest()
    cache_dir = tmp_path / ".pivot" / "cache"
    state_dir = tmp_path / ".pivot"
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    endpoint_url = f"http://{host}:{port}"
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint_url)

    try:
        bucket = "pivot-integrity-test"
        prefix = "cache/"
        s3 = cast(
            "_MotoS3Client",
            boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_url),
        )
        s3.create_bucket(Bucket=bucket)
        s3.put_object(
            Bucket=bucket,
            Key=keys.hash_to_key(prefix, expected_hash),
            Body=b"corrupt content",
        )
        remote = storage.S3Remote(f"s3://{bucket}/{prefix.rstrip('/')}")

        with state.StateDB(state_dir) as state_db:
            result = sync.pull(cache_dir, state_dir, remote, state_db, "origin", jobs=1)
            assert not state_db.remote_hash_exists("origin", expected_hash)
    finally:
        server.stop()
    assert result["failed"] == 1
    assert any("hash mismatch" in error.lower() for error in result["errors"])
    key = f"{bucket}/{keys.hash_to_key(prefix, expected_hash)}"
    assert any(
        f"Delete {key} from the remote, then run `pivot push`" in error
        and "has not run `pivot push`, `pivot pull`, `pivot fetch`, or `pivot status --remote`"
        in error
        and "delete `.pivot/state.lmdb/`" in error
        for error in result["errors"]
    ), result["errors"]
    assert not cache.get_cache_path(cache_dir / "files", expected_hash).exists()

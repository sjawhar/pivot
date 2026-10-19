import pytest

from pivot.remote import gcs, protocol, storage


@pytest.mark.parametrize("url,cls", [("s3://b/p/", storage.S3Remote), ("gs://b/p/", gcs.GCSRemote)])
def test_open_remote_picks_backend(url: str, cls: type[protocol.RemoteStorage]) -> None:
    assert isinstance(protocol.open_remote(url), cls)


@pytest.mark.parametrize("url", ["/local/path", "gcs://b", "s3:/b", "gs://"])
def test_open_remote_rejects(url: str) -> None:
    with pytest.raises(Exception, match="remote URL"):
        protocol.open_remote(url)


def test_both_backends_satisfy_protocol() -> None:
    for cls in (storage.S3Remote, gcs.GCSRemote):
        missing = [
            m
            for m in (
                "exists",
                "bulk_exists",
                "iter_hashes",
                "list_hashes",
                "upload_file",
                "download_file",
                "upload_batch",
                "download_batch",
            )
            if not hasattr(cls, m)
        ]
        assert missing == []

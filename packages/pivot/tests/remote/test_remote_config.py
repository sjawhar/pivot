from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from pivot import exceptions, project
from pivot.remote import config as remote_config

if TYPE_CHECKING:
    from pathlib import Path

# -----------------------------------------------------------------------------
# URL Validation Tests
# -----------------------------------------------------------------------------


_BOTH_SCHEMES = r"Expected format: s3://bucket/prefix or gs://bucket/prefix"


def test_validate_s3_url_basic() -> None:
    scheme, bucket, prefix = remote_config.validate_remote_url("s3://my-bucket/my-prefix")
    assert scheme is remote_config.RemoteScheme.S3
    assert bucket == "my-bucket"
    assert prefix == "my-prefix"


def test_validate_s3_url_no_prefix() -> None:
    _, bucket, prefix = remote_config.validate_remote_url("s3://my-bucket")
    assert bucket == "my-bucket"
    assert prefix == ""


def test_validate_s3_url_trailing_slash() -> None:
    _, bucket, prefix = remote_config.validate_remote_url("s3://my-bucket/prefix/")
    assert bucket == "my-bucket"
    assert prefix == "prefix/"


def test_validate_s3_url_nested_prefix() -> None:
    _, bucket, prefix = remote_config.validate_remote_url("s3://bucket/path/to/cache")
    assert bucket == "bucket"
    assert prefix == "path/to/cache"


@pytest.mark.parametrize(
    ("url", "bucket", "prefix"),
    [
        pytest.param("gs://my-bucket/my-prefix", "my-bucket", "my-prefix", id="prefix"),
        pytest.param("gs://my-bucket", "my-bucket", "", id="no-prefix"),
        pytest.param("gs://bucket/path/to/cache/", "bucket", "path/to/cache/", id="nested"),
    ],
)
def test_validate_gs_url(url: str, bucket: str, prefix: str) -> None:
    assert remote_config.validate_remote_url(url) == (remote_config.RemoteScheme.GS, bucket, prefix)


def test_validate_s3_url_invalid_scheme() -> None:
    with pytest.raises(
        exceptions.InvalidRemoteURLError, match=f"Invalid remote URL.*{_BOTH_SCHEMES}"
    ):
        remote_config.validate_remote_url("http://bucket/prefix")


def test_validate_s3_url_missing_bucket() -> None:
    with pytest.raises(
        exceptions.InvalidRemoteURLError, match=f"Invalid remote URL.*{_BOTH_SCHEMES}"
    ):
        remote_config.validate_remote_url("s3://")


def test_validate_s3_url_not_url() -> None:
    with pytest.raises(
        exceptions.InvalidRemoteURLError, match=f"Invalid remote URL.*{_BOTH_SCHEMES}"
    ):
        remote_config.validate_remote_url("just-a-string")


@pytest.mark.parametrize("url", ["gcs://bucket/prefix", "gs://", "gs:/bucket"])
def test_validate_gs_url_rejects(url: str) -> None:
    with pytest.raises(exceptions.InvalidRemoteURLError, match=_BOTH_SCHEMES):
        remote_config.validate_remote_url(url)


@pytest.mark.parametrize("scheme", list(remote_config.RemoteScheme))
def test_validate_remote_url_accepts_every_scheme(scheme: remote_config.RemoteScheme) -> None:
    """The URL grammar accepts exactly the schemes RemoteScheme names."""
    assert remote_config.validate_remote_url(f"{scheme}://bucket/p") == (scheme, "bucket", "p")


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("s3://My-Bucket/p", id="s3-uppercase"),
        pytest.param("gs://My-Bucket/p", id="gs-uppercase"),
        pytest.param("gs://bucket?alt=json/p", id="query"),
        pytest.param("s3://bucket#frag/p", id="fragment"),
        pytest.param("gs://-bucket/p", id="leading-hyphen"),
        pytest.param("s3://bucket./p", id="trailing-dot"),
        pytest.param("s3://my_bucket/p", id="s3-underscore"),
        pytest.param("gs://buck%2Fet/p", id="percent"),
    ],
)
def test_validate_remote_url_rejects_invalid_bucket_names(url: str) -> None:
    """A bucket name outside the cloud's naming rules is refused before any request is built."""
    with pytest.raises(exceptions.InvalidRemoteURLError, match=f"Bucket name .*{_BOTH_SCHEMES}"):
        remote_config.validate_remote_url(url)


@pytest.mark.parametrize(
    ("url", "bucket"),
    [
        pytest.param("gs://my_bucket.v2-a/p", "my_bucket.v2-a", id="gs-underscore-dot-hyphen"),
        pytest.param("s3://my.bucket-1/p", "my.bucket-1", id="s3-dot-hyphen"),
        pytest.param("s3://b/p", "b", id="one-character"),
    ],
)
def test_validate_remote_url_accepts_valid_bucket_names(url: str, bucket: str) -> None:
    _, parsed_bucket, _ = remote_config.validate_remote_url(url)
    assert parsed_bucket == bucket


def test_invalid_remote_url_suggests_only_supported_schemes() -> None:
    with pytest.raises(exceptions.InvalidRemoteURLError) as exc_info:
        remote_config.validate_remote_url("/local/path")
    assert exc_info.value.get_suggestion() == "Use format: s3://bucket/prefix or gs://bucket/prefix"


# -----------------------------------------------------------------------------
# Remote Configuration CRUD Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def config_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pivot_dir = tmp_path / ".pivot"
    pivot_dir.mkdir()
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    return tmp_path


def test_add_remote(config_project: Path) -> None:
    remote_config.add_remote("origin", "s3://bucket/prefix")

    config_path = config_project / ".pivot" / "config.yaml"
    assert config_path.exists()

    with config_path.open() as f:
        data = yaml.safe_load(f)

    assert data["remotes"]["origin"] == "s3://bucket/prefix"


def test_add_remote_validates_url(config_project: Path) -> None:
    with pytest.raises(exceptions.InvalidRemoteURLError):
        remote_config.add_remote("bad", "not-an-s3-url")


def test_add_remote_accepts_gs_url(config_project: Path) -> None:
    remote_config.add_remote("gcs", "gs://bucket/prefix")

    assert remote_config.list_remotes() == {"gcs": "gs://bucket/prefix"}


def test_add_multiple_remotes(config_project: Path) -> None:
    remote_config.add_remote("origin", "s3://bucket1/prefix")
    remote_config.add_remote("backup", "s3://bucket2/other")

    remotes = remote_config.list_remotes()
    assert remotes == {"origin": "s3://bucket1/prefix", "backup": "s3://bucket2/other"}


def test_add_remote_overwrites_existing(config_project: Path) -> None:
    remote_config.add_remote("origin", "s3://old-bucket/prefix")
    remote_config.add_remote("origin", "s3://new-bucket/prefix")

    remotes = remote_config.list_remotes()
    assert remotes["origin"] == "s3://new-bucket/prefix"


def test_remove_remote(config_project: Path) -> None:
    remote_config.add_remote("origin", "s3://bucket/prefix")
    remote_config.remove_remote("origin")

    remotes = remote_config.list_remotes()
    assert "origin" not in remotes


def test_remove_remote_not_found(config_project: Path) -> None:
    with pytest.raises(exceptions.RemoteNotFoundError, match="not found"):
        remote_config.remove_remote("nonexistent")


def test_remove_remote_clears_default(config_project: Path) -> None:
    remote_config.add_remote("origin", "s3://bucket/prefix")
    remote_config.set_default_remote("origin")
    remote_config.remove_remote("origin")

    assert remote_config.get_default_remote() is None


def test_list_remotes_empty(config_project: Path) -> None:
    remotes = remote_config.list_remotes()
    assert remotes == {}


def test_list_remotes(config_project: Path) -> None:
    remote_config.add_remote("a", "s3://bucket-a/prefix")
    remote_config.add_remote("b", "s3://bucket-b/prefix")

    remotes = remote_config.list_remotes()
    assert len(remotes) == 2
    assert "a" in remotes
    assert "b" in remotes


# -----------------------------------------------------------------------------
# Default Remote Tests
# -----------------------------------------------------------------------------


def test_set_default_remote(config_project: Path) -> None:
    remote_config.add_remote("origin", "s3://bucket/prefix")
    remote_config.set_default_remote("origin")

    assert remote_config.get_default_remote() == "origin"


def test_set_default_remote_not_found(config_project: Path) -> None:
    with pytest.raises(exceptions.RemoteNotFoundError, match="not found"):
        remote_config.set_default_remote("nonexistent")


def test_get_default_remote_none(config_project: Path) -> None:
    assert remote_config.get_default_remote() is None


# -----------------------------------------------------------------------------
# Get Remote URL Tests
# -----------------------------------------------------------------------------


def test_get_remote_url_by_name(config_project: Path) -> None:
    remote_config.add_remote("origin", "s3://bucket/prefix")

    url = remote_config.get_remote_url("origin")
    assert url == "s3://bucket/prefix"


def test_get_remote_url_default(config_project: Path) -> None:
    remote_config.add_remote("origin", "s3://bucket/prefix")
    remote_config.set_default_remote("origin")

    url = remote_config.get_remote_url()
    assert url == "s3://bucket/prefix"


def test_get_remote_url_single_remote_auto_default(config_project: Path) -> None:
    remote_config.add_remote("only", "s3://bucket/prefix")

    url = remote_config.get_remote_url()
    assert url == "s3://bucket/prefix"


def test_get_remote_url_multiple_no_default_error(config_project: Path) -> None:
    remote_config.add_remote("a", "s3://bucket-a/prefix")
    remote_config.add_remote("b", "s3://bucket-b/prefix")

    with pytest.raises(exceptions.RemoteNotFoundError, match="No default remote"):
        remote_config.get_remote_url()


def test_get_remote_url_not_found(config_project: Path) -> None:
    remote_config.add_remote("origin", "s3://bucket/prefix")

    with pytest.raises(exceptions.RemoteNotFoundError, match="not found"):
        remote_config.get_remote_url("nonexistent")


def test_get_remote_url_no_remotes(config_project: Path) -> None:
    with pytest.raises(exceptions.RemoteNotFoundError, match="No remotes configured"):
        remote_config.get_remote_url()

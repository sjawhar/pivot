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


def test_validate_s3_url_basic() -> None:
    bucket, prefix = remote_config.validate_s3_url("s3://my-bucket/my-prefix")
    assert bucket == "my-bucket"
    assert prefix == "my-prefix"


def test_validate_s3_url_no_prefix() -> None:
    bucket, prefix = remote_config.validate_s3_url("s3://my-bucket")
    assert bucket == "my-bucket"
    assert prefix == ""


def test_validate_s3_url_trailing_slash() -> None:
    bucket, prefix = remote_config.validate_s3_url("s3://my-bucket/prefix/")
    assert bucket == "my-bucket"
    assert prefix == "prefix/"


def test_validate_s3_url_nested_prefix() -> None:
    bucket, prefix = remote_config.validate_s3_url("s3://bucket/path/to/cache")
    assert bucket == "bucket"
    assert prefix == "path/to/cache"


def test_validate_s3_url_invalid_scheme() -> None:
    with pytest.raises(exceptions.InvalidRemoteURLError, match="Invalid S3 URL"):
        remote_config.validate_s3_url("http://bucket/prefix")


def test_validate_s3_url_missing_bucket() -> None:
    with pytest.raises(exceptions.InvalidRemoteURLError, match="Invalid S3 URL"):
        remote_config.validate_s3_url("s3://")


def test_validate_s3_url_not_url() -> None:
    with pytest.raises(exceptions.InvalidRemoteURLError, match="Invalid S3 URL"):
        remote_config.validate_s3_url("just-a-string")


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

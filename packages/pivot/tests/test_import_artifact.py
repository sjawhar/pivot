# pyright: reportMissingImports=false
from __future__ import annotations

import asyncio
import pathlib
from typing import TYPE_CHECKING

import pytest
import xxhash
import yaml

from pivot import exceptions, import_artifact
from pivot.remote import keys
from pivot.remote import storage as remote_mod
from pivot.storage import track

if TYPE_CHECKING:
    from pytest_mock import MockerFixture
    from types_aiobotocore_s3 import S3Client


def _lock_bytes(*, outs: list[dict[str, object]]) -> bytes:
    """Build a schema-v3 remote lock with identity-addressed outputs."""
    data = {
        "schema_version": 3,
        "code_manifest": {"stage": "hash"},
        "params": {},
        "deps": [],
        "outs": outs,
    }
    return yaml.safe_dump(data).encode()


def _config_bytes(remote_url: str) -> bytes:
    data = {
        "remotes": {"origin": remote_url},
        "default_remote": "origin",
    }
    return yaml.safe_dump(data).encode()


def _github_files(*names: str) -> list[dict[str, str]]:
    return [{"name": name, "type": "file"} for name in names]


def test_resolve_remote_path_exact_file(mocker: MockerFixture) -> None:
    repo_url = "https://github.com/org/repo"
    remote_url = "s3://bucket/prefix"
    lock_bytes = _lock_bytes(
        outs=[
            {
                "key": None,
                "tag": "data",
                "hash": "abc1230000000000",
            }
        ]
    )

    list_directory = mocker.patch(
        "pivot.import_artifact.github.list_directory",
        autospec=True,
    )
    list_directory.return_value = _github_files("train.lock")
    resolve_ref = mocker.patch(
        "pivot.import_artifact.github.resolve_ref",
        autospec=True,
    )
    resolve_ref.return_value = "deadbeef"

    async def _read_file(
        _owner: str, _repo: str, path: str, _ref: str, _token: str | None = None, **_kw: object
    ) -> bytes:
        if path == ".pivot/config.yaml":
            return _config_bytes(remote_url)
        if path == ".pivot/stages/train.lock":
            return lock_bytes
        raise AssertionError(f"Unexpected path: {path}")

    read_file = mocker.patch(
        "pivot.import_artifact.github.read_file",
        autospec=True,
    )
    read_file.side_effect = _read_file

    resolved = asyncio.run(import_artifact.resolve_remote_path(repo_url, "train", "main", None))

    assert resolved["stage"] == "train"
    assert resolved["path"] == "train"
    assert resolved["hash"] == "abc1230000000000"
    assert resolved["size"] == 0
    assert resolved["remote_url"] == remote_url
    assert resolved["rev_lock"] == "deadbeef"


def test_resolve_remote_path_identity_disambiguates_stages(mocker: MockerFixture) -> None:
    repo_url = "https://github.com/org/repo"
    remote_url = "s3://bucket/prefix"
    lock_bytes = _lock_bytes(
        outs=[
            {
                "key": None,
                "tag": "data",
                "hash": "abc1230000000000",
            }
        ]
    )

    list_directory = mocker.patch(
        "pivot.import_artifact.github.list_directory",
        autospec=True,
    )
    list_directory.return_value = _github_files("train.lock", "eval.lock")
    resolve_ref = mocker.patch(
        "pivot.import_artifact.github.resolve_ref",
        autospec=True,
    )
    resolve_ref.return_value = "deadbeef"

    async def _read_file(
        _owner: str, _repo: str, path: str, _ref: str, _token: str | None = None, **_kw: object
    ) -> bytes:
        if path == ".pivot/config.yaml":
            return _config_bytes(remote_url)
        if path in {".pivot/stages/train.lock", ".pivot/stages/eval.lock"}:
            return lock_bytes
        raise AssertionError(f"Unexpected path: {path}")

    read_file = mocker.patch(
        "pivot.import_artifact.github.read_file",
        autospec=True,
    )
    read_file.side_effect = _read_file

    # Artifact identities include their producer stage, so same-key outputs are unambiguous.
    resolved = asyncio.run(import_artifact.resolve_remote_path(repo_url, "train", "main", None))

    assert resolved["stage"] == "train"
    assert resolved["path"] == "train"
    assert resolved["hash"] == "abc1230000000000"


def test_resolve_remote_path_not_found(mocker: MockerFixture) -> None:
    repo_url = "https://github.com/org/repo"
    remote_url = "s3://bucket/prefix"
    lock_bytes = _lock_bytes(
        outs=[
            {
                "key": None,
                "tag": "data",
                "hash": "abc1230000000000",
            }
        ]
    )

    list_directory = mocker.patch(
        "pivot.import_artifact.github.list_directory",
        autospec=True,
    )
    list_directory.return_value = _github_files("train.lock")
    resolve_ref = mocker.patch(
        "pivot.import_artifact.github.resolve_ref",
        autospec=True,
    )
    resolve_ref.return_value = "deadbeef"

    async def _read_file(
        _owner: str, _repo: str, path: str, _ref: str, _token: str | None = None, **_kw: object
    ) -> bytes:
        if path == ".pivot/config.yaml":
            return _config_bytes(remote_url)
        if path == ".pivot/stages/train.lock":
            return lock_bytes
        raise AssertionError(f"Unexpected path: {path}")

    read_file = mocker.patch(
        "pivot.import_artifact.github.read_file",
        autospec=True,
    )
    read_file.side_effect = _read_file

    with pytest.raises(exceptions.PivotError, match="Available outputs"):
        asyncio.run(import_artifact.resolve_remote_path(repo_url, "missing", "main", None))


def test_read_remote_config_success(mocker: MockerFixture) -> None:
    repo_url = "https://github.com/org/repo"
    remote_url = "s3://bucket/prefix"

    read_file = mocker.patch(
        "pivot.import_artifact.github.read_file",
        autospec=True,
    )
    read_file.return_value = _config_bytes(remote_url)

    result = asyncio.run(import_artifact.read_remote_config(repo_url, "main", None))

    assert result == remote_url


async def test_import_artifact_creates_pvt(
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    moto_s3_bucket: str,
    aioboto3_s3_client: S3Client,
) -> None:
    repo_url = "https://github.com/org/repo"
    remote_url = f"s3://{moto_s3_bucket}/test-prefix/"
    content = b"hello"
    file_hash = xxhash.xxh64(content).hexdigest()
    lock_bytes = _lock_bytes(
        outs=[
            {
                "key": None,
                "tag": "data",
                "hash": file_hash,
            }
        ]
    )

    list_directory = mocker.patch(
        "pivot.import_artifact.github.list_directory",
        autospec=True,
    )
    list_directory.return_value = _github_files("train.lock")
    resolve_ref = mocker.patch(
        "pivot.import_artifact.github.resolve_ref",
        autospec=True,
    )
    resolve_ref.return_value = "deadbeef"

    async def _read_file(
        _owner: str, _repo: str, path: str, _ref: str, _token: str | None = None, **_kw: object
    ) -> bytes:
        if path == ".pivot/config.yaml":
            return _config_bytes(remote_url)
        if path == ".pivot/stages/train.lock":
            return lock_bytes
        raise AssertionError(f"Unexpected path: {path}")

    read_file = mocker.patch(
        "pivot.import_artifact.github.read_file",
        autospec=True,
    )
    read_file.side_effect = _read_file

    key = keys.hash_to_key("test-prefix/", file_hash)
    await aioboto3_s3_client.put_object(Bucket=moto_s3_bucket, Key=key, Body=content)

    mocker.patch(
        "pivot.remote.storage.S3Remote",
        autospec=True,
        return_value=remote_mod.S3Remote(remote_url),
    )

    result = await import_artifact.import_artifact(
        repo_url,
        "train",
        rev="main",
        project_root=tmp_path,
    )

    data_path = tmp_path / "train"
    pvt_path = track.get_pvt_path(data_path)
    assert pathlib.Path(result["data_path"]) == data_path
    assert pathlib.Path(result["pvt_path"]) == pvt_path
    assert result["downloaded"] is True
    assert data_path.read_bytes() == content
    pvt_data = track.read_pvt_file(pvt_path)
    assert pvt_data is not None
    assert pvt_data["hash"] == file_hash
    assert "source" in pvt_data
    assert pvt_data["source"]["stage"] == "train"
    assert pvt_data["source"]["path"] == "train"
    assert pvt_data["source"]["remote"] == remote_url
    assert pvt_data["size"] == len(content), "Size should come from downloaded file"


def test_import_artifact_no_download(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    repo_url = "https://github.com/org/repo"
    remote_url = "s3://bucket/prefix"
    content = b"hello"
    file_hash = xxhash.xxh64(content).hexdigest()
    lock_bytes = _lock_bytes(
        outs=[
            {
                "key": None,
                "tag": "data",
                "hash": file_hash,
            }
        ]
    )

    list_directory = mocker.patch(
        "pivot.import_artifact.github.list_directory",
        autospec=True,
    )
    list_directory.return_value = _github_files("train.lock")
    resolve_ref = mocker.patch(
        "pivot.import_artifact.github.resolve_ref",
        autospec=True,
    )
    resolve_ref.return_value = "deadbeef"

    async def _read_file(
        _owner: str, _repo: str, path: str, _ref: str, _token: str | None = None, **_kw: object
    ) -> bytes:
        if path == ".pivot/config.yaml":
            return _config_bytes(remote_url)
        if path == ".pivot/stages/train.lock":
            return lock_bytes
        raise AssertionError(f"Unexpected path: {path}")

    read_file = mocker.patch(
        "pivot.import_artifact.github.read_file",
        autospec=True,
    )
    read_file.side_effect = _read_file

    mocker.patch(
        "pivot.remote.storage.S3Remote",
        autospec=True,
        side_effect=AssertionError("S3Remote should not be instantiated"),
    )

    result = asyncio.run(
        import_artifact.import_artifact(
            repo_url,
            "train",
            rev="main",
            project_root=tmp_path,
            no_download=True,
        )
    )

    data_path = tmp_path / "train"
    pvt_path = track.get_pvt_path(data_path)
    assert pathlib.Path(result["data_path"]) == data_path
    assert pathlib.Path(result["pvt_path"]) == pvt_path
    assert result["downloaded"] is False
    assert not data_path.exists()
    pvt_data = track.read_pvt_file(pvt_path)
    assert pvt_data is not None
    assert "source" in pvt_data
    assert pvt_data["source"]["path"] == "train"


async def test_import_artifact_force_overwrites(
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    moto_s3_bucket: str,
    aioboto3_s3_client: S3Client,
) -> None:
    repo_url = "https://github.com/org/repo"
    remote_url = f"s3://{moto_s3_bucket}/test-prefix/"
    content = b"new"
    file_hash = xxhash.xxh64(content).hexdigest()
    lock_bytes = _lock_bytes(
        outs=[
            {
                "key": None,
                "tag": "data",
                "hash": file_hash,
            }
        ]
    )

    list_directory = mocker.patch(
        "pivot.import_artifact.github.list_directory",
        autospec=True,
    )
    list_directory.return_value = _github_files("train.lock")
    resolve_ref = mocker.patch(
        "pivot.import_artifact.github.resolve_ref",
        autospec=True,
    )
    resolve_ref.return_value = "deadbeef"

    async def _read_file(
        _owner: str, _repo: str, path: str, _ref: str, _token: str | None = None, **_kw: object
    ) -> bytes:
        if path == ".pivot/config.yaml":
            return _config_bytes(remote_url)
        if path == ".pivot/stages/train.lock":
            return lock_bytes
        raise AssertionError(f"Unexpected path: {path}")

    read_file = mocker.patch(
        "pivot.import_artifact.github.read_file",
        autospec=True,
    )
    read_file.side_effect = _read_file

    key = keys.hash_to_key("test-prefix/", file_hash)
    await aioboto3_s3_client.put_object(Bucket=moto_s3_bucket, Key=key, Body=content)

    mocker.patch(
        "pivot.remote.storage.S3Remote",
        autospec=True,
        return_value=remote_mod.S3Remote(remote_url),
    )

    data_path = tmp_path / "train"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text("old")
    track.write_pvt_file(
        track.get_pvt_path(data_path),
        track.PvtData(path="train", hash="old", size=3),
    )

    result = await import_artifact.import_artifact(
        repo_url,
        "train",
        rev="main",
        project_root=tmp_path,
        force=True,
    )

    assert pathlib.Path(result["data_path"]) == data_path
    assert data_path.read_bytes() == content
    pvt_path = track.get_pvt_path(data_path)
    pvt_data = track.read_pvt_file(pvt_path)
    assert pvt_data is not None
    assert pvt_data["hash"] == file_hash
    assert pvt_data["size"] == len(content), "Size should come from downloaded file"


@pytest.mark.parametrize(
    ("stage", "key", "directory", "listing_path", "artifact", "hint", "data_rel"),
    [
        pytest.param(
            "images",
            None,
            "images",
            ".pivot/stages/images=.listing",
            "images:part/a.csv",
            "images:{filename}",
            "images/part/a.csv",
            id="return-value",
        ),
        pytest.param(
            "build",
            "images",
            "build:images",
            ".pivot/stages/build=images.listing",
            "build:images/part/a.csv",
            "build:images/{filename}",
            "build/images/part/a.csv",
            id="typed-dict-field",
        ),
    ],
)
async def test_import_artifact_imports_one_file_from_remote_directory_listing(
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    moto_s3_bucket: str,
    aioboto3_s3_client: S3Client,
    stage: str,
    key: str | None,
    directory: str,
    listing_path: str,
    artifact: str,
    hint: str,
    data_rel: str,
) -> None:
    repo_url = "https://github.com/org/repo"
    remote_url = f"s3://{moto_s3_bucket}/test-prefix/"
    selected = b"a,b\n1,2\n"
    other = b"b\n"
    selected_hash = xxhash.xxh64(selected).hexdigest()
    other_hash = xxhash.xxh64(other).hexdigest()
    listing_bytes = (
        f"{selected_hash} {len(selected)} - part/a.csv\n"
        + f"{other_hash} {len(other)} - part/b.csv\n"
    ).encode()
    lock_bytes = _lock_bytes(
        outs=[
            {
                "key": key,
                "tag": "directory",
                "listing": xxhash.xxh64(listing_bytes).hexdigest(),
                "size": len(selected) + len(other),
                "nfiles": 2,
            }
        ]
    )

    list_directory = mocker.patch(
        "pivot.import_artifact.github.list_directory",
        autospec=True,
    )
    list_directory.return_value = _github_files(f"{stage}.lock")
    resolve_ref = mocker.patch(
        "pivot.import_artifact.github.resolve_ref",
        autospec=True,
    )
    resolve_ref.return_value = "deadbeef"

    async def _read_file(
        _owner: str, _repo: str, path: str, _ref: str, _token: str | None = None, **_kw: object
    ) -> bytes:
        if path == ".pivot/config.yaml":
            return _config_bytes(remote_url)
        if path == f".pivot/stages/{stage}.lock":
            return lock_bytes
        if path == listing_path:
            return listing_bytes
        raise AssertionError(f"Unexpected path: {path}")

    read_file = mocker.patch(
        "pivot.import_artifact.github.read_file",
        autospec=True,
    )
    read_file.side_effect = _read_file

    with pytest.raises(exceptions.PivotError) as refused:
        await import_artifact.import_artifact(
            repo_url,
            directory,
            rev="main",
            project_root=tmp_path,
            no_download=True,
        )
    assert f"Cannot import directory output '{directory}'" in str(refused.value)
    assert hint in str(refused.value), "Directory refusal should name the per-file artifact form"

    key_path = keys.hash_to_key("test-prefix/", selected_hash)
    await aioboto3_s3_client.put_object(Bucket=moto_s3_bucket, Key=key_path, Body=selected)
    mocker.patch(
        "pivot.remote.storage.S3Remote",
        autospec=True,
        return_value=remote_mod.S3Remote(remote_url),
    )

    result = await import_artifact.import_artifact(
        repo_url,
        artifact,
        rev="main",
        project_root=tmp_path,
    )

    data_path = tmp_path / data_rel
    assert pathlib.Path(result["data_path"]) == data_path
    assert data_path.read_bytes() == selected
    pvt_data = track.read_pvt_file(track.get_pvt_path(data_path))
    assert pvt_data is not None
    assert pvt_data["hash"] == selected_hash, "Import should use the listing member hash"
    assert pvt_data["size"] == len(selected), "Import should use the selected member size"
    assert "source" in pvt_data
    assert pvt_data["source"]["stage"] == stage
    assert pvt_data["source"]["path"] == artifact

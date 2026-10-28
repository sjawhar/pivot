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
    """Build a schema-v2 remote lock with identity-addressed outputs."""
    data = {
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

    remote_url_result, config_data = asyncio.run(
        import_artifact.read_remote_config(repo_url, "main", None)
    )

    assert remote_url_result == remote_url
    assert config_data["remotes"] == {"origin": remote_url}


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


_OTHER_TREE_PIPELINE = """\
import pathlib
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def tree() -> Annotated[dict[str, str], directory]:
    return {"generated.txt": "generated"}


with Pipeline("b", root=pathlib.Path(__file__).parent.parent) as pipeline:
    tree()
"""


def _helper_mock_resolved_import(mocker: MockerFixture) -> None:
    resolved = import_artifact.ResolvedImport(
        stage="source/build",
        path="source/build",
        hash="abc1230000000000",
        size=7,
        remote_url="s3://bucket/prefix",
        rev_lock="deadbeef",
    )
    mocker.patch("pivot.import_artifact.resolve_remote_path", autospec=True, return_value=resolved)


@pytest.mark.parametrize("no_download", [False, True])
async def test_import_artifact_refuses_destination_inside_any_pipeline_output(
    tmp_path: pathlib.Path, mocker: MockerFixture, no_download: bool
) -> None:
    """An import cannot create data or metadata inside another pipeline's directory output."""
    (tmp_path / ".pivot").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "pipeline.py").write_text(_OTHER_TREE_PIPELINE)
    _helper_mock_resolved_import(mocker)

    with pytest.raises(exceptions.PivotError, match="stage output"):
        await import_artifact.import_artifact(
            "https://github.com/org/source",
            "source/build",
            out="data/b/tree/imported.txt",
            no_download=no_download,
            project_root=tmp_path,
        )

    assert not (tmp_path / "data" / "b" / "tree" / "imported.txt").exists()
    assert not (tmp_path / "data" / "b" / "tree" / "imported.txt.pvt").exists()


@pytest.mark.parametrize("no_download", [False, True])
async def test_import_artifact_refuses_when_a_pipeline_fails_to_load(
    tmp_path: pathlib.Path, mocker: MockerFixture, no_download: bool
) -> None:
    """A pipeline.py that fails to load may declare the destination as an output, so the import
    refuses, naming that file, and writes nothing."""
    (tmp_path / ".pivot").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "pipeline.py").write_text('raise RuntimeError("broken pipeline")\n')
    _helper_mock_resolved_import(mocker)

    with pytest.raises(exceptions.PivotError, match=r"Failed to load pipeline .*other"):
        await import_artifact.import_artifact(
            "https://github.com/org/source",
            "source/build",
            out="data/raw/imported.txt",
            no_download=no_download,
            project_root=tmp_path,
        )

    assert not (tmp_path / "data" / "raw" / "imported.txt").exists()
    assert not (tmp_path / "data" / "raw" / "imported.txt.pvt").exists()


@pytest.mark.parametrize("no_download", [False, True])
async def test_import_artifact_force_updates_an_import_saved_inside_an_output(
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    no_download: bool,
) -> None:
    """An import made inside an output before the check is saved input: --force updates it in
    place and warns, naming the output. The downloading and metadata-only forms both work."""
    (tmp_path / ".pivot").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "pipeline.py").write_text(_OTHER_TREE_PIPELINE)
    _helper_mock_resolved_import(mocker)
    pvt_path = tmp_path / "data" / "b" / "tree" / "imported.txt.pvt"
    pvt_path.parent.mkdir(parents=True)
    track.write_pvt_file(pvt_path, track.PvtData(path="imported.txt", hash="0" * 16, size=1))

    async def _download(_url: str, _hash: str, data_path: pathlib.Path) -> int:
        data_path.write_bytes(b"downloaded")
        return 10

    mocker.patch("pivot.import_artifact._download_verified", autospec=True, side_effect=_download)

    result = await import_artifact.import_artifact(
        "https://github.com/org/source",
        "source/build",
        out="data/b/tree/imported.txt",
        force=True,
        no_download=no_download,
        project_root=tmp_path,
    )

    assert pathlib.Path(result["pvt_path"]) == pvt_path
    pvt_data = track.read_pvt_file(pvt_path)
    assert pvt_data is not None
    assert pvt_data["hash"] == "abc1230000000000"
    assert (
        "'data/b/tree/imported.txt' was imported inside a stage output: it is inside output "
        + "'data/b/tree' of stage 'b/tree'"
    ) in caplog.text
    if no_download:
        assert not (tmp_path / "data" / "b" / "tree" / "imported.txt").exists()
    else:
        assert (tmp_path / "data" / "b" / "tree" / "imported.txt").read_bytes() == b"downloaded"


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

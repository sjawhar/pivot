# pyright: reportMissingImports=false
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
import xxhash
import yaml

from pivot import exceptions, import_artifact
from pivot.remote import storage as remote_mod
from pivot.storage import track

if TYPE_CHECKING:
    import pathlib

    from pytest_mock import MockerFixture
    from types_aiobotocore_s3 import S3Client


def _lock_bytes(*, outs: list[dict[str, object]]) -> bytes:
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


def _make_pvt_data(
    *,
    path: str = "output.csv",
    hash: str = "aaa111",
    size: int = 5,
    repo: str = "https://github.com/org/repo",
    rev: str = "main",
    rev_lock: str = "oldsha",
    stage: str = "train",
    source_path: str = "data/output.csv",
    remote: str = "s3://bucket/prefix",
) -> track.PvtData:
    source = track.ImportSource(
        repo=repo,
        rev=rev,
        rev_lock=rev_lock,
        stage=stage,
        path=source_path,
        remote=remote,
    )
    return track.PvtData(path=path, hash=hash, size=size, source=source)


# ── check_for_update ──────────────────────────────────────────


def test_check_for_update_detects_new_commit(mocker: MockerFixture) -> None:
    resolve_ref = mocker.patch(
        "pivot.import_artifact.github.resolve_ref",
        autospec=True,
    )
    resolve_ref.return_value = "newsha999"

    pvt_data = _make_pvt_data(rev_lock="oldsha111")

    result = asyncio.run(import_artifact.check_for_update(pvt_data))

    assert result["available"] is True, "Should detect update when SHAs differ"
    assert result["current_rev"] == "oldsha111"
    assert result["latest_rev"] == "newsha999"


def test_check_for_update_no_change(mocker: MockerFixture) -> None:
    resolve_ref = mocker.patch(
        "pivot.import_artifact.github.resolve_ref",
        autospec=True,
    )
    resolve_ref.return_value = "samesha"

    pvt_data = _make_pvt_data(rev_lock="samesha")

    result = asyncio.run(import_artifact.check_for_update(pvt_data))

    assert result["available"] is False, "Should not detect update when SHAs match"
    assert result["current_rev"] == "samesha"
    assert result["latest_rev"] == "samesha"


# ── update_import ─────────────────────────────────────────────


def _setup_resolve_mocks(
    mocker: MockerFixture,
    *,
    remote_url: str,
    lock_outs: list[dict[str, object]],
    rev_lock: str = "newsha",
) -> None:
    """Set up standard mocks for resolve_remote_path used by update_import."""
    lock_bytes = _lock_bytes(outs=lock_outs)

    mocker.patch(
        "pivot.import_artifact.github.list_directory",
        autospec=True,
        return_value=["train.lock"],
    )
    mocker.patch(
        "pivot.import_artifact.github.resolve_ref",
        autospec=True,
        return_value=rev_lock,
    )

    async def _read_file(
        _owner: str, _repo: str, path: str, _ref: str, _token: str | None = None, **_kw: object
    ) -> bytes:
        if path == ".pivot/config.yaml":
            return _config_bytes(remote_url)
        if path == ".pivot/stages/train.lock":
            return lock_bytes
        raise AssertionError(f"Unexpected path: {path}")

    mocker.patch(
        "pivot.import_artifact.github.read_file",
        autospec=True,
        side_effect=_read_file,
    )


async def test_update_import_changed_output(
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    moto_s3_bucket: str,
    aioboto3_s3_client: S3Client,
) -> None:
    remote_url = f"s3://{moto_s3_bucket}/test-prefix/"
    old_content = b"old"
    new_content = b"new data here"
    old_hash = xxhash.xxh64(old_content).hexdigest()
    new_hash = xxhash.xxh64(new_content).hexdigest()

    # Write existing pvt + data file
    data_path = tmp_path / "output.csv"
    data_path.write_bytes(old_content)
    pvt_path = track.get_pvt_path(data_path)
    pvt_data = _make_pvt_data(
        path="output.csv",
        hash=old_hash,
        size=len(old_content),
        rev_lock="oldsha",
        source_path="data/output.csv",
        remote=remote_url,
    )
    track.write_pvt_file(pvt_path, pvt_data)

    # Remote now has new hash
    _setup_resolve_mocks(
        mocker,
        remote_url=remote_url,
        lock_outs=[{"path": "data/output.csv", "hash": new_hash}],
        rev_lock="newsha",
    )

    key = remote_mod._hash_to_key("test-prefix/", new_hash)
    await aioboto3_s3_client.put_object(Bucket=moto_s3_bucket, Key=key, Body=new_content)

    mocker.patch(
        "pivot.import_artifact.remote_storage.S3Remote",
        autospec=True,
        return_value=remote_mod.S3Remote(remote_url),
    )

    result = await import_artifact.update_import(pvt_path)

    assert result["downloaded"] is True, "Should mark as downloaded when hash changed"
    assert result["old_rev"] == "oldsha"
    assert result["new_rev"] == "newsha"
    assert data_path.read_bytes() == new_content

    # Verify pvt file updated
    updated_pvt = track.read_pvt_file(pvt_path)
    assert updated_pvt is not None
    assert updated_pvt["hash"] == new_hash
    assert updated_pvt["source"]["rev_lock"] == "newsha"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert updated_pvt["size"] == len(new_content), "Size should come from downloaded file"


def test_update_import_same_hash_no_redownload(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    remote_url = "s3://bucket/prefix"
    content = b"unchanged"
    file_hash = xxhash.xxh64(content).hexdigest()

    data_path = tmp_path / "output.csv"
    data_path.write_bytes(content)
    pvt_path = track.get_pvt_path(data_path)
    pvt_data = _make_pvt_data(
        path="output.csv",
        hash=file_hash,
        size=len(content),
        rev_lock="oldsha",
        source_path="data/output.csv",
        remote=remote_url,
    )
    track.write_pvt_file(pvt_path, pvt_data)

    # Remote has same hash but new rev_lock
    _setup_resolve_mocks(
        mocker,
        remote_url=remote_url,
        lock_outs=[{"path": "data/output.csv", "hash": file_hash}],
        rev_lock="newsha",
    )

    s3_mock = mocker.patch(
        "pivot.import_artifact.remote_storage.S3Remote",
        autospec=True,
        side_effect=AssertionError("S3Remote should not be instantiated when hash unchanged"),
    )

    result = asyncio.run(import_artifact.update_import(pvt_path))

    assert result["downloaded"] is False, "Should not re-download when hash unchanged"
    assert result["old_rev"] == "oldsha"
    assert result["new_rev"] == "newsha"
    s3_mock.assert_not_called()

    # Verify pvt file still updated with new rev_lock
    updated_pvt = track.read_pvt_file(pvt_path)
    assert updated_pvt is not None
    assert updated_pvt["source"]["rev_lock"] == "newsha"  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_update_import_with_rev_override(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    remote_url = "s3://bucket/prefix"
    content = b"data"
    file_hash = xxhash.xxh64(content).hexdigest()

    data_path = tmp_path / "output.csv"
    data_path.write_bytes(content)
    pvt_path = track.get_pvt_path(data_path)
    pvt_data = _make_pvt_data(
        path="output.csv",
        hash=file_hash,
        size=len(content),
        rev="main",
        rev_lock="oldsha",
        source_path="data/output.csv",
        remote=remote_url,
    )
    track.write_pvt_file(pvt_path, pvt_data)

    _setup_resolve_mocks(
        mocker,
        remote_url=remote_url,
        lock_outs=[{"path": "data/output.csv", "hash": file_hash}],
        rev_lock="v2sha",
    )

    mocker.patch(
        "pivot.import_artifact.remote_storage.S3Remote",
        autospec=True,
        side_effect=AssertionError("S3Remote should not be instantiated"),
    )

    result = asyncio.run(import_artifact.update_import(pvt_path, new_rev="v2"))

    assert result["new_rev"] == "v2sha"

    # Verify the rev field was updated to the override
    updated_pvt = track.read_pvt_file(pvt_path)
    assert updated_pvt is not None
    assert updated_pvt["source"]["rev"] == "v2", "Should update rev to override value"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert updated_pvt["source"]["rev_lock"] == "v2sha"  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_update_import_path_removed(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    remote_url = "s3://bucket/prefix"

    data_path = tmp_path / "output.csv"
    data_path.write_bytes(b"data")
    pvt_path = track.get_pvt_path(data_path)
    pvt_data = _make_pvt_data(
        path="output.csv",
        hash="aaa111",
        size=4,
        source_path="data/output.csv",
        remote=remote_url,
    )
    track.write_pvt_file(pvt_path, pvt_data)

    # Remote no longer has this path — lock file has different output
    _setup_resolve_mocks(
        mocker,
        remote_url=remote_url,
        lock_outs=[{"path": "data/other.csv", "hash": "bbb222"}],
        rev_lock="newsha",
    )

    with pytest.raises(exceptions.PivotError, match="not found in remote outputs"):
        asyncio.run(import_artifact.update_import(pvt_path))

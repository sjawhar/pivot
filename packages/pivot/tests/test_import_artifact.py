from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, cast

import pytest

from pivot import exceptions, import_artifact
from pivot.storage import track

if TYPE_CHECKING:
    from pivot.types import OutEntry


def test_maybe_token_prefers_explicit_token() -> None:
    assert import_artifact._maybe_token("https://github.com/acme/repo", "secret") == "secret"


def test_entry_display_and_size_helpers() -> None:
    file_entry = cast(
        "OutEntry", cast("object", {"path": "data/model.pkl", "hash": "abc", "size": 7})
    )
    dir_entry = cast(
        "OutEntry",
        cast(
            "object",
            {
                "path": "data/dir",
                "hash": "def",
                "manifest": [{"relpath": "a.txt", "hash": "h", "size": 3, "isexec": False}],
            },
        ),
    )
    unknown_entry = cast("OutEntry", cast("object", {"hash": "zzz", "key": "fallback"}))

    assert import_artifact._entry_display(file_entry) == "data/model.pkl"
    assert import_artifact._entry_size(file_entry) == 7
    assert import_artifact._entry_size(dir_entry) == 3
    assert import_artifact._entry_display(unknown_entry) == "fallback"


def test_iter_manifest_paths_joins_base_path() -> None:
    entry = cast(
        "OutEntry",
        cast(
            "object",
            {
                "path": "data/dir/",
                "hash": "abc",
                "manifest": [
                    {"relpath": "nested/file.txt", "hash": "x", "size": 1, "isexec": False}
                ],
            },
        ),
    )

    paths = import_artifact._iter_manifest_paths(entry)

    assert paths[0][0] == "data/dir/nested/file.txt"
    assert paths[0][1]["hash"] == "x"


@pytest.mark.asyncio
async def test_read_remote_config_uses_single_remote_as_default(mocker) -> None:
    mocker.patch.object(
        import_artifact.git_archive,
        "read_file_from_remote_repo",
        autospec=True,
        return_value=b"remotes:\n  origin: s3://bucket/cache\n",
    )

    result = await import_artifact.read_remote_config(
        "https://example.com/repo.git",
        "main",
        None,
    )

    assert result == "s3://bucket/cache"


@pytest.mark.asyncio
async def test_read_remote_config_errors_on_multiple_remotes_without_default(mocker) -> None:
    mocker.patch.object(
        import_artifact.git_archive,
        "read_file_from_remote_repo",
        autospec=True,
        return_value=b"remotes:\n  a: s3://bucket/a\n  b: s3://bucket/b\n",
    )

    with pytest.raises(exceptions.RemoteError, match="multiple remotes"):
        await import_artifact.read_remote_config("https://example.com/repo.git", "main", None)


@pytest.mark.asyncio
async def test_list_remote_lock_files_filters_non_lock_entries(mocker) -> None:
    mocker.patch.object(
        import_artifact.git_archive,
        "list_directory_from_remote_repo",
        autospec=True,
        return_value=["train.lock", "README.md", "eval.lock"],
    )

    stage_names = await import_artifact.list_remote_lock_files(
        "https://example.com/repo.git",
        "main",
        None,
    )

    assert stage_names == ["train", "eval"]


@pytest.mark.asyncio
async def test_list_remote_lock_files_raises_when_directory_missing(mocker) -> None:
    mocker.patch.object(
        import_artifact.git_archive,
        "list_directory_from_remote_repo",
        autospec=True,
        return_value=None,
    )

    with pytest.raises(exceptions.RemoteError, match="no .pivot/stages"):
        await import_artifact.list_remote_lock_files("https://example.com/repo.git", "main", None)


@pytest.mark.asyncio
async def test_read_remote_lock_file_returns_none_for_invalid_yaml(mocker) -> None:
    mocker.patch.object(
        import_artifact.git_archive,
        "read_file_from_remote_repo",
        autospec=True,
        return_value=b": bad-yaml",
    )

    data = await import_artifact.read_remote_lock_file(
        "https://example.com/repo.git",
        "train",
        "main",
        None,
    )

    assert data is None


@pytest.mark.asyncio
async def test_read_remote_lock_file_returns_parsed_lock_data(mocker) -> None:
    lock_yaml = (
        b"code_manifest: {}\n"
        b"params: {}\n"
        b"deps: []\n"
        b"outs:\n"
        b"  - key: model\n"
        b"    hash: abc123\n"
        b"    tag: data\n"
    )
    mocker.patch.object(
        import_artifact.git_archive,
        "read_file_from_remote_repo",
        autospec=True,
        return_value=lock_yaml,
    )

    data = await import_artifact.read_remote_lock_file(
        "https://example.com/repo.git",
        "train",
        "main",
        None,
    )

    assert data is not None
    assert data["outs"][0]["key"] == "model"


@pytest.mark.asyncio
async def test_check_for_update_detects_new_rev(mocker) -> None:
    pvt_data = track.PvtData(
        path="model.pkl",
        hash="abc",
        size=10,
        source=track.ImportSource(
            repo="https://example.com/repo.git",
            rev="main",
            rev_lock="oldrev",
            stage="train",
            path="models/model.pkl",
            remote="s3://bucket/cache",
        ),
    )
    mocker.patch.object(
        import_artifact.git_archive,
        "resolve_ref_from_remote_repo",
        autospec=True,
        return_value="newrev",
    )

    check = await import_artifact.check_for_update(pvt_data)

    assert check["available"] is True
    assert check["current_rev"] == "oldrev"
    assert check["latest_rev"] == "newrev"


@pytest.mark.asyncio
async def test_check_for_update_rejects_non_import_data() -> None:
    with pytest.raises(exceptions.PivotError, match="Not an import"):
        await import_artifact.check_for_update(track.PvtData(path="x", hash="h", size=1))


@pytest.mark.asyncio
async def test_update_import_rejects_invalid_pvt_file(tmp_path: pathlib.Path) -> None:
    bad_pvt = tmp_path / "bad.pvt"
    bad_pvt.write_text("not: valid")

    with pytest.raises(exceptions.PivotError, match="Invalid .pvt file"):
        await import_artifact.update_import(bad_pvt)


def _helper_git_read_side_effect(path: str) -> bytes:
    if path == ".pivot/config.yaml":
        return b"remotes:\n  origin: s3://bucket/cache\ndefault_remote: origin\n"
    if path == ".pivot/stages/train.lock":
        return (
            b"code_manifest: {}\n"
            b"params: {}\n"
            b"deps: []\n"
            b"outs:\n"
            b"  - key: model\n"
            b"    path: models/model.pkl\n"
            b"    hash: abc123def4567890\n"
            b"    size: 42\n"
            b"    tag: data\n"
        )
    raise AssertionError(f"unexpected path {path}")


@pytest.mark.asyncio
async def test_resolve_remote_path_happy_path(mocker) -> None:
    mocker.patch.object(
        import_artifact.git_archive,
        "resolve_ref_from_remote_repo",
        autospec=True,
        return_value="sha123",
    )
    mocker.patch.object(
        import_artifact.git_archive,
        "list_directory_from_remote_repo",
        autospec=True,
        return_value=["train.lock"],
    )

    def _helper_read(repo_url: str, path: str, rev: str) -> bytes:
        assert repo_url == "https://example.com/repo.git"
        assert rev == "sha123"
        return _helper_git_read_side_effect(path)

    mocker.patch.object(
        import_artifact.git_archive,
        "read_file_from_remote_repo",
        autospec=True,
        side_effect=_helper_read,
    )

    resolved = await import_artifact.resolve_remote_path(
        "https://example.com/repo.git",
        "models/model.pkl",
        "main",
        None,
    )

    assert resolved["stage"] == "train"
    assert resolved["path"] == "models/model.pkl"
    assert resolved["hash"] == "abc123def4567890"
    assert resolved["size"] == 42
    assert resolved["rev_lock"] == "sha123"


@pytest.mark.asyncio
async def test_resolve_remote_path_errors_for_missing_path(mocker) -> None:
    mocker.patch.object(
        import_artifact.git_archive,
        "resolve_ref_from_remote_repo",
        autospec=True,
        return_value="sha123",
    )
    mocker.patch.object(
        import_artifact.git_archive,
        "list_directory_from_remote_repo",
        autospec=True,
        return_value=["train.lock"],
    )
    mocker.patch.object(
        import_artifact.git_archive,
        "read_file_from_remote_repo",
        autospec=True,
        side_effect=lambda _repo, path, _rev: _helper_git_read_side_effect(path),
    )

    with pytest.raises(exceptions.PivotError, match="not found in remote outputs"):
        await import_artifact.resolve_remote_path(
            "https://example.com/repo.git",
            "missing/file.txt",
            "main",
            None,
        )


@pytest.mark.asyncio
async def test_import_artifact_no_download_writes_metadata_only(
    tmp_path: pathlib.Path,
    mocker,
) -> None:
    mocker.patch.object(
        import_artifact.git_archive,
        "resolve_ref_from_remote_repo",
        autospec=True,
        return_value="sha123",
    )
    mocker.patch.object(
        import_artifact.git_archive,
        "list_directory_from_remote_repo",
        autospec=True,
        return_value=["train.lock"],
    )
    mocker.patch.object(
        import_artifact.git_archive,
        "read_file_from_remote_repo",
        autospec=True,
        side_effect=lambda _repo, path, _rev: _helper_git_read_side_effect(path),
    )

    result = await import_artifact.import_artifact(
        repo_url="https://example.com/repo.git",
        path="models/model.pkl",
        rev="main",
        no_download=True,
        project_root=tmp_path,
    )

    pvt_path = pathlib.Path(result["pvt_path"])
    data_path = pathlib.Path(result["data_path"])
    assert result["downloaded"] is False
    assert pvt_path.exists()
    assert not data_path.exists()

"""`pivot import` against source repos written with the compositional API.

Stage names are pipeline-prefixed (``smoke/total``), so lock files live in nested
directories (``.pivot/stages/smoke/total.lock``), and lock outputs are identified by
``(stage, key)`` rather than by a workspace path. Imports therefore address an
artifact by its identity key (``smoke/total``, ``smoke/train:model``).
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import TYPE_CHECKING

import pytest
import yaml

import conftest
from pivot import exceptions, import_artifact, types
from pivot.remote import github
from pivot.storage import lock, track

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

_REMOTE_URL = "s3://pivot-source-bucket/cache"
_SCALE_HASH = "aaaaaaaaaaaaaaaa"
_TOTAL_HASH = "bbbbbbbbbbbbbbbb"
_MODEL_HASH = "cccccccccccccccc"
_METRICS_HASH = "dddddddddddddddd"


def _helper_lock_data(stage_name: str, outs: dict[str | None, str]) -> types.LockData:
    return types.LockData(
        code_manifest={"self:stage": "0123456789abcdef"},
        params={},
        dep_hashes={},
        output_hashes={
            types.ArtifactIdentity(stage_name, key): types.FileHash(hash=hash_)
            for key, hash_ in outs.items()
        },
    )


_SOURCE_LOCKS = {
    "smoke/scale": _helper_lock_data("smoke/scale", {None: _SCALE_HASH}),
    "smoke/total": _helper_lock_data("smoke/total", {None: _TOTAL_HASH}),
    "smoke/train": _helper_lock_data(
        "smoke/train", {"model": _MODEL_HASH, "metrics": _METRICS_HASH}
    ),
}


def _helper_git(repo: pathlib.Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _helper_write_pivot_state(root: pathlib.Path) -> None:
    """Write the committed .pivot/ state of a source project: config and lock files."""
    pivot_dir = root / ".pivot"
    pivot_dir.mkdir(parents=True)
    (pivot_dir / "config.yaml").write_text(yaml.safe_dump({"remotes": {"origin": _REMOTE_URL}}))
    stages_dir = lock.get_stages_dir(pivot_dir)
    for stage_name, lock_data in _SOURCE_LOCKS.items():
        lock.StageLock(stage_name, stages_dir).write(lock_data)


def _helper_source_repo(source_dir: pathlib.Path) -> str:
    """Create a git repo holding a new-API project's committed .pivot/ state. Returns HEAD."""
    source_dir.mkdir()
    conftest.init_git_repo(source_dir)
    # git archive --remote by commit SHA requires this for local repos
    _helper_git(source_dir, "config", "uploadArchive.allowUnreachable", "true")
    _helper_git(source_dir, "checkout", "-b", "main")
    _helper_write_pivot_state(source_dir)
    _helper_git(source_dir, "add", ".")
    _helper_git(source_dir, "commit", "-m", "source pipeline")
    return _helper_git(source_dir, "rev-parse", "HEAD")


async def test_resolve_remote_path_finds_pipeline_stage_output_by_identity(
    tmp_path: pathlib.Path,
) -> None:
    source_dir = tmp_path / "source"
    head = _helper_source_repo(source_dir)

    resolved = await import_artifact.resolve_remote_path(
        str(source_dir), "smoke/total", "main", None
    )

    assert resolved["stage"] == "smoke/total"
    assert resolved["path"] == "smoke/total"
    assert resolved["hash"] == _TOTAL_HASH
    assert resolved["rev_lock"] == head
    assert resolved["remote_url"] == _REMOTE_URL


async def test_resolve_remote_path_selects_keyed_output(tmp_path: pathlib.Path) -> None:
    source_dir = tmp_path / "source"
    _helper_source_repo(source_dir)

    resolved = await import_artifact.resolve_remote_path(
        str(source_dir), "smoke/train:model", "main", None
    )

    assert resolved["stage"] == "smoke/train"
    assert resolved["hash"] == _MODEL_HASH, "Should select the 'model' output, not 'metrics'"


async def test_resolve_remote_path_unknown_identity_lists_available(
    tmp_path: pathlib.Path,
) -> None:
    source_dir = tmp_path / "source"
    _helper_source_repo(source_dir)

    with pytest.raises(exceptions.PivotError) as exc_info:
        await import_artifact.resolve_remote_path(str(source_dir), "smoke/nope", "main", None)

    message = str(exc_info.value)
    assert "smoke/total" in message, "Error should list the source repo's artifact identities"
    assert "smoke/train:model" in message


async def test_import_artifact_records_identity_source(tmp_path: pathlib.Path) -> None:
    source_dir = tmp_path / "source"
    head = _helper_source_repo(source_dir)
    consumer_dir = tmp_path / "consumer"
    consumer_dir.mkdir()

    result = await import_artifact.import_artifact(
        str(source_dir),
        "smoke/total",
        rev="main",
        out="data/total.yaml",
        no_download=True,
        project_root=consumer_dir,
    )

    pvt_data = track.read_pvt_file(pathlib.Path(result["pvt_path"]))
    assert pvt_data is not None
    assert pvt_data["hash"] == _TOTAL_HASH
    assert "source" in pvt_data
    assert pvt_data["source"]["stage"] == "smoke/total"
    assert pvt_data["source"]["path"] == "smoke/total"
    assert pvt_data["source"]["rev_lock"] == head


async def test_resolve_remote_path_walks_nested_github_lock_dirs(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """GitHub's contents API lists one directory level; pipeline lock dirs must be walked."""
    _helper_write_pivot_state(tmp_path)
    repo_files = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in (tmp_path / ".pivot").rglob("*")
        if path.is_file()
    }
    listings = {
        ".pivot/stages": [github.DirectoryEntry(name="smoke", type="dir")],
        ".pivot/stages/smoke": [
            github.DirectoryEntry(name="scale.lock", type="file"),
            github.DirectoryEntry(name="total.lock", type="file"),
            github.DirectoryEntry(name="train.lock", type="file"),
        ],
    }

    async def _list_directory(
        owner: str,
        repo: str,
        path: str,
        ref: str,
        token: str | None = None,
        *,
        session: object = None,
    ) -> list[github.DirectoryEntry] | None:
        return listings.get(path)

    async def _read_file(
        owner: str,
        repo: str,
        path: str,
        ref: str,
        token: str | None = None,
        *,
        session: object = None,
    ) -> bytes | None:
        return repo_files.get(path)

    mocker.patch.object(github, "resolve_ref", autospec=True, return_value="f" * 40)
    mocker.patch.object(github, "list_directory", autospec=True, side_effect=_list_directory)
    mocker.patch.object(github, "read_file", autospec=True, side_effect=_read_file)

    resolved = await import_artifact.resolve_remote_path(
        "https://github.com/org/source", "smoke/total", "main", None
    )

    assert resolved["stage"] == "smoke/total"
    assert resolved["hash"] == _TOTAL_HASH


async def test_resolve_remote_path_walks_github_directory_named_lock(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """A GitHub Contents directory named `*.lock` is descended into, not read as a lock file."""
    pivot_dir = tmp_path / ".pivot"
    pivot_dir.mkdir()
    (pivot_dir / "config.yaml").write_text(
        yaml.safe_dump({"remotes": {"origin": _REMOTE_URL}, "default_remote": "origin"})
    )
    stages_dir = lock.get_stages_dir(pivot_dir)
    lock.StageLock("foo.lock/render", stages_dir).write(
        _helper_lock_data("foo.lock/render", {None: _TOTAL_HASH})
    )
    repo_files = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in pivot_dir.rglob("*")
        if path.is_file()
    }
    listings = {
        ".pivot/stages": [github.DirectoryEntry(name="foo.lock", type="dir")],
        ".pivot/stages/foo.lock": [github.DirectoryEntry(name="render.lock", type="file")],
    }

    async def _list_directory(
        owner: str,
        repo: str,
        path: str,
        ref: str,
        token: str | None = None,
        *,
        session: object = None,
    ) -> list[github.DirectoryEntry] | None:
        return listings.get(path)

    async def _read_file(
        owner: str,
        repo: str,
        path: str,
        ref: str,
        token: str | None = None,
        *,
        session: object = None,
    ) -> bytes | None:
        return repo_files.get(path)

    mocker.patch.object(github, "resolve_ref", autospec=True, return_value="f" * 40)
    mocker.patch.object(github, "list_directory", autospec=True, side_effect=_list_directory)
    mocker.patch.object(github, "read_file", autospec=True, side_effect=_read_file)

    resolved = await import_artifact.resolve_remote_path(
        "https://github.com/org/source", "foo.lock/render", "main", None
    )

    assert resolved["stage"] == "foo.lock/render"
    assert resolved["hash"] == _TOTAL_HASH

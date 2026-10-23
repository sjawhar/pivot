"""Imports retain Pivot 0.5.2 source layouts with a configured state directory."""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

import yaml

import conftest
from pivot import import_artifact
from pivot.remote import github

if TYPE_CHECKING:
    import pathlib

    import pytest
    from pytest_mock import MockerFixture


_HASH = "0123456789abcdef"
_LOCK = {
    "code_manifest": {},
    "params": {},
    "deps": [],
    "outs": [{"key": None, "tag": "data", "hash": _HASH, "size": 5}],
}


def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _source(tmp_path: pathlib.Path, state_dir: str) -> pathlib.Path:
    source = tmp_path / "source"
    source.mkdir()
    conftest.init_git_repo(source)
    _git(source, "config", "uploadArchive.allowUnreachable", "true")
    _git(source, "checkout", "-b", "main")
    (source / ".pivot" / "stages" / "smoke").mkdir(parents=True)
    (source / ".pivot" / "stages" / "smoke" / "total.lock").write_text(yaml.safe_dump(_LOCK))
    (source / ".pivot" / "config.yaml").write_text(
        "remotes:\n  origin: s3://source-cache\ncore:\n  state_dir: " + state_dir + "\n"
    )
    _git(source, "add", ".")
    _git(source, "commit", "-m", "source")
    return source


async def test_resolve_remote_path_uses_legacy_locks_when_configured_dir_is_absent(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = _source(tmp_path, "altstate")

    with caplog.at_level(logging.WARNING):
        resolved = await import_artifact.resolve_remote_path(
            str(source), "smoke/total", "main", None
        )

    assert resolved["hash"] == _HASH
    assert "Ignoring core.state_dir" in caplog.text
    assert "altstate/stages" in caplog.text


async def test_resolve_remote_path_uses_legacy_locks_when_state_dir_parent_is_a_symlink(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = _source(tmp_path, "lnk/sub")
    (source / "real").mkdir()
    (source / "real" / ".keep").write_text("")
    (source / "lnk").symlink_to("real")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "symlinked state directory")

    with caplog.at_level(logging.WARNING):
        resolved = await import_artifact.resolve_remote_path(
            str(source), "smoke/total", "main", None
        )

    assert resolved["hash"] == _HASH
    assert "Ignoring core.state_dir" in caplog.text
    assert "lnk/sub/stages" in caplog.text


async def test_resolve_github_source_uses_legacy_locks_when_state_dir_is_unlistable(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    config = b"remotes:\n  origin: s3://source-cache\ncore:\n  state_dir: link/stages\n"
    listings = {
        ".pivot/stages": [github.DirectoryEntry(name="smoke", type="dir")],
        ".pivot/stages/smoke": [github.DirectoryEntry(name="total.lock", type="file")],
    }

    async def list_directory(
        owner: str,
        repo: str,
        path: str,
        ref: str,
        token: str | None = None,
        *,
        session: object = None,
    ) -> list[github.DirectoryEntry] | None:
        return listings.get(path)

    async def read_file(
        owner: str,
        repo: str,
        path: str,
        ref: str,
        token: str | None = None,
        *,
        session: object = None,
    ) -> bytes | None:
        if path == ".pivot/config.yaml":
            return config
        if path == ".pivot/stages/smoke/total.lock":
            return yaml.safe_dump(_LOCK).encode()
        return None

    mocker.patch.object(github, "resolve_ref", autospec=True, return_value="f" * 40)
    mocker.patch.object(github, "list_directory", autospec=True, side_effect=list_directory)
    mocker.patch.object(github, "read_file", autospec=True, side_effect=read_file)

    with caplog.at_level(logging.WARNING):
        resolved = await import_artifact.resolve_remote_path(
            "https://github.com/org/source", "smoke/total", "main", None
        )

    assert resolved["hash"] == _HASH
    assert "Ignoring core.state_dir" in caplog.text
    assert "link/stages/stages" in caplog.text

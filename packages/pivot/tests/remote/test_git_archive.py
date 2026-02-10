from __future__ import annotations

import io
import subprocess
import tarfile
from typing import TYPE_CHECKING

from pivot.remote import git_archive

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _make_tar_bytes(files: dict[str, bytes]) -> bytes:
    """Create an in-memory tar archive with the given files."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


# resolve_ref_from_remote_repo


def test_resolve_ref_parses_ls_remote(mocker: MockerFixture) -> None:
    """git ls-remote output is parsed to extract the commit SHA."""
    mock_run = mocker.patch(
        "pivot.remote.git_archive.subprocess.run",
        autospec=True,
    )
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="abc123def456789012345678901234567890abcd\trefs/heads/main\n",
        stderr="",
    )

    result = git_archive.resolve_ref_from_remote_repo("git@example.com:repo.git", "main")

    assert result == "abc123def456789012345678901234567890abcd"
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert "ls-remote" in call_args[0][0]


def test_resolve_ref_prefers_deref_sha_for_annotated_tags(mocker: MockerFixture) -> None:
    """Annotated tags have both tag-object and deref ^{} lines; prefer the commit SHA."""
    mock_run = mocker.patch(
        "pivot.remote.git_archive.subprocess.run",
        autospec=True,
    )
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            "aaaa0000000000000000000000000000aaaaaaaa\trefs/tags/v1.0\n"
            "bbbb0000000000000000000000000000bbbbbbbb\trefs/tags/v1.0^{}\n"
        ),
        stderr="",
    )

    result = git_archive.resolve_ref_from_remote_repo("git@example.com:repo.git", "v1.0")

    assert result == "bbbb0000000000000000000000000000bbbbbbbb", (
        "Should return dereferenced commit SHA"
    )


def test_resolve_ref_lightweight_tag_no_deref(mocker: MockerFixture) -> None:
    """Lightweight tags have no ^{} line; the single SHA is returned."""
    mock_run = mocker.patch(
        "pivot.remote.git_archive.subprocess.run",
        autospec=True,
    )
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="cccc0000000000000000000000000000cccccccc\trefs/tags/v2.0\n",
        stderr="",
    )

    result = git_archive.resolve_ref_from_remote_repo("git@example.com:repo.git", "v2.0")

    assert result == "cccc0000000000000000000000000000cccccccc"


def test_resolve_ref_branch_not_found(mocker: MockerFixture) -> None:
    """Empty ls-remote output returns None when branch doesn't exist."""
    mock_run = mocker.patch(
        "pivot.remote.git_archive.subprocess.run",
        autospec=True,
    )
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    result = git_archive.resolve_ref_from_remote_repo("git@example.com:repo.git", "nonexistent")

    assert result is None


def test_resolve_ref_repo_not_found(mocker: MockerFixture) -> None:
    """CalledProcessError from ls-remote (e.g. repo not found) returns None."""
    mock_run = mocker.patch(
        "pivot.remote.git_archive.subprocess.run",
        autospec=True,
    )
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=128,
        stdout="",
        stderr="fatal: repository 'https://example.com/no-repo.git' not found",
    )

    result = git_archive.resolve_ref_from_remote_repo("https://example.com/no-repo.git", "main")

    assert result is None


# read_file_from_remote_repo


def test_read_file_extracts_tar(mocker: MockerFixture) -> None:
    """File content is correctly extracted from git archive tar stream."""
    tar_bytes = _make_tar_bytes({"pivot.yaml": b"stages:\n  train:\n"})
    mock_run = mocker.patch(
        "pivot.remote.git_archive.subprocess.run",
        autospec=True,
    )
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=tar_bytes, stderr=b""
    )

    result = git_archive.read_file_from_remote_repo(
        "git@example.com:repo.git", "pivot.yaml", "main"
    )

    assert result == b"stages:\n  train:\n"


def test_read_file_path_not_found(mocker: MockerFixture) -> None:
    """CalledProcessError from git archive (path not found) returns None."""
    mock_run = mocker.patch(
        "pivot.remote.git_archive.subprocess.run",
        autospec=True,
    )
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=128,
        cmd=["git", "archive"],
        stderr=b"fatal: path not found",
    )

    result = git_archive.read_file_from_remote_repo(
        "git@example.com:repo.git", "nonexistent.yaml", "main"
    )

    assert result is None


# list_directory_from_remote_repo


def test_list_directory_returns_file_names(mocker: MockerFixture) -> None:
    """Directory listing extracts file names from tar archive."""
    tar_bytes = _make_tar_bytes(
        {
            "data/train.csv": b"a,b\n1,2",
            "data/test.csv": b"a,b\n3,4",
        }
    )
    mock_run = mocker.patch(
        "pivot.remote.git_archive.subprocess.run",
        autospec=True,
    )
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=tar_bytes, stderr=b""
    )

    result = git_archive.list_directory_from_remote_repo(
        "git@example.com:repo.git", "data/", "main"
    )

    assert result is not None
    assert sorted(result) == ["test.csv", "train.csv"]


# timeout handling


def test_git_archive_timeout_on_read(mocker: MockerFixture) -> None:
    """TimeoutExpired during git archive returns None cleanly."""
    mock_run = mocker.patch(
        "pivot.remote.git_archive.subprocess.run",
        autospec=True,
    )
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["git", "archive"], timeout=30)

    result = git_archive.read_file_from_remote_repo(
        "git@example.com:repo.git", "pivot.yaml", "main", timeout=30
    )

    assert result is None


def test_git_archive_timeout_on_resolve(mocker: MockerFixture) -> None:
    """TimeoutExpired during git ls-remote returns None cleanly."""
    mock_run = mocker.patch(
        "pivot.remote.git_archive.subprocess.run",
        autospec=True,
    )
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["git", "ls-remote"], timeout=30)

    result = git_archive.resolve_ref_from_remote_repo(
        "git@example.com:repo.git", "main", timeout=30
    )

    assert result is None


def test_git_archive_timeout_on_list(mocker: MockerFixture) -> None:
    """TimeoutExpired during git archive list returns None cleanly."""
    mock_run = mocker.patch(
        "pivot.remote.git_archive.subprocess.run",
        autospec=True,
    )
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["git", "archive"], timeout=30)

    result = git_archive.list_directory_from_remote_repo(
        "git@example.com:repo.git", "data/", "main", timeout=30
    )

    assert result is None


# git not found


def test_resolve_ref_git_not_found(mocker: MockerFixture) -> None:
    """FileNotFoundError (git not installed) returns None."""
    mock_run = mocker.patch(
        "pivot.remote.git_archive.subprocess.run",
        autospec=True,
    )
    mock_run.side_effect = FileNotFoundError("git")

    result = git_archive.resolve_ref_from_remote_repo("git@example.com:repo.git", "main")

    assert result is None

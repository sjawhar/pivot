from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pivot import cli, project, types
from pivot.remote import sync as transfer
from pivot.storage import cache, track
from tests import helpers

if TYPE_CHECKING:
    import pathlib

    import click.testing
    from pytest import MonkeyPatch

    from conftest import GitRepo
    from pivot.remote.protocol import RemoteStorage


# =============================================================================
# CLI Help Tests
# =============================================================================


def test_get_help(runner: click.testing.CliRunner) -> None:
    """Shows help message."""
    result = runner.invoke(cli.cli, ["get", "--help"])

    assert result.exit_code == 0
    assert "Retrieve files or stage outputs" in result.output
    assert "--rev" in result.output


# =============================================================================
# CLI Argument Validation Tests
# =============================================================================


def test_get_requires_rev(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Requires --rev option."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("content")
    commit("initial")
    (repo_path / ".pivot").mkdir()

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(cli.cli, ["get", "file.txt"])

    assert result.exit_code != 0
    assert "Missing option" in result.output or "--rev" in result.output


def test_get_requires_targets(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Requires at least one target."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("content")
    commit("initial")
    (repo_path / ".pivot").mkdir()

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(cli.cli, ["get", "--rev", "HEAD"])

    assert result.exit_code != 0
    assert "Missing argument" in result.output or "TARGETS" in result.output


# =============================================================================
# CLI Basic Functionality Tests
# =============================================================================


def test_get_git_tracked_file(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Gets a git-tracked file from revision."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("original content")
    sha = commit("initial")

    # Modify file
    (repo_path / "file.txt").write_text("modified")

    # Create .pivot directory
    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "file.txt", "-o", str(repo_path / "restored.txt")],
    )

    assert result.exit_code == 0, result.output
    assert "Restored" in result.output
    assert (repo_path / "restored.txt").read_text() == "original content"


def test_get_git_tracked_file_with_force(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Overwrites existing file with --force."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("original")
    sha = commit("initial")

    # Create output file
    output_path = repo_path / "output.txt"
    output_path.write_text("existing")

    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "file.txt", "-o", str(output_path), "--force"],
    )

    assert result.exit_code == 0, result.output
    assert "Restored" in result.output
    assert output_path.read_text() == "original"


def test_get_restores_files_while_pipeline_is_broken(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """A pipeline.py that fails to import does not stop get restoring files, itself included."""
    repo_path, commit = git_repo
    good_pipeline = (
        "from pivot.compose import Pipeline\n\nwith Pipeline('p') as pipeline:\n    pass\n"
    )
    (repo_path / "pipeline.py").write_text(good_pipeline)
    (repo_path / "notes.txt").write_text("notes")
    sha = commit("initial")
    (repo_path / "pipeline.py").write_text(good_pipeline + "import nonexistent_module_xyz\n")
    (repo_path / "notes.txt").unlink()
    (repo_path / ".pivot" / "cache").mkdir(parents=True)
    monkeypatch.chdir(repo_path)
    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(
        cli.cli, ["get", "--rev", sha[:7], "--force", "pipeline.py", "notes.txt"]
    )

    assert result.exit_code == 0, result.output
    assert (repo_path / "pipeline.py").read_text() == good_pipeline
    assert (repo_path / "notes.txt").read_text() == "notes"


def test_get_restores_a_file_while_pipeline_root_is_refused(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """A root-refused pipeline stops stage lookup, not restoration of a tracked file."""
    repo_path, commit = git_repo
    (repo_path / "notes.txt").write_text("notes")
    sha = commit("initial")
    (repo_path / "pipeline.py").write_text(
        "from pathlib import Path\n\n"
        + "from pivot.compose import Pipeline\n\n"
        + 'with Pipeline("sub", root=Path(__file__).parent / "sub") as pipeline:\n'
        + "    pass\n"
    )
    (repo_path / "notes.txt").unlink()
    (repo_path / ".pivot" / "cache").mkdir(parents=True)
    monkeypatch.chdir(repo_path)
    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "--force", "notes.txt"])

    assert result.exit_code == 0, result.output
    assert (repo_path / "notes.txt").read_text() == "notes"


class _ObjectRemote:
    """A remote holding the cache objects ``objects`` maps by hash, which records each hash a
    download asks of it. It has only the url and download_batch a restore uses."""

    url: str = "s3://bucket/objects"

    objects: dict[str, bytes]
    asked: list[str]

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.asked = list[str]()

    async def download_batch(
        self, items: list[tuple[str, pathlib.Path]], *, readonly: bool = False
    ) -> list[types.TransferResult]:
        assert readonly, "get downloads cache objects read-only"
        results = list[types.TransferResult]()
        for file_hash, path in items:
            self.asked.append(file_hash)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.objects[file_hash])
            results.append(types.TransferResult(hash=file_hash, success=True))
        return results


def test_get_downloads_a_member_a_whole_tree_holds_where_links_are_refused(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """`pivot get` of tracked directory images, whose whole tree the cache holds, complete and
    marked, and whose member b.txt's object is gone, with SYMLINK, HARDLINK and COPY the
    checkout modes and the cache's register on a read-only mount
    (``helpers.register_on_a_read_only_mount``). The restore links nothing and copies each
    file, so it lacks b.txt's object: get downloads that object from the remote and restores
    images from it. pivot 0.10.4 read the tree as complete there, downloaded nothing and failed
    to restore from the cache."""
    repo_path, commit = git_repo
    (repo_path / ".pivot").mkdir()
    (repo_path / ".pivot" / "config.yaml").write_text(
        "cache:\n  checkout_mode: [symlink, hardlink, copy]\n"
    )
    monkeypatch.chdir(repo_path)
    monkeypatch.setattr(project, "_project_root_cache", repo_path)
    images = repo_path / "images"
    images.mkdir()
    (images / "a.txt").write_text("a")
    (images / "b.txt").write_text("b")
    helpers.track_then_remove(runner, repo_path, "images")
    sha = commit("track images")
    helpers.cache_whole_tree(repo_path, {"a.txt": "a", "b.txt": "b"})
    files = repo_path / ".pivot" / "cache" / "files"
    record = track.read_pvt_file(track.get_pvt_path(images))
    assert record is not None
    b_hash = next(
        entry["hash"]
        for entry in track.tracked_members(images, record, files)["files"]
        if entry["relpath"] == "b.txt"
    )
    cache.get_cache_path(files, b_hash).unlink()
    remote = _ObjectRemote({b_hash: b"b"})

    def open_remote(name: str | None) -> tuple[RemoteStorage, str]:
        assert name is None, "get names no remote"
        return cast("RemoteStorage", cast("object", remote)), "origin"

    monkeypatch.setattr(transfer, "create_remote_from_name", open_remote)
    helpers.register_on_a_read_only_mount(monkeypatch, repo_path / ".pivot" / "cache")

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "images"])

    assert result.exit_code == 0, result.output
    assert f"Restored: {images} (from remote)" in result.output.splitlines(), result.output
    assert remote.asked == [b_hash]
    assert not images.is_symlink()
    assert {path.name: path.read_text() for path in images.iterdir()} == {
        "a.txt": "a",
        "b.txt": "b",
    }


def test_get_skip_existing_without_force(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Skips existing files without --force."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("original")
    sha = commit("initial")

    # Create output file
    output_path = repo_path / "output.txt"
    output_path.write_text("existing")

    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "file.txt", "-o", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert "Skipped" in result.output
    assert output_path.read_text() == "existing"


# =============================================================================
# CLI Error Handling Tests
# =============================================================================


def test_get_invalid_revision(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Errors on invalid revision."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("content")
    commit("initial")
    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", "invalid-revision", "file.txt"],
    )

    assert result.exit_code != 0
    assert "RevisionNotFoundError" in result.output or "Cannot resolve" in result.output


def test_get_target_not_found(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Errors when target not found at revision."""
    repo_path, commit = git_repo
    (repo_path / "file.txt").write_text("content")
    sha = commit("initial")
    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "nonexistent.txt"],
    )

    assert result.exit_code != 0
    assert "TargetNotFoundError" in result.output or "not found" in result.output


# =============================================================================
# CLI Multiple Targets Tests
# =============================================================================


def test_get_multiple_git_tracked_files(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Gets multiple git-tracked files from revision."""
    repo_path, commit = git_repo
    (repo_path / "file1.txt").write_text("content1")
    (repo_path / "file2.txt").write_text("content2")
    (repo_path / "file3.txt").write_text("content3")
    sha = commit("initial")

    # Modify files
    (repo_path / "file1.txt").write_text("modified1")
    (repo_path / "file2.txt").write_text("modified2")
    (repo_path / "file3.txt").write_text("modified3")

    # Create .pivot directory
    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "--force", "file1.txt", "file2.txt", "file3.txt"],
    )

    assert result.exit_code == 0, result.output
    assert "Restored" in result.output
    # Verify all files restored to original content
    assert (repo_path / "file1.txt").read_text() == "content1"
    assert (repo_path / "file2.txt").read_text() == "content2"
    assert (repo_path / "file3.txt").read_text() == "content3"


def test_get_with_output_single_file_succeeds(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Get with -o and single file succeeds."""
    repo_path, commit = git_repo
    (repo_path / "original.txt").write_text("original content")
    sha = commit("initial")

    # Create .pivot directory
    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "original.txt", "-o", str(repo_path / "renamed.txt")],
    )

    assert result.exit_code == 0, result.output
    assert "Restored" in result.output
    assert (repo_path / "renamed.txt").exists()
    assert (repo_path / "renamed.txt").read_text() == "original content"


def test_get_with_output_multiple_files_fails(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Get with -o and multiple files fails with error."""
    repo_path, commit = git_repo
    (repo_path / "file1.txt").write_text("content1")
    (repo_path / "file2.txt").write_text("content2")
    sha = commit("initial")

    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "file1.txt", "file2.txt", "-o", "output.txt"],
    )

    # Should fail because -o is incompatible with multiple targets
    assert result.exit_code != 0
    # Error message should mention incompatibility
    assert "incompatible" in result.output.lower() or "single" in result.output.lower()


def test_get_partial_success_with_missing_file(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Get with mix of existing and missing files handles gracefully."""
    repo_path, commit = git_repo
    (repo_path / "exists.txt").write_text("exists")
    sha = commit("initial")

    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    # Try to get one existing and one non-existing file
    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "exists.txt", "missing.txt"],
    )

    # Should report error for missing file
    assert result.exit_code != 0
    assert "missing.txt" in result.output.lower() or "not found" in result.output.lower()


# =============================================================================
# CLI Edge Case Tests
# =============================================================================


def test_get_empty_file(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Get handles empty files correctly."""
    repo_path, commit = git_repo
    (repo_path / "empty.txt").write_text("")
    sha = commit("empty file")

    # Modify to non-empty
    (repo_path / "empty.txt").write_text("now has content")

    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "--force", "empty.txt"],
    )

    assert result.exit_code == 0, result.output
    assert "Restored" in result.output
    assert (repo_path / "empty.txt").read_text() == ""


def test_get_binary_file(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Get handles binary files correctly."""
    repo_path, commit = git_repo
    binary_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    (repo_path / "image.png").write_bytes(binary_content)
    sha = commit("binary file")

    # Modify binary
    (repo_path / "image.png").write_bytes(b"corrupted")

    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "--force", "image.png"],
    )

    assert result.exit_code == 0, result.output
    assert "Restored" in result.output
    assert (repo_path / "image.png").read_bytes() == binary_content


def test_get_file_in_subdirectory(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Get handles files in subdirectories."""
    repo_path, commit = git_repo
    (repo_path / "subdir").mkdir()
    (repo_path / "subdir" / "nested.txt").write_text("nested content")
    sha = commit("nested file")

    # Remove subdirectory
    (repo_path / "subdir" / "nested.txt").unlink()
    (repo_path / "subdir").rmdir()

    (repo_path / ".pivot" / "cache").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "subdir/nested.txt"],
    )

    assert result.exit_code == 0, result.output
    assert "Restored" in result.output
    assert (repo_path / "subdir" / "nested.txt").exists()
    assert (repo_path / "subdir" / "nested.txt").read_text() == "nested content"

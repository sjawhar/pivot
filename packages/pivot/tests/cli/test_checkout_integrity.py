from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli
from pivot.storage import cache, track

if TYPE_CHECKING:
    from collections.abc import Generator

    import click.testing


@pytest.mark.parametrize(
    "document",
    [
        pytest.param("path: payload\n", id="truncated-fields"),
        pytest.param("path: [unclosed\n", id="truncated-yaml"),
        pytest.param("path: payload\nhash: 123\nsize: 1\n", id="unusable-field"),
    ],
)
@pytest.mark.parametrize("command", [["checkout", "--all", "--force"], ["verify"]])
def test_unreadable_pvt_fails_naming_the_sidecar(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    document: str,
    command: list[str],
) -> None:
    """Discovery must not silently omit tracked data or certify it as verified."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(
            "from pivot import Pipeline, stage\n\n@stage\ndef alive() -> str:\n"
            + "    return 'alive'\n\nwith Pipeline('test') as pipeline:\n    alive()\n"
        )
        reproduced = runner.invoke(cli.cli, ["repro"])
        assert reproduced.exit_code == 0, reproduced.output
        sidecar = root / "payload.pvt"
        sidecar.write_text(document)

        result = runner.invoke(cli.cli, command)

        assert result.exit_code == 1, result.output
        assert "payload.pvt" in result.output
        assert "Restored" not in result.output
        assert "Verification passed" not in result.output
        assert not (root / "payload").exists()
        assert sidecar.read_text() == document


@pytest.mark.parametrize("mode", ["copy", "hardlink", "symlink"])
def test_checkout_refuses_directory_with_wrong_cached_content(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mode: str
) -> None:
    """A successful file operation does not prove the directory holds its recorded bytes."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        payload = root / "payload"
        payload.mkdir()
        (payload / "a.txt").write_text("original")
        (payload / "b.txt").write_text("second")
        tracked = runner.invoke(cli.cli, ["track", "payload"])
        assert tracked.exit_code == 0, tracked.output
        pvt = track.read_pvt_file(root / "payload.pvt")
        assert pvt is not None
        members = track.tracked_members(payload, pvt, root / ".pivot/cache/files")
        damaged = cache.get_cache_path(root / ".pivot/cache/files", members["files"][0]["hash"])
        damaged.chmod(0o644)
        damaged.write_text("tampered")  # Same size, different hash.
        shutil.rmtree(payload)

        result = runner.invoke(cli.cli, ["checkout", "--all", "--force", "--checkout-mode", mode])

        assert result.exit_code == 1, result.output
        assert "payload" in result.output
        assert "recorded listing" in result.output
        assert "Restored" not in result.output
        assert not payload.exists(), "A failed restore must not publish its staging directory"
        assert not list(root.glob(".pivot_restore_*"))


@pytest.mark.parametrize("same_content", [False, True], ids=["different-bytes", "same-bytes"])
def test_checkout_refuses_collapsed_materialization(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    same_content: bool,
) -> None:
    """A filesystem rename that folds case must not publish a partial directory."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        payload = root / "payload"
        payload.mkdir()
        (payload / "A.txt").write_text("first")
        (payload / "a.txt").write_text("first" if same_content else "other")
        tracked = runner.invoke(cli.cli, ["track", "payload"])
        assert tracked.exit_code == 0, tracked.output
        shutil.rmtree(payload)
        payload.mkdir()
        (payload / "keep.txt").write_text("old workspace")
        replace = pathlib.Path.replace

        def folded_replace(path: pathlib.Path, target: str | os.PathLike[str]) -> pathlib.Path:
            destination = pathlib.Path(target)
            if destination.name == "A.txt" and destination.parent.name.startswith(
                ".pivot_restore_"
            ):
                destination = destination.with_name("a.txt")
            return replace(path, destination)

        monkeypatch.setattr(pathlib.Path, "replace", folded_replace)
        result = runner.invoke(cli.cli, ["checkout", "--all", "--force", "--checkout-mode", "copy"])

        assert result.exit_code == 1, result.output
        assert "expected 2 file(s), found 1" in result.output
        assert "Restored" not in result.output
        assert {member.name: member.read_text() for member in payload.iterdir()} == {
            "keep.txt": "old workspace"
        }
        assert not list(root.glob(".pivot_restore_*"))


def test_update_leaves_unreadable_tracked_payload_members_alone(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """An ordinary tracked directory can hold .pvt-named content without making it an import."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        payload = root / "payload"
        payload.mkdir()
        (payload / "notes.pvt").write_text("ordinary payload")
        tracked = runner.invoke(cli.cli, ["track", "payload"])
        assert tracked.exit_code == 0, tracked.output

        result = runner.invoke(cli.cli, ["update", "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "No imports found" in result.output
        assert (payload / "notes.pvt").read_text() == "ordinary payload"


@contextlib.contextmanager
def _helper_casefold_directory(root: pathlib.Path) -> Generator[pathlib.Path]:
    """Mount ext4 with casefold enabled; CI has the same mount grant as its timestamp tests."""
    image = root / "casefold.ext4"
    with image.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024)
    subprocess.run(["/usr/sbin/mkfs.ext4", "-q", "-F", "-O", "casefold", str(image)], check=True)
    mount = root / "casefold"
    mount.mkdir()
    subprocess.run(["sudo", "-n", "mount", "-o", "loop", str(image), str(mount)], check=True)
    try:
        subprocess.run(
            ["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", str(mount)], check=True
        )
        directory = mount / "folded"
        directory.mkdir()
        subprocess.run(["chattr", "+F", str(directory)], check=True)
        yield directory
    finally:
        subprocess.run(["sudo", "-n", "umount", str(mount)], check=True)


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not os.environ.get("CI"),
    reason="requires /dev/loop-control and sudo; loop devices are unavailable locally; enforced in CI",
)
@pytest.mark.parametrize("same_content", [False, True], ids=["different-bytes", "same-bytes"])
def test_checkout_refuses_case_collapsed_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, same_content: bool
) -> None:
    """Two listed names must remain two files, even when their hashes are identical."""
    with isolated_pivot_dir(runner, tmp_path) as root:
        payload = root / "payload"
        payload.mkdir()
        (payload / "A.txt").write_text("first")
        (payload / "a.txt").write_text("first" if same_content else "other")
        tracked = runner.invoke(cli.cli, ["track", "payload"])
        assert tracked.exit_code == 0, tracked.output
        shutil.rmtree(payload)
        with _helper_casefold_directory(root) as destination:
            probe = destination / "CaseProbe"
            probe.write_text("casefold enabled")
            assert (destination / "caseprobe").read_text() == "casefold enabled"
            probe.unlink()
            shutil.move(root / "payload.pvt", destination / "payload.pvt")
            shutil.move(root / "payload.pvt.listing", destination / "payload.pvt.listing")

            result = runner.invoke(
                cli.cli, ["checkout", "--all", "--force", "--checkout-mode", "copy"]
            )

            assert result.exit_code == 1, result.output
            assert "payload" in result.output
            assert "expected 2 file(s), found 1" in result.output
            assert "Restored" not in result.output
            assert not (destination / "payload").exists()
            assert not list(destination.glob(".pivot_restore_*"))

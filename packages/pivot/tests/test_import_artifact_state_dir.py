"""`pivot import` reads a source's locks where the source's Pivot wrote them."""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

import pytest
import yaml

import conftest
from pivot import import_artifact
from pivot.remote import github
from tests import helpers

if TYPE_CHECKING:
    import pathlib

    from pytest_mock import MockerFixture

# Imports retain Pivot 0.5.2 source layouts with a configured state directory.

_HASH = "0123456789abcdef"


_LOCK = {
    "code_manifest": {},
    "params": {},
    "deps": [],
    "outs": [{"key": None, "tag": "data", "hash": _HASH, "size": 5}],
}


def _git_ok(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _source(tmp_path: pathlib.Path, state_dir: str) -> pathlib.Path:
    source = tmp_path / "source"
    source.mkdir()
    conftest.init_git_repo(source)
    _git_ok(source, "config", "uploadArchive.allowUnreachable", "true")
    _git_ok(source, "checkout", "-b", "main")
    (source / ".pivot" / "stages" / "smoke").mkdir(parents=True)
    (source / ".pivot" / "stages" / "smoke" / "total.lock").write_text(yaml.safe_dump(_LOCK))
    (source / ".pivot" / "config.yaml").write_text(
        "remotes:\n  origin: s3://source-cache\ncore:\n  state_dir: " + state_dir + "\n"
    )
    _git_ok(source, "add", ".")
    _git_ok(source, "commit", "-m", "source")
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
    _git_ok(source, "add", ".")
    _git_ok(source, "commit", "-m", "symlinked state directory")

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


# `pivot import` reads a source's locks where Pivot 0.5.2 read them.
#
# `pivot import --no-download` from a local git source whose `.pivot/config.yaml` has a problem,
# or that 0.5.2 configured with a valid `core.state_dir` while its locks stayed in `.pivot/stages`.
# Pivot 0.5.2 read every one of these sources' locks from `.pivot/stages` at the revision and
# exited 0. The head does the same and prints one warning naming the source, except for the
# control, which has no state-directory problem and prints none.
#
# No decoy and no source worktree file is read: every decoy lock, and the source worktree's own
# lock after the commit, records `_DECOY_HASH`, so a read of any of them puts that hash in the
# `.pvt` file.

_DECOY_HASH = "fedcba9876543210"


_DECOY_LOCK = {**_LOCK, "outs": [{"key": None, "tag": "data", "hash": _DECOY_HASH, "size": 5}]}


_REMOTES = "remotes:\n  origin: s3://source-cache\n"


_WARNING = "Ignoring core.state_dir"


# name -> (config text, symlink to create in the source or None).
# {E} is the decoy directory beside the source.
_CASES: dict[str, tuple[str, tuple[str, str] | None]] = {
    "dup-key": ("core:\n  max_workers: 1\n  max_workers: 2\n" + _REMOTES, None),
    "dup-state-dir": ("core:\n  state_dir: {E}\n  state_dir: elsewhere\n" + _REMOTES, None),
    "non-string": ("core:\n  state_dir: 5\n" + _REMOTES, None),
    "empty": ("core:\n  state_dir: ''\n" + _REMOTES, None),
    "absolute": ("core:\n  state_dir: {E}\n" + _REMOTES, None),
    "escaping": ("core:\n  state_dir: ../elsewhere\n" + _REMOTES, None),
    "symlink-out": ("core:\n  state_dir: st-link\n" + _REMOTES, ("st-link", "../elsewhere")),
    "control-none": ("core:\n  max_workers: 1\n" + _REMOTES, None),
    "valid-052-layout": ("core:\n  state_dir: altstate\n" + _REMOTES, None),
    "symlink-parent": ("core:\n  state_dir: lnk/sub\n" + _REMOTES, ("lnk", "real")),
}


def _git_in(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> None:
    result = helpers.run_cli(cwd, home, "git", *args)
    assert result.returncode == 0, (args, result.stderr)


@pytest.mark.parametrize("case", list(_CASES))
def test_import_reads_the_052_lock_location(tmp_path: pathlib.Path, case: str) -> None:
    home = tmp_path / "home"
    home.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    decoy = tmp_path / "elsewhere"
    (decoy / "stages" / "test").mkdir(parents=True)
    (decoy / "stages" / "test" / "writes_state.lock").write_text(yaml.safe_dump(_DECOY_LOCK))
    _git_in(source, home, "init", "-q", "-b", "main")
    _git_in(source, home, "config", "user.email", "t@example.com")
    _git_in(source, home, "config", "user.name", "t")
    _git_in(source, home, "config", "uploadArchive.allowUnreachable", "true")
    source_lock = source / ".pivot" / "stages" / "test" / "writes_state.lock"
    source_lock.parent.mkdir(parents=True)
    source_lock.write_text(yaml.safe_dump(_LOCK))
    text, link = _CASES[case]
    (source / ".pivot" / "config.yaml").write_text(text.replace("{E}", str(decoy)))
    if link is not None:
        (source / link[0]).symlink_to(link[1])
    if case == "symlink-parent":
        # The configured directory's real copy, with a lock that 0.5.2 never reads.
        real_lock = source / "real" / "sub" / "stages" / "test" / "writes_state.lock"
        real_lock.parent.mkdir(parents=True)
        real_lock.write_text(yaml.safe_dump(_DECOY_LOCK))
    _git_in(source, home, "add", "-A")
    _git_in(source, home, "commit", "-qm", "source")
    # Import reads the revision, never the source worktree.
    source_lock.write_text(yaml.safe_dump(_DECOY_LOCK))
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    _git_in(consumer, home, "init", "-q", "-b", "main")
    assert helpers.run_pivot(consumer, home, "init").returncode == 0

    result = helpers.run_pivot(
        consumer,
        home,
        "import",
        str(source),
        "test/writes_state",
        "--out",
        "got.txt",
        "--no-download",
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Traceback" not in output, output
    pvt = (consumer / "got.txt.pvt").read_text()
    assert _HASH in pvt, pvt
    assert _DECOY_HASH not in pvt, pvt
    expected_warnings = 0 if case == "control-none" else 1
    assert result.stderr.count(_WARNING) == expected_warnings, result.stderr

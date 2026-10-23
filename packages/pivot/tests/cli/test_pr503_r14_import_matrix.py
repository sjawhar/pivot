"""`pivot import` reads a source's locks where Pivot 0.5.2 read them.

`pivot import --no-download` from a local git source whose `.pivot/config.yaml` has a problem,
or that 0.5.2 configured with a valid `core.state_dir` while its locks stayed in `.pivot/stages`.
Pivot 0.5.2 read every one of these sources' locks from `.pivot/stages` at the revision and
exited 0. The head does the same and prints one warning naming the source, except for the
control, which has no state-directory problem and prints none.

No decoy and no source worktree file is read: every decoy lock, and the source worktree's own
lock after the commit, records `_DECOY_HASH`, so a read of any of them puts that hash in the
`.pvt` file.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest
import yaml

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_HASH = "0123456789abcdef"
_DECOY_HASH = "fedcba9876543210"
_LOCK = {
    "code_manifest": {},
    "params": {},
    "deps": [],
    "outs": [{"key": None, "tag": "data", "hash": _HASH, "size": 5}],
}
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


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIVOT")}
    env["HOME"] = str(home)
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


def _git(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> None:
    result = _run(cwd, home, "git", *args)
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
    _git(source, home, "init", "-q", "-b", "main")
    _git(source, home, "config", "user.email", "t@example.com")
    _git(source, home, "config", "user.name", "t")
    _git(source, home, "config", "uploadArchive.allowUnreachable", "true")
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
    _git(source, home, "add", "-A")
    _git(source, home, "commit", "-qm", "source")
    # Import reads the revision, never the source worktree.
    source_lock.write_text(yaml.safe_dump(_DECOY_LOCK))
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    _git(consumer, home, "init", "-q", "-b", "main")
    assert _run(consumer, home, _PIVOT, "init").returncode == 0

    result = _run(
        consumer,
        home,
        _PIVOT,
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

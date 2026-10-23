"""Commands on a project that Pivot 0.5.2 laid out, after core.state_dir is set.

Pivot 0.5.2 (`init`, `config set core.state_dir`, `repro`, then `git add -A` and a commit) left
locks, cache and run history in `.pivot`; in the configured state directory only `locks/` and
`state.lmdb`, with no `.gitignore`, so a relative one is committed with `.pivot/config.yaml`.
This test writes that layout. The default `.pivot` layout is byte-identical to 0.5.2's, so the
head builds it. Each case (none, a relative `altstate`, an absolute directory outside the
project) then runs one command per copy. Every command keeps 0.5.2's exit code (0). Configured
layouts read the legacy locks and cache and print the migration warning once; the unconfigured
layout prints none.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""
_WARNING = "Pivot 0.5.2 stored locks and cache"

# name -> core.state_dir value ({outside} is a directory beside the project).
_CASES: dict[str, str | None] = {
    "none": None,
    "altstate": "altstate",
    "abs-outside": "{outside}/st",
}

# Commands, with whether each reads stage locks or cache objects.
_COMMANDS: dict[str, tuple[tuple[str, ...], bool]] = {
    "status": (("status",), True),
    "verify": (("verify",), True),
    "list": (("list",), False),
    "checkout-deleted": (("checkout", "test/process"), True),
    "get-rev": (("get", "--rev", "HEAD", "test/process", "-o", "restored.txt"), True),
    "repro": (("repro",), True),
    "params-diff": (("params", "diff"), True),
    "config-get-cache": (("config", "get", "cache.dir"), False),
}


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIVOT")}
    env["HOME"] = str(home)
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    directory = tmp_path_factory.mktemp("legacy-base")
    home = directory / "home"
    home.mkdir()
    root = directory / "project"
    root.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
    ):
        assert _run(root, home, "git", *args).returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert _run(root, home, _PIVOT, "repro").returncode == 0
    assert _run(root, home, "git", "add", "-A").returncode == 0
    assert _run(root, home, "git", "commit", "-qm", "legacy layout").returncode == 0
    return directory


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("command", list(_COMMANDS))
def test_legacy_layout_command_matrix(
    base: pathlib.Path, tmp_path: pathlib.Path, case: str, command: str
) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(base, copy, symlinks=True)
    (copy / "outside").mkdir()
    root = copy / "project"
    state_dir = _CASES[case]
    if state_dir is not None:
        value = state_dir.format(outside=copy / "outside")
        (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")
        # 0.5.2 wrote only its state database and artifact locks to the configured directory.
        configured = root / value
        (configured / "locks").mkdir(parents=True)
        shutil.copytree(root / ".pivot" / "state.lmdb", configured / "state.lmdb")
        assert _run(root, copy / "home", "git", "add", "-A").returncode == 0
        assert _run(root, copy / "home", "git", "commit", "-qm", "configured").returncode == 0
    if command == "checkout-deleted":
        (root / "data" / "test" / "process.txt").unlink()
    args, reads_legacy = _COMMANDS[command]

    result = _run(root, copy / "home", _PIVOT, *args)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Traceback" not in output, output
    if state_dir is not None and reads_legacy:
        assert result.stderr.count(_WARNING) == 1, result.stderr
    else:
        assert _WARNING not in result.stderr, result.stderr
    match command:
        case "status":
            assert "1 cached" in result.stdout, result.stdout
        case "repro":
            # 0.5.2 skipped the stage from its committed lock ("0 ran, 1 cached").
            assert "test/process cached" in result.stdout, result.stdout
            assert "0 ran, 1 cached" in output, output
        case "checkout-deleted":
            assert (root / "data" / "test" / "process.txt").read_text() == "INPUT\n"
        case "get-rev":
            assert (root / "restored.txt").read_text() == "INPUT\n"
        case _:
            pass

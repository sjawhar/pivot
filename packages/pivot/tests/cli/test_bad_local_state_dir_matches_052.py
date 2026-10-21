"""Every command in `pivot --help` exits as on 0.5.2 when the local `core.state_dir` cannot be used.

Each case replaces `.pivot/config.yaml` of a reproduced, committed project. Either the file
cannot be read (`core: [unclosed`), or its `core.state_dir` is not a path (`5`, `True`, `0x10`),
or it names a directory that cannot exist or be created: a regular file, `/root/x` for a
non-root user, and a child of a read-only directory. A command 0.5.2 ran still runs with 0.5.2's
exit code. A command 0.5.2 failed still fails, in one `Error:` line and without a traceback.

`_052_EXIT_CODES` holds the released package's codes. Running this file captures them again:

    uv run --no-project --isolated --with pivot==0.5.2 --with pytest python <this file>

Re-capture whenever a command or a case is added. Pass `--pivot <binary>` to print another
build's codes with the same scenarios.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

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

_CASES = {
    "invalid-yaml": "core: [unclosed\n",
    "integer": "core:\n  state_dir: 5\n",
    "boolean": "core:\n  state_dir: True\n",
    "hex": "core:\n  state_dir: 0x10\n",
    "regular-file": "core:\n  state_dir: afile\n",
    "unwritable-absolute": "core:\n  state_dir: /root/x\n",
    "read-only-parent": "core:\n  state_dir: ro/state\n",
}

# One invocation per command in `pivot --help`, each group with its first subcommand. Every
# argument keeps the command local: no remote is configured, and import reads a missing path.
_COMMANDS: dict[str, tuple[str, ...]] = {
    "check-ignore": ("check-ignore", "cache.pyc"),
    "checkout": ("checkout",),
    "commit": ("commit",),
    "completion": ("completion", "bash"),
    "config": ("config", "list"),
    "dag": ("dag",),
    "diff": ("diff", "data/raw/input.txt"),
    "doctor": ("doctor",),
    "fetch": ("fetch",),
    "fingerprint": ("fingerprint", "reset"),
    "get": ("get", "--rev", "HEAD", "data/raw/input.txt"),
    "history": ("history",),
    "import": ("import", "/nonexistent-source-repo", "out.txt"),
    "init": ("init", "--force"),
    "list": ("list",),
    "metrics": ("metrics", "show"),
    "params": ("params", "show"),
    "plots": ("plots", "show"),
    "pull": ("pull",),
    "push": ("push",),
    "remote": ("remote", "list"),
    "repro": ("repro",),
    "run": ("run", "test/process"),
    "show": ("show",),
    "status": ("status",),
    "track": ("track", "data/raw/input.txt"),
    "update": ("update",),
    "verify": ("verify",),
}

# The tail of the one warning a command that runs prints about the unusable configuration.
_DEFAULT_WARNING = "; commands that do not need it use the default .pivot."

# Captured from pivot 0.5.2 by running this file (see the module docstring).
_052_EXIT_CODES: dict[str, dict[str, int]] = {
    "invalid-yaml": {
        "check-ignore": 0,
        "checkout": 1,
        "commit": 1,
        "completion": 0,
        "config": 1,
        "dag": 0,
        "diff": 1,
        "doctor": 1,
        "fetch": 1,
        "fingerprint": 1,
        "get": 1,
        "history": 1,
        "import": 1,
        "init": 0,
        "list": 0,
        "metrics": 1,
        "params": 1,
        "plots": 0,
        "pull": 1,
        "push": 1,
        "remote": 0,
        "repro": 1,
        "run": 1,
        "show": 1,
        "status": 1,
        "track": 1,
        "update": 0,
        "verify": 1,
    },
    "integer": {
        "check-ignore": 0,
        "checkout": 1,
        "commit": 1,
        "completion": 0,
        "config": 1,
        "dag": 0,
        "diff": 1,
        "doctor": 1,
        "fetch": 1,
        "fingerprint": 1,
        "get": 1,
        "history": 1,
        "import": 1,
        "init": 0,
        "list": 0,
        "metrics": 1,
        "params": 1,
        "plots": 0,
        "pull": 1,
        "push": 1,
        "remote": 0,
        "repro": 1,
        "run": 1,
        "show": 1,
        "status": 1,
        "track": 1,
        "update": 0,
        "verify": 1,
    },
    "boolean": {
        "check-ignore": 0,
        "checkout": 1,
        "commit": 1,
        "completion": 0,
        "config": 1,
        "dag": 0,
        "diff": 1,
        "doctor": 1,
        "fetch": 1,
        "fingerprint": 1,
        "get": 1,
        "history": 1,
        "import": 1,
        "init": 0,
        "list": 0,
        "metrics": 1,
        "params": 1,
        "plots": 0,
        "pull": 1,
        "push": 1,
        "remote": 0,
        "repro": 1,
        "run": 1,
        "show": 1,
        "status": 1,
        "track": 1,
        "update": 0,
        "verify": 1,
    },
    "hex": {
        "check-ignore": 0,
        "checkout": 1,
        "commit": 1,
        "completion": 0,
        "config": 1,
        "dag": 0,
        "diff": 1,
        "doctor": 1,
        "fetch": 1,
        "fingerprint": 1,
        "get": 1,
        "history": 1,
        "import": 1,
        "init": 0,
        "list": 0,
        "metrics": 1,
        "params": 1,
        "plots": 0,
        "pull": 1,
        "push": 1,
        "remote": 0,
        "repro": 1,
        "run": 1,
        "show": 1,
        "status": 1,
        "track": 1,
        "update": 0,
        "verify": 1,
    },
    "regular-file": {
        "check-ignore": 0,
        "checkout": 1,
        "commit": 0,
        "completion": 0,
        "config": 0,
        "dag": 0,
        "diff": 0,
        "doctor": 0,
        "fetch": 1,
        "fingerprint": 1,
        "get": 0,
        "history": 1,
        "import": 1,
        "init": 0,
        "list": 0,
        "metrics": 0,
        "params": 0,
        "plots": 0,
        "pull": 1,
        "push": 1,
        "remote": 0,
        "repro": 1,
        "run": 1,
        "show": 1,
        "status": 1,
        "track": 0,
        "update": 0,
        "verify": 0,
    },
    "unwritable-absolute": {
        "check-ignore": 0,
        "checkout": 1,
        "commit": 0,
        "completion": 0,
        "config": 0,
        "dag": 0,
        "diff": 0,
        "doctor": 0,
        "fetch": 1,
        "fingerprint": 1,
        "get": 0,
        "history": 1,
        "import": 1,
        "init": 0,
        "list": 0,
        "metrics": 0,
        "params": 0,
        "plots": 0,
        "pull": 1,
        "push": 1,
        "remote": 0,
        "repro": 1,
        "run": 1,
        "show": 1,
        "status": 1,
        "track": 0,
        "update": 0,
        "verify": 0,
    },
    "read-only-parent": {
        "check-ignore": 0,
        "checkout": 1,
        "commit": 0,
        "completion": 0,
        "config": 0,
        "dag": 0,
        "diff": 0,
        "doctor": 0,
        "fetch": 1,
        "fingerprint": 1,
        "get": 0,
        "history": 1,
        "import": 1,
        "init": 0,
        "list": 0,
        "metrics": 0,
        "params": 0,
        "plots": 0,
        "pull": 1,
        "push": 1,
        "remote": 0,
        "repro": 1,
        "run": 1,
        "show": 1,
        "status": 1,
        "track": 0,
        "update": 0,
        "verify": 0,
    },
}


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


def _build_base(directory: pathlib.Path, pivot: str) -> pathlib.Path:
    """A committed project that ``pivot`` has reproduced, beside its own HOME."""
    home = directory / "home"
    home.mkdir(parents=True)
    root = directory / "project"
    root.mkdir()
    for git_args in (
        ("init", "-q"),
        ("config", "user.email", "test@example.com"),
        ("config", "user.name", "test"),
    ):
        assert _run(root, home, "git", *git_args).returncode == 0
    assert _run(root, home, pivot, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert _run(root, home, pivot, "repro").returncode == 0
    assert _run(root, home, "git", "add", "-A").returncode == 0
    assert _run(root, home, "git", "commit", "-qm", "base").returncode == 0
    return directory


def _case(
    base: pathlib.Path, destination: pathlib.Path, case: str
) -> tuple[pathlib.Path, pathlib.Path]:
    """Copy the base project to ``destination`` and give it the case's config."""
    copy = destination / "copy"
    shutil.copytree(base, copy, symlinks=True)
    root = copy / "project"
    if case == "regular-file":
        (root / "afile").write_text("not a directory\n")
    elif case == "read-only-parent":
        (root / "ro").mkdir()
        (root / "ro").chmod(0o555)
    (root / ".pivot" / "config.yaml").write_text(_CASES[case])
    return copy / "home", root


@pytest.fixture(scope="session")
def base_project(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return _build_base(tmp_path_factory.mktemp("base"), _PIVOT)


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("command", list(_COMMANDS))
def test_command_exits_as_on_052(
    base_project: pathlib.Path, tmp_path: pathlib.Path, case: str, command: str
) -> None:
    """Same exit code as 0.5.2; a failure is one `Error:` line, never a traceback."""
    if os.geteuid() == 0 and case in {"unwritable-absolute", "read-only-parent"}:
        pytest.skip("root ignores permission bits")
    home, root = _case(base_project, tmp_path, case)

    result = _run(root, home, _PIVOT, *_COMMANDS[command])

    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    assert result.returncode == _052_EXIT_CODES[case][command], output
    if result.returncode != 0:
        errors = [line for line in result.stderr.splitlines() if line.startswith("Error")]
        assert len(errors) == 1, result.stderr


def test_the_table_covers_every_help_command() -> None:
    """A command added to `pivot --help` must be added here, with 0.5.2's codes."""
    from pivot import cli

    assert set(_COMMANDS) == set(cli._LAZY_COMMANDS)
    assert set(_052_EXIT_CODES) == set(_CASES)
    for codes in _052_EXIT_CODES.values():
        assert set(codes) <= set(_COMMANDS)


def _capture(pivot: str) -> None:
    """Print each case's exit codes under ``pivot`` as the literal table, then one line per run."""
    with tempfile.TemporaryDirectory(prefix="cap") as tmp:
        base = _build_base(pathlib.Path(tmp) / "base", pivot)
        runs = [(case, command) for case in _CASES for command in _COMMANDS]

        def run_one(index: int) -> tuple[str, str, subprocess.CompletedProcess[str]]:
            case, command = runs[index]
            home, root = _case(base, pathlib.Path(tmp) / str(index), case)
            return case, command, _run(root, home, pivot, *_COMMANDS[command])

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run_one, range(len(runs))))
        # The read-only-parent case leaves a 0555 directory that would stop the cleanup.
        subprocess.run(["chmod", "-R", "u+w", tmp], check=False)
    table = {case: dict[str, int]() for case in _CASES}
    for case, command, result in results:
        table[case][command] = result.returncode
    print(json.dumps(table, indent=4))
    for case, command, result in results:
        errors = [line for line in result.stderr.splitlines() if line.startswith("Error")]
        traceback = "Traceback" in result.stdout + result.stderr
        first = errors[0][:140] if errors else ""
        warnings = result.stderr.count(_DEFAULT_WARNING)
        detail = f"rc={result.returncode} errors={len(errors)} {traceback=} {warnings=} {first!r}"
        print(f"{case:20} {command:13} {detail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pivot", default=_PIVOT)
    _capture(parser.parse_args().pivot)

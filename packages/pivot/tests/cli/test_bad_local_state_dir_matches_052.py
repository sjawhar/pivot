"""Every command in `pivot --help` exits as on 0.5.2 when the local `core.state_dir` cannot be used.

Each case replaces `.pivot/config.yaml` of a reproduced, committed project. Either the file
cannot be read (`core: [unclosed`), or its `core.state_dir` is not a path (`5`, `True`, `0x10`),
or it names a directory that cannot exist or be created: a regular file, `/root/x` for a
non-root user, and a child of a read-only directory. A command 0.5.2 ran still runs with 0.5.2's
exit code. A command 0.5.2 failed still fails, in one `Error:` line and without a traceback.

`_052_EXIT_CODES` holds the released package's codes. Running this file from `packages/pivot`
captures them again:

    uv run --no-project --isolated --with pivot==0.5.2 --with pytest \
        python -m tests.cli.test_bad_local_state_dir_matches_052

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
import tempfile

import pytest

from tests import helpers

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def process(value: str) -> str:
    return value.upper()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt", t=str))
"""

_CASES = {
    "empty": "core:\n  state_dir: ''\n",
    "invalid-yaml": "core: [unclosed\n",
    "integer": "core:\n  state_dir: 5\n",
    "boolean": "core:\n  state_dir: True\n",
    "hex": "core:\n  state_dir: 0x10\n",
    "regular-file": "core:\n  state_dir: afile\n",
    "unwritable-absolute": "core:\n  state_dir: /root/x\n",
    "read-only-parent": "core:\n  state_dir: ro/state\n",
    # Three layouts of a configured directory this build filled, then made unwritable. Its
    # `state.lmdb` stays writable in each; Pivot 0.5.2 ran a command only in the first, where
    # the lock directory it opens is there too.
    "lost-write-permission": "core:\n  state_dir: s1\n",
    "lost-write-permission-no-locks": "core:\n  state_dir: s1\n",
    "unwritable-empty-database": "core:\n  state_dir: s1\n",
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
_DEFAULT_WARNING = "; Pivot uses the default .pivot."

# Captured from pivot 0.5.2 by running this file (see the module docstring).
_052_EXIT_CODES: dict[str, dict[str, int]] = {
    "empty": {
        "check-ignore": 0,
        "checkout": 0,
        "commit": 0,
        "completion": 0,
        "config": 0,
        "dag": 0,
        "diff": 0,
        "doctor": 0,
        "fetch": 1,
        "fingerprint": 0,
        "get": 0,
        "history": 0,
        "import": 1,
        "init": 0,
        "list": 0,
        "metrics": 0,
        "params": 0,
        "plots": 0,
        "pull": 1,
        "push": 1,
        "remote": 0,
        "repro": 0,
        "run": 0,
        "show": 1,
        "status": 0,
        "track": 0,
        "update": 0,
        "verify": 0,
    },
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
    "lost-write-permission": {
        "check-ignore": 0,
        "checkout": 0,
        "commit": 0,
        "completion": 0,
        "config": 0,
        "dag": 0,
        "diff": 0,
        "doctor": 0,
        "fetch": 1,
        "fingerprint": 0,
        "get": 0,
        "history": 0,
        "import": 1,
        "init": 0,
        "list": 0,
        "metrics": 0,
        "params": 0,
        "plots": 0,
        "pull": 1,
        "push": 1,
        "remote": 0,
        "repro": 0,
        "run": 0,
        "show": 0,
        "status": 0,
        "track": 0,
        "update": 0,
        "verify": 0,
    },
    "lost-write-permission-no-locks": {
        "check-ignore": 0,
        "checkout": 1,
        "commit": 0,
        "completion": 0,
        "config": 0,
        "dag": 0,
        "diff": 0,
        "doctor": 0,
        "fetch": 1,
        "fingerprint": 0,
        "get": 0,
        "history": 0,
        "import": 1,
        "init": 0,
        "list": 0,
        "metrics": 0,
        "params": 0,
        "plots": 0,
        "pull": 1,
        "push": 1,
        "remote": 0,
        "repro": 0,
        "run": 0,
        "show": 0,
        "status": 0,
        "track": 0,
        "update": 0,
        "verify": 0,
    },
    "unwritable-empty-database": {
        "check-ignore": 0,
        "checkout": 1,
        "commit": 0,
        "completion": 0,
        "config": 0,
        "dag": 0,
        "diff": 0,
        "doctor": 0,
        "fetch": 1,
        "fingerprint": 0,
        "get": 0,
        "history": 0,
        "import": 1,
        "init": 0,
        "list": 0,
        "metrics": 0,
        "params": 0,
        "plots": 0,
        "pull": 1,
        "push": 1,
        "remote": 0,
        "repro": 0,
        "run": 0,
        "show": 1,
        "status": 0,
        "track": 0,
        "update": 0,
        "verify": 0,
    },
}


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
        assert helpers.run_cli(root, home, "git", *git_args).returncode == 0
    assert helpers.run_cli(root, home, pivot, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    assert helpers.run_cli(root, home, pivot, "repro").returncode == 0
    assert helpers.run_cli(root, home, "git", "add", "-A").returncode == 0
    assert helpers.run_cli(root, home, "git", "commit", "-qm", "base").returncode == 0
    return directory


def _case(
    base: pathlib.Path, destination: pathlib.Path, case: str, pivot: str = helpers.PIVOT
) -> tuple[pathlib.Path, pathlib.Path]:
    """Copy the base project to ``destination`` and give it the case's config."""
    copy = destination / "copy"
    shutil.copytree(base, copy, symlinks=True)
    home = copy / "home"
    root = copy / "project"
    if case == "regular-file":
        (root / "afile").write_text("not a directory\n")
    elif case == "read-only-parent":
        (root / "ro").mkdir()
        (root / "ro").chmod(0o555)
    (root / ".pivot" / "config.yaml").write_text(_CASES[case])
    if case.startswith(("lost-write-permission", "unwritable-empty-database")):
        if case == "unwritable-empty-database":
            (root / "s1" / "state.lmdb").mkdir(parents=True)
        else:
            assert helpers.run_cli(root, home, pivot, "repro").returncode == 0
            # The lock directory Pivot 0.5.2 opens under the state directory. Each build's own
            # run leaves a different layout there, so the case names this one.
            (root / "s1" / "locks").mkdir(exist_ok=True)
        if case == "lost-write-permission-no-locks":
            for name in ("stages", "cache", "locks"):
                shutil.rmtree(root / "s1" / name, ignore_errors=True)
        (root / "s1").chmod(0o555)
    return home, root


@pytest.fixture(scope="session")
def base_project(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return _build_base(tmp_path_factory.mktemp("base"), helpers.PIVOT)


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("command", list(_COMMANDS))
def test_command_exits_as_on_052(
    base_project: pathlib.Path, tmp_path: pathlib.Path, case: str, command: str
) -> None:
    """Same exit code as 0.5.2; a failure is one `Error:` line, never a traceback."""
    if os.geteuid() == 0 and case in {
        "unwritable-absolute",
        "read-only-parent",
        "lost-write-permission",
        "lost-write-permission-no-locks",
        "unwritable-empty-database",
    }:
        pytest.skip("root ignores permission bits")
    home, root = _case(base_project, tmp_path, case)
    result = helpers.run_pivot(root, home, *_COMMANDS[command])
    output = result.stdout + result.stderr
    assert "Traceback" not in output, output
    # Rulings (coordinator): `show` exits 0 where the base project's run is in `.pivot` and the
    # directory 0.5.2 reads holds none, where 0.5.2 finds no runs and exits 1: it reads a run that
    # exists and creates no state. Under '', 0.5.2 reads the project root (11:16Z). Under an
    # unwritable `s1` holding only an empty database, Pivot reads the runs of both state
    # directories (10-28); every other command there keeps 0.5.2's code, `checkout`'s 1 included.
    show_reads_pivot_runs = command == "show" and case in {"empty", "unwritable-empty-database"}
    # Ruling (coordinator, 10-28): #516 makes `status` read the hash cache read-only and
    # best-effort, so a configured path whose parent is read-only, or that runs through a regular
    # file, no longer fails it. Main 27488ddc exits 0 there too; 0.5.2 exits 1.
    status_reads_no_state = command == "status" and case in {"regular-file", "read-only-parent"}
    expected = (
        0 if show_reads_pivot_runs or status_reads_no_state else _052_EXIT_CODES[case][command]
    )
    assert result.returncode == expected, output
    if result.returncode != 0:
        errors = [line for line in result.stderr.splitlines() if line.startswith("Error")]
        assert len(errors) == 1, result.stderr


@pytest.mark.parametrize("command", list(_COMMANDS))
def test_every_created_configured_state_directory_has_the_template(
    base_project: pathlib.Path, tmp_path: pathlib.Path, command: str
) -> None:
    """A command that creates a configured state directory preserves its state files from git."""
    from pivot import state_dir as state_dir_mod

    copy = tmp_path / "copy"
    shutil.copytree(base_project, copy, symlinks=True)
    home = copy / "home"
    root = copy / "project"
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: fresh\n")

    _ = helpers.run_pivot(root, home, *_COMMANDS[command])

    fresh = root / "fresh"
    if fresh.exists():
        assert (fresh / ".gitignore").read_text() == state_dir_mod.STATE_GITIGNORE_CONTENT


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
            home, root = _case(base, pathlib.Path(tmp) / str(index), case, pivot)
            return case, command, helpers.run_cli(root, home, pivot, *_COMMANDS[command])

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run_one, range(len(runs))))
        # Two cases leave a 0555 directory that would stop the cleanup.
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
    parser.add_argument("--pivot", default=helpers.PIVOT)
    _capture(parser.parse_args().pivot)
